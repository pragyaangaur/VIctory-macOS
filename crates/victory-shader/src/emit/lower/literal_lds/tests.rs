use super::*;
use crate::emit::halves_tests::{literal_lds::paired, parts};

/// Exercise final native admission only. A refusal never enters the lowerer.
fn admission(n: &NativeProgram) -> Result<(), Stop> {
    let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
    let st = crate::structure::split_literal(&p).unwrap();
    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let io = crate::io::raise(&p, n).unwrap();
    let x = parts(n, &p, &st, &res, &io, true, true);
    let StageIo::Cs(cs) = n.io else {
        unreachable!()
    };
    literal_publish::prove(
        &x,
        64, // The LDS proof must validate the native dimensions independently.
        cs.lds_bytes() / 4,
        &HashMap::new(),
    )
    .map(|proof| assert!(proof.is_some()))
}

fn refusal(n: &NativeProgram, expected: &str) {
    let Err(Stop::Fail(err)) = admission(n) else {
        panic!("expected {expected}")
    };
    assert!(err.to_string().contains(expected), "{err}");
}

fn replace(n: &mut NativeProgram, before: u32, after: u32) {
    let word = n
        .code
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .find(|w| **w == before.to_le_bytes())
        .unwrap();
    word.copy_from_slice(&after.to_le_bytes());
}

#[test]
fn literal_lds_admission_refuses_wide_scalar_publication() {
    use crate::emit::halves_tests::literal_publish::wide::fixture;

    for width in [8_u32, 16] {
        let without_lds = fixture(width, ResourceType::Cbv, &[0x7d84_0100], true);
        assert!(admission(&without_lds).is_ok(), "width {width}");

        let mut n = paired([8, 8, 1], 512);
        let load = 0xf420_0016 | width.trailing_zeros() << 18 | 80 << 6;
        replace(&mut n, 0xf424_0796, load); // table CBV to s[80..]
        replace(&mut n, 0x3000_0000, 0xfa00_0004); // NULL offset, immediate byte 4
        replace(&mut n, 0x4a0c_141e, 0x4a0c_1450); // v6 = s80 + v10
        replace(&mut n, 0x4a0e_161f, 0x4a0e_1651); // v7 = s81 + v11
        refusal(&n, "a wide scalar publication with guest LDS");
    }
}

#[test]
fn literal_lds_admission_requires_ticket_and_guest_bounds() {
    let n = paired([8, 8, 1], 32_256);
    assert!(admission(&n).is_ok());
    crate::emit::with_thread_ticket(false, || refusal(&n, "one ticketed wave64"));
    refusal(&paired([8, 8, 1], 32_768), "one ticketed wave64");
    let n = paired([8, 8, 1], 512);
    for (threads, reason) in [
        ([0, 8, 1], "outside one complete wave64"),
        ([32, 1, 1], "outside one complete wave64"),
        (
            [1_073_741_840, 4, 1],
            "overflow in a literal LDS group size",
        ),
    ] {
        let mut invalid = n.clone();
        let StageIo::Cs(cs) = &mut invalid.io else {
            unreachable!()
        };
        cs.threads = threads;
        refusal(&invalid, reason);
    }
    for (from, to, why) in [
        (0xd838_0100, 0xd838_8000, "out-of-range paired LDS"), // second component reaches 512
        (0x3404_1083, 0x3404_1081, "unaligned or out-of-range"), // two-byte lane spacing
        (0xd838_0100, 0xd838_0000, "overlapping writes"),      // components alias
        (0x0007_0602, 0x0007_0605, "overlapping writes"),      // all lanes write the same base
        (0xd838_0100, 0xd8dc_0100, "dominating initialization"), // first phase reads
        (0x3404_1083, 0x3404_3083, "exact entry value"),       // data without address provenance
    ] {
        let mut program = n.clone();
        replace(&mut program, from, to);
        refusal(&program, why);
    }
}

#[test]
fn literal_lds_admission_classifies_each_reachable_operation_and_cap_edge() {
    let n = paired([64, 1, 1], 512);
    for opcode in [
        0xd800_0000, // ds_add_u32
        0xd87c_0000, // ds_write_b16
        0xd83c_0100, // ds_write2st64_b32
        0xd83a_0100, // GDS
        0xd834_0000, // single dword
    ] {
        let mut program = n.clone();
        replace(&mut program, 0xd838_0100, opcode);
        refusal(&program, "unsupported operation in paired literal LDS");
    }
    let mut direct = n.clone();
    replace(&mut direct, 0x7e0a_0280, 0x7e0a_02fe); // src_lds_direct
    refusal(&direct, "unsupported operation in paired literal LDS");
    let mut divergent = n;
    replace(&mut divergent, 0xbe94_0383, 0xb914_0004); // hardware state as trip count
    refusal(&divergent, "branch without common group control flow");
}

#[test]
fn literal_lds_admission_requires_a_defined_thread_id_layout() {
    let mut n = paired([64, 1, 1], 512);
    let StageIo::Cs(cs) = &mut n.io else {
        unreachable!()
    };
    cs.rsrc2 |= 3 << 11;
    refusal(&n, "reserved thread ID layout");
}

#[test]
fn literal_lds_admission_requires_full_dominating_initialization() {
    for setup in [
        &[0xbefe_0481][..], // only lane 0 executes the first store
        &[
            0xf420_0496,
            0xfa00_0000,
            0xbf8c_c07f, // s18 = table CBV word 0
            0xbf07_8012,
            0xbf85_0003, // if s18 != 0, skip the first store and wait
        ],
    ] {
        let mut n = paired([64, 1, 1], 512);
        let mut code: Vec<u32> = n
            .code
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect();
        let store = code.iter().position(|&w| w == 0xd838_0100).unwrap();
        code.insert(store + 3, 0xbefe_04c1); // full EXEC before the partner read
        code.splice(store..store, setup.iter().copied());
        n.code = code.into_iter().flat_map(u32::to_le_bytes).collect();
        refusal(&n, "a paired LDS read without a dominating initialization");
    }
}
