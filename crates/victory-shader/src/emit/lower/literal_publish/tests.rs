use super::*;
use crate::emit::halves_tests::{literal_publish::program, native, parts};
use crate::{io, ir, resource, structure};

fn admission(
    n: &NativeProgram,
    inspect: impl FnOnce(&mut Program, &mut Structured, &mut Resources),
) -> (Program, Result<Option<Proof>, Stop>) {
    let mut p = ir::build(&n.code, true, n.vgprs).unwrap();
    let mut st = structure::split_literal(&p).unwrap();
    let mut res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let io = io::raise(&p, n).unwrap();
    inspect(&mut p, &mut st, &mut res);
    let x = parts(n, &p, &st, &res, &io, true, true);
    let result = prove(&x, 64, 0, &HashMap::new());
    (p, result)
}

#[test]
fn literal_publication_numeric_source_admission() {
    // Query the proof itself. These extra definitions have no memory or control uses.
    let mut words = vec![
        0xbebc_1f00, // PC pair in s[60:61]
        0xbebe_0301, // s62 copies the second root
        0xbebf_0302, // s63 copies group X
    ];
    words.extend(program(&[0x7d84_0100]));
    for root in [
        RootType::Constants,
        RootType::Table,
        RootType::ResourceHeap,
        RootType::SamplerHeap,
    ] {
        let mut n = native(&words, true, 64, 0);
        n.layout.push(crate::native::RootParam {
            root,
            user_data: 1,
            slot: 1,
            ..n.layout[0]
        });
        let StageIo::Cs(cs) = &mut n.io else {
            unreachable!()
        };
        cs.rsrc2 = 2 << 1 | 1 << 7;
        let (p, result) = admission(&n, |_, _, _| {});
        let proof = result.ok().flatten().unwrap();
        for (dword, expected) in [
            (60, false),
            (61, false),
            (62, root == RootType::Constants),
            (63, true),
        ] {
            let v = p.blocks[0]
                .insts
                .iter()
                .find_map(|i| i.write(Reg::S(dword)))
                .unwrap();
            assert_eq!(proof.common.contains(&v), expected, "{root:?} s{dword}");
        }
        let table = p.inputs.iter().find(|r| r.0 == Reg::S(0)).unwrap().1;
        assert!(!proof.common.contains(&table));
    }
}

#[test]
fn literal_publication_descriptor_only_admission_checks_all_uses() {
    // Use the original descriptor prologue and final store. Query only admission.
    let fixture = program(&[0x7d84_0100]);
    for (source, allowed) in [(0x7e0c_0280, true), (0x7e0c_0228, false)] {
        let mut words = fixture[..5].to_vec();
        words.extend([0x3420_0085, source]); // store a zero, or descriptor word zero
        words.extend(&fixture[fixture.len() - 3..]);
        let n = native(&words, true, 64, 0);
        let (_, result) = admission(&n, |_, _, _| {});
        if allowed {
            assert!(matches!(result, Ok(None)));
        } else {
            assert!(
                matches!(result, Err(Stop::Fail(crate::ShaderError::Unsupported { what, .. })) if what.contains("descriptor word read as numeric data"))
            );
        }
    }
}

#[test]
fn literal_publication_checks_each_operand_role() {
    let n = native(&program(&[0x7d84_0100]), true, 64, 0);
    // Exercise the validator on role metadata only. The original store's
    // descriptor remains valid when its first word also names a numeric input.
    let (_, result) = admission(&n, |p, _, res| {
        let d = res
            .descriptors
            .iter()
            .find(|d| p.blocks[d.site.0].insts[d.site.1].name() == "buffer_store_dword")
            .unwrap();
        let inst = &mut p.blocks[d.site.0].insts[d.site.1];
        let Fields::Buffer {
            rsrc: Operand::Scalar { code, .. },
            soffset,
            ..
        } = &mut inst.fields
        else {
            unreachable!()
        };
        *soffset = Operand::Scalar {
            code: *code,
            dwords: 1,
        };
    });
    assert!(
        matches!(result, Err(Stop::Fail(crate::ShaderError::Unsupported { what, .. }))
        if what.contains("without guest-exact bits"))
    );
}

#[test]
fn literal_publication_requires_bounded_code_contents() {
    let fixture = program(&[0x7d84_0100]);
    // Drive source admission only. The source metadata varies; no negative
    // program is lowered. The observed word passes through a lane read to a store.
    for (read, width) in [
        (&[0xf404_0902, 0xfa00_0000, 0x7e0c_0225][..], 8), // scalar pair, v6 = s37
        (&[0xdc34_8000, 0x0604_0010][..], 8),              // global dwordx2 to v[6:7]
        (&[0xdc28_8000, 0x0604_0010][..], 2),              // global ushort to v6
        (&[0xdc20_8000, 0x0604_0010][..], 1),              // global ubyte to v6
    ] {
        for strict in [false, true] {
            let mut words = if strict {
                fixture[..fixture.len() - 4].to_vec()
            } else {
                fixture[..5].to_vec()
            };
            words.push(0x3420_0085); // v16 = lane * 32
            let at = 4 * words.len();
            words.extend(read);
            words.extend([0x7e4c_0506, 0x8726_8126, 0x7e0c_0226]); // v6 = first(v6) & 1
            words.extend(&fixture[fixture.len() - 3..]); // store v6 and end
            let n = native(&words, true, 64, 0);
            let end = n.code.len() as u32;
            for (offset, dynamic, proved) in [
                (end - width, false, true),
                (end - width + 1, false, false),
                (end - 4, false, width <= 4),
                (1, false, width <= 2),
                (3, false, width == 1),
                (u32::MAX - 3, false, false),
                (0, true, false),
            ] {
                let (_, result) = admission(&n, |p, _, res| {
                    let (_, address) = res
                        .scalar
                        .iter_mut()
                        .find(|((b, i), _)| p.blocks[*b].insts[*i].machine.offset == at)
                        .unwrap();
                    *address = Address {
                        param: CODE,
                        offset,
                        dynamic: dynamic
                            .then(|| p.inputs.iter().find(|r| r.0 == Reg::V(0)).unwrap().1),
                    };
                });
                let admitted = match result {
                    Ok(proof) => {
                        assert_eq!(proof.is_some(), strict);
                        true
                    }
                    Err(Stop::Fail(crate::ShaderError::Unsupported { what, .. })) => {
                        assert!(
                            what.contains("code read without a proved byte range"),
                            "{what}"
                        );
                        false
                    }
                    _ => panic!("unexpected admission result"),
                };
                assert_eq!(
                    admitted,
                    !strict || proved,
                    "width {width}, offset {offset}, dynamic {dynamic}, strict {strict}"
                );
            }
        }
    }
}

#[test]
fn literal_publication_wide_admission_checks_each_run() {
    use crate::emit::halves_tests::literal_publish::wide::fixture;
    for width in [8, 16] {
        let n = fixture(width, ResourceType::Srv, &[0x7d84_0100], true);
        // Only query the final-IR proof. No refused form reaches emission.
        for (case, admitted) in [
            ("last aligned immediate", true),
            ("negative", false),
            ("unaligned", false),
            ("outside immediate", false),
            ("register", false),
            ("guard", false),
            ("source", false),
            ("missing middle", false),
            ("missing last", false),
            ("extra destination", false),
            ("unaligned destination", false),
            ("special destination", false),
        ] {
            let (_, result) = admission(&n, |p, st, res| {
                let desc = res
                    .descriptors
                    .iter_mut()
                    .find(|d| {
                        p.blocks[d.site.0].insts[d.site.1]
                            .write(Reg::S(80))
                            .is_some()
                    })
                    .unwrap();
                let (b, i) = desc.site;
                let count = p
                    .blocks
                    .iter()
                    .flat_map(|block| &block.insts)
                    .find_map(|inst| {
                        (inst.machine.words[0] == 0xbf06_8214)
                            .then(|| inst.read(Reg::S(20)))
                            .flatten()
                    })
                    .unwrap();
                let inst = &mut p.blocks[b].insts[i];
                let Fields::Smem {
                    data: Some(Operand::Scalar { code, .. }),
                    offset,
                    soffset,
                    ..
                } = &mut inst.fields
                else {
                    unreachable!()
                };
                match case {
                    "last aligned immediate" => *offset = 0xffffc,
                    "negative" => *offset = -4,
                    "unaligned" => *offset = 6,
                    "outside immediate" => *offset = 1 << 20,
                    "register" => {
                        *soffset = Operand::Scalar {
                            code: 20,
                            dwords: 1,
                        };
                        inst.reads.push((Reg::S(20), count));
                    }
                    "guard" => st.guards[b][i] = Some(Bool::FALSE),
                    "source" => desc.source = Source::Constant(vec![0; 4]),
                    "missing middle" => inst.writes.retain(|&(r, _)| r != Reg::S(84)),
                    "missing last" => inst.writes.retain(|&(r, _)| r != Reg::S(79 + width as u8)),
                    "extra destination" => inst.writes.push((Reg::S(99), count)),
                    "unaligned destination" | "special destination" => {
                        let first = if case == "unaligned destination" { 81 } else { 100 };
                        for (reg, _) in &mut inst.writes {
                            let Reg::S(word) = reg else { unreachable!() };
                            *word = first + (*word - *code);
                        }
                        *code = first;
                    }
                    _ => unreachable!(),
                }
            });
            if admitted {
                assert!(matches!(result, Ok(Some(_))), "{case}, width {width}");
            } else {
                assert!(
                    matches!(result, Err(Stop::Fail(ShaderError::Unsupported { what, .. }))
                    if what.contains("unsupported source or width") || what.contains("cached or predicated")),
                    "{case}, width {width}"
                );
            }
        }
    }
}
