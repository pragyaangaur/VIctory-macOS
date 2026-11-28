use super::*;

const MASKS: [u64; 5] = [
    u64::MAX,
    0xffff_ffff,
    0xffff_ffff_0000_0000,
    0,
    0x5050_5050_5050_5050,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Case {
    Interval,
    LastRegister,
    SeparateMove,
    KeyEscapes,
    KeyPhi,
    M0Escapes,
    MaskEscapes,
    SavedEscapes,
    FlagEscapes,
    WrongIndex,
    WrongSelect,
    WrongRetry,
    OverwriteIndex,
    ExtraWork,
    ExtraAccess,
    WrongM0,
    OutOfFile,
    BothIndexed,
}

fn program(max: u32, active: u64, square: bool, case: Case) -> NativeProgram {
    let vop1 = |op: u32, dst: u32, src: u32| 0x7E00_0000 | dst << 17 | op << 9 | src;
    let vop2 = |op: u32, dst: u32, src0: u32, src1: u32| op << 25 | dst << 17 | src1 << 9 | src0;
    let vcmp = |op: u32, src0: u32, src1: u32| 0x7C00_0000 | op << 17 | src1 << 9 | src0;
    let sop1 = |op: u32, dst: u32, src: u32| 0xBE80_0000 | dst << 16 | op << 8 | src;
    let sop2_64 =
        |op: u32, dst: u32, a: u32, b: u32| 0x8000_0000 | op << 23 | dst << 16 | b << 8 | a;
    let set_branch = |words: &mut [u32], op: u32, at: usize, dest: usize| {
        let rel = i16::try_from(dest as isize - at as isize - 1).unwrap();
        words[at] = 0xBF80_0000 | op << 16 | u32::from(rel as u16);
    };
    let v = |r: u32| 256 + r;
    let c = |n: u32| 128 + n;
    let exec = 126;
    let m0 = 124;
    let vcc = 106;
    let base = if case == Case::LastRegister { 29 } else { 8 };
    let mut w = PROLOGUE.to_vec();
    if case == Case::SeparateMove {
        // A separate in-file relative read of v0. v7 is reused by the selector below.
        w.extend([sop1(3, m0, c(0)), vop1(0x43, 7, v(0))]);
    }
    if square {
        w.extend([vop2(0x1a, 1, c(3), 1), vop2(0x25, 0, v(1), 0)]); // v0 = y * 8 + x
    }
    let (bits, selected) = match active {
        u64::MAX => (0, 0),
        0xffff_ffff => (32, 0),
        0xffff_ffff_0000_0000 => (32, 32),
        0 => (0, 1),
        0x5050_5050_5050_5050 => (5, 4),
        _ => unreachable!(),
    };
    w.extend([
        vop2(0x16, 1, c(5), 0),   // v1 = lane >> 5
        vop2(0x1d, 2, v(1), 0),   // v2 = lane ^ v1
        vop2(0x1b, 2, c(1), 2),   // v2 = lo
        vop2(0x16, 3, c(1), 0),   // v3 = lane >> 1
        vop2(0x1b, 3, c(3), 3),   // v3 &= 3
        vop2(0x13, 3, c(max), 3), // v3 = hi
        vop1(1, base, c(10)),     // three private array slots
        vop1(1, base + 1, c(20)),
        vop1(1, base + 2, c(30)),
        vop1(1, 11, 40), // v11: unavailable first dword of UAV descriptor
        vop1(1, 6, c(0)),
        vop2(0x1b, 7, c(bits), 0),
        vcmp(0xd2, c(selected), 7), // EXEC from the flat thread ID
        vcmp(0xc3, v(2), 3),        // v_cmp_le_u32 vcc, v2, v3
        sop1(0x24, 8, vcc),         // save full EXEC and select lanes with lo <= hi
        0,                          // skip loop on the empty selection
        sop1(4, 10, exec),          // keep the pending lanes of the outer loop
    ]);
    let after_entry_branch = w.len() - 2;
    let outer = w.len();
    w.push(sop1(4, exec, 10)); // outer iteration restores pending lanes
    w.push(sop1(4, 12, exec)); // save outer iteration's active set
    for write in [false, true] {
        let waterfall = w.len();
        let key = if !write && case == Case::KeyPhi {
            24
        } else {
            20
        };
        w.extend([
            vop1(2, key, v(2)),  // s20 = current index of first active lane
            vcmp(0xc2, key, 2),  // compare the same index
            sop1(0x24, 14, vcc), // save remaining and select matches
            sop1(3, m0, key),
            if write {
                vop1(0x42, base, v(5))
            } else {
                vop1(0x43, 5, v(base))
            },
            sop2_64(0x13, exec, exec, 14), // retry exactly the remaining nonmatches
            0,
        ]);
        if !write {
            match case {
                Case::WrongIndex => w[waterfall + 1] = vcmp(0xc2, 20, 3),
                Case::WrongSelect => w[waterfall + 2] = sop1(0x24, 14, exec),
                Case::WrongM0 => w[waterfall + 3] = sop1(3, m0, c(1)),
                Case::OutOfFile => w[waterfall + 4] = vop1(0x43, 5, v(30)),
                Case::BothIndexed => w[waterfall + 4] = vop1(0x44, 5, v(8)),
                Case::OverwriteIndex => w[waterfall + 4] = vop1(0x43, 2, v(8)),
                Case::WrongRetry => w[waterfall + 5] = sop2_64(0x13, exec, vcc, exec),
                Case::ExtraWork => w.insert(waterfall + 1, vop2(0x25, 6, c(1), 6)),
                Case::ExtraAccess => w.insert(waterfall + 5, vop1(0x43, 6, v(8))),
                _ => {}
            }
        }
        if write && case == Case::OverwriteIndex {
            // Isolate the read's stability rejection from admission of the separate write loop.
            w.insert(waterfall + 5, vop1(1, 6, v(6)));
        }
        let back = w.len() - 1;
        set_branch(&mut w, 9, back, waterfall);
        w.push(sop1(4, exec, 12)); // next work starts with the iteration's mask
        if !write {
            let observed = match case {
                Case::KeyEscapes => Some(20),
                Case::M0Escapes => Some(m0),
                Case::MaskEscapes => Some(vcc),
                Case::SavedEscapes => Some(14),
                Case::FlagEscapes => Some(253),
                _ => None,
            };
            if let Some(reg) = observed {
                w.push(vop1(1, 6, reg));
            }
            w.push(vop2(0x25, 5, v(0), 5)); // update with a lane-specific value
            w.push(vop2(0x25, 5, c(1), 5));
        }
    }
    w.extend([
        vop2(0x25, 2, c(1), 2),      // current++
        vcmp(0xd4, v(2), 3),         // v_cmpx_gt_u32 exec, v2, v3
        sop2_64(0x15, 10, 10, exec), // retire done lanes
        0,
    ]);
    let back = w.len() - 1;
    set_branch(&mut w, 5, back, outer);
    let done = w.len();
    set_branch(&mut w, 8, after_entry_branch, done);
    w.extend([
        sop1(4, exec, 8),
        vop2(0x1a, 16, c(5), 0), // v16 = 32 bytes per lane
    ]);
    for (slot, reg) in (base as u8..=base as u8 + 2).enumerate() {
        w.extend(store(reg, 4 * slot as u32));
    }
    if case == Case::KeyPhi {
        w.push(vop1(1, 6, 24)); // join the loop's key with the entry value
    }
    if !matches!(
        case,
        Case::Interval | Case::LastRegister | Case::SeparateMove
    ) {
        w.extend(store(6, 12));
    }
    w.push(S_ENDPGM);
    let mut n = native(&w, true, 64, 0);
    if square {
        n.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 11,
            threads: [8, 8, 1],
        });
    }
    n
}

fn lane_result(lane: u32) -> [u32; 3] {
    let lo = (lane ^ (lane >> 5)) & 1;
    let hi = ((lane >> 1) & 3).min(2);
    let mut array = [10, 20, 30];
    for i in lo..=hi {
        array[i as usize] += lane + 1;
    }
    array
}

// CPU model of the two equality waterfalls. Each outer iteration reads
// all active lanes' slots and then updates the same lane/slot pairs.
fn grouped(mut pending: u64) -> [[u32; 3]; 64] {
    let mut array = [[10, 20, 30]; 64];
    let mut index = std::array::from_fn::<_, 64, _>(|lane| (lane ^ (lane >> 5)) & 1);
    let hi = std::array::from_fn::<_, 64, _>(|lane| ((lane >> 1) & 3).min(2));
    for lane in 0..64 {
        if index[lane] > hi[lane] {
            pending &= !(1 << lane);
        }
    }
    while pending != 0 {
        let mut values = [0; 64];
        for write in [false, true] {
            let mut remaining = pending;
            while remaining != 0 {
                let first = remaining.trailing_zeros() as usize;
                let slot = index[first];
                for lane in 0..64 {
                    if remaining & (1 << lane) == 0 || index[lane] != slot {
                        continue;
                    }
                    if write {
                        array[lane][slot] = values[lane] + lane as u32 + 1;
                    } else {
                        values[lane] = array[lane][slot];
                    }
                    remaining &= !(1 << lane);
                }
            }
        }
        for lane in 0..64 {
            if pending & (1 << lane) != 0 {
                index[lane] += 1;
                if index[lane] > hi[lane] {
                    pending &= !(1 << lane);
                }
            }
        }
    }
    array
}

/// Range, resource tracking and normal lowering follow the runtime entry point.
fn lower_program(n: &NativeProgram, split: bool) -> Result<lower::Lowered, crate::ShaderError> {
    let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs)?;
    let st = st?;
    let res = resource::track(&p, &n.layout, n.stage, Some(&st))?;
    let io = io::raise(&p, n)?;
    lower::lower_pruned(
        &lower::Parts {
            native: n,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: split,
            uav_size_level0: false,
            halves_rendezvous: true,
            host_wave32: split,
        },
        None,
        false,
        false,
    )
}

#[test]
fn index_groups_match_independent_lane_intervals() {
    for active in MASKS {
        let actual = grouped(active);
        for (lane, row) in actual.iter().enumerate() {
            let want = if active & (1 << lane) == 0 {
                [10, 20, 30]
            } else {
                lane_result(lane as u32)
            };
            assert_eq!(*row, want, "EXEC={active:#x}, lane={lane}");
        }
    }
    assert_eq!(lane_result(1), [10, 20, 30]);
    assert_eq!(lane_result(6), [17, 27, 37]);
    assert_eq!(lane_result(38), [10, 59, 69]);
}

#[test]
fn bounded_indexed_moves_emit_without_literal_retry() {
    for square in [false, true] {
        for active in MASKS {
            let n = program(2, active, square, Case::Interval);
            lower_program(&n, false).expect("native wave64");
            let split = lower_program(&n, true).expect("indexed wave32");
            // The two lane-local moves require neither a host wave size nor a group exchange.
            assert_eq!(split.wave_size, None);
            assert_eq!(split.barriers, 0);
        }
    }
    let sentinel = program(3, u64::MAX, false, Case::Interval);
    for split in [false, true] {
        let why = lower_program(&sentinel, split)
            .err()
            .expect("lane 6 visits descriptor slot 3");
        assert!(
            why.to_string()
                .contains("a descriptor's address or format read as data"),
            "{why}"
        );
    }
}

#[test]
fn proved_last_register_is_in_file() {
    let n = program(2, u64::MAX, false, Case::LastRegister);
    let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
    let res = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap())).unwrap();
    assert_eq!(res.waterfall.reads.len(), 2);
    lower_program(&n, true).expect("wave32, array ends on final VGPR");
}

#[test]
fn index_proof_uses_the_bounded_per_lane_ir() {
    for max in [2, 3] {
        let n = program(max, u64::MAX, false, Case::Interval);
        let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
        let res = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap())).unwrap();
        assert_eq!(res.waterfall.reads.len(), 2);
        assert_eq!(res.waterfall.compares.len(), 2);
        assert!(res.waterfall.loads.is_empty());
        let expected: Vec<_> = (8..=8 + max as u8).map(ir::Reg::V).collect();
        for i in p.blocks.iter().flat_map(|b| &b.insts) {
            let slots = match i.name() {
                "v_movrels_b32" => i
                    .reads
                    .iter()
                    .take_while(|(r, _)| matches!(r, ir::Reg::V(_)))
                    .map(|r| r.0)
                    .collect::<Vec<_>>(),
                "v_movreld_b32" => i.writes.iter().map(|r| r.0).collect(),
                _ => continue,
            };
            assert_eq!(slots, expected);
        }
        let raw = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
        let st = structure::structure(&raw).unwrap();
        assert_range_refusal(resource::track(&raw, &n.layout, n.stage, Some(&st)).unwrap_err());
        let literal = structure::split_literal(&raw).unwrap();
        let res = resource::track(&raw, &n.layout, n.stage, Some(&literal)).unwrap();
        assert!(res.waterfall.reads.is_empty());
    }
}

fn assert_range_refusal(error: crate::ShaderError) {
    assert!(
        matches!(
            error,
            crate::ShaderError::Unsupported {
                what: "an indexed-waterfall program with a relative VGPR operand not proved inside its register file",
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn waterfall_validation_precedes_host_selection() {
    let n = program(2, u64::MAX, false, Case::OutOfFile);
    let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
    let error = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap())).unwrap_err();
    assert_range_refusal(error);

    let n = program(2, u64::MAX, false, Case::SeparateMove);
    let (mut p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
    let st = st.unwrap();
    assert_eq!(
        resource::track(&p, &n.layout, n.stage, Some(&st))
            .unwrap()
            .waterfall
            .reads
            .len(),
        2
    );
    let separate = p.blocks[0]
        .insts
        .iter()
        .position(|i| i.name() == "v_movrels_b32")
        .unwrap();
    assert_eq!(p.blocks[0].insts[separate].m0_max, Some(0));
    // The actual read stays v0. Missing or too loose a proof must veto the whole rewrite.
    for bound in [None, Some(32)] {
        p.blocks[0].insts[separate].m0_max = bound;
        assert_range_refusal(resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap_err());
    }
}

#[test]
fn index_waterfalls_require_isolated_stable_in_file_access() {
    for case in [
        Case::KeyEscapes,
        Case::KeyPhi,
        Case::M0Escapes,
        Case::MaskEscapes,
        Case::SavedEscapes,
        Case::FlagEscapes,
        Case::WrongIndex,
        Case::WrongSelect,
        Case::WrongRetry,
        Case::OverwriteIndex,
        Case::ExtraWork,
        Case::WrongM0,
        Case::BothIndexed,
    ] {
        let n = program(2, u64::MAX, false, case);
        let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
        let res = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap()))
            .unwrap_or_else(|e| panic!("{case:?}: {e}"));
        let first = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "v_readfirstlane_b32")
            .unwrap()
            .writes[0]
            .1;
        assert!(
            !res.waterfall.reads.contains(&first),
            "{case:?} was rewritten"
        );
        if case == Case::KeyPhi {
            assert!(
                p.blocks
                    .iter()
                    .flat_map(|b| &b.phis)
                    .any(|phi| phi.args.contains(&first))
            );
        }
        let why = lower_program(&n, true)
            .err()
            .unwrap_or_else(|| panic!("{case:?} unexpectedly emitted"));
        assert!(
            why.to_string()
                .contains("a first-lane read of a value that may differ"),
            "{case:?}: {why}"
        );
    }
}

#[test]
fn two_bounded_relative_moves_keep_the_per_half_waterfall() {
    let n = program(2, u64::MAX, false, Case::ExtraAccess);
    let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
    let res = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap())).unwrap();
    let first = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|i| i.name() == "v_readfirstlane_b32")
        .unwrap()
        .writes[0]
        .1;
    assert!(
        !res.waterfall.reads.contains(&first),
        "two reads need the original loop"
    );
    let split = lower_program(&n, true).expect("bounded moves lower independently in each half");
    assert_eq!(split.barriers, 0, "no exchange across host halves");
}

#[cfg(windows)]
#[test]
fn indexed_wave32_matches_every_lane_on_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::indexed::indexed_wave32_matches_every_lane_on_gpu",
    ) {
        return;
    }
    for square in [false, true] {
        for (active, case) in MASKS
            .into_iter()
            .map(|m| (m, Case::Interval))
            .chain([
                (u64::MAX, Case::LastRegister),
                (u64::MAX, Case::ExtraAccess),
            ])
        {
            let n = program(2, active, square, case);
            let form = [(
                "halves",
                shader(lower_program(&n, true).expect("indexed wave32")),
            )];
            let input = vec![0x1357_9bdf; 64 * 8];
            let mut expected = input.clone();
            for (lane, row) in expected.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                if active & (1 << lane) != 0 {
                    row[..3].copy_from_slice(&lane_result(lane as u32));
                    if case == Case::ExtraAccess {
                        let lo = (lane ^ (lane >> 5)) & 1;
                        let hi = ((lane >> 1) & 3).min(2);
                        // Last visit reads the final slot before updating it.
                        row[3] = if lo <= hi { 10 * (hi as u32 + 1) } else { 0 };
                    }
                }
            }
            let views = [
                View::Texture(1, 1, &[[0.0; 4]]),
                View::Buffer {
                    data: &[0; 4],
                    stride: Stride::Raw,
                },
                View::RawUavData(&input),
            ];
            let (wrong, dispatches) = compare_dispatches(&form, 1, 8, &expected, &views);
            assert!(dispatches > 0, "no indexed dispatch");
            assert!(
                wrong.is_empty(),
                "{case:?}, square={square}, EXEC={active:#x}: {}",
                wrong.join("\n")
            );
        }
    }
}

/// Both proofs feed real reads; the lane-select waterfall activates the
/// whole-program admission check on the three preceding scalar-indexed reads.
fn range_proof_program() -> NativeProgram {
    use crate::range::tests::{SELECT_TWO_OR_ELEVEN, equality_words, selection_words};

    let mut words = PROLOGUE.to_vec();
    let equality = equality_words();
    words.extend(&equality[..equality.len() - 1]);
    words.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    for (reg, offset) in [(20, 0), (21, 4), (22, 8)] {
        words.extend(store(reg, offset));
    }
    let selection = selection_words(SELECT_TWO_OR_ELEVEN);
    words.extend(&selection[..selection.len() - 1]);
    words.push(0xbefe_04c1); // restore all 64 lanes after the waterfall
    words.extend(store(18, 12));
    words.push(S_ENDPGM);
    native(&words, true, 64, 0)
}

#[test]
fn range_proof_fixture_compiles_with_all_four_reads_in_file() {
    let n = range_proof_program();
    let (p, st) = crate::range::build(&n.code, n.wave64, n.vgprs).unwrap();
    let relative: Vec<_> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| i.name() == "v_movrels_b32")
        .collect();
    assert_eq!(
        relative.iter().map(|i| i.m0_max).collect::<Vec<_>>(),
        [Some(4), Some(5), Some(6), Some(11)]
    );
    assert!(relative.iter().all(|i| i.relative_in_file(p.vgprs)));
    let facts = resource::track(&p, &n.layout, n.stage, Some(&st.unwrap()))
        .unwrap()
        .waterfall;
    assert_eq!((facts.reads.len(), facts.compares.len()), (1, 1));
    lower_program(&n, true).unwrap();
}

#[cfg(windows)]
#[test]
fn range_proof_reads_match_every_lane_on_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::indexed::range_proof_reads_match_every_lane_on_gpu",
    ) {
        return;
    }
    let form = [(
        "halves",
        shader(lower_program(&range_proof_program(), true).unwrap()),
    )];
    let input = vec![0x1357_9bdf; 64 * 8];
    let mut expected = input.clone();
    for (lane, row) in expected.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        // Second scalar trip reads v29, v30 and v31. Even lanes select v31, odd lanes v22.
        row[..4].copy_from_slice(&[29, 30, 31, if lane % 2 == 0 { 31 } else { 22 }]);
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Raw,
        },
        View::RawUavData(&input),
    ];
    let (wrong, dispatches) = compare_dispatches(&form, 1, 8, &expected, &views);
    assert!(dispatches > 0, "no range proof dispatch");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
