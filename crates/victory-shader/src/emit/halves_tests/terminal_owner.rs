use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Change {
    None,
    LoadedCoordinate,
    MissingInit,
    WrongPhiInput,
    SecondEntry,
    LiveJoin,
    RestoreExec,
    EnlargeExec,
    EmptyExec,
    HighExec,
    Barrier,
    ReadLane,
    UnmaskedVote,
    ScalarDecision,
    MaskDecision,
}

fn fixture(leader: Option<u32>, change: Change) -> (NativeProgram, usize) {
    let mut words = PROLOGUE.to_vec();
    words.extend([
        0x8F0F_8801, // s15 = group_id.x * 256
        0x3404_0284, // v2 = y * 16
        0x4A04_0500, // v2 = x + v2
        0x3406_0482, // v3 = 4 * v2
        0x4A08_040F, // v4 = s15 + v2
        0x4A08_0889, // v4 += 9
        0xD834_0000,
        0x0000_0403, // LDS[v2] = v4
        S_WAITCNT_LGKM,
        0xBF8A_0000,
        if change == Change::MissingInit {
            0x7E0A_0280
        } else {
            0x7E0C_0280
        }, // initialize flag
        0xBE94_047E, // save full EXEC in s[20:21]
    ]);
    match leader {
        Some(leader) if change == Change::LoadedCoordinate => {
            words.extend([0xD8D8_0000, 0x0D00_0003, S_WAITCNT_LGKM]); // v13 = LDS[v3]
            words.push(0x7DA4_1A80 | leader);
        }
        Some(leader) => words.push(0x7DA4_0480 | leader), // EXEC &= v2 == leader
        None => words.extend([
            0x3618_04FF,
            0xffff_ffdf, // v12 = v2 & ~32
            0x7DA4_1880, // EXEC &= (v12 == 0), keeps {0, 32}
        ]),
    }
    let select = words.len();
    words.extend([
        s_cbranch_execz(1),
        if change == Change::WrongPhiInput {
            0x7E0C_0304
        } else {
            0x7E0C_02C1
        }, // v4 or -1
        S_MOV_EXEC_S20,
        0xBF8A_0000,
        0x7DA4_0CC1, // EXEC &= v6 == -1
    ]);
    let empty = words.len();
    words.extend([
        0,
        0x3420_0885, // v16 = output byte address = v4 * 32
        0x7E14_0283, // v10 running sum = 3
        0xBE92_0380, // s18 iteration = 0
    ]);
    match change {
        Change::RestoreExec => words.push(S_MOV_EXEC_S20),
        Change::EnlargeExec => words.push(0xBEFE_04C1), // EXEC = -1
        Change::EmptyExec => words.push(0xBEFE_0480),   // EXEC = 0
        Change::HighExec => words.push(0xBEFF_0380),    // EXEC_HI = 0, clears leader 40
        Change::Barrier => words.push(0xBF8A_0000),
        Change::ReadLane => words.extend([0xD760_0013, 0x0001_0109]), // v_readlane_b32 s19, v9, 0
        _ => {}
    }
    let loop_start = words.len();
    words.extend([
        0x7E0A_0212, // v5 = s18
        0x340A_0A82, // v5 *= 4
        0x4A0A_0B03, // v5 = v3 + v5
        0xD8D8_0000,
        0x0900_0005, // v9 = LDS[flat + iteration]
        S_WAITCNT_LGKM,
        0x4A12_130A, // v9 = v10 + v9
        0x7E26_0509, // s19 = firstlane(v9)
        0x8012_8112, // s18 += 1
        0x7E14_0213, // v10 = s19
        0xBF09_FF13,
        200, // SCC = s19 >= 200
    ]);
    let branch = if change == Change::UnmaskedVote {
        words.push(0xBEEA_0414); // VCC = saved full EXEC, including nonowners
        0xBF86_0000 // s_cbranch_vccz
    } else {
        0xBF84_0000 // s_cbranch_scc0
    };
    let latch = words.len();
    words.push(branch | ((loop_start as i32 - latch as i32 - 1) as u32 & 0xffff));
    if matches!(change, Change::ScalarDecision | Change::MaskDecision) {
        if change == Change::ScalarDecision {
            words.extend([0xBF09_FF13, 250]); // s_cmp_ge_u32 s19, 250 (205 vs 308)
        } else {
            words.push(0xBE98_087E); // s_not_b64 s[24:25], exec sets SCC; owner bit is zero
        }
        words.push(0xBF85_0002); // s_cbranch_scc1 skips the observable store at +8
        words.extend(store(10, 8));
    }
    let store_start = words.len();
    words.extend(store(10, 0));
    words.push(0x7E16_0212); // v11 = s18
    words.extend(store(11, 4));
    words.push(S_ENDPGM);
    let exit = if change == Change::LiveJoin {
        store_start
    } else {
        words.len() - 1
    };
    words[empty] = s_cbranch_execz(exit - empty - 1);
    if change == Change::SecondEntry {
        words[select] = s_cbranch_execz(empty - select); // bypass reaches the live continuation
    }
    let mut program = native_for_groups(&words, true, 256, 1024, 2);
    let StageIo::Cs(cs) = &mut program.io else {
        unreachable!()
    };
    cs.rsrc2 |= 1 << 11; // x and y per-thread IDs
    cs.threads = [16, 16, 1];
    (program, latch * 4)
}

#[test]
fn terminal_owner_first_lane_lowers() {
    let mut failures = Vec::new();
    for leader in [Some(0), Some(40), None] {
        let (program, latch) = fixture(leader, Change::None);
        assert_eq!(
            lower_halves(&program, false, true, false)
                .unwrap()
                .wave_size,
            Some(64)
        );
        for retry in [false, true] {
            let result = lower_halves(&program, true, true, retry);
            if leader.is_some() {
                match result {
                    Ok(shader) => assert_eq!(
                        (shader.wave_size, shader.owner_reads, shader.barriers),
                        (Some(32), 1, 3)
                    ),
                    Err(e) => failures.push(format!("{leader:?}, retry={retry}: {e}")),
                }
            } else {
                let Err(crate::ShaderError::Unsupported { offset, what }) = result else {
                    panic!("two halves must retain the named first-lane refusal")
                };
                assert_eq!(offset, latch);
                assert!(what.contains("a first-lane read of a value that may differ"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn terminal_owner_rejects_unproved_participation() {
    for change in [
        Change::LoadedCoordinate,
        Change::MissingInit,
        Change::WrongPhiInput,
        Change::SecondEntry,
        Change::LiveJoin,
        Change::RestoreExec,
        Change::EnlargeExec,
        Change::EmptyExec,
        Change::HighExec,
        Change::Barrier,
        Change::ReadLane,
    ] {
        let (n, latch) = fixture(Some(40), change);
        let Err(crate::ShaderError::Unsupported { offset, what }) =
            lower_halves(&n, true, true, false)
        else {
            panic!("{change:?}: expected the normal-path refusal")
        };
        assert_eq!(offset, latch, "{change:?}");
        assert!(
            what.contains(if change == Change::HighExec {
                "a lane mask whose lanes depend"
            } else {
                "a first-lane read"
            }),
            "{change:?}: {what}"
        );
    }
    let (vote, _) = fixture(Some(40), Change::UnmaskedVote);
    assert_eq!(
        lower_halves(&vote, true, true, false).unwrap().owner_reads,
        0
    );
    let (n, latch) = fixture(Some(40), Change::None);
    let p = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
    let mut st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &n).unwrap();
    let (b, i) = p
        .blocks
        .iter()
        .enumerate()
        .find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.name() == "v_readfirstlane_b32")
                .map(|i| (b, i))
        })
        .unwrap();
    st.guards[b][i] = Some(structure::Bool::FALSE);
    let guarded = parts(&n, &p, &st, &res, &io, true, true);
    let Err(crate::ShaderError::Unsupported { offset, what }) =
        lower::lower_pruned(&guarded, None, false, false)
    else {
        panic!("a per-site guard can clear the selected owner")
    };
    assert_eq!(offset, latch);
    assert!(what.contains("a first-lane read"));
    let Err(crate::ShaderError::Unsupported { offset, what }) =
        super::super::with_thread_ticket(false, || lower_halves(&n, true, true, false))
    else {
        panic!("a tiled group needs lane tickets");
    };
    assert_eq!(offset, latch);
    assert!(what.contains("a first-lane read"));
}

#[test]
fn terminal_owner_scc_decision_preserves_conditional_store() {
    for change in [Change::ScalarDecision, Change::MaskDecision] {
        let (n, latch) = fixture(Some(40), change);
        let p = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
        let (b, decision) = p
            .blocks
            .iter()
            .enumerate()
            .find(|(_, block)| {
                block
                    .insts
                    .last()
                    .is_some_and(|i| i.name() == "s_cbranch_scc1")
            })
            .unwrap();
        let ir::Exit::Branch { taken, not_taken } = decision.exit else {
            unreachable!()
        };
        let store_offset = |b: usize| {
            p.blocks[b].insts.iter().find_map(|i| match i.fields {
                gfx10::Fields::Buffer { offset, .. } if i.name() == "buffer_store_dword" => {
                    Some(offset)
                }
                _ => None,
            })
        };
        assert_eq!(
            (store_offset(not_taken), store_offset(taken)),
            (Some(8), Some(0))
        );
        let result = lower_halves(&n, true, true, false);
        if change == Change::ScalarDecision {
            let (base, _) = fixture(Some(40), Change::None);
            let base = lower_halves(&base, true, true, false).unwrap();
            let decision = result.unwrap();
            assert_eq!(decision.owner_reads, 1);
            assert_eq!(decision.shape.branches, base.shape.branches + 1);
        } else {
            let st = structure::structure(&p).unwrap();
            assert!(matches!(
                st.exits[b],
                structure::LaneExit::Branch {
                    when: structure::Bool::FALSE,
                    ..
                }
            ));
            let Err(crate::ShaderError::Unsupported { offset, what }) = result else {
                panic!("the other 63 bits set SCC even though the owner's bit is clear");
            };
            assert_eq!(offset, latch);
            assert!(what.contains("a first-lane read"));
        }
    }
}

#[cfg(windows)]
#[test]
fn terminal_owner_matches_two_groups_on_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::terminal_owner::terminal_owner_matches_two_groups_on_gpu",
    ) {
        return;
    }
    for (leader, group0, group1) in [(0, [220, 14], [268, 1]), (40, [205, 4], [308, 1])] {
        let (n, _) = fixture(Some(leader), Change::None);
        let forms = [(
            "owner wave32",
            shader(lower_halves(&n, true, true, false).unwrap()),
        )];
        let input = vec![0x1357_9bdf; 512 * 8];
        let mut expected = input.clone();
        for (group, result) in [group0, group1].into_iter().enumerate() {
            let row = group * 256 + leader as usize + 9;
            let visits = super::super::loop_limit().min(result[1]);
            // Sum of the first visits LDS words, starting at this group's payload.
            let sum = 3 + visits * row as u32 + visits * (visits - 1) / 2;
            assert!(visits < result[1] || sum == result[0]);
            expected[8 * row..8 * row + 2].copy_from_slice(&[sum, visits]);
        }
        let views = [
            View::Texture(1, 1, &[[0.0; 4]]),
            View::Buffer {
                data: &[0; 4],
                stride: Stride::Raw,
            },
            View::RawUavData(&input),
        ];
        eprintln!(
            "leader={leader}, cap={}, checking {} dwords",
            super::super::loop_limit(),
            expected.len()
        );
        let (wrong, dispatches) = compare_dispatches(&forms, 2, 8, &expected, &views);
        assert!(dispatches > 0, "no dispatch for leader {leader}");
        assert!(wrong.is_empty(), "leader {leader}: {}", wrong.join("\n"));
    }
}
