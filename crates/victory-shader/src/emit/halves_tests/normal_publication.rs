use super::*;

mod saved_exec;

/// Group 0 conditionally loads a pair. Group 1 either skips or reads its own pair.
/// The first merged scalar chooses the loop latch. The join reads both SGPRs after EXEC is restored.
fn program(mask: &[u32], per_group: bool) -> Vec<u32> {
    let mut w = two_latch_first_lane(&MASK_LANES_8_9, LatchChoice::Scalar(40));
    let iteration = w.iter().position(|&x| x == 0xbf06_1a18).unwrap();
    w.splice(iteration..iteration, store(2, 12));
    // The added per-iteration store shifts both back edges and the exit.
    let header = w.iter().position(|&x| x == 0xbf06_821c).unwrap();
    w[header + 1] += 2;
    for word in &mut w[iteration..] {
        if *word >> 16 == 0xbf82 && *word & 0x8000 != 0 {
            *word -= 2;
        }
    }
    let mut prefix = vec![
        0x8f0f_8601, // s15 = group_id.x << 6
        0x4a22_000f, // v17 = group base + lane
        0x3420_2285, // v16 = byte offset of output row
        0xf408_0b02,
        0xfa00_0020, // load the fixed table SRV
        S_WAITCNT_LGKM,
        0xbe9e_0391, // old s30 = 17
        0x801f_9501, // old s31 = group_id.x + 21
    ];
    // A compare on the global ID makes the second group empty.
    if per_group {
        prefix.push(0x8f12_8301); // s18 = group_id.x * 8
        prefix.push(mask[0]); // use local lanes; both groups can enter
    } else if mask == MASK_LANES_12_44 {
        prefix.extend([0x3606_22ff, 95]); // v3 = global ID & 95
    } else {
        prefix.push(mask[0] | (17 << 9));
    }
    prefix.push(mask[1]);
    prefix.extend([
        0xbe96_246a, // save and intersect EXEC
        0xbf88_0005, // skip the body on empty EXEC
        0x7e1c_0500, // firstlane(v0) requires converged execution in this arm
        0xbefe_0480, // clear EXEC inside the visited arm; SMEM still executes
        0xf424_0796,
        if per_group { 0x2400_0008 } else { 0xfa00_0008 },
        S_WAITCNT_LGKM,
        0xbefe_0416, // restore full EXEC
        0x7e0c_021e, // v6 = merged s30
        0x7e0e_021f, // v7 = merged s31
    ]);
    prefix.extend([
        0x7e24_0506, // s18 = firstlane(v6), full EXEC after the join
        0x8712_8112, // low bit of the merged first word
        0x801a_121a, // an even first word chooses the +1 loop latch
    ]);
    prefix.extend(store(6, 4));
    prefix.extend(store(7, 8));
    w.splice(header..header, prefix);
    w
}

#[test]
fn predicated_fixed_pair_publishes_both_components() {
    for (mask, _) in TWO_LATCH_CASES {
        for dynamic in [false, true] {
            let mut w = program(mask, dynamic);
            let n = native_for_groups(&w, true, 64, 0, 2);
            let split = lowered_with(&n, true, true).expect("predicated normal publication");
            eprintln!(
                "mask={mask:x?} per_group={dynamic} cap={} DXIL MD5={:02x?}",
                super::super::loop_limit(),
                super::super::container::md5(&split.dxil)
            );
            assert_eq!(split.wave_size, Some(32));
            let at = (!dynamic).then_some(8);
            assert_eq!(live_sources(&split), [(at, 0), (at, 1)]);
            let [load] = split.raw_loads[0][..] else {
                panic!("one SRV load")
            };
            let (cfg, fences) = &split.barrier_cfg;
            let [join] = cfg[load][..] else {
                panic!("load has one exit")
            };
            assert_eq!(fences.iter().filter(|&&b| b == load).count(), 4);
            assert!(
                cfg.iter()
                    .enumerate()
                    .any(|(head, edges)| edges == &[load, join]
                        && fences.iter().filter(|&&b| b == head).count() >= 2)
            );
            let pair = w.iter().position(|&x| x == 0xf424_0796).unwrap();
            w[pair..pair + 2].copy_from_slice(&[0xbe9e_0383, 0xbe9f_0385]);
            let control = lowered_with(&native_for_groups(&w, true, 64, 0, 2), true, true).unwrap();
            assert_eq!(
                split.shape.branches,
                control.shape.branches + 1,
                "whole-group visit"
            );
            assert_eq!(
                split.barriers,
                control.barriers + if dynamic { 10 } else { 6 }
            );
        }
    }
}

#[test]
fn predicated_publication_rejects_wrong_visit_guards_and_partial_results() {
    let mut w = program(&MASK_LANES_40_41, false);
    let branch = w.iter().position(|&x| x == 0xbf88_0005).unwrap();
    w[branch] = 0xbf89_0005; // body on EXEC empty; some inactive lanes do not prove emptiness
    let n = native_for_groups(&w, true, 64, 0, 2);
    let error = lowered_with(&n, true, true).err().expect("unproved visit");
    assert!(
        error.to_string().contains("without a proved predicated"),
        "{error}"
    );

    let n = native_for_groups(&program(&MASK_LANES_40_41, false), true, 64, 0, 2);
    for (fault, refusal) in [
        ("guard", "guarded or incomplete scalar-buffer publication"),
        (
            "component",
            "guarded or incomplete scalar-buffer publication",
        ),
        ("edge", "without a proved predicated"),
    ] {
        let mut p = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
        let mut st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &n).unwrap();
        let (b, i) = res
            .descriptors
            .iter()
            .map(|d| d.site)
            .find(|&(b, i)| p.blocks[b].insts[i].name() == "s_buffer_load_dwordx2")
            .unwrap();
        match fault {
            "guard" => st.guards[b][i] = Some(structure::Bool::FALSE),
            "component" => {
                p.blocks[b].insts[i].writes.pop();
            }
            _ => {
                let d = p.blocks[b].preds[0];
                let branch = p.blocks[d].insts.last().unwrap();
                let mask = st.masks[&branch.read(ir::Reg::S(ir::EXEC)).unwrap()];
                let structure::LaneExit::Branch { when, .. } = &mut st.exits[d] else {
                    unreachable!()
                };
                *when = mask; // EXECZ must skip on zero, even after path simplification
            }
        }
        let error = lower::lower(&parts(&n, &p, &st, &res, &io, true, true))
            .err()
            .expect("unproved publication");
        assert!(error.to_string().contains(refusal), "{error}");
    }
}

#[cfg(windows)]
#[test]
fn predicated_publication_matches_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::normal_publication::predicated_publication_matches_gpu",
    ) {
        return;
    }
    for (mask, _) in TWO_LATCH_CASES {
        for per_group in [false, true] {
            check_publication_gpu(
                &program(mask, per_group),
                [mask != MASK_EMPTY, mask != MASK_EMPTY && per_group],
                &format!("mask={mask:x?} per_group={per_group}"),
            );
        }
    }
}

#[cfg(windows)]
fn check_publication_gpu(words: &[u32], visits: [bool; 2], case: &str) {
    let limit = super::super::loop_limit();
    let input = [5, 7, 112, 197, 227, 251];
    let start = vec![0x1357_2468; 2 * 64 * 8];
    let n = native_for_groups(words, true, 64, 0, 2);
    let forms = [(
        "publication wave32",
        shader(lowered_with(&n, true, true).unwrap()),
    )];
    eprintln!(
        "{case} cap={limit} DXIL MD5={:02x?}",
        super::super::container::md5(&forms[0].1.dxil)
    );
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &input,
            stride: Stride::Raw,
        },
        View::RawUavData(&start),
    ];
    let mut want = start.clone();
    for (lane, row) in want.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let group = lane / 64;
        let pair = if visits[group] {
            [input[2 + 2 * group], input[3 + 2 * group]]
        } else {
            [17, 21 + group as u32]
        };
        row[1..3].copy_from_slice(&pair);
        let iterations = if pair[0] % 2 == 0 { 2 } else { 1 };
        row[3] = 14 + iterations.min(limit);
        if iterations < limit {
            row[0] = 14 + iterations;
        }
    }
    let (errors, ran) = compare_dispatches(&forms, 2, 8, &want, &views);
    assert!(ran > 0, "physical wave32 required");
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn predicated_publication_rejects_a_guarded_raw_view() {
    let mut w = program(&MASK_LANES_8_9, false);
    w.splice(PROLOGUE.len()..PROLOGUE.len(), [0xf400_0c02, 0xfa00_0000]); // unrelated scalar read
    let mut n = native_for_groups(&w, true, 64, 0, 2);
    n.layout.push(RootParam {
        root: RootType::Descriptor,
        user_data: 4,
        slot: 1,
        sgprs: 2,
        table_offset_dwords: 0,
        base_register: 0,
    });
    let mut used = n.used[1];
    used.root = RootType::Descriptor;
    used.user_data = 4;
    n.used.push(used);
    let error = lower_with_resources(&n, true, true, false, |p, res| {
        let mut view = res
            .descriptors
            .iter()
            .find(|d| p.blocks[d.site.0].insts[d.site.1].name() == "s_buffer_load_dwordx2")
            .unwrap()
            .clone();
        // Exercise the plan/emitter boundary directly. The planner sees its
        // fixed SRV; the final view for the same site takes the bounds-checked
        // root-memory route. No contradictory metadata is ever dispatched.
        view.source = resource::Source::Built {
            address: resource::Address {
                param: 1,
                offset: 0,
                dynamic: None,
            },
            bits: 0,
            words: [0; 2],
        };
        res.descriptors.push(view);
        let (_, other) = res
            .scalar
            .iter_mut()
            .find(|((b, i), _)| p.blocks[*b].insts[*i].name() == "s_load_dword")
            .unwrap();
        other.param = resource::MEMORY;
    })
    .err()
    .expect("conditional raw view");
    assert!(
        error.to_string().contains("through a guarded raw view"),
        "{error}"
    );
}

#[test]
fn nested_publication_requires_the_inner_exec_to_stay_inside_the_outer() {
    for restore in [false, true] {
        let mut w = program(&MASK_LANES_40_41, true);
        let body = w.iter().position(|&x| x == 0x8f12_8301).unwrap();
        let mut outer = vec![
            0x7d88_00a0, // compare the local lane ID with 32
            0xbea0_246a, // save outer EXEC in s[32:33]
            0,           // skip the entire remaining program if no lane entered
        ];
        if restore {
            outer.push(0xbefe_0420); // allow lanes outside the outer arm again
        }
        // A group-ID branch rejoins before the inner guard, preserving its
        // dominating outer predicate on both physical paths.
        outer.extend([0xbf06_8001, 0xbf85_0001, 0xbf80_0000]);
        outer[2] = s_cbranch_execz(w.len() + outer.len() - 1 - (body + 3));
        w.splice(body..body, outer);
        let n = native_for_groups(&w, true, 64, 0, 2);
        let result = lowered_with(&n, true, true);
        if restore {
            let error = result.err().expect("unproved nested EXEC");
            assert!(
                error.to_string().contains("without a proved predicated"),
                "{error}"
            );
        } else {
            assert_eq!(live_sources(&result.unwrap()), [(None, 0), (None, 1)]);
        }
    }
}

/// Exercise only the visit proof on normal IR. Neither negative program enters code generation.
fn visit_admitted(w: &[u32], heads: &[usize]) -> bool {
    let n = native_for_groups(w, true, 64, 0, 2);
    let p = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    io::raise(&p, &n).unwrap();
    let gates: Vec<_> = heads
        .iter()
        .map(|&d| {
            assert_eq!(p.blocks[d].insts.last().unwrap().name(), "s_cbranch_execz");
            let structure::LaneExit::Branch {
                not_taken: body, ..
            } = st.exits[d]
            else {
                panic!("EXEC branch")
            };
            assert_eq!(p.blocks[body].preds, [d]);
            (body, d)
        })
        .collect();
    let body = gates.last().unwrap().0;
    assert!(res.descriptors.iter().any(|desc| desc.site.0 == body
        && p.blocks[body].insts[desc.site.1].name() == "s_buffer_load_dwordx2"));
    let order = ir::reverse_postorder(&p.blocks);
    let mut ranks = vec![usize::MAX; p.blocks.len()];
    for (i, &b) in order.iter().enumerate() {
        ranks[b] = i;
    }
    let idom = ir::idoms(&p.blocks, &order, &ranks);
    lower::publication_gates(&p, &st, &gates, &Default::default(), &idom)
}

#[test]
fn quad_expansion_of_empty_exec_cannot_authorize_a_scalar_read() {
    let mut w = program(&MASK_EMPTY, false);
    let at = w.iter().position(|&x| x == 0xbf88_0005).unwrap();
    assert!(visit_admitted(&w, &[0]));
    w.insert(at, 0xbefe_0a7e); // WQM of the empty EXEC still has zero visitors
    assert!(!visit_admitted(&w, &[0]));
    w.remove(at);
    w.insert(PROLOGUE.len(), 0xbefe_0a7e); // WQM of the full entry EXEC is exact
    assert!(visit_admitted(&w, &[0]));
}

#[test]
fn divergent_mask_phi_cannot_authorize_a_scalar_read() {
    let mut w = program(&MASK_LANES_40_41, true);
    let pos = w.iter().position(|&x| x == 0x8f12_8301).unwrap();
    let mut prefix = [
        0x7d88_00a0,
        0xbea0_246a,
        0, // outer mask = lanes 0..31
        0x2c06_0081,
        0x7d84_0684,
        0xbea4_246a, // A = lanes 8 and 9; save outer EXEC
        0xbf88_0002, // whole guest wave takes the nonempty arm
        0xbefe_087e,
        0x88fe_817e, // that arm sets EXEC = !A | 1
        0x87fe_247e, // join: EXEC &= saved outer mask
    ];
    prefix[2] = s_cbranch_execz(w.len() + prefix.len() - 1 - (pos + 3));
    w.splice(pos..pos, prefix);
    let at = w.iter().position(|&x| x == 0xbf88_0005).unwrap();
    w[at - 3..at].copy_from_slice(&[0xbf80_0000, 0xbf80_0000, 0xbe96_047e]);
    // Guest takes one predecessor for the wave, leaving 30 visitors. With
    // per-lane selection, lanes 8/9 clear their own bits and all others skip.
    assert!(!visit_admitted(&w, &[0, 3]));
    w[pos + 8] = 0xbf80_0000; // EXEC = !A; the per-lane mask phi can fold to false
    assert!(!visit_admitted(&w, &[0, 3]));
}
