use super::*;
use crate::emit::halves_tests::{live_sources, native, parts};

/// Each outer entry adds ten. The inner loop runs old_index+2 times, then
/// preserves its progress in a per-lane UAV word before the two-way latch.
fn table_loop(words: [u32; 4]) -> NativeProgram {
    fixture(words, false)
}

fn fixture(words: [u32; 4], next_index: bool) -> NativeProgram {
    fixture_with_input(words, next_index, true)
}

fn fixture_with_input(words: [u32; 4], next_index: bool, publish: bool) -> NativeProgram {
    fixture_address(words, next_index, publish, false)
}

fn fixture_address(
    words: [u32; 4],
    next_index: bool,
    publish: bool,
    paired: bool,
) -> NativeProgram {
    let mut code = vec![
        0xbe84_0300,
        0xbe85_0380,
        0xf408_0a02,
        0xfa00_0040,
        0xbf8c_c07f, // UAV
    ];
    if publish {
        code.extend([
            0xf408_0b02,
            0xfa00_0020,
            0xbf8c_c07f, // SRV s[44:47]
            0xf420_0696,
            0xfa00_0000,
            0xbf8c_c07f, // s26 = actual published input
        ]);
    } else {
        code.push(0xbe9a_03bf); // s26 = 63
    }
    code.extend([
        0x3420_0085, // v16 = lane * 32
        0xbe88_1f00, // s[8:9] = PC
    ]);
    let pc = code.len() * 4;
    if paired {
        code.extend([0x800a_ff08, 0, 0x820b_8009]); // s[10:11] = PC + table distance
    }
    code.extend([
        0xbe94_0380, // s20 = 0, index
        0xbe95_0380, // s21 = 0, progress
        0xbe96_0380, // no previous table word
    ]);
    let outer = code.len();
    code.extend([
        0x8015_8a15, // progress += 10
        0x7d84_0080, // VCC = lane zero
        0xbe9c_046a, // s[28:29] = VCC
    ]);
    let skip = code.len();
    code.push(0);
    let latch = code.len();
    if paired {
        // Two separately formed addresses share the adjusted PC. Both use the
        // same full displacement. Only the second result controls the branch.
        // The constant guard skips the first use at runtime. Its block is
        // still in the graph and must receive an independent certificate.
        code.extend([0xbf06_8080, 0xbf85_0004]);
        for dest in [36, 22] {
            code.extend([
                0x800c_1e0a,
                0x820d_1f0b,
                0xf400_0006 | (dest << 6),
                0xfa00_0000,
            ]);
        }
    } else {
        code.extend([0xf400_0584, 0]);
    }
    code.extend([0xbf8c_c07f, 0xbf06_8016]); // s22 = table word; == 0
    code.push(0xbe9c_0480); // direct latch entry enables no masked store
    let branch = |at: usize, to: usize, opcode: u32| {
        opcode | u32::from(i16::try_from(to as isize - at as isize - 1).unwrap() as u16)
    };
    code.push(branch(code.len(), outer, 0xbf85_0000));
    code[skip] = branch(skip, code.len(), 0xbf82_0000);
    let shift: &[u32] = if paired {
        // Sign extend the old dword and shift the pair left by two.
        &[0xbea2_0314, 0x9123_9f22, 0x8f9e_8222]
    } else {
        &[0x8f17_8214]
    };
    if next_index {
        code.push(0x8014_8114);
        code.extend(shift);
    } else {
        code.extend(shift);
        code.push(0x8014_8114);
    }
    code.extend([
        0x8018_8114, // s24 = index + 1 (inner count)
        0x8015_8115, // inner: ++progress
        0x8098_8118, // --s24
        0xbf07_8018, // s24 != 0
        0xbf85_fffc, // if set, inner
    ]);
    for (register, offset) in [(21, 0), (22, 4), (26, 8)] {
        code.extend([0x7e0c_0200 | register, 0xe070_1000 | offset, 0x800a_0610]);
    }
    code.extend([
        0x7e0c_0215, // v6 = progress
        0xbefe_041c, // joined mask from the selected incoming edge
        0xe070_100c,
        0x800a_0610, // guarded store at slot 3
        0xbefe_04c1, // restore every lane
    ]);
    code.push(0xbf06_8414); // index == 4
    code.push(branch(code.len(), latch, 0xbf84_0000));
    code.push(0xbf81_0000);
    let distance = (code.len() * 4 - pc) as u32;
    if paired {
        code[pc / 4 + 1] = distance;
    } else {
        code[latch + 1] = (23 << 25) | distance;
    }
    code.extend(words);
    native(&code, true, 64, 0)
}

fn emit(n: &NativeProgram) -> Result<Lowered, ShaderError> {
    let p = crate::ir::build(&n.code, true, n.vgprs)?;
    let st = crate::structure::split_literal(&p)?;
    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st))?;
    let io = crate::io::raise(&p, n)?;
    lower_literal(&parts(n, &p, &st, &res, &io, true, true), None, false)
}

#[test]
fn immutable_table_paired_addresses() {
    for (words, next, publish) in [
        ([2, 7, 5, 0], false, true),
        ([0, 0, 0, 9], false, true),
        ([0, 2, 7, 5], true, false),
    ] {
        let n = fixture_address(words, next, publish, true);
        let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
        let st = crate::structure::split_literal(&p).unwrap();
        let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
        let facts = immutable::prove(&p, &st, &res, &n);
        let start = usize::from(next);
        let expected: Set<_> = words[start..start + 3].iter().copied().collect();
        assert_eq!(facts.reads.len(), 2, "both uses of the shared base");
        assert!(facts.reads.values().all(|content| *content == expected));
        assert_eq!(facts.jumps.len(), 1);
        assert_eq!(facts.carries.len(), 3, "one fixed and two dynamic carries");
        let (latch, selected) = facts.jumps[0];
        let Exit::Branch { taken, not_taken } = p.blocks[latch].exit else {
            panic!("two-way latch")
        };
        assert_eq!(selected, if words[start] == 0 { taken } else { not_taken });
        let q = crate::ir::folded(&p, &facts.jumps).unwrap().unwrap();
        let st = crate::structure::split_literal(&q).unwrap();
        let res = crate::resource::track(&q, &n.layout, n.stage, Some(&st)).unwrap();
        let final_facts = immutable::prove(&q, &st, &res, &n);
        assert_eq!(final_facts.reads.len(), 2);
        assert!(
            final_facts
                .reads
                .values()
                .all(|content| *content == expected)
        );
        assert_eq!(final_facts.carries.len(), 3);
        assert!(final_facts.jumps.is_empty());
        let io = crate::io::raise(&q, &n).unwrap();
        let x = parts(&n, &q, &st, &res, &io, true, true);
        let proof = super::super::literal_publish::prove(&x, 64, 0, &HashMap::new())
            .ok()
            .unwrap();
        assert_eq!(proof.is_some(), publish);
        if let Some(proof) = proof {
            for &(site, carry) in &final_facts.carries {
                let high = &q.blocks[site.0].insts[site.1];
                assert!(!proof.common.contains(&carry));
                assert!(high.writes.iter().all(|(_, v)| !proof.common.contains(v)));
            }
        }
        let emitted = emit(&n).unwrap();
        assert_eq!(live_sources(&emitted).len(), usize::from(publish));
    }
}

#[test]
fn immutable_table_paired_requires_complete_operands_and_uses() {
    let n = fixture_address([2, 7, 5, 0], false, true, true);
    let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
    let st = crate::structure::split_literal(&p).unwrap();
    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let facts = immutable::prove(&p, &st, &res, &n);
    assert_eq!(facts.reads.len(), 2);
    let site = |name, reg| {
        p.blocks
            .iter()
            .enumerate()
            .find_map(|(b, block)| {
                block
                    .insts
                    .iter()
                    .position(|i| i.name() == name && i.write(reg).is_some())
                    .map(|i| (b, i))
            })
            .unwrap()
    };
    let fixed_hi = site("s_addc_u32", Reg::S(11));
    let outer_hi = site("s_addc_u32", Reg::S(13));
    let fixed_lo = site("s_add_u32", Reg::S(10));
    let shift = site("s_lshl_b64", Reg::S(30));
    let sign = site("s_ashr_i32", Reg::S(35));
    let load = site("s_load_dword", Reg::S(36));
    let second = site("s_load_dword", Reg::S(22));
    let adjusted = p.blocks[fixed_lo.0].insts[fixed_lo.1]
        .write(Reg::S(10))
        .unwrap();
    let pure = p.blocks[shift.0].insts[shift.1].write(Reg::Scc).unwrap();
    let refuse = |query: &Program, st: &Structured, res: &Resources| {
        let f = immutable::prove(query, st, res, &n);
        assert!(f.reads.is_empty() && f.carries.is_empty() && f.jumps.is_empty());
    };
    // Each query changes just a proof input on the original native graph.
    for (b, i) in [fixed_hi, outer_hi] {
        for replacement in [None, Some(pure)] {
            let mut q = p.clone();
            let add = &mut q.blocks[b].insts[i];
            add.reads.retain(|(r, _)| *r != Reg::Scc);
            if let Some(v) = replacement {
                add.reads.push((Reg::Scc, v));
            }
            refuse(&q, &st, &res);
        }
    }
    for ((b, i), arg, replacement) in [
        (fixed_hi, 0, Operand::Scalar { code: 8, dwords: 1 }), // low PC as high
        (fixed_hi, 1, Operand::Const(129)),                    // wrong high adjustment
        (outer_hi, 1, Operand::Const(253)), // explicit SCC data in addition to implicit carry
        (
            outer_hi,
            1,
            Operand::Scalar {
                code: 30,
                dwords: 1,
            },
        ), // low shift word as high
        (sign, 1, Operand::Const(128)),     // no sign extension
        (
            shift,
            1,
            Operand::Scalar {
                code: 60,
                dwords: 1,
            },
        ), // unknown count
    ] {
        let mut q = p.clone();
        let Fields::Sop { src, .. } = &mut q.blocks[b].insts[i].fields else {
            unreachable!()
        };
        src[arg] = replacement;
        refuse(&q, &st, &res);
    }
    let mut q = p.clone();
    let later_low = p.blocks[second.0].insts[second.1].read(Reg::S(12)).unwrap();
    q.blocks[load.0].insts[load.1]
        .reads
        .iter_mut()
        .find(|(r, _)| *r == Reg::S(12))
        .unwrap()
        .1 = later_low;
    refuse(&q, &st, &res); // use before its definition
    let mut q = p.clone();
    let skipped_low = p.blocks[load.0].insts[load.1].read(Reg::S(12)).unwrap();
    q.blocks[second.0].insts[second.1]
        .reads
        .iter_mut()
        .find(|(r, _)| *r == Reg::S(12))
        .unwrap()
        .1 = skipped_low;
    refuse(&q, &st, &res); // the optional arm does not dominate the join
    let mut q = p.clone();
    let Fields::Smem { soffset, .. } = &mut q.blocks[load.0].insts[load.1].fields else {
        unreachable!()
    };
    *soffset = Operand::Scalar {
        code: 12,
        dwords: 1,
    };
    refuse(&q, &st, &res); // base low also used as numeric displacement
    for escaped in [
        adjusted,
        p.blocks[outer_hi.0].insts[outer_hi.1]
            .write(Reg::Scc)
            .unwrap(),
    ] {
        let mut q = p.clone();
        q.blocks[sign.0].insts[sign.1]
            .reads
            .push((Reg::S(60), escaped));
        refuse(&q, &st, &res);
        let mut s = st.clone();
        s.restored.insert(pure, escaped);
        refuse(&p, &s, &res);
        let mut r = res.clone();
        r.descriptors[0].source = Source::Select {
            scc: escaped,
            args: vec![r.descriptors[0].source.clone()],
        };
        refuse(&p, &st, &r);
        let mut r = res.clone();
        r.descriptors[0].source = Source::Built {
            address: Address {
                param: CODE,
                offset: 0,
                dynamic: Some(escaped),
            },
            bits: 0,
            words: [0; 2],
        };
        refuse(&p, &st, &r);
    }
    let mixed = fixture_address([2, 0, 5, 9], false, true, true);
    let facts = immutable::prove(&p, &st, &res, &mixed);
    assert_eq!(facts.reads.len(), 2);
    assert!(facts.jumps.is_empty());
    // Strict admission without a fold still checks every CODE producer.
    let io = crate::io::raise(&p, &mixed).unwrap();
    let mut r = res.clone();
    r.descriptors[0].source = Source::Built {
        address: Address {
            param: CODE,
            offset: u32::MAX,
            dynamic: None,
        },
        bits: 0,
        words: [0; 2],
    };
    let x = parts(&mixed, &p, &st, &r, &io, true, true);
    assert!(
        matches!(super::super::literal_publish::prove(&x, 64, 0, &HashMap::new()),
        Err(Stop::Fail(ShaderError::Unsupported{what, ..})) if what.contains("proved byte range"))
    );
}

#[test]
fn immutable_table_two_counted_edges() {
    for words in [[2, 7, 5, 0], [0, 0, 0, 9]] {
        let lower = emit(&table_loop(words)).unwrap_or_else(|error| panic!("{words:?}: {error}"));
        assert_eq!(live_sources(&lower), [(Some(0), 0)]);
    }
}

#[test]
fn immutable_table_old_and_new_index() {
    for (words, next) in [
        ([2, 0, 5, 9], false),
        ([0, 2, 4, 6], false),
        ([2, 4, 6, 0], true),
        ([0, 2, 4, 6], true),
    ] {
        let n = fixture(words, next);
        let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
        let st = crate::structure::split_literal(&p).unwrap();
        let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
        let facts = immutable::prove(&p, &st, &res, &n);
        let start = usize::from(next);
        let expected: Set<_> = words[start..start + 3].iter().copied().collect();
        assert_eq!(facts.reads.len(), 1, "{words:?}, next={next}");
        assert_eq!(facts.reads.values().next(), Some(&expected));
        assert_eq!(facts.jumps.len(), usize::from(!expected.contains(&0)));

        // Mixed words still select both counted edges. Common immutable
        // contents prove the dynamic choice without folding the latch.
        let lower = emit(&n).unwrap_or_else(|error| panic!("{words:?}, next={next}: {error}"));
        assert_eq!(live_sources(&lower), [(Some(0), 0)]);
    }
}

fn trace(words: [u32; 4], next_index: bool, cap: usize) -> Vec<(u32, u32)> {
    let (mut progress, mut previous) = (10, 0);
    let mut trace = Vec::new();
    for index in 0..4.min(cap) {
        progress += (index + 2).min(cap) as u32;
        trace.push((progress, previous));
        if index == 3 {
            break;
        }
        previous = words[index + usize::from(next_index)];
        if previous == 0 {
            progress += 10;
        }
    }
    trace
}

#[test]
fn immutable_table_guest_visits_and_caps() {
    assert_eq!(
        trace([2, 7, 5, 0], false, 16384),
        [(12, 0), (15, 2), (19, 7), (24, 5)]
    );
    assert_eq!(
        trace([0, 0, 0, 9], false, 16384),
        [(12, 0), (25, 0), (39, 0), (54, 0)]
    );
    assert_eq!(
        trace([2, 0, 5, 9], false, 16384),
        [(12, 0), (15, 2), (29, 0), (34, 5)]
    );
    assert_eq!(trace([0, 2, 7, 5], true, 1), [(11, 0)]);
    assert_eq!(trace([0, 2, 7, 5], true, 2), [(12, 0), (14, 2)]);
}

#[cfg(windows)]
#[test]
fn immutable_table_matches_gpu() {
    check_gpu(
        "emit::lower::retry::tests::immutable_table_matches_gpu",
        false,
    );
}

#[cfg(windows)]
#[test]
fn immutable_table_paired_matches_gpu() {
    check_gpu(
        "emit::lower::retry::tests::immutable_table_paired_matches_gpu",
        true,
    );
}

#[cfg(windows)]
fn check_gpu(name: &str, paired: bool) {
    use crate::emit::halves_tests::{compare_dispatches, in_bounded_gpu_child, shader};
    use crate::emit::host::{Stride, View};
    use crate::emit::validator::Validator;
    if !in_bounded_gpu_child(name) {
        return;
    }
    let mut validator = Validator::load(Validator::default_path().as_deref());
    if let Err(reason) = &validator {
        eprintln!("DXIL validator unavailable: {reason}");
    }
    const MARK: u32 = 0xa25c_19d3;
    let seed = [MARK; 512];
    let input = [63; 8];
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &input,
            stride: Stride::Raw,
        },
        View::RawUavData(&seed),
    ];
    for (words, next, publish) in [
        ([2, 7, 5, 0], false, true),
        ([0, 0, 0, 9], false, true),
        ([0, 2, 7, 5], true, !paired),
        ([2, 0, 5, 9], false, true),
    ] {
        let n = fixture_address(words, next, publish, paired);
        let result = trace(words, next, crate::emit::loop_limit() as usize);
        let &(progress, previous) = result.last().unwrap();
        let mut expected = seed;
        for lane in expected.as_chunks_mut::<8>().0 {
            lane[..3].copy_from_slice(&[progress, previous, input[0]]);
        }
        // Lane zero writes on outer entries; other lanes retain the sentinel.
        expected[3] = result
            .iter()
            .rev()
            .find(|(_, prior)| *prior == 0)
            .unwrap()
            .0;
        let legal = shader(emit(&n).unwrap());
        if let Ok(v) = &mut validator {
            v.validate(&mut legal.dxil.clone()).unwrap();
        }
        let forms = [("publication wave32", legal)];
        let (errors, dispatches) = compare_dispatches(&forms, 1, 8, &expected, &views);
        assert!(dispatches > 0, "no physical wave32 device");
        assert!(errors.is_empty(), "{}", errors.join("\n"));
    }
}

#[test]
fn immutable_table_rebuilds_exits_joins_and_strict_sources() {
    for (words, selected) in [([2, 7, 5, 0], false), ([0, 0, 0, 9], true)] {
        let n = table_loop(words);
        let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
        let track = |p: &Program| {
            let st = crate::structure::split_literal(p).unwrap();
            let res = crate::resource::track(p, &n.layout, n.stage, Some(&st)).unwrap();
            (st, res)
        };
        let (st, res) = track(&p);
        let before = immutable::prove(&p, &st, &res, &n);
        let [((b, _), _)] = before.reads.iter().collect::<Vec<_>>()[..] else {
            panic!("one CODE read")
        };
        let Exit::Branch { taken, not_taken } = p.blocks[*b].exit else {
            panic!("two targets")
        };
        let target = if selected { taken } else { not_taken };
        assert_eq!(before.jumps, [(*b, target)]);
        let q = crate::ir::folded(&p, &before.jumps).unwrap().unwrap();
        let (st, res) = track(&q);
        let after = immutable::prove(&q, &st, &res, &n);
        assert!(after.jumps.is_empty());
        let (&(latch, _), contents) = after.reads.iter().next().unwrap();
        assert_eq!(*contents, words[..3].iter().copied().collect());
        let Exit::Jump(to) = q.blocks[latch].exit else {
            panic!("one counted edge")
        };
        assert_eq!(q.blocks[to].offset, p.blocks[target].offset);
        let order = crate::ir::reverse_postorder(&q.blocks);
        let rank = |b| order.iter().position(|&i| i == b).unwrap();
        assert!(rank(to) <= rank(latch));
        if !selected {
            let block = &q.blocks[to];
            let phi = |r| {
                block
                    .phis
                    .iter()
                    .find(|p| p.reg == Reg::S(r))
                    .unwrap()
                    .value
            };
            assert_eq!(st.mask_inputs[&(phi(28), phi(29))].len(), block.preds.len());
        }
        let mut inner = 0;
        for (b, block) in q.blocks.iter().enumerate() {
            let preds: Vec<_> = q
                .blocks
                .iter()
                .enumerate()
                .filter(|(_, p)| p.exit.successors().contains(&b))
                .map(|(p, _)| p)
                .collect();
            assert_eq!(block.preds, preds);
            assert!(block.phis.iter().all(|p| p.args.len() == preds.len()));
            if let LaneExit::Branch {
                taken, not_taken, ..
            } = st.exits[b]
                && (rank(taken) <= rank(b) || rank(not_taken) <= rank(b))
            {
                inner += 1;
            }
        }
        assert_eq!(
            inner, 1,
            "one conditional counter with its own ordinary exit"
        );
        let io = crate::io::raise(&q, &n).unwrap();
        let x = parts(&n, &q, &st, &res, &io, true, true);
        assert!(matches!(
            super::super::literal_publish::prove(&x, 64, 0, &HashMap::new()),
            Ok(Some(_))
        ));
    }
}

#[test]
fn immutable_table_declines_incomplete_index_proofs() {
    for paired in [false, true] {
        check_incomplete_index_proofs(fixture_address([2, 7, 5, 0], false, true, paired));
    }
}

fn check_incomplete_index_proofs(n: NativeProgram) {
    let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
    let st = crate::structure::split_literal(&p).unwrap();
    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let no_read =
        |p: &Program, r: &Resources| assert!(immutable::prove(p, &st, r, &n).reads.is_empty());
    let mut unknown = res.clone();
    let entry = unknown
        .scalar
        .iter_mut()
        .find(|(_, a)| a.param == CODE)
        .unwrap();
    let site = entry.0;
    entry.1.param = MEMORY;
    no_read(&p, &unknown);
    let (b, i) = p
        .blocks
        .iter()
        .enumerate()
        .find_map(|(b, block)| {
            block
                .phis
                .iter()
                .position(|p| p.reg == Reg::S(20))
                .map(|i| (b, i))
        })
        .unwrap();
    let mut missing = p.clone();
    missing.blocks[b].phis[i].args.pop();
    no_read(&missing, &res);
    let mut changed = p.clone();
    let phi = &mut changed.blocks[b].phis[i];
    phi.args.fill(phi.value);
    no_read(&changed, &res);
    // An extra entry into the body carries a value with no numeric proof.
    let mut extra = p.clone();
    let inner = extra
        .blocks
        .iter()
        .position(|block| {
            !block.preds.contains(&0) && block.phis.iter().any(|phi| phi.reg == Reg::S(20))
        })
        .unwrap();
    let Exit::Jump(outer) = extra.blocks[0].exit else {
        panic!("ordinary preheader")
    };
    extra.blocks[0].exit = Exit::Branch {
        taken: inner,
        not_taken: outer,
    };
    extra.blocks[inner].preds.insert(0, 0);
    for phi in &mut extra.blocks[inner].phis {
        phi.args.insert(0, p.inputs[0].1);
    }
    no_read(&extra, &res);
    // Removing the exit test's numeric input admits an unbounded sequence.
    // These are direct admission queries, never submitted for lowering.
    let mut unbounded = p.clone();
    unbounded
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.insts)
        .find(|i| i.name() == "s_cmp_eq_u32" && i.read(Reg::S(20)).is_some())
        .unwrap()
        .reads
        .clear();
    no_read(&unbounded, &res);
    for (original, word, replacement) in [
        ("s_add_u32", 0x9314_8114u32, "s_mul_i32"),
        ("s_cmp_eq_u32", 0xbf08_8414, "s_cmp_gt_u32"),
    ] {
        let mut other = p.clone();
        let inst = other
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.insts)
            .find(|i| i.name() == original && i.read(Reg::S(20)).is_some())
            .unwrap();
        let bytes = [word, 0xbf81_0000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut machine = crate::gfx10::decode(&bytes).unwrap().remove(0);
        assert_eq!(machine.name(), Some(replacement));
        machine.offset = inst.machine.offset;
        inst.fields = machine.fields(true);
        inst.machine = machine;
        no_read(&other, &res);
    }
    assert_eq!(p.blocks[site.0].insts[site.1].name(), "s_load_dword");
}

#[test]
fn immutable_table_validates_remaining_producers_on_folded_legacy_graph() {
    let n = fixture_with_input([2, 7, 5, 0], false, false);
    assert!(
        emit(&n).is_ok(),
        "legal folded program with zero publications"
    );
    let src = crate::ir::build(&n.code, true, n.vgprs).unwrap();
    let tracked = |p: &Program| {
        let st = crate::structure::split_literal(p).unwrap();
        crate::resource::track(p, &n.layout, n.stage, Some(&st)).unwrap()
    };
    let facts = immutable::prove(
        &src,
        &crate::structure::split_literal(&src).unwrap(),
        &tracked(&src),
        &n,
    );
    assert_eq!(facts.jumps.len(), 1);
    let q = crate::ir::folded(&src, &facts.jumps).unwrap().unwrap();
    let res = tracked(&q);
    let st = crate::structure::split_literal(&q).unwrap();
    let io = crate::io::raise(&q, &n).unwrap();
    let x = parts(&n, &q, &st, &res, &io, true, true);
    assert!(matches!(
        super::super::literal_publish::prove(&x, 64, 0, &HashMap::new()),
        Ok(None)
    ));
    let site = q
        .blocks
        .iter()
        .enumerate()
        .find_map(|(b, block)| {
            block
                .insts
                .iter()
                .position(|i| i.write(Reg::S(26)).is_some())
                .map(|i| (b, i))
        })
        .unwrap();
    // Query the final range gate only. Supply each producer class at the
    // independent definition site. The original computed table stays live.
    for (op, buffer, fixed_displacement) in [
        ([0xf404_0684, 0xfa00_0000], false, true), // scalar pair
        ([0xdc34_8000, 0x0604_0010], false, true), // global pair
        ([0xf424_0696, 0xfa00_0004], true, true),  // scalar buffer pair, displacement four
        ([0xf424_0696, 0x2800_0004], true, false), // scalar buffer pair, additional SGPR
    ] {
        let mut p = q.clone();
        let inst = &mut p.blocks[site.0].insts[site.1];
        let bytes = [op[0], op[1], 0xbf81_0000u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut machine = crate::gfx10::decode(&bytes).unwrap().remove(0);
        let offset = inst.machine.offset;
        machine.offset = offset;
        inst.fields = machine.fields(true);
        inst.machine = machine;
        let unknown = p.inputs[0].1;
        let end = u32::try_from(n.code.len()).unwrap();
        let last_base = end - 8 - if buffer { 4 } else { 0 };
        for (base, dynamic, ok) in [
            (last_base, None, true),
            (last_base, Some(unknown), false),
            (last_base + 4, None, false),
        ] {
            let mut resources = res.clone();
            let address = Address {
                param: CODE,
                offset: base,
                dynamic,
            };
            if buffer {
                resources.descriptors.push(crate::resource::Descriptor {
                    site,
                    role: Role::Resource,
                    source: Source::Built {
                        address,
                        bits: 0,
                        words: [u32::MAX, 0x1101_4fac],
                    },
                });
            } else {
                resources.scalar.push((site, address));
            }
            let facts = immutable::prove(&p, &st, &resources, &n);
            assert_eq!(
                facts.reads.len(),
                1,
                "the independent table still has its proof"
            );
            let result =
                super::super::literal_publish::folded_code(&p, &resources, &n, &facts.reads);
            if ok && fixed_displacement {
                result.unwrap();
            } else {
                assert!(
                    matches!(result,Err(ShaderError::Unsupported{offset:at,what}) if at==offset && what.contains("proved byte range"))
                );
            }
        }
    }
}

#[test]
fn immutable_table_folded_roles_without_handles_or_publication() {
    let mut n = fixture_with_input([2, 7, 5, 0], false, false);
    // Keep the CODE reader and the numeric mask consumer in the original
    // loop. Removing its output stores and descriptor read leaves no handles.
    for inst in crate::gfx10::decode(&n.code).unwrap() {
        if matches!(inst.name(), Some("buffer_store_dword" | "s_load_dwordx4")) {
            for word in n.code[inst.offset..inst.offset + 4 * inst.words.len()]
                .as_chunks_mut::<4>()
                .0
            {
                word.copy_from_slice(&0xbf80_0000u32.to_le_bytes());
            }
        }
    }
    let p = crate::ir::build(&n.code, true, n.vgprs).unwrap();
    let st = crate::structure::split_literal(&p).unwrap();
    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    let jump = immutable::prove(&p, &st, &res, &n).jumps;
    assert_eq!(jump.len(), 1);
    let q = crate::ir::folded(&p, &jump).unwrap().unwrap();
    let st = crate::structure::split_literal(&q).unwrap();
    let res = crate::resource::track(&q, &n.layout, n.stage, Some(&st)).unwrap();
    assert!(res.descriptors.is_empty());
    assert!(
        res.scalar
            .iter()
            .any(|(_, a)| a.param == CODE && a.dynamic.is_some())
    );
    let io = crate::io::raise(&q, &n).unwrap();
    let x = parts(&n, &q, &st, &res, &io, true, true);
    assert!(matches!(
        super::super::literal_publish::prove(&x, 64, 0, &HashMap::new()),
        Ok(None)
    ));
    super::super::literal_publish::folded_roles(&x, &Set::default()).unwrap();
    let defs = || q.blocks.iter().flat_map(|b| &b.insts);
    let pc = defs().find(|i| i.name() == "s_getpc_b64").unwrap().writes[0].1;
    let root = res
        .params
        .iter()
        .find(|(_, p, d)| (*p, *d) == (0, 0))
        .unwrap()
        .0;
    let integer = defs().find_map(|i| i.write(Reg::S(26))).unwrap();
    // These are source-role queries only. No guest encoding or addresses are
    // changed, and none of the substituted metadata goes to lowering.
    for (value, allowed) in [(integer, true), (pc, false), (root, false)] {
        let mut query = q.clone();
        let consumer = query
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.insts)
            .find(|i| i.write(Reg::S(EXEC)).is_some() && i.read(Reg::S(28)).is_some())
            .unwrap();
        consumer
            .reads
            .iter_mut()
            .find(|(reg, _)| *reg == Reg::S(28))
            .unwrap()
            .1 = value;
        let offset = consumer.machine.offset;
        let checked = super::super::literal_publish::folded_roles(
            &Parts {
                p: &query,
                ..x.clone()
            },
            &Set::default(),
        );
        if allowed {
            checked.unwrap();
        } else {
            assert!(
                matches!(checked, Err(ShaderError::Unsupported{offset:at,what}) if at==offset && what.contains("without guest-exact bits"))
            );
        }
    }
}
