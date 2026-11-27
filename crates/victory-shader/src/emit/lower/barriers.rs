//! Where group barriers can go when the host splits a compute group into
//! waves the console's lockstep did not have: the blocks every lane of the
//! group runs together ([`in_step`]), and the barriers a group that is one
//! console wave needs between its lanes' LDS accesses
//! ([`split_wave_barriers`]).

use super::*;

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barriers between one lane's LDS access and another's. Here a read
/// (or atomic) of LDS that a write or atomic may have changed since the last
/// barrier gets one before it, and so does a write (or atomic) after a read
/// or atomic since then, so that it cannot overwrite what another lane has
/// yet to read. (A write after a write, which the console orders too, gets
/// none.) Barriers need every lane of the group, so they go only in blocks
/// every lane runs together, the same number of times ([`in_step`]): once,
/// or in a loop on a count every host wave computes alike. Such a block that
/// leads into code only some lanes run syncs at its end where that code
/// could otherwise read a write, or overwrite a read, made before it.
///
/// `None`: a read in divergent code may see a write made in divergent code
/// since the last barrier. A write in divergent code after a read there gets
/// no barrier, as before.
pub(crate) fn split_wave_barriers(p: &Program, st: &Structured) -> Option<Set<(usize, usize)>> {
    /// LDS accesses since the last barrier.
    #[derive(Clone, Copy, Default, PartialEq)]
    struct Since {
        read: bool,
        written: bool,
    }
    impl Since {
        fn or(self, o: Since) -> Since {
            Since {
                read: self.read || o.read,
                written: self.written || o.written,
            }
        }
    }
    // (reads, writes) of an LDS access; an atomic does both.
    let access = |inst: &Inst| {
        let name = inst.name();
        if !matches!(inst.fields, Fields::Ds { .. })
            || name.starts_with("ds_swizzle")
            || name.contains("permute")
        {
            return None;
        }
        Some((
            !name.starts_with("ds_write"),
            !name.starts_with("ds_read"),
        ))
    };
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let step = step_and_cap(p, st, true).0;
    let divergent_writes = (0..n).any(|b| {
        !step[b]
            && p.blocks[b]
                .insts
                .iter()
                .any(|i| access(i).is_some_and(|(_, w)| w))
    });
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut out = vec![Since::default(); n];
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut since = p.blocks[b]
                .preds
                .iter()
                .fold(Since::default(), |a, &q| a.or(out[q]));
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                if inst.name() == "s_barrier" {
                    since = Since::default();
                }
                let Some((reads, writes)) = access(inst) else {
                    continue;
                };
                // A read of what another lane may have written; a write over
                // what another lane may have yet to read.
                let stale = reads && since.written;
                let early = writes && since.read;
                if stale || early {
                    if step[b] {
                        at.insert((b, i));
                        since = Since::default();
                    } else if stale {
                        return None;
                    }
                }
                since = since.or(Since {
                    read: reads,
                    written: writes,
                });
            }
            if step[b]
                && succ(b).iter().any(|&s| !step[s])
                && (since.written || since.read && divergent_writes)
            {
                at.insert((b, p.blocks[b].insts.len()));
                since = Since::default();
            }
            // What reaches a block only grows, so this ends; barriers placed
            // for more than reaches it are still right.
            let since = out[b].or(since);
            changed |= std::mem::replace(&mut out[b], since) != since;
        }
        if !changed {
            return Some(at);
        }
    }
}

/// Per block, whether every lane of the group runs it together, the same
/// number of times, however the host splits the group into waves, so that a
/// group barrier can go there: outside the region of every branch whose
/// condition may differ between waves ([`halves::branch_regions`]). That is
/// a per-lane branch, or one on a scalar not computed alike in every wave.
/// Every block each lane runs once ([`once_per_lane`]) is one; so is a loop
/// whose count every wave computes alike, from the group's inputs,
/// constants and scalar loads through scalar code (not from lane masks,
/// other lanes' values or hardware state). Such a scalar is the same in
/// every console wave of the group too.
///
/// Scalars are assumed alike until shown otherwise (a loop counter needs
/// itself), and the regions and the scalars are narrowed together to a
/// fixed point. With an indirect branch, the blocks each lane runs once.
///
/// The other half is per block whether a lane may have left the program
/// early at a capped back edge's iteration limit before reaching it. Lowering
/// caps an unconditional back edge with an early return at the loop
/// iteration limit (`LOOP_LIMIT`): a lane whose count reaches it leaves the
/// program from the latch. The header's counter is shared by *every* back
/// edge to it, so the cap check holds for every lane only when the latch
/// and the header run together and so does each back edge to the header — a
/// conditional one with a uniform condition. Otherwise one wave's count can
/// reach the limit at the latch while another goes on, and that wave leaves
/// there: no block reachable from that loop is run by every lane of the
/// group, the same number of times, so those blocks are not in step either.
/// The escape reaches past the loop too, along the sibling paths of every
/// divergent branch the latch is reachable from: a lane that took the other
/// arm never passes the latch's cap, so a barrier after that branch can
/// still be reached while a lane in the loop arm has left at the cap. The
/// walk back from the latch stops at a block every lane runs together,
/// though: every lane reconverges there before the cap exit, so a divergent
/// branch earlier than that, whose arms already reconverged there, is no
/// escape.
///
/// With `need_step` false the step half is left out unless the program has
/// an `s_barrier` after a capped back edge, the only reader of the escape
/// half.
pub(crate) fn step_and_cap(
    p: &Program,
    st: &Structured,
    need_step: bool,
) -> (Vec<bool>, Vec<bool>) {
    let n = p.blocks.len();
    // Reverse postorder ranks, as lowering numbers them: a Jump to a block at
    // or before the jump is a back edge.
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    let back = |b: usize, t: usize| rpo[b] != usize::MAX && rpo[t] <= rpo[b];
    // The unconditional back edges, (latch, header): lowering counts the
    // loop's iterations and at the limit leaves the program from the latch
    // with an early return.
    let capped: Vec<(usize, usize)> = (0..n)
        .filter_map(|b| match st.exits[b] {
            LaneExit::Jump(t) if back(b, t) => Some((b, t)),
            _ => None,
        })
        .collect();
    if !need_step
        && (capped.is_empty()
            || !p
                .blocks
                .iter()
                .any(|b| b.insts.iter().any(|i| i.name() == "s_barrier")))
    {
        return (Vec::new(), vec![false; n]);
    }
    let Some(regions) = halves::branch_regions(p, st) else {
        return (
            if need_step {
                once_per_lane(p)
            } else {
                Vec::new()
            },
            vec![false; n],
        );
    };
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    // Every back edge to each header: the header's iteration counter is
    // shared by all of them.
    let mut back_to: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        match st.exits[b] {
            LaneExit::Jump(t) if back(b, t) => back_to[t].push(b),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                if back(b, taken) {
                    back_to[taken].push(b);
                }
                if back(b, not_taken) {
                    back_to[not_taken].push(b);
                }
            }
            _ => {}
        }
    }
    let scalar = |r: Reg| matches!(r, Reg::S(_) | Reg::Scc);
    // Not EXEC or VCC: lane masks.
    let mask = |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    // Scalar writes no host wave can compute alike: other lanes' values,
    // hardware state, atomics, coherent loads (the group may write them).
    let own = |inst: &Inst| {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { .. } => {
                name.starts_with("s_getreg")
                    || name.starts_with("s_sendmsg")
                    || name.contains("waveid")
            }
            Fields::Smem { cache, .. } => {
                !(name.starts_with("s_load") || name.starts_with("s_buffer_load")) || cache.glc
            }
            _ => true,
        }
    };
    let together = |when: Bool, alike: &Set<Value>| {
        st.bools
            .support(when)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(v) if alike.contains(v)))
    };
    let mut alike: Set<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| scalar(*r))
        .map(|(_, v)| *v)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| scalar(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1)),
                )
        }))
        .collect();
    loop {
        let mut step = vec![true; n];
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = st.exits[*d] else {
                continue;
            };
            if !together(when, &alike) {
                for (b, &r) in region.iter().enumerate() {
                    step[b] &= !r;
                }
            }
        }
        // A capped back edge's cap check holds for every lane only when the
        // latch and the header run together and so does every back edge to
        // the header (a conditional one with a uniform condition): one the
        // lanes do not run together can put one wave's count over the limit
        // at the latch while another goes on, and that wave leaves the
        // program there — so no block reachable from the loop is run by
        // every lane of the group, the same number of times.
        let cap_uniform = |latch: usize, header: usize, step: &[bool], alike: &Set<Value>| {
            step[latch]
                && step[header]
                && back_to[header].iter().all(|&b| {
                    if !step[b] {
                        return false;
                    }
                    match st.exits[b] {
                        LaneExit::Jump(t) => t == header,
                        LaneExit::Branch {
                            when,
                            taken,
                            not_taken,
                        } => {
                            (taken == header || not_taken == header)
                                && together(when, alike)
                        }
                        _ => false,
                    }
                })
        };
        let mut escaped = vec![false; n];
        // Reverse adjacency, to walk back from each latch to the branches
        // that can reach it.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (b, ss) in succ.iter().enumerate() {
            for &s in ss {
                preds[s].push(b);
            }
        }
        // Mark every block forward-reachable from `from`.
        let mark = |from: usize, escaped: &mut Vec<bool>| {
            let mut seen = vec![false; n];
            let mut stack = vec![from];
            seen[from] = true;
            while let Some(b) = stack.pop() {
                escaped[b] = true;
                for &s in &succ[b] {
                    if !seen[s] {
                        seen[s] = true;
                        stack.push(s);
                    }
                }
            }
        };
        loop {
            let mut changed = false;
            for &(latch, header) in &capped {
                if cap_uniform(latch, header, &step, &alike) {
                    continue;
                }
                // Whether the latch is in step, before the walk from the
                // header below clears it.
                let latch_in_step = step[latch];
                let mut seen = vec![false; n];
                let mut stack = vec![header];
                seen[header] = true;
                while let Some(b) = stack.pop() {
                    escaped[b] = true;
                    if std::mem::replace(&mut step[b], false) {
                        changed = true;
                    }
                    for &s in &succ[b] {
                        if !seen[s] {
                            seen[s] = true;
                            stack.push(s);
                        }
                    }
                }
                // The sibling paths of every divergent branch the latch is
                // reachable from: a lane that took another arm of such a
                // branch never passes the latch's cap, so it can still reach
                // a barrier after the branch while a lane in the loop arm
                // has left the program at the cap. (The branch's own block
                // every lane runs, and a uniform branch sends every lane the
                // same way, so neither is marked here.)
                let arms = |d: usize, escaped: &mut Vec<bool>| {
                    if let LaneExit::Branch { when, .. } = st.exits[d]
                        && !together(when, &alike)
                    {
                        for &s in &succ[d] {
                            mark(s, escaped);
                        }
                    }
                };
                let mut seen = vec![false; n];
                let mut stack = vec![latch];
                if latch_in_step {
                    // The latch is in step, but a back edge to the header
                    // the lanes do not run together shares its counter all
                    // the same: walk back from those sources too.
                    stack.extend(back_to[header].iter().copied().filter(|&b| !step[b]));
                }
                while let Some(b) = stack.pop() {
                    if std::mem::replace(&mut seen[b], true) {
                        continue;
                    }
                    arms(b, &mut escaped);
                    for &p in &preds[b] {
                        arms(p, &mut escaped);
                        // Walk back only through blocks the lanes do not run
                        // together: at a block every lane runs together they
                        // all reconverge before the cap exit, so an earlier
                        // divergent branch whose arms already reconverged
                        // there cannot strand a lane past the cap.
                        if !step[p] {
                            stack.push(p);
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut next: Set<Value> = p
            .inputs
            .iter()
            .filter(|(r, _)| scalar(*r))
            .map(|(_, v)| *v)
            .collect();
        for (b, block) in p.blocks.iter().enumerate() {
            // Every host wave enters by the same edge.
            let entered = step[b] && block.preds.iter().all(|&q| step[q]);
            for phi in &block.phis {
                if scalar(phi.reg) && entered && phi.args.iter().all(|a| alike.contains(a)) {
                    next.insert(phi.value);
                }
            }
            for inst in &block.insts {
                let from = inst
                    .reads
                    .iter()
                    .all(|&(r, v)| scalar(r) && !mask(r) && alike.contains(&v));
                if from && !own(inst) {
                    next.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                }
            }
        }
        if next.len() == alike.len() {
            return (step, escaped);
        }
        alike = next;
    }
}

/// Per block, whether each lane of the group runs it exactly once: no cycle
/// passes through it, and every path from the entry to an end does. A group
/// barrier needs every lane of the group, so it can go only in these.
pub(crate) fn once_per_lane(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}
