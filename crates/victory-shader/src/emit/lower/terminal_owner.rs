//! A terminal EXEC split after a group barrier. Only the physical thread with
//! the selected in-group coordinate may reach the scalar continuation.
use super::*;

pub(super) struct Owner {
    pub(super) gate: usize,
    selector: usize,
    entry: usize,
    end: usize,
    body: Set<usize>,
    pub(super) sites: Set<Site>,
    #[cfg(test)]
    pub(super) emitted: Vec<usize>,
}

fn operands(i: &Inst) -> Option<&[Operand]> {
    let Fields::Valu(v) = &i.fields else {
        return None;
    };
    (v.dpp.is_none()
        && v.sdwa.is_none()
        && v.abs == 0
        && v.neg == 0
        && v.neg_hi == 0
        && v.opsel == 0
        && v.opsel_hi == 0
        && !v.clamp
        && v.omod == 0
        && v.carry.is_none())
    .then_some(v.src.as_slice())
}

fn immediate(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(k) => crate::resource::inline(k),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

fn vector(i: &Inst, op: Operand) -> Option<Value> {
    let Operand::Vector { reg, dwords: 1 } = op else {
        return None;
    };
    i.read(Reg::V(reg))
}

fn pair(i: &Inst, reg: u8) -> Option<(Value, Value)> {
    Some((i.read(Reg::S(reg))?, i.read(Reg::S(reg + 1))?))
}

/// The initial two EXEC words, including exact saves/restores through scalar pairs.
/// A partially overwritten pair or an unchecked phi never proves full participation.
fn full_pairs(p: &Program, order: &[usize]) -> Set<(Value, Value)> {
    let input = |reg| {
        p.inputs
            .iter()
            .find_map(|&(r, v)| (r == Reg::S(reg)).then_some(v))
    };
    let mut full = Set::default();
    if let (Some(lo), Some(hi)) = (input(EXEC), input(EXEC + 1)) {
        full.insert((lo, hi));
    }
    for i in order.iter().flat_map(|&b| &p.blocks[b].insts) {
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, dwords: 2 }),
            src,
            ..
        } = &i.fields
        else {
            continue;
        };
        let from = match (i.name(), src.as_slice()) {
            (
                "s_mov_b64",
                [
                    Operand::Scalar {
                        code: src,
                        dwords: 2,
                    },
                ],
            ) => *src,
            ("s_and_saveexec_b64", _) if *code < EXEC => EXEC,
            _ => continue,
        };
        if pair(i, from).is_some_and(|v| full.contains(&v))
            && let (Some(lo), Some(hi)) = (i.write(Reg::S(*code)), i.write(Reg::S(*code + 1)))
        {
            full.insert((lo, hi));
        }
    }
    full
}

fn full_exec(pairs: &Set<(Value, Value)>, i: &Inst) -> bool {
    pair(i, EXEC).is_some_and(|v| pairs.contains(&v))
}

struct Selection {
    exec: (Value, Value),
    expected: u32,
    source: Value,
    empty: usize,
    live: usize,
}

/// A real lane-local negative test of the two EXEC words written by this
/// comparison, with the same empty/nonempty successors as the guest branch.
fn selection(x: &Parts, b: usize, full: &Set<(Value, Value)>) -> Option<Selection> {
    let block = &x.p.blocks[b];
    let [.., cmp, branch] = block.insts.as_slice() else {
        return None;
    };
    let LaneExit::Branch {
        when,
        taken,
        not_taken,
    } = x.st.exits[b]
    else {
        return None;
    };
    let [k, v] = operands(cmp)? else { return None };
    let (lo, hi) = pair(branch, EXEC)?;
    (cmp.name() == "v_cmpx_eq_u32"
        && full_exec(full, cmp)
        && x.st.guards[b][block.insts.len() - 2].is_none()
        && cmp.write(Reg::S(EXEC)) == Some(lo)
        && cmp.write(Reg::S(EXEC + 1)) == Some(hi)
        && branch.name() == "s_cbranch_execz"
        && block.exit == (Exit::Branch { taken, not_taken })
        && x.st.bools.node(when) == Some((Atom::Lane(lo), Bool::TRUE, Bool::FALSE))
        && x.st.reach[b] == Bool::TRUE)
        .then_some(Selection {
            exec: (lo, hi),
            expected: immediate(*k)?,
            source: vector(cmp, *v)?,
            empty: taken,
            live: not_taken,
        })
}

fn flat_coordinate(x: &Parts, flat: Value, width: u32, full: &Set<(Value, Value)>) -> bool {
    let entry = &x.p.blocks[0].insts;
    let def = |v| {
        entry.iter().enumerate().find_map(|(at, i)| {
            (i.writes.iter().any(|&(_, w)| w == v)
                && full_exec(full, i)
                && x.st.guards[0][at].is_none())
            .then_some(i)
        })
    };
    let coordinate = |i: &Inst, op, dim| {
        vector(i, op).is_some_and(|v| x.io.inputs.contains(&(v, Input::ThreadIdInGroup(dim))))
    };
    let Some(i) = def(flat) else { return false };
    let shift = |op| immediate(op).is_some_and(|k| 1u32.checked_shl(k) == Some(width));
    match (i.name(), operands(i)) {
        ("v_lshl_add_u32", Some(&[y, k, x_id])) => {
            coordinate(i, y, 1) && shift(k) && coordinate(i, x_id, 0)
        }
        ("v_add_nc_u32", Some(&[a, b])) => [(a, b), (b, a)].into_iter().any(|(a, b)| {
            coordinate(i, a, 0)
                && vector(i, b).and_then(def).is_some_and(|s| {
                    s.name() == "v_lshlrev_b32"
                        && matches!(operands(s), Some(&[k, y]) if shift(k) && coordinate(s, y, 1))
                })
        }),
        _ => false,
    }
}

pub(super) fn find(x: &Parts, order: &[usize]) -> Option<Owner> {
    let StageIo::Cs(cs) = x.native.io else {
        return None;
    };
    let [width, height, 1] = cs.threads else {
        return None;
    };
    let group = width.checked_mul(height)?;
    if !group.is_multiple_of(64)
        || x.st.literal
        || !x.res.waterfall.compares.is_empty()
        || !x.res.waterfall.reads.is_empty()
    {
        return None;
    }
    // Most compute programs have no such terminal barrier/compare at all.
    let mut gates = order.iter().copied().filter(|&b| matches!(x.p.blocks[b].insts.as_slice(),
        [.., barrier, cmp, branch] if barrier.name() == "s_barrier" && cmp.name() == "v_cmpx_eq_u32" && branch.name() == "s_cbranch_execz"));
    gates.clone().next()?;
    let full = full_pairs(x.p, order);
    gates.find_map(|gate| {
        let retest = selection(x, gate, &full)?;
        let (entry, end) = (retest.live, retest.empty);
        let bb = &x.p.blocks[gate];
        if retest.expected != u32::MAX
            || end == entry
            || x.p.blocks[end].exit != Exit::End
            || !x.p.blocks[end].phis.is_empty()
            || !x.p.blocks[end].insts.iter().all(|i| i.name() == "s_endpgm")
        {
            return None;
        }
        let phi = bb.phis.iter().find(|p| p.value == retest.source)?;
        let [a, b] = bb.preds[..] else { return None };
        let [z, v] = phi.args[..] else { return None };
        let (selector, zero, body, one) = match (a, b) {
            (a, b) if x.p.blocks[b].preds == [a] => (a, z, b, v),
            (a, b) if x.p.blocks[a].preds == [b] => (b, v, a, z),
            _ => return None,
        };
        let select = selection(x, selector, &full)?;
        let mov = |b: usize, v, val| {
            x.p.blocks[b].insts.iter().enumerate().find_map(|(at, i)| {
                (i.write(phi.reg) == Some(v)
                    && i.name() == "v_mov_b32"
                    && x.st.guards[b][at].is_none()
                    && matches!(operands(i), Some(&[k]) if immediate(k) == Some(val)))
                .then_some(i)
            })
        };
        let init = mov(selector, zero, 0)?;
        let set = mov(body, one, retest.expected)?;
        if x.st.exits[body] != LaneExit::Jump(gate)
            || (select.empty, select.live) != (gate, body)
            || select.expected >= group
            || !flat_coordinate(x, select.source, width, &full)
            || !full_exec(&full, init)
            || set.read(phi.reg) != Some(zero)
            || pair(set, EXEC) != Some(select.exec)
        {
            return None;
        }
        let mut owner = Owner {
            gate,
            selector,
            entry,
            end,
            body: Set::default(),
            sites: Set::default(),
            #[cfg(test)]
            emitted: Vec::new(),
        };
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            if b == end || !owner.body.insert(b) {
                continue;
            }
            let block = &x.p.blocks[b];
            if b == gate
                || b == 0
                || block.exit == Exit::Indirect
                || !owner_branch(x, b, retest.exec)
                || block
                    .phis
                    .iter()
                    .any(|p| matches!(p.reg, Reg::S(126 | 127)))
            {
                return None;
            }
            for (j, i) in block.insts.iter().enumerate() {
                if i.writes
                    .iter()
                    .any(|&(r, _)| matches!(r, Reg::S(126 | 127)))
                    || x.st.guards[b][j].is_some()
                    || !lane_local(i)
                    || reads_mask_as_data(x.st, i)
                {
                    return None;
                }
                if i.name() == "v_readfirstlane_b32" {
                    if pair(i, EXEC) != Some(retest.exec)
                        || !matches!(operands(i), Some(&[Operand::Vector { dwords: 1, .. }]))
                    {
                        return None;
                    }
                    owner.sites.insert((b, j));
                }
            }
            stack.extend(block.exit.successors());
        }
        (!owner.sites.is_empty()
            && owner.body.iter().all(|&b| {
                x.p.blocks[b]
                    .preds
                    .iter()
                    .all(|&q| owner.body.contains(&q) || (q == gate && b == entry))
            }))
        .then_some(owner)
    })
}

/// With only one active guest bit, a VCC vote is local only when its other
/// bits are clear. Check the exact pair consumed by the scalar branch.
fn owner_branch(x: &Parts, b: usize, exec: (Value, Value)) -> bool {
    let LaneExit::Branch { when, .. } = x.st.exits[b] else {
        return true;
    };
    let block = &x.p.blocks[b];
    let Some((branch, defs)) = block.insts.split_last() else {
        return false;
    };
    match branch.name() {
        "s_cbranch_scc0" | "s_cbranch_scc1" => {
            let Some(scc) = branch.read(Reg::Scc) else {
                return false;
            };
            let (lo, hi) = if branch.name() == "s_cbranch_scc1" {
                (Bool::FALSE, Bool::TRUE)
            } else {
                (Bool::TRUE, Bool::FALSE)
            };
            // A mask-result SCC votes over all 64 bits. Reach can simplify its
            // owner bit to a constant even when the guest vote is the opposite.
            x.st.bools.node(when) == Some((Atom::Uniform(scc), lo, hi))
                && defs.iter().any(|i| {
                    let name = i.name();
                    i.write(Reg::Scc) == Some(scc)
                        && name.starts_with("s_cmp_")
                        && (name.ends_with("_i32") || name.ends_with("_u32"))
                })
        }
        "s_cbranch_vccz" | "s_cbranch_vccnz" => {
            let Some((lo, hi)) = pair(branch, VCC) else {
                return false;
            };
            x.st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_) | Atom::Lane(_))) &&
                defs.iter().any(|i| matches!(i.name(), "s_and_b64" | "s_andn2_b64")
                    && i.write(Reg::S(VCC)) == Some(lo) && i.write(Reg::S(VCC+1)) == Some(hi)
                    && matches!(&i.fields, Fields::Sop { src, .. } if matches!(src.first(), Some(Operand::Scalar { code: EXEC, dwords: 2 })))
                    && pair(i, EXEC) == Some(exec))
        }
        _ => false,
    }
}

/// No inter-lane source, implicit derivative, atomic lock or barrier can
/// require participation by a lane that took the terminal exit.
fn lane_local(i: &Inst) -> bool {
    let n = i.name();
    match &i.fields {
        Fields::Valu(v) => {
            v.dpp.is_none()
                && !n.contains("perm")
                && !n.contains("lane")
                && !n.starts_with("v_mbcnt")
                || n == "v_readfirstlane_b32"
        }
        Fields::Ds { .. } => n.starts_with("ds_read") && !n.contains("lane"),
        Fields::Buffer { lds: false, .. } => n.contains("load") || n.contains("store"),
        Fields::Image { .. } => matches!(
            n,
            "image_load" | "image_store" | "image_gather4_lz" | "image_sample_lz"
        ),
        Fields::Smem { .. } => n.starts_with("s_load_") || n.starts_with("s_buffer_load_"),
        Fields::Sopp { .. } => {
            n == "s_endpgm"
                || n == "s_branch"
                || n.starts_with("s_cbranch_")
                || n.starts_with("s_waitcnt")
                || n == "s_nop"
        }
        Fields::Sop { .. } => !n.contains("saveexec") && !n.contains("wqm"),
        _ => false,
    }
}

impl Owner {
    pub(super) fn overlaps_meet(&self, meet: &meet::Meet) -> bool {
        meet.regions.iter().any(|r| {
            self.body.contains(&r.branch)
                || self.body.iter().any(|&b| r.blocks[b])
                || r.blocks[self.end]
                || r.join
                    .is_some_and(|b| b == self.end || self.body.contains(&b))
        })
    }

    pub(super) fn preserved(
        &self,
        p: &Program,
        st: &Structured,
        exits: &[LaneExit],
        plan: &predicate::PredicatePlan,
    ) -> bool {
        // Completed prefix skip guards restore their outer predicate at their joins.
        // A wider or future rewrite must prove physical participation separately.
        if !plan
            .exits
            .values()
            .all(|e| matches!(e, predicate::FlatExit::SkipGuard { .. }))
            || !plan.step[self.gate]
            || !plan.step[self.selector]
            || exits[self.gate] != st.exits[self.gate]
            || exits[self.end] != LaneExit::End
        {
            return false;
        }
        for mut b in [self.gate, self.selector] {
            while b != 0 {
                if plan.skip_bodies.contains_key(&b) {
                    return false;
                }
                b = plan.skip_joins.get(&b).copied().unwrap_or(plan.idom[b]);
            }
        }
        if self.body.iter().any(|&b| {
            exits[b] != st.exits[b]
                || plan.step[b]
                || plan.skip_bodies.contains_key(&b)
                || plan.skip_joins.contains_key(&b)
        }) {
            return false;
        }
        // Check incoming edges again on the CFG actually emitted.
        exits.iter().enumerate().all(|(q, &exit)| {
            let ok = |b| {
                !self.body.contains(&b)
                    || self.body.contains(&q)
                    || (q == self.gate && b == self.entry)
            };
            match exit {
                LaneExit::Jump(b) => ok(b),
                LaneExit::Branch {
                    taken, not_taken, ..
                } => ok(taken) && ok(not_taken),
                LaneExit::End | LaneExit::Indirect => true,
            }
        }) && p.blocks[self.entry].preds == [self.gate]
    }
}

#[cfg(test)]
impl Owner {
    pub(super) fn verify_emission(&self, l: &Lower) {
        assert_eq!(self.emitted.len(), self.sites.len());
        let mut blocks = vec![l.tail[self.gate].unwrap(), l.block[self.entry].unwrap()];
        blocks.extend(&self.emitted);
        let (edges, sites) = l.s.m.written_cfg(&blocks);
        let (&[gate, entry], reads) = sites.split_at(2) else {
            unreachable!()
        };
        let reach = |start, stop| {
            let mut reached = Set::default();
            let mut work = vec![start];
            while let Some(b) = work.pop() {
                if Some(b) != stop && reached.insert(b) {
                    work.extend(&edges[b]);
                }
            }
            reached
        };
        let &[a, b] = edges[gate].as_slice() else {
            panic!("terminal owner lost its branch")
        };
        let arms = [reach(a, None), reach(b, None)];
        let live = usize::from(!arms[0].contains(&entry));
        assert!(arms[live].contains(&entry) && !arms[1 - live].contains(&entry));
        let bypass = reach(0, Some(gate));
        assert!(
            reads.iter().all(|r| arms[live].contains(r)
                && !arms[1 - live].contains(r)
                && !bypass.contains(r))
        );
        assert!(
            l.s.m
                .written_call_blocks("dx.op.barrier")
                .iter()
                .all(|b| !arms[live].contains(b))
        );
    }
}
