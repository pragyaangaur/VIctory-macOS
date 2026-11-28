//! Selective divergent-branch & lockstep-loop predication for wave64-on-32-lane
//! (`halves`) compute shaders.
//!
//! Cross-half exchanges (`exchange_halves`) and split-wave LDS barriers
//! (`split_wave_barriers`) emit group barriers (`dx.op.barrier`), which
//! require every thread in the threadgroup to reach the block together
//! (`in_step`). Standard compute patterns such as an early-out bounds check
//! (`if (dtid >= limit) return;`), a one-sided guard (`if (cond) { ... }`,
//! including bodies containing uniform loops), an acyclic `if/else` diamond,
//! or a wave-scalarization / per-lane loop (`while (exec != 0)`) put their
//! bodies outside `in_step`.
//!
//! When a wave64-on-32-lane compute shader has such a region needing `in_step`
//! (cross-half wave ops, mask-as-data reads, `v_readfirstlane`/`v_readlane`,
//! or split-wave LDS accesses), this pass rewrites its control flow into
//! predicated lockstep execution:
//! * **Early-out (`if (cond) return;`)**: replaces `CondBr(cond, end, cont)`
//!   with `Br(cont)` and carries `active &= !cond` into every block dominated
//!   by `cont`.
//! * **Skip-guard (`if (cond) { body } -> J`)**: replaces `CondBr(cond, body, J)`
//!   with `Br(body)` without linearizing `body` (so uniform loops and branches
//!   inside `body` keep their CFG), marks `(d, J)` skipped, and lowers `J`'s
//!   phis via `select(act_body, v_body, v_d)`.
//! * **Acyclic SESE diamond (`if (cond) { A } else { B } -> J`)**: linearizes
//!   strictly acyclic two-sided regions (`<= 64` blocks) in topological RPO
//!   order and lowers merge phis into `select` chains.
//! * **Lockstep loop (`header .. latch -> (header, exit)`)**: turns a
//!   single-latch, single-exit divergent loop (with no non-mask SGPRs live out)
//!   into a group-uniform loop whose latch tests `group_any_true(next_active)`
//!   and carries `loop_active` through an `i1` phi at `header`.
//!
//! Every side effect and potentially faulting load in a predicated block is
//! guarded by the block's `active` predicate through `Lower::guard`, while
//! barriers and cross-half exchanges execute at `self.level` in 100%
//! group-uniform control flow.

use super::*;

/// Maximum number of blocks inside one linearized SESE diamond.
const MAX_DIAMOND_BLOCKS: usize = 64;

/// Maximum number of blocks inside one lockstep-predicated loop.
const MAX_LOCKSTEP_LOOP_BLOCKS: usize = 32;

/// How a predicated block's CFG exit is rewritten.
#[derive(Clone, Copy, Debug)]
pub(super) enum FlatExit {
    /// Early-out branch `d -> (end, cont)` rewritten to unconditional
    /// `Br(cont)`, with lanes staying active along `(d, cont)`.
    EarlyOut { cont: usize },
    /// One-sided skip-guard `d -> (body, join)` rewritten to unconditional
    /// `Br(body)`, bypassing `(d, join)` so `body` (including any uniform
    /// loops inside it) runs in lockstep under `active &= body_cond`.
    SkipGuard { body: usize },
    /// Block inside (or at the head of) a linearized SESE diamond jumping
    /// unconditionally to the next block in topological RPO order.
    Linear { next: usize },
    /// Latch of a lockstep-predicated loop branching on
    /// `group_any_true(next_active) && iter < LOOP_LIMIT`.
    LockstepLatch {
        header: usize,
        exit: usize,
        stay_when_true: bool,
    },
    /// The exit branch of a while-shaped lockstep-predicated loop, at a
    /// block other than its latch (which jumps back to `header`): branching
    /// to `stay` on `group_any_true(next_active) && iter < LOOP_LIMIT`, else
    /// to `exit`.
    LockstepExit {
        header: usize,
        stay: usize,
        exit: usize,
        stay_when_true: bool,
    },
}

/// Predication plan computed before `split_wave_barriers` and `Lower::run`.
pub(super) struct PredicatePlan {
    /// Rewritten exits for predicated branches, linearized diamond blocks,
    /// and lockstep loop latches.
    pub(super) exits: Map<usize, FlatExit>,
    /// Blocks inside linearized diamonds (strictly between `d` and `join`).
    pub(super) in_diamond: Set<usize>,
    /// Join blocks of linearized diamonds (`join -> head branch block d`).
    pub(super) diamond_joins: Map<usize, usize>,
    /// Body entry blocks of `SkipGuard`s (`body -> head branch block d`).
    pub(super) skip_bodies: Map<usize, usize>,
    /// Join blocks of `SkipGuard`s (`join -> head branch block d`).
    pub(super) skip_joins: Map<usize, usize>,
    /// Bypassed CFG edges `(d, join)` dropped from DXIL phis in `fill_phis`.
    pub(super) skipped_edges: Set<(usize, usize)>,
    /// Blocks whose DXIL predecessors were linearized (internal diamond merges
    /// and diamond joins), so their IR phis lower to `select` chains.
    pub(super) linear_merges: Set<usize>,
    /// Lockstep-predicated loop headers (`header -> latch`).
    pub(super) lockstep_headers: Map<usize, usize>,
    /// Lockstep-predicated loop exits (`exit -> header`) so `block_active`
    /// restores to the outer predicate at loop exit.
    pub(super) lockstep_exits: Map<usize, usize>,
    /// Blocks belonging to any lockstep-predicated loop.
    pub(super) in_lockstep_loop: Set<usize>,
    /// Latch blocks promoted to group-uniform branches for `in_step_with`.
    pub(super) uniform_branches: Set<usize>,
    /// Blocks bypassed by `EarlyOut` that have no remaining live predecessors.
    pub(super) dead_blocks: Set<usize>,
    /// Immediate dominator per IR block (on the original CFG).
    pub(super) idom: Vec<usize>,
    /// Continuation blocks of `EarlyOut` branches (`cont -> branch block d`).
    pub(super) early_conts: Map<usize, usize>,
    /// Trivial `End` blocks bypassed by at least one `EarlyOut`.
    pub(super) skipped_ends: Set<usize>,
    /// Scalar-buffer reads whose results codegen must publish to both halves.
    pub(super) scalar_publications: Set<Site>,
    /// Common SCC branches, established before assuming any publication.
    publication_scc: Set<usize>,
    /// Original immediate postdominators, before flattening any branch.
    original_ipdom: Vec<Option<usize>>,
    /// First-lane sites physically reached by a single guest thread.
    pub(super) owner: Option<terminal_owner::Owner>,
    /// Updated `in_step` vector reflecting predicated control flow.
    pub(super) step: Vec<bool>,
}

impl PredicatePlan {
    pub(super) fn empty(step: Vec<bool>) -> Self {
        PredicatePlan {
            exits: Map::default(),
            in_diamond: Set::default(),
            diamond_joins: Map::default(),
            skip_bodies: Map::default(),
            skip_joins: Map::default(),
            skipped_edges: Set::default(),
            linear_merges: Set::default(),
            lockstep_headers: Map::default(),
            lockstep_exits: Map::default(),
            in_lockstep_loop: Set::default(),
            uniform_branches: Set::default(),
            dead_blocks: Set::default(),
            idom: Vec::new(),
            early_conts: Map::default(),
            skipped_ends: Set::default(),
            scalar_publications: Set::default(),
            publication_scc: Set::default(),
            original_ipdom: Vec::new(),
            owner: None,
            step,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.exits.is_empty()
    }

    /// Computes a selective predication plan for a wave64 compute program on a
    /// 32-lane host. Returns an empty plan (zero codegen overhead) when no
    /// non-`in_step` block needs cross-half or split-wave LDS execution.
    pub(super) fn build(
        p: &Program,
        st: &Structured,
        base_step: Vec<bool>,
        one_wave_lds: bool,
        host_wave32: bool,
        mapped_wave: Option<(&Parts, bool, &HashSet<Value>)>,
        ticketed_group: Option<&Parts>,
    ) -> Self {
        let n = p.blocks.len();
        if n <= 1 {
            return Self::empty(base_step);
        }
        let needs = candidate_blocks(p, st, one_wave_lds);
        if !(0..n).any(|b| needs[b] && !base_step[b]) {
            return Self::empty(base_step);
        }

        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; n];
        for (idx, &b) in order.iter().enumerate() {
            rpo[b] = idx;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &rpo);

        let mut plan = PredicatePlan {
            idom,
            ..Self::empty(base_step)
        };
        let mut eff_exits = st.exits.clone();
        // Each latch uses the same lowered counter at its header. A complete,
        // mapped console wave can prove both latch choices from scalar
        // broadcasts. Recheck participation after each CFG transformation.
        let owner = ticketed_group.and_then(|x| terminal_owner::find(x, &order));
        let wave = mapped_wave
            .map(|x| (x, two_latch_blocks(p, &rpo)))
            .filter(|(_, blocks)| !blocks.is_empty())
            .map(|((x, tickets, first_alone), blocks)| {
                let res = x.res;
                // These fixed-address loads lower to data embedded in the
                // program or supplied as dispatch root constants.
                let immutable = res
                    .scalar
                    .iter()
                    .filter_map(|&(site, address)| {
                        (matches!(address.param, CODE | USER_DATA) && address.dynamic.is_none())
                            .then_some(site)
                    })
                    .collect();
                let before = blocks
                    .iter()
                    .map(|&b| rpo[b])
                    .min()
                    .expect("nonempty loops");
                let buffers = if tickets {
                    fixed_scalar_buffers(x, |kind| kind == ResourceType::Srv)
                        .filter(|((b, _), ..)| rpo[*b] < before)
                        .map(|(site, offset, ..)| (site, offset))
                        .collect()
                } else {
                    Vec::new()
                };
                let per_lane = res.waterfall.reads.iter().chain(first_alone).copied().collect();
                (per_lane, blocks, immutable, buffers)
            });
        let update_step = |plan: &mut PredicatePlan, exits: &[LaneExit]| {
            let mut step = barriers::in_step_with(p, st, exits, &plan.uniform_branches);
            if let Some((per_lane, loop_blocks, immutable, buffers)) = &wave {
                plan.scalar_publications.clear();
                let analyze = |plan: &Self| {
                    barriers::step_and_cap_with(
                        p,
                        st,
                        exits,
                        &plan.uniform_branches,
                        true,
                        Some(barriers::WaveReads {
                            plan,
                            per_lane,
                            immutable,
                            loop_blocks,
                            base_step: &step,
                        }),
                    )
                };
                // Establish arrival and offset without any published-value facts.
                let (mut wave_step, _, alike) = analyze(plan);
                plan.publication_scc = plan
                    .exits
                    .keys()
                    .copied()
                    .filter(|&d| {
                        let Some(branch) = p.blocks[d].insts.last() else {
                            return false;
                        };
                        let LaneExit::Branch { when, .. } = st.exits[d] else {
                            return false;
                        };
                        let (lo, hi) = match branch.name() {
                            "s_cbranch_scc0" => (Bool::TRUE, Bool::FALSE),
                            "s_cbranch_scc1" => (Bool::FALSE, Bool::TRUE),
                            _ => return false,
                        };
                        branch.read(Reg::Scc).is_some_and(|v| {
                            alike.contains(&v)
                                && st.bools.node(when) == Some((Atom::Uniform(v), lo, hi))
                        })
                    })
                    .collect();
                for &(site @ (b, _), offset) in buffers {
                    if step[b]
                        && wave_step[b]
                        && !plan.dead_blocks.contains(&b)
                        && offset.is_none_or(|v| alike.contains(&v))
                    {
                        plan.scalar_publications.insert(site);
                    }
                }
                if !plan.scalar_publications.is_empty() {
                    wave_step = analyze(plan).0;
                }
                for &b in loop_blocks {
                    step[b] |= wave_step[b];
                }
            }
            step
        };
        if wave.is_some() {
            plan.step = update_step(&mut plan, &eff_exits);
        }

        // Iteratively predicate eligible divergent branches and loops whose
        // regions contain candidate blocks needing `in_step`.
        loop {
            let mut changed = false;
            let pdom = post_dominators(&eff_exits);
            let ipdom = immediate_post_dominators(&eff_exits, &pdom);
            if wave.is_some() && plan.exits.is_empty() {
                plan.original_ipdom.clone_from(&ipdom);
            }

            for &d in &order {
                if owner.as_ref().is_some_and(|o| o.gate == d)
                    || !plan.step[d]
                    || plan.exits.contains_key(&d)
                    || plan.dead_blocks.contains(&d)
                {
                    continue;
                }
                let LaneExit::Branch {
                    taken, not_taken, ..
                } = eff_exits[d]
                else {
                    continue;
                };
                if taken == not_taken || rpo[taken] <= rpo[d] || rpo[not_taken] <= rpo[d] {
                    continue;
                }
                if plan.step[taken] && plan.step[not_taken] {
                    continue;
                }

                // Shape 1: acyclic early-out `if (cond) return;`.
                if let Some((cont, end_blk)) =
                    match_early_out(p, &eff_exits, d, taken, not_taken, &rpo, &needs, &plan.step)
                {
                    plan.exits.insert(d, FlatExit::EarlyOut { cont });
                    plan.early_conts.insert(cont, d);
                    plan.skipped_ends.insert(end_blk);
                    plan.skipped_edges.insert((d, end_blk));
                    eff_exits[d] = LaneExit::Jump(cont);
                    if p.blocks[end_blk].preds.iter().all(|&q| {
                        plan.dead_blocks.contains(&q)
                            || plan.skipped_edges.contains(&(q, end_blk))
                    }) {
                        plan.dead_blocks.insert(end_blk);
                    }
                    plan.step = update_step(&mut plan, &eff_exits);
                    changed = true;
                    break;
                }

                // Shape 2: one-sided skip-guard `if (cond) { body } -> J`
                // (allows uniform loops and nested branches inside `body`).
                if let Some(join) = ipdom[d]
                    && let Some(body) = match_skip_guard(
                        p,
                        &eff_exits,
                        d,
                        taken,
                        not_taken,
                        join,
                        &rpo,
                        &needs,
                        &plan,
                    )
                {
                    plan.exits.insert(d, FlatExit::SkipGuard { body });
                    plan.skip_bodies.insert(body, d);
                    plan.skip_joins.insert(join, d);
                    plan.skipped_edges.insert((d, join));
                    eff_exits[d] = LaneExit::Jump(body);
                    plan.step = update_step(&mut plan, &eff_exits);
                    changed = true;
                    break;
                }

                // Shape 3: acyclic two-sided SESE diamond `d -> J`.
                if let Some(join) = ipdom[d]
                    && let Some(region) = match_sese_diamond(
                        p,
                        &eff_exits,
                        d,
                        taken,
                        not_taken,
                        join,
                        &rpo,
                        &needs,
                        &plan,
                    )
                {
                    plan.exits.insert(d, FlatExit::Linear { next: region[0] });
                    eff_exits[d] = LaneExit::Jump(region[0]);
                    for (idx, &b) in region.iter().enumerate() {
                        let next = region.get(idx + 1).copied().unwrap_or(join);
                        plan.in_diamond.insert(b);
                        plan.exits.insert(b, FlatExit::Linear { next });
                        eff_exits[b] = LaneExit::Jump(next);
                        if !p.blocks[b].phis.is_empty() || p.blocks[b].preds.len() > 1 {
                            plan.linear_merges.insert(b);
                        }
                    }
                    plan.diamond_joins.insert(join, d);
                    plan.linear_merges.insert(join);
                    plan.step = update_step(&mut plan, &eff_exits);
                    changed = true;
                    break;
                }
            }

            if !changed && host_wave32 {
                // Shape 4: single-latch, single-exit divergent loop promoted
                // to group-uniform lockstep execution (`group_any_true`).
                for &h in &order {
                    if plan.dead_blocks.contains(&h)
                        || plan.lockstep_headers.contains_key(&h)
                        || plan.in_diamond.contains(&h)
                    {
                        continue;
                    }
                    if let Some((latch, exit, stay_when_true, body)) =
                        match_lockstep_loop(p, &eff_exits, h, &rpo, &needs, &plan)
                    {
                        plan.exits.insert(
                            latch,
                            FlatExit::LockstepLatch {
                                header: h,
                                exit,
                                stay_when_true,
                            },
                        );
                        plan.lockstep_headers.insert(h, latch);
                        plan.lockstep_exits.insert(exit, h);
                        plan.uniform_branches.insert(latch);
                        plan.in_lockstep_loop.extend(body);
                        plan.step = update_step(&mut plan, &eff_exits);
                        changed = true;
                        break;
                    }
                }
            }

            if !changed && host_wave32 {
                // Shape 5: single-exit while-shaped divergent loop (the exit
                // branch at a block other than the latch, which jumps back to
                // the header) promoted to group-uniform lockstep execution.
                for &h in &order {
                    if plan.dead_blocks.contains(&h)
                        || plan.lockstep_headers.contains_key(&h)
                        || plan.in_diamond.contains(&h)
                    {
                        continue;
                    }
                    if let Some(lp) =
                        match_lockstep_while_loop(p, &eff_exits, h, &rpo, &needs, &plan)
                    {
                        plan.exits.insert(
                            lp.exiting,
                            FlatExit::LockstepExit {
                                header: h,
                                stay: lp.stay,
                                exit: lp.exit,
                                stay_when_true: lp.stay_when_true,
                            },
                        );
                        plan.early_conts.insert(lp.stay, lp.exiting);
                        plan.lockstep_headers.insert(h, lp.latch);
                        plan.lockstep_exits.insert(lp.exit, h);
                        plan.uniform_branches.insert(lp.exiting);
                        plan.in_lockstep_loop.extend(lp.body);
                        plan.step = update_step(&mut plan, &eff_exits);
                        changed = true;
                        break;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        for &b in &plan.dead_blocks {
            plan.step[b] = false;
        }
        if let Some(owner) = owner {
            if !owner.preserved(p, st, &eff_exits, &plan) {
                // Keep the established predicate plan when a rewrite would invalidate ownership.
                return Self::build(p, st, barriers::in_step(p, st), one_wave_lds, host_wave32, mapped_wave, None);
            }
            plan.owner = Some(owner);
        }
        plan
    }
}

/// A fixed buffer in the descriptor table and its scalar byte offset.
/// Waterfalls can replace a scalar with lane-local data.
pub(super) fn fixed_scalar_buffers<'a>(
    x: &'a Parts,
    accepts: fn(ResourceType) -> bool,
) -> impl Iterator<Item = (Site, Option<Value>, Site, u8)> + 'a {
    let addresses: Map<_, _> = x.res.scalar.iter().copied().collect();
    x.res.descriptors.iter().filter_map(move |desc| {
        let (b, i) = desc.site;
        let inst = &x.p.blocks[b].insts[i];
        if desc.role != Role::Resource || !inst.name().starts_with("s_buffer_load_dword") {
            return None;
        }
        let Fields::Smem { cache, soffset, .. } = inst.fields else {
            return None;
        };
        let Source::Loaded { load, first } = desc.source else {
            return None;
        };
        let address = *addresses.get(&load)?;
        let binding = table_kind(x.native, address, first)?;
        if cache != Default::default()
            || address.dynamic.is_some()
            || x.res.waterfall.loads.contains(&load)
            || !accepts(binding.kind)
            || binding.dwords != 4
            || !address.offset.is_multiple_of(4)
            || address.offset / 4 + u32::from(first) != binding.table_offset_dwords
        {
            return None;
        }
        let offset = match soffset {
            Operand::Scalar { code: NULL, .. } => None,
            _ => Some(sgpr_operand(inst, soffset).filter(|v| !x.res.waterfall.reads.contains(v))?),
        };
        Some((desc.site, offset, load, first))
    })
}

/// Natural loops with exactly two unconditional back edges to the same
/// header. Structurisation already checked that the reachable CFG is reducible.
fn two_latch_blocks(p: &Program, rpo: &[usize]) -> Set<usize> {
    let mut blocks = Set::default();
    for (header, block) in p.blocks.iter().enumerate() {
        let mut back = block
            .preds
            .iter()
            .copied()
            .filter(|&q| rpo[q] != usize::MAX && rpo[q] >= rpo[header]);
        let (Some(a), Some(b), None) = (back.next(), back.next(), back.next()) else {
            continue;
        };
        if ![a, b]
            .iter()
            .all(|&q| p.blocks[q].exit == Exit::Jump(header))
        {
            continue;
        }
        let mut body = Set::from_iter([header]);
        let mut stack = vec![a, b];
        while let Some(q) = stack.pop() {
            if body.insert(q) {
                stack.extend(p.blocks[q].preds.iter().filter(|&&v| rpo[v] != usize::MAX));
            }
        }
        blocks.extend(body);
    }
    blocks
}

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Blocks that contain an operation needing `in_step` in `halves` mode
/// (cross-half wave ops, mask-as-data reads and their transitive mask sources,
/// `v_readlane`/`v_readfirstlane`, or split-wave LDS accesses).
fn candidate_blocks(p: &Program, st: &Structured, one_wave_lds: bool) -> Vec<bool> {
    let mut needs = vec![false; p.blocks.len()];
    let mut mask_def: Map<Value, (usize, Option<usize>)> = Map::default();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            if st.masks.contains_key(&phi.value) || st.highs.contains_key(&phi.value) {
                mask_def.insert(phi.value, (b, None));
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, w) in &inst.writes {
                if st.masks.contains_key(&w) || st.highs.contains_key(&w) {
                    mask_def.insert(w, (b, Some(i)));
                }
            }
        }
    }

    let mut mask_worklist: Vec<Value> = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            let name = inst.name();
            if matches!(
                name,
                "v_readfirstlane_b32"
                    | "v_readlane_b32"
                    | "s_or_saveexec_b64"
                    | "s_nand_saveexec_b64"
                    | "v_mbcnt_lo_u32_b32"
                    | "v_mbcnt_hi_u32_b32"
                    | "s_bcnt1_i32_b64"
                    | "s_ff1_i32_b64"
                    | "s_flbit_i32_b64"
            ) {
                needs[b] = true;
            }
            let is_mask_user = name.starts_with("s_cbranch")
                || name.starts_with("v_cndmask")
                || matches!(
                    name,
                    "s_and_b64"
                        | "s_or_b64"
                        | "s_xor_b64"
                        | "s_andn2_b64"
                        | "s_orn2_b64"
                        | "s_nand_b64"
                        | "s_nor_b64"
                        | "s_xnor_b64"
                        | "s_not_b64"
                        | "s_mov_b64"
                        | "s_and_saveexec_b64"
                        | "s_or_saveexec_b64"
                        | "s_xor_saveexec_b64"
                        | "s_andn2_saveexec_b64"
                        | "s_orn2_saveexec_b64"
                        | "s_nand_saveexec_b64"
                        | "s_nor_saveexec_b64"
                        | "s_xnor_saveexec_b64"
                );
            if !is_mask_user {
                for &(_, v) in &inst.reads {
                    if st.masks.contains_key(&v) || st.highs.contains_key(&v) {
                        needs[b] = true;
                        mask_worklist.push(v);
                    }
                }
            }
            if one_wave_lds
                && matches!(inst.fields, Fields::Ds { .. })
                && !name.starts_with("ds_swizzle")
                && !name.contains("permute")
            {
                needs[b] = true;
            }
        }
    }

    // Walk backward through pure scalar mask ops (`s_not_b64`, `s_and_b64`,
    // `s_mov_b64`, etc.) so source mask-defining blocks are marked as well.
    let mut seen_masks: Set<Value> = Set::default();
    while let Some(v) = mask_worklist.pop() {
        let lo = st.highs.get(&v).copied().unwrap_or(v);
        if !seen_masks.insert(lo) {
            continue;
        }
        let Some(&(def_b, inst_idx)) = mask_def.get(&lo) else {
            continue;
        };
        needs[def_b] = true;
        if let Some(i) = inst_idx {
            let inst = &p.blocks[def_b].insts[i];
            if matches!(inst.fields, Fields::Sop { .. })
                && inst.write(Reg::S(EXEC)).is_none()
            {
                for &(_, r) in &inst.reads {
                    if st.masks.contains_key(&r) || st.highs.contains_key(&r) {
                        mask_worklist.push(r);
                    }
                }
            }
        }
    }
    needs
}

fn is_trivial_epilogue_inst(i: &Inst) -> bool {
    match i.name() {
        "s_endpgm" | "s_nop" | "s_waitcnt" | "s_waitcnt_depctr" | "s_branch" => true,
        "s_mov_b64" | "s_or_b64" => {
            i.writes.iter().all(|&(r, _)| matches!(r, Reg::S(EXEC) | Reg::S(127) | Reg::Scc))
        }
        _ => false,
    }
}

/// Whether `b` is a side-effect-free epilogue ending in `s_endpgm` (directly
/// or through up to 4 trivial jump trampolines).
fn is_trivial_end(p: &Program, exits: &[LaneExit], mut b: usize) -> bool {
    for _ in 0..5 {
        if !p.blocks[b].insts.iter().all(is_trivial_epilogue_inst) {
            return false;
        }
        match exits[b] {
            LaneExit::End => return true,
            LaneExit::Jump(next) if next != b && p.blocks[b].phis.is_empty() => b = next,
            _ => return false,
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn match_early_out(
    p: &Program,
    exits: &[LaneExit],
    d: usize,
    taken: usize,
    not_taken: usize,
    rpo: &[usize],
    needs: &[bool],
    step: &[bool],
) -> Option<(usize, usize)> {
    let (end_blk, cont) = if is_trivial_end(p, exits, taken) {
        (taken, not_taken)
    } else if is_trivial_end(p, exits, not_taken) {
        (not_taken, taken)
    } else {
        return None;
    };
    if cont == end_blk || rpo[cont] <= rpo[d] || p.blocks[cont].preds != [d] {
        return None;
    }
    let mut seen = vec![false; p.blocks.len()];
    let mut stack = vec![cont];
    let mut useful = false;
    while let Some(b) = stack.pop() {
        if b == end_blk || std::mem::replace(&mut seen[b], true) {
            continue;
        }
        if needs[b] && !step[b] {
            useful = true;
            break;
        }
        match exits[b] {
            LaneExit::Jump(t) => stack.push(t),
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => stack.extend([t, f]),
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    useful.then_some((cont, end_blk))
}

/// Live predecessors of `b` under `eff_exits` (excluding dead blocks).
fn eff_preds(exits: &[LaneExit], dead: &Set<usize>, target: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for (q, e) in exits.iter().enumerate() {
        if dead.contains(&q) {
            continue;
        }
        let hits = match *e {
            LaneExit::Jump(t) => t == target,
            LaneExit::Branch {
                taken, not_taken, ..
            } => taken == target || not_taken == target,
            LaneExit::End | LaneExit::Indirect => false,
        };
        if hits {
            out.push(q);
        }
    }
    out
}

/// Matches a one-sided skip-guard `d -> (body, join)` where `join == ipdom[d]`.
/// Unlike `match_sese_diamond`, `body` is NOT linearized, so uniform loops and
/// nested branches inside `body` keep their native CFG structure.
#[allow(clippy::too_many_arguments)]
fn match_skip_guard(
    p: &Program,
    exits: &[LaneExit],
    d: usize,
    taken: usize,
    not_taken: usize,
    join: usize,
    rpo: &[usize],
    needs: &[bool],
    plan: &PredicatePlan,
) -> Option<usize> {
    let body = if taken == join && not_taken != join {
        not_taken
    } else if not_taken == join && taken != join {
        taken
    } else {
        return None;
    };
    if join == d
        || rpo[join] <= rpo[d]
        || rpo[body] <= rpo[d]
        || plan.diamond_joins.contains_key(&join)
        || plan.skip_joins.contains_key(&join)
        || plan.skip_bodies.contains_key(&body)
        || plan.in_diamond.contains(&body)
        || plan.exits.contains_key(&body)
    {
        return None;
    }
    // `join` cannot be a loop header itself.
    if eff_preds(exits, &plan.dead_blocks, join)
        .iter()
        .any(|&q| rpo[q] >= rpo[join])
    {
        return None;
    }
    // Collect all blocks reachable from `body` before `join`.
    let mut region_set = Set::default();
    let mut stack = vec![body];
    while let Some(b) = stack.pop() {
        if b == join || !region_set.insert(b) {
            continue;
        }
        if plan.in_diamond.contains(&b) || plan.dead_blocks.contains(&b) {
            return None;
        }
        match exits[b] {
            LaneExit::Jump(t) => {
                if t != join {
                    stack.push(t);
                }
            }
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => {
                for s in [t, f] {
                    if s != join {
                        stack.push(s);
                    }
                }
            }
            LaneExit::End | LaneExit::Indirect => return None,
        }
    }
    if region_set.is_empty() {
        return None;
    }
    // Single-entry: every live predecessor of `body` is `d` or inside `region_set`;
    // every live predecessor of any other block in `region_set` is inside `region_set`.
    for &b in &region_set {
        for q in eff_preds(exits, &plan.dead_blocks, b) {
            let ok = if b == body {
                q == d || region_set.contains(&q)
            } else {
                region_set.contains(&q)
            };
            if !ok {
                return None;
            }
        }
    }
    // Single-exit: every live predecessor of `join` is `d` or inside `region_set`.
    let join_preds = eff_preds(exits, &plan.dead_blocks, join);
    if !join_preds.iter().all(|&q| q == d || region_set.contains(&q)) {
        return None;
    }
    if !join_preds.iter().any(|&q| q != d) {
        return None;
    }
    // Every IR phi at `join` must come from original predecessors `d` or `region_set`.
    if !p.blocks[join]
        .preds
        .iter()
        .filter(|q| !plan.dead_blocks.contains(q) && !plan.skipped_edges.contains(&(**q, join)))
        .all(|&q| q == d || region_set.contains(&q))
    {
        return None;
    }
    // Only predicate if at least one block in `region_set` needs `in_step`.
    if !region_set.iter().any(|&b| needs[b] && !plan.step[b]) {
        return None;
    }
    Some(body)
}

#[allow(clippy::too_many_arguments)]
fn match_sese_diamond(
    p: &Program,
    exits: &[LaneExit],
    d: usize,
    taken: usize,
    not_taken: usize,
    join: usize,
    rpo: &[usize],
    needs: &[bool],
    plan: &PredicatePlan,
) -> Option<Vec<usize>> {
    if join == d || rpo[join] <= rpo[d] {
        return None;
    }
    if plan.diamond_joins.contains_key(&join)
        || plan.skip_joins.contains_key(&join)
        || p.blocks[join].preds.iter().any(|&q| rpo[q] >= rpo[join])
    {
        return None;
    }
    let mut region_set = Set::default();
    let mut stack = Vec::new();
    for t in [taken, not_taken] {
        if t != join {
            stack.push(t);
        }
    }
    while let Some(b) = stack.pop() {
        if b == join || !region_set.insert(b) {
            continue;
        }
        if region_set.len() > MAX_DIAMOND_BLOCKS
            || plan.in_diamond.contains(&b)
            || plan.exits.contains_key(&b)
            || plan.dead_blocks.contains(&b)
        {
            return None;
        }
        match exits[b] {
            LaneExit::Jump(t) => {
                if rpo[t] <= rpo[b] {
                    return None;
                }
                if t != join {
                    stack.push(t);
                }
            }
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => {
                if rpo[t] <= rpo[b] || rpo[f] <= rpo[b] {
                    return None;
                }
                for s in [t, f] {
                    if s != join {
                        stack.push(s);
                    }
                }
            }
            LaneExit::End | LaneExit::Indirect => return None,
        }
    }
    if region_set.is_empty() {
        return None;
    }
    for &b in &region_set {
        if !p.blocks[b]
            .preds
            .iter()
            .filter(|q| !plan.dead_blocks.contains(q) && !plan.skipped_edges.contains(&(**q, b)))
            .all(|&q| q == d || region_set.contains(&q))
        {
            return None;
        }
    }
    if !p.blocks[join]
        .preds
        .iter()
        .filter(|q| !plan.dead_blocks.contains(q) && !plan.skipped_edges.contains(&(**q, join)))
        .all(|&q| q == d || region_set.contains(&q))
    {
        return None;
    }
    if !region_set.iter().any(|&b| needs[b] && !plan.step[b]) {
        return None;
    }
    let mut region: Vec<usize> = region_set.into_iter().collect();
    region.sort_unstable_by_key(|&b| rpo[b]);
    Some(region)
}

/// Matches a single-latch, single-exit natural loop `(header, latch, exit)`
/// whose outside entry predecessors are all `in_step`, containing at least one
/// candidate block not yet in `in_step`, and with no non-mask SGPR written
/// inside the loop read outside the loop.
fn match_lockstep_loop(
    p: &Program,
    exits: &[LaneExit],
    h: usize,
    rpo: &[usize],
    needs: &[bool],
    plan: &PredicatePlan,
) -> Option<(usize, usize, bool, Set<usize>)> {
    let preds = eff_preds(exits, &plan.dead_blocks, h);
    let mut back_preds = Vec::new();
    let mut out_preds = Vec::new();
    for q in preds {
        if rpo[q] >= rpo[h] {
            back_preds.push(q);
        } else {
            out_preds.push(q);
        }
    }
    if back_preds.len() != 1 || out_preds.is_empty() {
        return None;
    }
    let latch = back_preds[0];
    if plan.exits.contains_key(&latch)
        || plan.in_diamond.contains(&latch)
        || plan.in_lockstep_loop.contains(&h)
        || plan.in_lockstep_loop.contains(&latch)
    {
        return None;
    }
    // Every outside predecessor entering the loop must already be `in_step`.
    if !out_preds.iter().all(|&q| plan.step[q]) {
        return None;
    }
    let LaneExit::Branch {
        taken, not_taken, ..
    } = exits[latch]
    else {
        return None;
    };
    let (exit, stay_when_true) = if taken == h && not_taken != h {
        (not_taken, true)
    } else if not_taken == h && taken != h {
        (taken, false)
    } else {
        return None;
    };
    if rpo[exit] <= rpo[h]
        || plan.lockstep_exits.contains_key(&exit)
        || plan.diamond_joins.contains_key(&exit)
    {
        return None;
    }

    // Collect the natural loop body by reverse DFS from `latch` stopping at `h`.
    let mut body = Set::default();
    body.insert(h);
    let mut stack = vec![latch];
    while let Some(b) = stack.pop() {
        if !body.insert(b) {
            continue;
        }
        if body.len() > MAX_LOCKSTEP_LOOP_BLOCKS
            || plan.in_diamond.contains(&b)
            || plan.in_lockstep_loop.contains(&b)
            || plan.dead_blocks.contains(&b)
        {
            return None;
        }
        for q in eff_preds(exits, &plan.dead_blocks, b) {
            if !body.contains(&q) {
                if rpo[q] < rpo[h] {
                    return None; // side entry into loop body
                }
                stack.push(q);
            }
        }
    }

    // Single exit: every edge from `body` stays inside `body`, except `latch -> exit`,
    // and no nested cycle exists inside `body`.
    for &b in &body {
        let succs: Vec<usize> = match exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => vec![t, f],
            LaneExit::End | LaneExit::Indirect => return None,
        };
        for t in succs {
            if b == latch && t == exit {
                continue;
            }
            if !body.contains(&t) {
                return None; // second exit out of loop
            }
            if rpo[t] <= rpo[b] && !(b == latch && t == h) {
                return None; // nested cycle inside loop
            }
        }
    }
    // `exit` must only be entered by `latch` under `eff_exits`.
    if eff_preds(exits, &plan.dead_blocks, exit) != [latch] {
        return None;
    }
    if !body.iter().any(|&b| needs[b] && !plan.step[b]) {
        return None;
    }

    // Soundness check: no non-EXEC/VCC SGPR or SCC written inside `body` may
    // be read outside `body` (inactive lanes spinning in lockstep overwrite
    // scalar registers without per-lane guarding).
    let is_mask_reg = |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    let mut loop_scalars: Set<Value> = Set::default();
    for &b in &body {
        for phi in &p.blocks[b].phis {
            if matches!(phi.reg, Reg::S(_) | Reg::Scc) && !is_mask_reg(phi.reg) {
                loop_scalars.insert(phi.value);
            }
        }
        for inst in &p.blocks[b].insts {
            for &(r, v) in &inst.writes {
                if matches!(r, Reg::S(_) | Reg::Scc) && !is_mask_reg(r) {
                    loop_scalars.insert(v);
                }
            }
        }
    }
    if !loop_scalars.is_empty() {
        for (b, block) in p.blocks.iter().enumerate() {
            if body.contains(&b) || plan.dead_blocks.contains(&b) {
                continue;
            }
            for phi in &block.phis {
                if phi.args.iter().any(|v| loop_scalars.contains(v)) {
                    return None;
                }
            }
            for inst in &block.insts {
                if inst.reads.iter().any(|(_, v)| loop_scalars.contains(v)) {
                    return None;
                }
            }
        }
    }

    Some((latch, exit, stay_when_true, body))
}

/// A while-shaped loop [`match_lockstep_while_loop`] found.
struct WhileLoop {
    /// The block jumping back to the header, unconditionally.
    latch: usize,
    /// The block whose branch leaves the loop.
    exiting: usize,
    /// Its successor inside the loop, and the one outside.
    stay: usize,
    exit: usize,
    /// Whether the branch's condition holds where it stays.
    stay_when_true: bool,
    body: Set<usize>,
}

/// A single-exit loop whose one exit branch is at a block other than its
/// latch, the latch jumping back to the header `h` (a `while` loop as the
/// console compiler lays one out, the test before the latch's work): it
/// runs in lockstep like [`match_lockstep_loop`]'s, the exit branch taken
/// by the whole group while any lane stays (`FlatExit::LockstepExit`).
fn match_lockstep_while_loop(
    p: &Program,
    exits: &[LaneExit],
    h: usize,
    rpo: &[usize],
    needs: &[bool],
    plan: &PredicatePlan,
) -> Option<WhileLoop> {
    let preds = eff_preds(exits, &plan.dead_blocks, h);
    let (back, out): (Vec<usize>, Vec<usize>) = preds.into_iter().partition(|&q| rpo[q] >= rpo[h]);
    let [latch] = back[..] else {
        return None;
    };
    if out.is_empty()
        || exits[latch] != LaneExit::Jump(h)
        || plan.exits.contains_key(&latch)
        || plan.in_diamond.contains(&latch)
        || plan.in_lockstep_loop.contains(&h)
        || plan.in_lockstep_loop.contains(&latch)
        || !out.iter().all(|&q| plan.step[q])
    {
        return None;
    }
    // The natural loop body, by reverse DFS from `latch` stopping at `h`.
    let mut body = Set::default();
    body.insert(h);
    let mut stack = vec![latch];
    while let Some(b) = stack.pop() {
        if !body.insert(b) {
            continue;
        }
        if body.len() > MAX_LOCKSTEP_LOOP_BLOCKS
            || plan.in_diamond.contains(&b)
            || plan.in_lockstep_loop.contains(&b)
            || plan.dead_blocks.contains(&b)
        {
            return None;
        }
        for q in eff_preds(exits, &plan.dead_blocks, b) {
            if !body.contains(&q) {
                if rpo[q] < rpo[h] {
                    return None; // side entry into loop body
                }
                stack.push(q);
            }
        }
    }
    // One edge leaves the body, from a branch whose other side stays; no
    // cycle inside the body but the latch's.
    let mut leaving = None;
    for &b in &body {
        let succs: Vec<usize> = match exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => return None,
        };
        for t in succs {
            if !body.contains(&t) {
                if leaving.replace((b, t)).is_some() {
                    return None; // a second exit
                }
            } else if rpo[t] <= rpo[b] && !(b == latch && t == h) {
                return None; // nested cycle inside loop
            }
        }
    }
    let (exiting, exit) = leaving?;
    let LaneExit::Branch {
        taken, not_taken, ..
    } = exits[exiting]
    else {
        return None;
    };
    let (stay, stay_when_true) = if not_taken == exit && body.contains(&taken) {
        (taken, true)
    } else if taken == exit && body.contains(&not_taken) {
        (not_taken, false)
    } else {
        return None;
    };
    if exiting == latch
        || plan.exits.contains_key(&exiting)
        || plan.early_conts.contains_key(&stay)
        || eff_preds(exits, &plan.dead_blocks, stay) != [exiting]
        || eff_preds(exits, &plan.dead_blocks, exit) != [exiting]
        || rpo[exit] <= rpo[h]
        || plan.lockstep_exits.contains_key(&exit)
        || plan.diamond_joins.contains_key(&exit)
        || !body.iter().any(|&b| needs[b] && !plan.step[b])
    {
        return None;
    }
    // As for `match_lockstep_loop`: no scalar the body writes (but EXEC and
    // VCC) is read outside it, since lanes spinning inactive in lockstep
    // still run the body's scalar instructions.
    let is_mask_reg =
        |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    let mut loop_scalars: Set<Value> = Set::default();
    for &b in &body {
        for phi in &p.blocks[b].phis {
            if matches!(phi.reg, Reg::S(_) | Reg::Scc) && !is_mask_reg(phi.reg) {
                loop_scalars.insert(phi.value);
            }
        }
        for inst in &p.blocks[b].insts {
            for &(r, v) in &inst.writes {
                if matches!(r, Reg::S(_) | Reg::Scc) && !is_mask_reg(r) {
                    loop_scalars.insert(v);
                }
            }
        }
    }
    for (b, block) in p.blocks.iter().enumerate() {
        if body.contains(&b) || plan.dead_blocks.contains(&b) {
            continue;
        }
        if block
            .phis
            .iter()
            .any(|phi| phi.args.iter().any(|v| loop_scalars.contains(v)))
            || block
                .insts
                .iter()
                .any(|i| i.reads.iter().any(|(_, v)| loop_scalars.contains(v)))
        {
            return None;
        }
    }
    Some(WhileLoop {
        latch,
        exiting,
        stay,
        exit,
        stay_when_true,
        body,
    })
}

fn post_dominators(exits: &[LaneExit]) -> Vec<Vec<u64>> {
    let n = exits.len();
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in exits.iter().enumerate() {
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
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for (w, p) in v.iter_mut().zip(&pdom[t]) {
                    *w &= p;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    pdom
}

/// Immediate strict post-dominator per block (none for exits or unreachable).
fn immediate_post_dominators(exits: &[LaneExit], pdom: &[Vec<u64>]) -> Vec<Option<usize>> {
    let n = exits.len();
    let contains = |set: &[u64], x: usize| (set[x / 64] >> (x % 64)) & 1 == 1;
    let count = |set: &[u64]| -> u32 {
        let mut c = 0;
        for b in 0..n {
            if contains(set, b) {
                c += 1;
            }
        }
        c
    };
    let sizes: Vec<u32> = (0..n).map(|b| count(&pdom[b])).collect();
    (0..n)
        .map(|b| {
            let mut best: Option<(u32, usize)> = None;
            for (cand, &sz) in sizes.iter().enumerate() {
                if cand != b
                    && contains(&pdom[b], cand)
                    && best.is_none_or(|(best_sz, _)| sz > best_sz)
                {
                    best = Some((sz, cand));
                }
            }
            best.map(|(_, cand)| cand)
        })
        .collect()
}

/// A whole-pair copy, or the saved destination of AND_SAVEEXEC, can restore
/// a known EXEC even when the instruction's narrowed EXEC depends on a phi.
fn restored_publication_exec(
    p: &Program,
    st: &Structured,
    idom: &[usize],
    head: usize,
    branch: &Inst,
    direct: &Set<Value>,
) -> Option<()> {
    if !p.wave64 || st.literal || !matches!(branch.name(), "s_cbranch_execz" | "s_cbranch_execnz") {
        return None;
    }
    let read_pair =
        |inst: &Inst, code| Some((inst.read(Reg::S(code))?, inst.read(Reg::S(code + 1))?));
    let mut code = EXEC;
    let mut pair = read_pair(branch, code)?;
    let mask = st.mask_bits(pair.0, Some(pair.1))?;
    for b in std::iter::successors(Some(head), |&b| (b != 0).then(|| idom[b])) {
        for inst in p.blocks[b].insts.iter().rev() {
            // SSA identities require both dwords to come from this very
            // definition. No phi or one-dword update can match the pair.
            if inst.write(Reg::S(code)) != Some(pair.0)
                || inst.write(Reg::S(code + 1)) != Some(pair.1)
            {
                continue;
            }
            let Fields::Sop {
                dst:
                    Some(Operand::Scalar {
                        code: dst,
                        dwords: 2,
                    }),
                src,
                ..
            } = &inst.fields
            else {
                return None;
            };
            if *dst != code {
                return None;
            }
            code = match (inst.name(), src.as_slice()) {
                (
                    "s_mov_b64",
                    [
                        Operand::Scalar {
                            code: from @ (0..=104 | EXEC),
                            dwords: 2,
                        },
                    ],
                ) => *from,
                ("s_and_saveexec_b64", _) if code <= 104 => EXEC,
                _ => return None,
            };
            pair = read_pair(inst, code)?;
            if code == EXEC && direct.contains(&pair.0) && direct.contains(&pair.1) {
                return (st.mask_bits(pair.0, Some(pair.1)) == Some(mask)).then_some(());
            }
        }
    }
    None
}

/// Check the guest branches after identifying the flattened bodies supplying activity.
pub(in crate::emit) fn publication_gates(
    p: &Program,
    st: &Structured,
    gates: &[(usize, usize)],
    common_scc: &Set<usize>,
    idom: &[usize],
) -> bool {
    // BDD simplification can erase a mask phi. Walk the actual SSA operands
    // on the proved path; a phi never establishes a wave-wide predecessor.
    let mut direct: Set<_> = p.inputs.iter().map(|&(_, v)| v).collect();
    let exact_bits = |m, direct: &Set<Value>| {
        st.bools.support(m).into_iter().all(|atom| match atom {
            Atom::Lane(v) => direct.contains(&v),
            Atom::Constant(_) => true,
            _ => false,
        })
    };
    let mut outer = None;
    let mut enclosing = Vec::new();
    for &(body, d) in gates {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[d]
        else {
            return false;
        };
        let head = &p.blocks[d];
        for (i, inst) in head.insts.iter().enumerate() {
            // WQM of the original full wave is an identity; the compute
            // emitter approximates expansion of any other mask.
            let full_wqm = || {
                matches!(inst.reads.as_slice(), [lo, hi] if
                lo.0 == Reg::S(EXEC) && hi.0 == Reg::S(EXEC + 1)
                && p.inputs.contains(lo) && p.inputs.contains(hi))
            };
            if inst.reads.iter().all(|(_, v)| direct.contains(v))
                && st.guards[d][i].is_none_or(|g| exact_bits(g, &direct))
                && (inst.name() != "s_wqm_b64" || full_wqm())
            {
                direct.extend(inst.writes.iter().map(|&(_, v)| v));
            }
        }
        let Some(branch) = head.insts.last() else {
            return false;
        };
        if !branch.reads.iter().all(|(_, v)| direct.contains(v))
            && restored_publication_exec(p, st, idom, d, branch, &direct).is_none()
        {
            return false;
        }
        let taken_edge = body == taken;
        if !common_scc.contains(&d) {
            let nonempty = match branch.name() {
                "s_cbranch_execz" => not_taken,
                "s_cbranch_execnz" => taken,
                _ => return false,
            };
            if body != nonempty {
                return false;
            }
            let mask = branch
                .read(Reg::S(EXEC))
                .zip(branch.read(Reg::S(EXEC + 1)))
                .and_then(|(lo, hi)| st.mask_bits(lo, Some(hi)));
            let Some(mask) = mask else { return false };
            // Quad approximates expansion and Phi selects by lane in normal
            // code. Neither is an exact source for a wave-wide EXEC visit.
            if !exact_bits(mask, &direct) || !exact_bits(when, &direct) {
                return false;
            }
            // A disjoint restored EXEC could enter a nested guest arm
            // even though no lane meets both flattened edge predicates.
            if outer.is_some_and(|m| st.bools.meet(&[(mask, true), (m, false)])) {
                return false;
            }
            // The structurizer can simplify an exit using lane reachability.
            // It must still match the full EXEC bit on the inherited mask.
            let mismatch = |exec, edge| {
                st.bools
                    .meet(&[enclosing.as_slice(), &[(mask, exec), (when, edge)]].concat())
            };
            if mismatch(true, !taken_edge) || mismatch(false, taken_edge) {
                return false;
            }
            outer = Some(mask);
        }
        enclosing.push((when, taken_edge));
    }
    true
}

impl Lower<'_> {
    /// Per-lane active predicate for IR block `b`, if `b` executes under
    /// predication (`None` means all lanes in the workgroup are active).
    pub(super) fn pred_active(&self, b: usize) -> Option<V> {
        self.block_active.get(b).copied().flatten()
    }

    /// Find the single-entry body supplying this block's emitted predicate.
    /// Through intervening merges it must also dominate the block, which
    /// must postdominate the body: every original surviving visit gets here.
    fn scalar_publication_guard(&self, b: usize) -> Option<(usize, usize)> {
        let active = self.pred_active(b)?;
        let plan = &self.pred;
        let mut dominators = std::iter::successors(Some(b), |&q| (q != 0).then(|| plan.idom[q]));
        let (body, head) = dominators.find_map(|q| {
            let d = plan
                .early_conts
                .get(&q)
                .or_else(|| plan.skip_bodies.get(&q))?;
            Some((q, *d))
        })?;
        if self.x.p.blocks[body].preds != [head]
            || self.pred_edge_active.get(&(head, body)) != Some(&active)
            || self.on_cycle(b)
        {
            return None;
        }
        if body != b {
            let postdominators = std::iter::successors(Some(body), |&q| plan.original_ipdom[q]);
            if self.pred_active(body) != Some(active)
                || self.on_cycle(body)
                || !postdominators.take(plan.idom.len()).any(|q| q == b)
            {
                return None;
            }
        }
        Some((body, head))
    }

    fn proved_scalar_visit(&self, mut b: usize) -> bool {
        let mut gates = Vec::new();
        while b != 0 {
            let Some((body, d)) = self.scalar_publication_guard(b) else {
                return false;
            };
            gates.push((body, d));
            b = d;
        }
        gates.reverse();
        publication_gates(
            self.x.p,
            self.x.st,
            &gates,
            &self.pred.publication_scc,
            &self.pred.idom,
        )
    }

    pub(super) fn published_scalar_words(
        &mut self,
        h: V,
        mut addr: V,
        count: u32,
    ) -> Result<Vec<V>, Why> {
        if !self.across_here() {
            return Err("a scalar-buffer publication outside converged instruction control flow");
        }
        let active = self.pred_active(self.cur_ir);
        let visit = if let Some(act) = active {
            if self.x.st.literal || !self.proved_scalar_visit(self.cur_ir) {
                return Err(
                    "a scalar-buffer publication without a proved predicated address or visit",
                );
            }
            // Inactive lanes need not hold the guest address. Broadcast an
            // actual visitor's common byte offset before all lanes load it.
            if self.s.m.int_value(addr).is_none() {
                addr = self.first_lane_across(addr, Some(act))?;
            }
            Some(self.group_any_true(act, self.cur)?)
        } else {
            None
        };
        let guard = self.begin_guarded(visit);
        let load_block = self.cur;
        let mut words = self.raw_run(h, addr, count);
        if self.cur != load_block {
            return Err("a scalar-buffer publication through a guarded raw view");
        }
        // Both ticketed halves enter together, including logical lane zero
        // when its EXEC bit is clear. SALU takes its value for each component.
        for x in &mut words {
            *x = self.first_lane_across(*x, None)?;
        }
        if guard.is_some() {
            let zero = self.s.m.i32(0);
            let pairs: Vec<_> = words.into_iter().map(|x| (x, zero)).collect();
            words = self.end_guarded(guard, &pairs);
        }
        Ok(words)
    }

    /// Computes `self.block_active[b]` at the start of IR block `b` (when
    /// `self.pred` is non-empty) and lowers phis at linearized merges or
    /// `SkipGuard` joins.
    pub(super) fn enter_predicated_block(&mut self, b: usize) -> Result<bool, Stop> {
        if self.pred.is_empty() {
            return Ok(false);
        }
        // 1. Compute `block_active[b]`.
        let active = if let Some(&latch) = self.pred.lockstep_headers.get(&b) {
            let outer = self.pred_outer_active(b);
            self.loop_outer_active.insert(b, outer);
            let phi = self.s.m.phi(self.cur, self.s.t.i1);
            self.loop_act_phis.push((phi, b, latch));
            Some(phi)
        } else if let Some(&d) = self.pred.skip_joins.get(&b) {
            self.pred_active(d)
        } else if let Some(&d) = self.pred.diamond_joins.get(&b) {
            self.pred_active(d)
        } else if let Some(&h) = self.pred.lockstep_exits.get(&b) {
            self.loop_outer_active.get(&h).copied().flatten()
        } else if let Some(&d) = self.pred.skip_bodies.get(&b) {
            self.pred_edge_active.get(&(d, b)).copied()
        } else if self.pred.in_diamond.contains(&b) {
            let mut acc: Option<V> = None;
            for &q in &self.x.p.blocks[b].preds {
                if let Some(&e) = self.pred_edge_active.get(&(q, b)) {
                    acc = Some(match acc {
                        None => e,
                        Some(prev) => self.s.m.bin(self.cur, Bin::Or, prev, e),
                    });
                }
            }
            acc
        } else if let Some(&d) = self.pred.early_conts.get(&b) {
            self.pred_edge_active.get(&(d, b)).copied()
        } else if self.pred.skipped_ends.contains(&b) {
            None
        } else {
            self.pred_outer_active(b)
        };
        self.block_active[b] = active;

        if let Some(&d) = self.pred.skip_joins.get(&b) {
            self.lower_skip_join_phis(b, d)?;
            return Ok(true);
        }

        if !self.pred.linear_merges.contains(&b) {
            return Ok(false);
        }

        // 2. Lower IR phis in linearized merge block `b` directly as `select`
        // chains over `b`'s original IR predecessors.
        let preds = &self.x.p.blocks[b].preds;
        let mut pred_order: Vec<(usize, usize)> = preds.iter().copied().enumerate().collect();
        pred_order.sort_by_key(|&(_, q)| self.rpo[q]);
        let mut wave_any_edge: Map<usize, V> = Map::default();
        for phi in &self.x.p.blocks[b].phis {
            if matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                self.vals.insert(phi.value, Val::Missing("a lane mask"));
                continue;
            }
            if let Some(&why) = self.bad.get(&phi.value) {
                self.vals.insert(phi.value, Val::Missing(why));
                continue;
            }
            let bit = phi.reg == Reg::Scc;
            let scalar = matches!(phi.reg, Reg::S(_) | Reg::Scc);
            let mut acc: Option<V> = None;
            let mut missing: Option<Why> = None;
            for &(n, q) in &pred_order {
                if self.pred.dead_blocks.contains(&q) || self.tail[q].is_none() {
                    continue;
                }
                let Some(&edge_act) = self.pred_edge_active.get(&(q, b)) else {
                    continue;
                };
                let arg = phi.args[n];
                let val = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => Some(x),
                    _ if self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg) => Some(if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    }),
                    (Some(Val::Missing(why)), _) => {
                        missing.get_or_insert(why);
                        None
                    }
                    _ => {
                        missing.get_or_insert("a join of mismatched values");
                        None
                    }
                };
                if let Some(x) = val {
                    acc = Some(match acc {
                        None => x,
                        Some(prev) if prev == x => prev,
                        Some(prev) => {
                            let cond = if scalar {
                                match wave_any_edge.get(&q).copied() {
                                    Some(v) => v,
                                    None => {
                                        let v = self.pred_wave_any(b, edge_act)?;
                                        wave_any_edge.insert(q, v);
                                        v
                                    }
                                }
                            } else {
                                edge_act
                            };
                            self.s.m.select(self.cur, cond, x, prev)
                        }
                    });
                }
            }
            let result = match (acc, missing) {
                (Some(x), None) => {
                    if bit {
                        Val::Bit(x)
                    } else {
                        Val::I32(x)
                    }
                }
                (_, Some(why)) => Val::Missing(why),
                (None, None) => {
                    if bit {
                        Val::Bit(self.s.m.bool(false))
                    } else {
                        Val::I32(self.s.m.i32(0))
                    }
                }
            };
            self.vals.insert(phi.value, result);
        }
        Ok(true)
    }

    /// Whether any lane in the console wave satisfies `act` at in-step block
    /// `b`: used when merging wave-wide scalar registers (`Reg::S`, `Reg::Scc`)
    /// at a predicated branch join so every lane sees the scalar value written
    /// when any lane of the wave entered the branch.
    fn pred_wave_any(&mut self, b: usize, act: V) -> Result<V, Stop> {
        let offset = self.x.p.blocks[b].offset;
        let cur = self.cur;
        let zero = self.s.m.i32(0);
        if self.halves && !self.one_host_wave() {
            let (lo, hi) = self
                .ballot_across_halves(act, cur)
                .map_err(|w| unsupported(offset, w))?;
            let both = self.s.m.bin(cur, Bin::Or, lo, hi);
            return Ok(self.s.m.cmp(cur, pred::NE, both, zero));
        }
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        self.s.flags |= flags::WAVE_OPS;
        let r = self
            .s
            .op(cur, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
        let lo = self.s.m.extract(cur, r, 0);
        let mask = if !self.halves {
            let hi = self.s.m.extract(cur, r, 1);
            self.s.m.bin(cur, Bin::Or, lo, hi)
        } else {
            lo
        };
        Ok(self.s.m.cmp(cur, pred::NE, mask, zero))
    }

    fn pred_outer_active(&self, b: usize) -> Option<V> {
        let idom = self.pred.idom.get(b).copied().unwrap_or(b);
        if idom != b && idom != usize::MAX {
            self.pred_active(idom)
        } else {
            None
        }
    }

    /// Lowers phis at a `SkipGuard` join block `b` (where edge `d -> b` was
    /// bypassed through `body`): selects `act_body ? v_body : v_d` for VGPRs
    /// (and `wave_any(act_body) ? v_body : v_d` for wave-wide scalars),
    /// creating a DXIL phi over the live body predecessors when there are two
    /// or more.
    fn lower_skip_join_phis(&mut self, b: usize, d: usize) -> Result<(), Stop> {
        let FlatExit::SkipGuard { body } = self.pred.exits[&d] else {
            return Ok(());
        };
        let act_body = self.pred_edge_active[&(d, body)];
        let mut wave_any_body: Option<V> = None;
        let preds = &self.x.p.blocks[b].preds;
        let skip_idx = preds.iter().position(|&q| q == d);
        let body_preds: Vec<(usize, usize)> = preds
            .iter()
            .enumerate()
            .filter_map(|(n, &q)| {
                (q != d
                    && !self.pred.dead_blocks.contains(&q)
                    && !self.pred.skipped_edges.contains(&(q, b))
                    && self.tail[q].is_some())
                .then_some((n, q))
            })
            .collect();

        for (i, phi) in self.x.p.blocks[b].phis.iter().enumerate() {
            if matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                self.vals.insert(phi.value, Val::Missing("a lane mask"));
                continue;
            }
            if let Some(&why) = self.bad.get(&phi.value) {
                self.vals.insert(phi.value, Val::Missing(why));
                continue;
            }
            let bit = phi.reg == Reg::Scc;
            let scalar = matches!(phi.reg, Reg::S(_) | Reg::Scc);
            let zero = |l: &mut Self| {
                if bit {
                    l.s.m.bool(false)
                } else {
                    l.s.m.i32(0)
                }
            };
            let v_d = match skip_idx {
                Some(n_d) => {
                    let arg = phi.args[n_d];
                    match (self.vals.get(&arg).copied(), bit) {
                        (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => Ok(x),
                        _ if self.never_taken(d, b) || self.dead_on_edge(b, n_d, phi.reg) => {
                            Ok(zero(self))
                        }
                        (Some(Val::Missing(why)), _) => Err(why),
                        _ => Err("a join of mismatched values"),
                    }
                }
                None => Ok(zero(self)),
            };
            let v_body = match body_preds.as_slice() {
                [] => Ok(zero(self)),
                &[(n_q, q)] => {
                    let arg = phi.args[n_q];
                    match (self.vals.get(&arg).copied(), bit) {
                        (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => Ok(x),
                        _ if self.never_taken(q, b) || self.dead_on_edge(b, n_q, phi.reg) => {
                            Ok(zero(self))
                        }
                        (Some(Val::Missing(why)), _) => Err(why),
                        _ => Err("a join of mismatched values"),
                    }
                }
                _ => {
                    let ty = if bit { self.s.t.i1 } else { self.s.t.i32 };
                    let v = self.s.m.phi(self.cur, ty);
                    self.data_phis.push((v, b, i));
                    Ok(v)
                }
            };
            let result = match (v_body, v_d) {
                (Ok(vb), Ok(vd)) => {
                    let sel = if vb == vd {
                        vb
                    } else {
                        let cond = if scalar {
                            match wave_any_body {
                                Some(v) => v,
                                None => {
                                    let v = self.pred_wave_any(b, act_body)?;
                                    wave_any_body = Some(v);
                                    v
                                }
                            }
                        } else {
                            act_body
                        };
                        self.s.m.select(self.cur, cond, vb, vd)
                    };
                    if bit {
                        Val::Bit(sel)
                    } else {
                        Val::I32(sel)
                    }
                }
                (Err(why), _) | (_, Err(why)) => Val::Missing(why),
            };
            self.vals.insert(phi.value, result);
        }
        Ok(())
    }

    /// Evaluates a mask phi (`Atom::Phi`) at a linearized merge or `SkipGuard`
    /// join block `b`.
    pub(super) fn predicated_mask_phi(
        &mut self,
        id: Value,
        b: usize,
        i: usize,
        at: usize,
    ) -> Result<V, Why> {
        if let Some(&d) = self.pred.skip_joins.get(&b) {
            let FlatExit::SkipGuard { body } = self.pred.exits[&d] else {
                return Err("a lane-mask join without a skip guard");
            };
            let act_body = self.pred_edge_active[&(d, body)];
            let phi = &self.x.p.blocks[b].phis[i];
            let preds = &self.x.p.blocks[b].preds;
            let m_d = match preds.iter().position(|&q| q == d) {
                Some(n_d) => match (self.x.st.masks.get(&phi.args[n_d]), self.tail[d]) {
                    (Some(&m), Some(db)) => self.boolean(m, db)?,
                    _ => self.s.m.bool(false),
                },
                None => self.s.m.bool(false),
            };
            let body_preds: Vec<(usize, usize)> = preds
                .iter()
                .enumerate()
                .filter_map(|(n, &q)| {
                    (q != d
                        && !self.pred.dead_blocks.contains(&q)
                        && !self.pred.skipped_edges.contains(&(q, b))
                        && self.tail[q].is_some())
                    .then_some((n, q))
                })
                .collect();
            let m_body = match body_preds.as_slice() {
                [] => self.s.m.bool(false),
                &[(n_q, q)] => {
                    let qb = self.tail[q].expect("tail");
                    match self.x.st.masks.get(&phi.args[n_q]) {
                        Some(&m) => self.boolean(m, qb)?,
                        None => self.s.m.bool(false),
                    }
                }
                _ => {
                    let blk = self.block[b].ok_or("an unreachable join")?;
                    let v = self.s.m.phi(blk, self.s.t.i1);
                    self.pending_masks.push((v, b, i, None));
                    v
                }
            };
            let v = self.s.m.select(at, act_body, m_body, m_d);
            self.mask_phis.insert(Atom::Phi(id), v);
            return Ok(v);
        }

        let phi = &self.x.p.blocks[b].phis[i];
        let preds = &self.x.p.blocks[b].preds;
        let mut acc: Option<V> = None;
        for (n, &q) in preds.iter().enumerate() {
            if self.pred.dead_blocks.contains(&q) {
                continue;
            }
            let Some(qb) = self.tail[q] else { continue };
            let Some(&edge_act) = self.pred_edge_active.get(&(q, b)) else {
                continue;
            };
            let arg = phi.args[n];
            let m_val = match self.x.st.masks.get(&arg) {
                Some(&m) => self.boolean(m, qb)?,
                None => self.s.m.bool(false),
            };
            acc = Some(match acc {
                None => m_val,
                Some(prev) => self.s.m.select(at, edge_act, m_val, prev),
            });
        }
        let v = acc.unwrap_or_else(|| self.s.m.bool(false));
        self.mask_phis.insert(Atom::Phi(id), v);
        Ok(v)
    }

    /// Populates `loop_act` phis at all lockstep-predicated loop headers after
    /// all IR blocks and critical edges have been lowered.
    pub(super) fn fill_predicated_phis(&mut self) {
        let p = self.x.p;
        for (phi, h, latch) in std::mem::take(&mut self.loop_act_phis) {
            let mut incoming = Vec::new();
            for &q in &p.blocks[h].preds {
                if self.pred.dead_blocks.contains(&q) || self.pred.skipped_edges.contains(&(q, h)) {
                    continue;
                }
                let Some(qb) = self.tail[q] else { continue };
                let from = if q == latch {
                    qb
                } else {
                    self.edges.get(&(q, h)).copied().unwrap_or(qb)
                };
                let act = self
                    .pred_edge_active
                    .get(&(q, h))
                    .copied()
                    .or_else(|| self.pred_active(q))
                    .unwrap_or_else(|| self.s.m.bool(true));
                incoming.push((act, from));
            }
            self.s.m.set_phi(phi, incoming);
        }
    }

    /// Records per-edge active predicates at the end of IR block `b` and, if
    /// `b` has a flattened or lockstep exit in `self.pred`, returns its
    /// replacement terminator.
    pub(super) fn exit_predicated_block(&mut self, b: usize) -> Result<Option<Term>, Stop> {
        let Some(&flat) = self.pred.exits.get(&b) else {
            return Ok(None);
        };
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let cur = self.cur;
        let base_act = self.pred_active(b);
        let and_base = |l: &mut Self, cond: V| match base_act {
            Some(a) => l.s.m.bin(cur, Bin::And, a, cond),
            None => cond,
        };
        match flat {
            FlatExit::LockstepLatch {
                header,
                exit,
                stay_when_true,
            } => {
                let LaneExit::Branch { when, .. } = self.x.st.exits[b] else {
                    return Ok(None);
                };
                let cond = self
                    .boolean(when, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let stay = if stay_when_true {
                    cond
                } else {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(cur, Bin::Xor, cond, t)
                };
                let next_act = and_base(self, stay);
                self.pred_edge_active.insert((b, header), next_act);
                let any = self
                    .group_any_true(next_act, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let ok = self.count_iteration(header, pred::IULT);
                let go = self.s.m.bin(cur, Bin::And, any, ok);
                let exit_blk = self.edge(b, exit);
                let from = if Some(exit_blk) == self.block[exit] {
                    cur
                } else {
                    exit_blk
                };
                self.enter_loop(b, exit, from);
                Ok(Some(Term::CondBr(
                    go,
                    self.block[header].expect("reachable"),
                    exit_blk,
                )))
            }
            FlatExit::LockstepExit {
                header,
                stay,
                exit,
                stay_when_true,
            } => {
                let LaneExit::Branch { when, .. } = self.x.st.exits[b] else {
                    return Ok(None);
                };
                let cond = self
                    .boolean(when, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let stays = if stay_when_true {
                    cond
                } else {
                    let t = self.s.m.bool(true);
                    self.s.m.bin(cur, Bin::Xor, cond, t)
                };
                let next_act = and_base(self, stays);
                self.pred_edge_active.insert((b, stay), next_act);
                let any = self
                    .group_any_true(next_act, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let ok = self.count_iteration(header, pred::IULT);
                let go = self.s.m.bin(cur, Bin::And, any, ok);
                let exit_blk = self.edge(b, exit);
                let from = if Some(exit_blk) == self.block[exit] {
                    cur
                } else {
                    exit_blk
                };
                self.enter_loop(b, exit, from);
                Ok(Some(Term::CondBr(
                    go,
                    self.block[stay].expect("reachable"),
                    exit_blk,
                )))
            }
            FlatExit::EarlyOut { cont } => {
                self.record_branch_edge_actives(b, cur, offset, base_act)?;
                self.enter_loop(b, cont, cur);
                Ok(Some(Term::Br(self.block[cont].expect("reachable"))))
            }
            FlatExit::SkipGuard { body } => {
                self.record_branch_edge_actives(b, cur, offset, base_act)?;
                self.enter_loop(b, body, cur);
                Ok(Some(Term::Br(self.block[body].expect("reachable"))))
            }
            FlatExit::Linear { next } => {
                self.record_branch_edge_actives(b, cur, offset, base_act)?;
                self.enter_loop(b, next, cur);
                Ok(Some(Term::Br(self.block[next].expect("reachable"))))
            }
        }
    }

    fn record_branch_edge_actives(
        &mut self,
        b: usize,
        cur: usize,
        offset: usize,
        base_act: Option<V>,
    ) -> Result<(), Stop> {
        let and_base = |l: &mut Self, cond: V| match base_act {
            Some(a) => l.s.m.bin(cur, Bin::And, a, cond),
            None => cond,
        };
        match self.x.st.exits[b] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                let cond = self
                    .boolean(when, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let t = self.s.m.bool(true);
                let not_cond = self.s.m.bin(cur, Bin::Xor, cond, t);
                let act_taken = and_base(self, cond);
                let act_not_taken = and_base(self, not_cond);
                self.pred_edge_active.insert((b, taken), act_taken);
                self.pred_edge_active.insert((b, not_taken), act_not_taken);
            }
            LaneExit::Jump(target) => {
                let act = base_act.unwrap_or_else(|| self.s.m.bool(true));
                self.pred_edge_active.insert((b, target), act);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
        Ok(())
    }
}
