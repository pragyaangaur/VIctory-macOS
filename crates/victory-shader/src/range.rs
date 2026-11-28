//! Upper bounds of the index of register-relative moves (`v_movrel*`,
//! `v_swaprel_b32`), proven on the per-lane form, so that a move names only
//! the registers its M0 can reach rather than every one up to the program's
//! VGPR count.
//!
//! A value is bounded when it is a constant; a copy (`s_mov_b32`,
//! `v_mov_b32`, or an unguarded `v_readfirstlane_b32`: the first active
//! lane is then a lane that runs there) of a bounded value; `min`, `and` or
//! `med3` (unsigned) with a bounded operand; a sum of bounded values that
//! cannot wrap; a plain dword selection with both value arms bounded; or a
//! phi whose every incoming value is bounded on its edge,
//! where an unsigned compare of that value with a bounded one (`v_cmp*`,
//! `s_cmp*`: `lt`, `le`, `eq` true or `gt`, `ge` false) that every lane
//! taking the edge has passed also bounds it: a loop counter against a
//! bounded limit. A scalar counter with a constant seed and positive fixed
//! stride is bounded if its latch exits at an exactly reachable equality.
//! A VGPR write under a guard keeps the old value in the
//! other lanes, so it is bounded only if both are. Anything else (memory,
//! signed compares, what a relative move writes) is unbounded, and a move
//! with an unbounded index keeps today's range.

use std::collections::{HashMap, HashSet};

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{self, EXEC, Inst, M0, Program, Reg, Value};
use crate::structure::{self, Atom, LaneExit, Structured};

/// [`ir::build`] and [`structure::structure`], built again with the register
/// ranges of relative moves bounded where their index is.
///
/// A move can write the registers its own index comes from (the full range
/// of a `v_movreld` covers a loop's counter and limit), so the bounds are
/// found as an induction: assume each move reaches only its first register,
/// widen to what the analysis finds under the assumption until the
/// assumption implies itself. Then no run takes a move past it (the first
/// move that would has run, until then, as the assumption says), and what
/// the analysis found holds.
pub fn build(
    code: &[u8],
    wave64: bool,
    vgprs: u16,
) -> Result<(Program, Result<Structured, ShaderError>), ShaderError> {
    let p = ir::build(code, wave64, vgprs)?;
    let st = structure::structure(&p);
    let mut assumed: HashMap<usize, u8> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| relative(i))
        .map(|i| (i.machine.offset, 0))
        .collect();
    if st.is_err() {
        assumed.clear();
    }
    // Each round grows a bound or drops a move; few programs need more than two.
    for _ in 0..8 {
        if assumed.is_empty() {
            break;
        }
        let q = ir::build_bounded(code, wave64, vgprs, &assumed)?;
        let Ok(s) = structure::structure(&q) else {
            break;
        };
        let found = relative_bounds(&q, &s);
        let proven: HashMap<usize, u8> = assumed
            .iter()
            .filter_map(|(&o, &m)| Some((o, u8::try_from(found[&o]).ok().filter(|&f| f <= m)?)))
            .collect();
        if proven.len() == assumed.len() {
            if proven == assumed {
                return Ok((q, Ok(s)));
            }
            let q = ir::build_bounded(code, wave64, vgprs, &proven)?;
            let s = structure::structure(&q);
            return Ok((q, s));
        }
        assumed = assumed
            .iter()
            .filter_map(|(&o, &m)| Some((o, u8::try_from(found[&o]).ok()?.max(m))))
            .collect();
    }
    Ok((p, st))
}

fn relative(inst: &Inst) -> bool {
    ir::relative_operands(inst.name()) != (false, false)
}

/// The largest M0 each relative move can see, by instruction offset
/// (`u32::MAX`: unbounded).
fn relative_bounds(p: &Program, st: &Structured) -> HashMap<usize, u32> {
    let mut r = Ranges {
        p,
        st,
        defs: HashMap::new(),
        memo: HashMap::new(),
        busy: HashSet::new(),
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, phi) in block.phis.iter().enumerate() {
            r.defs.insert(phi.value, Def::Phi(b, i));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for w in &inst.writes {
                r.defs.insert(w.1, Def::Inst(b, i));
            }
        }
    }
    // Copies of a block (irreducible loops) share offsets: the largest wins.
    let mut out: HashMap<usize, u32> = HashMap::new();
    for inst in p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| relative(i))
    {
        let hi = inst.read(Reg::S(M0)).map_or(u32::MAX, |m0| r.hi(m0));
        let e = out.entry(inst.machine.offset).or_insert(0);
        *e = (*e).max(hi);
    }
    out
}

#[derive(Clone, Copy)]
enum Def {
    Phi(usize, usize),
    Inst(usize, usize),
}

struct Ranges<'a> {
    p: &'a Program,
    st: &'a Structured,
    defs: HashMap<Value, Def>,
    memo: HashMap<Value, u32>,
    /// Values being bounded: met again, a cycle, unbounded.
    busy: HashSet<Value>,
}

impl Ranges<'_> {
    /// The largest unsigned value `v` takes in any lane that defines it.
    fn hi(&mut self, v: Value) -> u32 {
        if let Some(&h) = self.memo.get(&v) {
            return h;
        }
        if !self.busy.insert(v) {
            return u32::MAX;
        }
        let p = self.p;
        let h = match self.defs.get(&v).copied() {
            Some(Def::Phi(b, i)) => {
                let block = &p.blocks[b];
                let recurrence = self.recurrence_hi(b, i);
                let mut h = 0;
                for (k, &pred) in block.preds.iter().enumerate() {
                    let arg = block.phis[i].args[k];
                    let bound = recurrence
                        .filter(|&(update, _)| update == arg)
                        .map_or(u32::MAX, |(_, h)| h);
                    h = h.max(self.edge_hi(arg, pred, b, bound));
                }
                h
            }
            Some(Def::Inst(b, i)) => {
                let inst = &p.blocks[b].insts[i];
                let reg = inst.writes.iter().find(|w| w.1 == v).map(|w| w.0);
                // Under a guard EXEC may be empty there, and the first lane
                // then is lane 0, which may not run there.
                let lane0 = inst.name() == "v_readfirstlane_b32" && self.st.guards[b][i].is_some();
                let h = if reg.is_some() && reg == dst(inst) && !lane0 {
                    self.op_hi(inst)
                } else {
                    u32::MAX
                };
                match (reg, self.st.guards[b][i]) {
                    (Some(r @ Reg::V(_)), Some(_)) => {
                        h.max(inst.read(r).map_or(u32::MAX, |o| self.hi(o)))
                    }
                    _ => h,
                }
            }
            None => u32::MAX,
        };
        self.busy.remove(&v);
        self.memo.insert(v, h);
        h
    }

    /// A two-input scalar phi seeded by a constant and updated only on a
    /// backedge testing that exact sum for inequality with a constant limit.
    fn recurrence_hi(&self, header: usize, i: usize) -> Option<(Value, u32)> {
        let block = &self.p.blocks[header];
        let phi = &block.phis[i];
        let [a, b] = phi.args[..] else { return None };
        if !matches!(phi.reg, Reg::S(0..=105)) {
            return None;
        }
        let seed = |v| {
            let (mov, src) = self.scalar_def(v)?;
            match (mov.name(), src) {
                ("s_mov_b32", [op]) => unsigned_immediate(op),
                _ => None,
            }
        };
        let (start, update, pred) = match (seed(a), seed(b)) {
            (Some(n), None) => (n, b, block.preds[1]),
            (None, Some(n)) => (n, a, block.preds[0]),
            _ => return None,
        };
        let latch = &self.p.blocks[pred];
        let ir::Exit::Branch { taken, not_taken } = latch.exit else {
            return None;
        };
        let branch = latch.insts.last()?;
        if branch.name() != "s_cbranch_scc0" || taken != header || not_taken == header {
            return None;
        }
        let scc = branch.read(Reg::Scc)?;
        let Def::Inst(cmp_block, ci) = *self.defs.get(&scc)? else {
            return None;
        };
        let cmp = &self.p.blocks[cmp_block].insts[ci];
        if cmp_block != pred || cmp.name() != "s_cmp_eq_u32" {
            return None;
        }
        let Fields::Sop { src, .. } = &cmp.fields else {
            return None;
        };
        let (compared, limit) = value_and_constant(cmp, src)?;
        if compared != update || limit <= start {
            return None;
        }
        let mut v = update;
        let mut stride = 0u32;
        while v != phi.value {
            let (add, src) = self.scalar_def(v)?;
            if !matches!(add.name(), "s_add_i32" | "s_add_u32") {
                return None;
            }
            let (prev, step) = value_and_constant(add, src)?;
            stride = stride.checked_add(step)?;
            v = prev;
        }
        // Every partial sum is nonnegative and the last one fits at the limit.
        (stride != 0 && (limit - start).is_multiple_of(stride)).then(|| (update, limit - stride))
    }

    /// Inputs of an ordinary scalar dword definition, excluding its SCC result.
    fn scalar_def(&self, v: Value) -> Option<(&Inst, &[Operand])> {
        let Def::Inst(b, i) = *self.defs.get(&v)? else {
            return None;
        };
        let inst = &self.p.blocks[b].insts[i];
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let reg = dst(inst)?;
        (matches!(reg, Reg::S(0..=105)) && inst.write(reg) == Some(v)).then_some((inst, src))
    }

    /// The largest value `a` has in lanes going from `pred` to `to`.
    fn edge_hi(&mut self, a: Value, pred: usize, to: usize, mut best: u32) -> u32 {
        let mut facts = vec![(self.st.reach[pred], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.st.exits[pred]
            && taken != not_taken
        {
            facts.push((when, taken == to));
        }
        let mut atoms: Vec<Atom> = facts
            .iter()
            .flat_map(|&(f, _)| self.st.bools.support(f))
            .collect();
        atoms.dedup();
        for atom in atoms {
            let (Atom::Lane(c) | Atom::Uniform(c)) = atom else {
                continue;
            };
            if let Some(holds) = self.st.bools.decides(&facts, atom)
                && let Some(h) = self.compare_bound(c, a, holds)
            {
                best = best.min(h);
            }
        }
        if best == u32::MAX { self.hi(a) } else { best }
    }

    /// What an unsigned compare whose result `c` is `holds` says of the
    /// largest value of its operand `a`.
    fn compare_bound(&mut self, c: Value, a: Value, holds: bool) -> Option<u32> {
        let Some(&Def::Inst(b, i)) = self.defs.get(&c) else {
            return None;
        };
        let p = self.p;
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let (op, src) = match &inst.fields {
            Fields::Valu(v) if v.sdwa.is_none() && v.dpp.is_none() => (
                name.strip_prefix("v_cmpx_")
                    .or_else(|| name.strip_prefix("v_cmp_"))?,
                &v.src,
            ),
            Fields::Sop { src, .. } if inst.write(Reg::Scc) == Some(c) => {
                (name.strip_prefix("s_cmp_")?, src)
            }
            _ => return None,
        };
        let op = op.strip_suffix("_u32")?;
        let [x, y] = &src[..] else { return None };
        let (x, y) = (value(inst, x), value(inst, y));
        // `a op other`, with the result as it holds.
        let (op, other) = match (x == Some(a), y == Some(a)) {
            (true, false) => (op, &src[1]),
            (false, true) => (
                match op {
                    "lt" => "gt",
                    "le" => "ge",
                    "gt" => "lt",
                    "ge" => "le",
                    o => o,
                },
                &src[0],
            ),
            _ => return None,
        };
        let below = match (op, holds) {
            ("lt", true) | ("ge", false) => true,
            ("le", true) | ("gt", false) | ("eq", true) => false,
            _ => return None,
        };
        let h = self.operand_hi(inst, other);
        Some(if below { h.saturating_sub(1) } else { h })
    }

    /// The largest result of `inst`, from its operands.
    fn op_hi(&mut self, inst: &Inst) -> u32 {
        let src = match &inst.fields {
            Fields::Valu(v)
                if v.sdwa.is_none()
                    && v.dpp.is_none()
                    && !v.clamp
                    && v.omod == 0
                    && v.neg | v.abs | v.opsel == 0 =>
            {
                &v.src
            }
            Fields::Sop { src, .. } => src,
            _ => return u32::MAX,
        };
        if let ("v_cndmask_b32", [x, y, _mask]) = (inst.name(), &src[..]) {
            return self.operand_hi(inst, x).max(self.operand_hi(inst, y));
        }
        let s: Vec<u32> = src.iter().map(|o| self.operand_hi(inst, o)).collect();
        match (inst.name(), &s[..]) {
            ("s_mov_b32" | "v_mov_b32" | "v_readfirstlane_b32", &[x]) => x,
            ("s_min_u32" | "v_min_u32" | "s_and_b32" | "v_and_b32", &[x, y]) => x.min(y),
            ("s_max_u32" | "v_max_u32", &[x, y]) => x.max(y),
            // The median is at most the larger of any two operands.
            ("v_med3_u32", &[x, y, z]) => x.max(y).min(x.max(z)).min(y.max(z)),
            // A sum that may wrap is unbounded.
            ("s_add_u32" | "s_add_i32" | "v_add_nc_u32", &[x, y]) => x.saturating_add(y),
            _ => u32::MAX,
        }
    }

    fn operand_hi(&mut self, inst: &Inst, op: &Operand) -> u32 {
        unsigned_immediate(op).unwrap_or_else(|| value(inst, op).map_or(u32::MAX, |v| self.hi(v)))
    }
}

fn unsigned_immediate(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c @ 128..=192) => Some(u32::from(c - 128)),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

fn value_and_constant(inst: &Inst, src: &[Operand]) -> Option<(Value, u32)> {
    let [x, y] = src else { return None };
    if let Some(c) = unsigned_immediate(y) {
        Some((value(inst, x)?, c))
    } else {
        Some((value(inst, y)?, unsigned_immediate(x)?))
    }
}

/// The value a one-dword register operand reads.
fn value(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Vector { reg, dwords: 1 } => inst.read(Reg::V(reg)),
        // Ordinary shaders cannot write trap temporaries; their nominal SSA writes supply no bound.
        Operand::Scalar { code, dwords: 1 } if !matches!(code, 108..=123 | EXEC) => {
            inst.read(Reg::S(code))
        }
        _ => None,
    }
}

/// The register an instruction's one-dword result goes to.
fn dst(inst: &Inst) -> Option<Reg> {
    match &inst.fields {
        Fields::Valu(v) => match v.dst? {
            Operand::Vector { reg, dwords: 1 } => Some(Reg::V(reg)),
            Operand::Scalar { code, dwords: 1 } => Some(Reg::S(code)),
            _ => None,
        },
        Fields::Sop {
            dst: Some(Operand::Scalar { code, dwords: 1 }),
            ..
        } => Some(Reg::S(*code)),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    const V: u32 = 256;

    const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
        0x7E00_0000 | vdst << 17 | op << 9 | src0
    }

    const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
        op << 25 | vdst << 17 | vsrc1 << 9 | src0
    }

    /// `s_mov_b32 m0, s0`.
    const M0_S0: u32 = 0xBEFC_0300;

    fn build_words(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (p, st) = build(&code, false, 16).unwrap();
        st.unwrap();
        p
    }

    fn vgprs(regs: &[(Reg, Value)]) -> Vec<u8> {
        regs.iter()
            .filter_map(|r| match r.0 {
                Reg::V(v) => Some(v),
                _ => None,
            })
            .collect()
    }

    fn wave64_program(words: &[u32]) -> (Program, Structured) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (p, st) = build(&code, true, 32).unwrap();
        (p, st.unwrap())
    }

    pub(crate) fn equality_words() -> Vec<u32> {
        let mut w: Vec<_> = (26..32).map(|r| vop1(1, r, 128 + r)).collect();
        w.push(0xbe80_0380); // s0 = 0
        for dst in 20..23 {
            w.extend([0x8100_8100, M0_S0, vop1(0x43, dst, V + 25)]); // ++s0, read v[25+s0]
        }
        w.extend([0xbf06_8600, 0xbf84_fff5, S_ENDPGM]); // repeat unless s0 == 6
        w
    }

    #[test]
    fn recurrence_links_and_selector_arms_respect_scalar_register_roles() {
        for (code, ordinary, bound) in [
            (105, true, 2),
            (108, false, u32::MAX),
            (123, false, u32::MAX),
            (M0, false, 2),
        ] {
            let (p, st) = wave64_program(&[
                0x8100_8181 | u32::from(code) << 16, // scalar result of 1 + 1
                vop1(1, 3, 128),                     // v3 = 0
                vop2(1, 4, u32::from(code), 3),      // select scalar or zero
                0xbe83_0300 | u32::from(code),       // copy the scalar into s3
                vop2(1, 5, 3, 3),                    // select copied scalar or zero
                S_ENDPGM,
            ]);
            let insts = &p.blocks[0].insts;
            let scalar = insts[0].write(Reg::S(code)).unwrap();
            let direct = insts[2].write(Reg::V(4)).unwrap();
            let copied = insts[4].write(Reg::V(5)).unwrap();
            let mut ranges = Ranges {
                p: &p,
                st: &st,
                defs: insts
                    .iter()
                    .enumerate()
                    .flat_map(|(i, inst)| {
                        inst.writes.iter().map(move |&(_, v)| (v, Def::Inst(0, i)))
                    })
                    .collect(),
                memo: HashMap::new(),
                busy: HashSet::new(),
            };
            assert_eq!(
                (
                    ranges.scalar_def(scalar).is_some(),
                    ranges.hi(direct),
                    ranges.hi(copied)
                ),
                (ordinary, bound, bound),
                "scalar register {code}"
            );
        }
    }

    #[test]
    fn equality_latch_bounds_each_increment() {
        let (p, _) = wave64_program(&equality_words());
        let moves: Vec<_> = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| relative(i))
            .collect();
        // Trips read v26..v28 then v29..v31. The last read reaches the final register.
        assert_eq!(
            moves.iter().map(|i| i.m0_max).collect::<Vec<_>>(),
            [Some(4), Some(5), Some(6)]
        );
        assert!(moves.iter().all(|i| i.relative_in_file(p.vgprs)));
    }

    #[test]
    fn unchecked_latches_do_not_prove_in_file_accesses() {
        let mut cases = Vec::new();
        for (case, word, replacement) in [
            ("skipped limit", 16, 0xbf06_8500), // trip advances by 3, limit 5
            ("variable step", 7, 0x8100_0100),  // add s1
            ("unrelated compare", 16, 0xbf06_8601), // compare s1
            ("continue on equal", 17, 0xbf85_fff5),
            ("unknown entry", 6, 0xbe80_0301), // seed from input s1
        ] {
            let mut w = equality_words();
            w[word] = replacement;
            cases.push((case, w));
        }
        let mut w = equality_words();
        for n in [7, 10, 13] {
            w[n] = 0x8100_8000; // add zero
        }
        cases.push(("zero stride", w));
        let mut w = equality_words();
        w.splice(6..7, [0xbe80_03ff, u32::MAX - 2]); // first trip wraps to zero
        cases.push(("wrap", w));
        let mut w = equality_words();
        w.splice(7..8, [0x8100_ff00, u32::MAX]); // the sum of three steps exceeds u32::MAX
        w[18] = 0xbf84_fff4;
        cases.push(("overflowing stride", w));
        let mut w = equality_words();
        w.insert(13, 0xbe81_0300); // save s0 before its last increment
        w[17] = 0xbf06_8601; // test that previous version instead of the update
        w[18] = 0xbf84_fff4;
        cases.push(("stale counter", w));
        let mut w = equality_words();
        w.splice(16..16, [0xbf06_8001, 0xbf85_fff5]); // if s1 == 0, bypass the latch
        w[19] = 0xbf84_fff3;
        cases.push(("second unchecked incoming edge", w));
        for (case, words) in cases {
            let (p, _) = wave64_program(&words);
            let moves: Vec<_> = p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| relative(i))
                .collect();
            assert_eq!(moves.len(), 3, "{case}");
            assert!(moves.iter().all(|i| !i.relative_in_file(p.vgprs)), "{case}");
        }
    }

    pub(crate) const SELECT_TWO_OR_ELEVEN: u32 = vop2(1, 2, 130, 3);

    pub(crate) fn selection_words(select: u32) -> Vec<u32> {
        let mut w: Vec<_> = (20..32).map(|r| vop1(1, r, 128 + r)).collect();
        w.extend([
            vop1(1, 3, 139),
            vop2(0x1b, 1, 129, 0),
            0x7d84_0280, // VCC = (tid & 1) == 0
            select,
            vop1(2, 20, V + 2),
            0x7d84_0414,
            0xbe8e_246a,
            0xbefc_0314, // select first key
            vop1(0x43, 18, V + 20),
            0x89fe_0e7e,
            0xbf89_fff9,
            S_ENDPGM,
        ]);
        w
    }

    #[test]
    fn conditional_selection_bounds_both_waterfall_arms() {
        for (select, bound) in [
            (vop2(0x1b, 2, 139, 0), Some(11)), // AND control
            (SELECT_TWO_OR_ELEVEN, Some(11)),  // select 2 or 11
            (vop2(1, 2, 140, 3), Some(12)),    // select 12 or 11, past the file
            (vop2(1, 2, 130, 0), None),        // one unknown arm
        ] {
            let w = selection_words(select);
            let (p, st) = wave64_program(&w);
            let mov = p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| relative(i))
                .unwrap();
            assert_eq!(mov.m0_max, bound);
            let tracker = crate::resource::track(&p, &[], crate::HwStage::Cs, Some(&st));
            if bound == Some(11) {
                assert!(mov.relative_in_file(p.vgprs));
                let facts = tracker.unwrap().waterfall;
                assert_eq!((facts.reads.len(), facts.compares.len()), (1, 1));
            } else {
                assert!(!mov.relative_in_file(p.vgprs));
                assert!(tracker.is_err());
            }
        }
    }

    #[test]
    fn a_bound_must_survive_what_the_bounded_moves_write() {
        // v1 = v0 & 3, vN = v0 & 1; v[2 + v1] = v7; then v[8 + vN] is read.
        // The first move reaches v2-v5: with N = 4 it may overwrite the
        // second index, which then has no bound (it would, were the first
        // move taken to write only v2); with N = 6 the second reads v8-v9.
        for (n, second) in [
            (4, (8..16).chain([1]).collect::<Vec<u8>>()),
            (6, vec![8, 9, 1]),
        ] {
            let p = build_words(&[
                vop2(0x1B, 1, 131, 0), // v_and_b32 v1, 3, v0
                vop2(0x1B, n, 129, 0), // v_and_b32 vN, 1, v0
                vop1(0x02, 0, V + 1),  // v_readfirstlane_b32 s0, v1
                M0_S0,
                vop1(0x42, 2, V + 7), // v_movreld_b32 v2, v7
                vop1(0x02, 0, V + n), // v_readfirstlane_b32 s0, vN
                M0_S0,
                vop1(0x43, 1, V + 8), // v_movrels_b32 v1, v8
                S_ENDPGM,
            ]);
            let i = &p.blocks[0].insts;
            assert_eq!(vgprs(&i[4].writes), [2, 3, 4, 5]);
            assert_eq!(vgprs(&i[7].reads), second, "index in v{n}");
        }
    }

    #[test]
    fn a_loop_counter_bounds_the_index() {
        // for (v7 = 0; v7 < 4; v7++) v[2 + v7] = v0. Without the bound the
        // move could overwrite its own counter.
        let p = build_words(&[
            vop1(0x01, 7, 128),                      // v_mov_b32 v7, 0
            vop1(0x02, 0, V + 7),                    // v_readfirstlane_b32 s0, v7
            M0_S0,                                   //
            vop1(0x42, 2, V),                        // v_movreld_b32 v2, v0
            vop2(0x25, 7, 129, 7),                   // v_add_nc_u32 v7, 1, v7
            0x7C00_0000 | 0xC4 << 17 | 7 << 9 | 132, // v_cmp_gt_u32 vcc, 4, v7
            0xBF87_FFFA,                             // s_cbranch_vccnz to the readfirstlane
            S_ENDPGM,
        ]);
        let movreld = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "v_movreld_b32")
            .unwrap();
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // A limit of 5 lets the index reach 4: v2-v6.
        let p = build_words(&[
            vop1(0x01, 7, 128),
            vop1(0x02, 0, V + 7),
            M0_S0,
            vop1(0x42, 2, V),
            vop2(0x25, 7, 129, 7),
            0x7C00_0000 | 0xC4 << 17 | 7 << 9 | 133, // v_cmp_gt_u32 vcc, 5, v7
            0xBF87_FFFA,
            S_ENDPGM,
        ]);
        let movreld = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "v_movreld_b32")
            .unwrap();
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_first_lane_where_exec_may_be_empty_is_unbounded() {
        // In lanes with v0 < 2: v1 = v0 & 3; then EXEC = lanes with v0 < 1,
        // possibly none, and m0 = the first lane's v1: with none, lane 0's,
        // which may never have set v1. Read before narrowing EXEC, the index
        // is bounded.
        for (narrow_first, reads) in [
            (true, (8..16).collect::<Vec<u8>>()),
            (false, vec![8, 9, 10, 11]),
        ] {
            let narrow = [
                0x7C00_0000 | 0xC4 << 17 | 129, // v_cmp_gt_u32 vcc, 1, v0
                0xBE83_3C6A,                    // s_and_saveexec_b32 s3, vcc_lo
            ];
            let first = vop1(0x02, 0, V + 1); // v_readfirstlane_b32 s0, v1
            let mut words = vec![
                0x7C00_0000 | 0xC4 << 17 | 130, // v_cmp_gt_u32 vcc, 2, v0
                0xBE82_3C6A,                    // s_and_saveexec_b32 s2, vcc_lo
                0xBF88_0007,                    // s_cbranch_execz to the end
                vop2(0x1B, 1, 131, 0),          // v_and_b32 v1, 3, v0
            ];
            if narrow_first {
                words.extend(narrow);
                words.push(first);
            } else {
                words.push(first);
                words.extend(narrow);
            }
            words.extend([
                M0_S0,
                0xBEFE_0303,          // s_mov_b32 exec_lo, s3
                vop1(0x43, 4, V + 8), // v_movrels_b32 v4, v8
                0xBEFE_0302,          // s_mov_b32 exec_lo, s2
                S_ENDPGM,
            ]);
            let p = build_words(&words);
            let movrels = p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.name() == "v_movrels_b32")
                .unwrap();
            assert_eq!(
                vgprs(&movrels.reads)[..reads.len()],
                reads[..],
                "{narrow_first}"
            );
        }
    }
}
