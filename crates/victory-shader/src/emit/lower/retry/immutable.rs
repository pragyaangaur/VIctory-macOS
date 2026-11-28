//! Literal SCC branches over a small, bounded set of immutable dwords.
use super::*;

mod paired;

struct Definitions<'a> {
    p: &'a Program,
    insts: Map<Value, (Site, Reg)>,
    phis: Map<Value, (usize, &'a Phi)>,
    rank: Vec<usize>,
    idom: Vec<usize>,
}

impl<'a> Definitions<'a> {
    fn new(p: &'a Program) -> Self {
        let mut rank = vec![usize::MAX; p.blocks.len()];
        let mut insts = Map::default();
        let mut phis = Map::default();
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            rank[b] = n;
            let block = &p.blocks[b];
            phis.extend(block.phis.iter().map(|phi| (phi.value, (b, phi))));
            for (i, inst) in block.insts.iter().enumerate() {
                insts.extend(inst.writes.iter().map(|&(r, v)| (v, ((b, i), r))));
            }
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &rank);
        Self {
            p,
            insts,
            phis,
            rank,
            idom,
        }
    }

    fn inst(&self, v: Value) -> Option<(&'a Inst, Reg)> {
        let &((b, i), r) = self.insts.get(&v)?;
        Some((&self.p.blocks[b].insts[i], r))
    }

    fn at(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn dominates(&self, def: usize, mut b: usize) -> bool {
        while self.rank[b] > self.rank[def] {
            b = self.idom[b];
        }
        b == def
    }

    fn before(&self, def: Site, at: Site) -> bool {
        if def.0 == at.0 {
            def.1 < at.1
        } else {
            self.dominates(def.0, at.0)
        }
    }

    fn read_at(&self, at: Site, reg: Reg) -> Option<Value> {
        let v = self.at(at).read(reg)?;
        let available = if let Some(&(def, r)) = self.insts.get(&v) {
            r == reg && self.before(def, at)
        } else {
            let &(b, phi) = self.phis.get(&v)?;
            phi.reg == reg && self.dominates(b, at.0)
        };
        available.then_some(v)
    }

    // Only ordinary SGPR dwords and exact eq/ne comparisons. A failed or
    // overflowing transfer is unknown even when the emitter could wrap it.
    fn number(&self, v: Value, state: &[(Value, Option<u32>)], fuel: &mut usize) -> Option<u32> {
        *fuel = fuel.checked_sub(1)?;
        if let Some(&(_, n)) = state.iter().find(|s| s.0 == v) {
            return n;
        }
        let (inst, reg) = self.inst(v)?;
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let mut arg = |i| self.operand(self.insts.get(&v)?.0, *src.get(i)?, state, fuel);
        match (reg, inst.name()) {
            (Reg::S(0..=105), "s_mov_b32") => arg(0),
            (Reg::S(0..=105), "s_add_u32") => arg(0)?.checked_add(arg(1)?),
            (Reg::S(0..=105), "s_add_i32") => {
                Some((arg(0)? as i32).checked_add(arg(1)? as i32)? as u32)
            }
            (Reg::S(0..=105), "s_lshl_b32") => arg(0)?.checked_mul(1u32.checked_shl(arg(1)?)?),
            (Reg::Scc, "s_cmp_eq_u32" | "s_cmp_eq_i32") => Some(u32::from(arg(0)? == arg(1)?)),
            (Reg::Scc, "s_cmp_lg_u32" | "s_cmp_lg_i32") => Some(u32::from(arg(0)? != arg(1)?)),
            _ => None,
        }
    }

    fn operand(
        &self,
        site: Site,
        op: Operand,
        state: &[(Value, Option<u32>)],
        fuel: &mut usize,
    ) -> Option<u32> {
        match op {
            Operand::Literal(_) | Operand::Const(_) => scalar_constant(op),
            Operand::Scalar { code: NULL, .. } => Some(0),
            Operand::Scalar {
                code: c @ 0..=105,
                dwords: 1,
            } => self.number(self.read_at(site, Reg::S(c))?, state, fuel),
            _ => None,
        }
    }

    /// The only loop state needed for this numeric displacement. Every phi
    /// is initially unknown and gets its actual predecessor operand on entry.
    fn index_phis(&self, mut pending: Vec<Value>) -> Option<Vec<Value>> {
        let mut seen = Set::default();
        let mut values = Vec::new();
        while let Some(v) = pending.pop() {
            if !seen.insert(v) {
                continue;
            }
            if seen.len() > 64 {
                return None;
            }
            if let Some(&(b, phi)) = self.phis.get(&v) {
                if !matches!(phi.reg, Reg::S(0..=105))
                    || phi.args.len() != self.p.blocks[b].preds.len()
                {
                    return None;
                }
                for (&arg, &pred) in phi.args.iter().zip(&self.p.blocks[b].preds) {
                    let def = self
                        .insts
                        .get(&arg)
                        .filter(|(_, r)| *r == phi.reg)
                        .map(|&((b, _), _)| b)
                        .or_else(|| {
                            self.phis
                                .get(&arg)
                                .and_then(|&(b, incoming)| (incoming.reg == phi.reg).then_some(b))
                        })?;
                    if *self.rank.get(pred)? == usize::MAX
                        || !self.p.blocks[pred].exit.successors().contains(&b)
                    {
                        return None;
                    }
                    if !self.dominates(def, pred) {
                        return None;
                    }
                }
                values.push(v);
                pending.extend(phi.args.iter().copied());
            } else if let Some((inst, Reg::S(0..=105))) = self.inst(v)
                && matches!(
                    inst.name(),
                    "s_mov_b32" | "s_add_u32" | "s_add_i32" | "s_lshl_b32"
                )
            {
                pending.extend(inst.reads.iter().map(|r| r.1));
            }
        }
        Some(values)
    }

    fn taken(&self, b: usize, state: &[(Value, Option<u32>)]) -> Option<bool> {
        let branch = self.p.blocks[b].insts.last()?;
        let site = (b, self.p.blocks[b].insts.len() - 1);
        let bit = self.number(self.read_at(site, Reg::Scc)?, state, &mut 64)? != 0;
        match branch.name() {
            "s_cbranch_scc1" => Some(bit),
            "s_cbranch_scc0" => Some(!bit),
            _ => None,
        }
    }

    /// Scalar load from the original PC pair plus an independently numeric
    /// byte displacement. The resource address alone can denote a legacy
    /// relocated value, which is insufficient for a content proof.
    fn contents(
        &self,
        native: &NativeProgram,
        site: Site,
        a: Address,
        first: u32,
        budget: &mut usize,
    ) -> Option<(Set<u32>, Option<paired::Roles>)> {
        let (b, i) = site;
        let inst = &self.p.blocks[b].insts[i];
        let Fields::Smem {
            base: Operand::Scalar { code, dwords: 2 },
            soffset,
            offset,
            ..
        } = inst.fields
        else {
            return None;
        };
        if inst.name() != "s_load_dword" || a.param != CODE {
            return None;
        }
        let dynamic = a.dynamic?;
        let lo = self.read_at(site, Reg::S(code))?;
        let (pc, _) = self.inst(lo)?;
        let paired = if pc.name() == "s_getpc_b64" {
            let hi = self.read_at(site, Reg::S(code.checked_add(1)?))?;
            if pc.writes != [(Reg::S(code), lo), (Reg::S(code + 1), hi)] {
                return None;
            }
            None
        } else {
            Some(self.paired(site, code, soffset, dynamic)?)
        };
        let mut roots = Vec::new();
        let base = if let Some(pair) = &paired {
            roots.push(pair.index);
            if let Operand::Scalar {
                code: c @ 0..=105, ..
            } = pair.count
            {
                roots.push(self.read_at(pair.shift_site, Reg::S(c))?);
            }
            pair.base
        } else {
            roots.push(dynamic);
            u32::try_from(pc.machine.offset).ok()?.checked_add(4)?
        };
        if paired.is_some()
            && let Operand::Scalar {
                code: c @ 0..=105, ..
            } = soffset
        {
            roots.push(self.read_at(site, Reg::S(c))?);
        }
        let phis = self.index_phis(roots)?;
        let mut pending = vec![(0usize, phis.iter().map(|&v| (v, None)).collect::<Vec<_>>())];
        let mut seen = Set::default();
        let mut words = Set::default();
        while let Some((from, state)) = pending.pop() {
            if !seen.insert((from, state.clone())) {
                continue;
            }
            // A proof budget limits compilation only. Partial traces never qualify.
            *budget = budget.checked_sub(1)?;
            if seen.len() > self.p.blocks.len().saturating_mul(16).min(8192) {
                return None;
            }
            if from == b {
                let delta = if let Some(pair) = &paired {
                    let index = self.number(pair.index, &state, &mut 64)?;
                    let count = self.operand(pair.shift_site, pair.count, &state, &mut 64)?;
                    paired::displacement(index, count)?
                } else {
                    self.number(dynamic, &state, &mut 64)?
                };
                let address = a.offset.checked_add(delta)?;
                let guest = base
                    .checked_add(if paired.is_some() { delta } else { 0 })?
                    .checked_add(self.operand(site, soffset, &state, &mut 64)?)?
                    .checked_add_signed(offset)?;
                if address != guest {
                    return None;
                }
                words.insert(table_word(&native.code, first, address)?);
            }
            let block = &self.p.blocks[from];
            let selected = match block.exit {
                // A constant guard may hide a statically reachable second user
                // of the same PC. Cover that path too when certifying the pair.
                Exit::Branch { .. } if paired.is_some() && self.taken(from, &[]).is_some() => None,
                Exit::Branch { taken, not_taken } => self
                    .taken(from, &state)
                    .map(|cond| if cond { taken } else { not_taken }),
                _ => None,
            };
            for to in block.exit.successors() {
                // A selected conditional backedge may instead leave through
                // its forward successor when that header exhausts its cap.
                if selected.is_some_and(|s| s != to && self.rank[s] > self.rank[from]) {
                    continue;
                }
                let mut next = state.clone();
                for (v, value) in &mut next {
                    let &(header, phi) = self.phis.get(v)?;
                    if header == to {
                        let pred = self.p.blocks[to].preds.iter().position(|&p| p == from)?;
                        *value = self.number(*phi.args.get(pred)?, &state, &mut 64);
                    }
                }
                pending.push((to, next));
            }
        }
        (!words.is_empty()).then_some((words, paired.map(|p| p.roles)))
    }
}

fn table_word(code: &[u8], first: u32, address: u32) -> Option<u32> {
    if address < first || !address.is_multiple_of(4) {
        return None;
    }
    let bytes = code.get(address as usize..address.checked_add(4)? as usize)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

pub(in crate::emit::lower) struct Facts {
    pub(in crate::emit::lower) reads: Map<Site, Set<u32>>,
    pub(in crate::emit::lower) carries: Set<(Site, Value)>,
    pub(super) jumps: Vec<(usize, usize)>,
}

pub(in crate::emit::lower) fn prove(
    p: &Program,
    st: &Structured,
    res: &Resources,
    native: &NativeProgram,
) -> Facts {
    let mut facts = Facts {
        reads: Map::default(),
        jumps: Vec::new(),
        carries: Set::default(),
    };
    if !res
        .scalar
        .iter()
        .any(|(_, a)| a.param == CODE && a.dynamic.is_some())
    {
        return facts;
    }
    let defs = Definitions::new(p);
    // A dynamic read below the emitter's first table word would be clamped.
    // code_end is a conservative floor even when an independent fixed read
    // makes the emitted table start earlier.
    let Some(end) = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| i.machine.offset + 4 * i.machine.words.len())
        .max()
        .and_then(|n| u32::try_from(n).ok())
    else {
        return facts;
    };
    let mut budget = 65_536;
    let mut paired_reads = Map::default();
    let mut roles = paired::Roles::default();
    for &(site, a) in &res.scalar {
        if defs.rank[site.0] != usize::MAX
            && let Some((words, paired)) = defs.contents(native, site, a, end, &mut budget)
        {
            if let Some(pair) = paired {
                roles.extend(pair);
                paired_reads.insert(site, words);
            } else {
                facts.reads.insert(site, words);
            }
        }
    }
    let legacy = !facts.reads.is_empty();
    if !paired_reads.is_empty() && roles.valid(&defs, st, res) {
        facts.carries = roles.carries;
        facts.reads.extend(paired_reads);
    }
    // Only the refused literal shape needs executable edge simplification.
    if facts.reads.is_empty()
        || !p.blocks.iter().enumerate().any(|(b, block)| {
            matches!(block.exit,
        Exit::Branch { taken, not_taken } if defs.rank[b] != usize::MAX && taken != not_taken &&
        defs.rank[taken] <= defs.rank[b] && defs.rank[not_taken] <= defs.rank[b])
        })
    {
        return facts;
    }
    for (b, block) in p.blocks.iter().enumerate() {
        let Exit::Branch { taken, not_taken } = block.exit else {
            continue;
        };
        let Some(branch) = block.insts.last() else {
            continue;
        };
        if !matches!(branch.name(), "s_cbranch_scc0" | "s_cbranch_scc1") {
            continue;
        }
        let cond = legacy.then(|| defs.taken(b, &[])).flatten().or_else(|| {
            let cmp_value = defs.read_at((b, block.insts.len() - 1), Reg::Scc)?;
            let (cmp, Reg::Scc) = defs.inst(cmp_value)? else {
                return None;
            };
            let Fields::Sop { src, .. } = &cmp.fields else {
                return None;
            };
            let [a, b] = src[..] else { return None };
            let (values, constant) = [(a, b), (b, a)].into_iter().find_map(|(load, k)| {
                let Operand::Scalar { code, dwords: 1 } = load else {
                    return None;
                };
                let cmp_site = defs.insts.get(&cmp_value)?.0;
                let (site, _) = defs.insts.get(&defs.read_at(cmp_site, Reg::S(code))?)?;
                Some((
                    facts.reads.get(site)?,
                    defs.operand(cmp_site, k, &[], &mut 64)?,
                ))
            })?;
            let eq = match cmp.name() {
                "s_cmp_eq_u32" | "s_cmp_eq_i32" => true,
                "s_cmp_lg_u32" | "s_cmp_lg_i32" => false,
                _ => return None,
            };
            let taken_if_equal = eq == (branch.name() == "s_cbranch_scc1");
            if values.contains(&constant) {
                (values.len() == 1).then_some(taken_if_equal)
            } else {
                Some(!taken_if_equal)
            }
        });
        if let Some(take) = cond {
            let (selected, other) = if take {
                (taken, not_taken)
            } else {
                (not_taken, taken)
            };
            if defs.rank[selected] > defs.rank[b] || defs.rank[other] <= defs.rank[b] {
                facts.jumps.push((b, selected));
            }
        }
    }
    facts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_table_declines_overflowing_transfers() {
        let program = |op| {
            let words = [0xbe8c_03ff, u32::MAX, op, 0xbf81_0000];
            let code = words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            crate::ir::build(&code, true, 4).unwrap()
        };
        for op in [0x800c_810c, 0x8f0c_810c] {
            // unsigned +1, left shift 1
            let p = program(op);
            let defs = Definitions::new(&p);
            let value = p.blocks[0].insts[1].write(Reg::S(12)).unwrap();
            assert_eq!(defs.number(value, &[], &mut 64), None);
        }
    }

    #[test]
    fn immutable_table_paired_shift_width() {
        for (index, count, expected) in [
            (1, 0, Some(1)),
            (1, 31, Some(1 << 31)),
            (1, 32, None),
            (1, 63, None),
            (1, 64, Some(1)),
            (1, 65, Some(2)),
            (2, 63, None),
            (0, 63, Some(0)),
            (u32::MAX, 0, None),
            (u32::MAX, 1, None),
        ] {
            assert_eq!(paired::displacement(index, count), expected);
        }
    }

    #[test]
    fn immutable_table_original_dwords_only() {
        let bytes = [0, 0, 0, 0, 0x23, 0x45, 0x67, 0x89, 0xff];
        assert_eq!(table_word(&bytes, 4, 4), Some(0x8967_4523));
        for (first, at) in [(4, 0), (0, 1), (0, 8), (0, 12), (0, u32::MAX - 3)] {
            assert_eq!(table_word(&bytes, first, at), None);
        }
    }
}
