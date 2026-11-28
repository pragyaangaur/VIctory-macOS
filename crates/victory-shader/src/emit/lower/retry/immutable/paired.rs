//! A PC pair has unknown guest bits. Only a complete low ADD / high ADDC
//! represents the same relative addition for every relocation.
use super::*;

#[derive(Default)]
pub(super) struct Roles {
    values: Set<Value>,
    reads: Set<(Site, Reg, Value)>,
    pub(super) carries: Set<(Site, Value)>,
}

pub(super) struct Paired {
    pub(super) base: u32,
    pub(super) shift_site: Site,
    pub(super) index: Value,
    pub(super) count: Operand,
    pub(super) roles: Roles,
}

struct Add {
    site: Site,
    result: Value,
    carry: Value,
    src: [Operand; 2],
}

impl Definitions<'_> {
    fn add(&self, value: Value, name: &str) -> Option<Add> {
        let &(site, reg) = self.insts.get(&value)?;
        let inst = self.at(site);
        let Fields::Sop {
            dst:
                Some(Operand::Scalar {
                    code: c @ 0..=105,
                    dwords: 1,
                }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        let carry = inst.write(Reg::Scc)?;
        (inst.name() == name
            && reg == Reg::S(*c)
            && inst.writes == [(reg, value), (Reg::Scc, carry)])
        .then_some(Add {
            site,
            result: value,
            carry,
            src: src.as_slice().try_into().ok()?,
        })
    }

    fn sgpr(&self, site: Site, op: Operand) -> Option<(Reg, Value)> {
        let Operand::Scalar {
            code: c @ 0..=105,
            dwords: 1,
        } = op
        else {
            return None;
        };
        let r = Reg::S(c);
        Some((r, self.read_at(site, r)?))
    }

    fn add_pair(&self, words: [Value; 2], roles: &mut Roles) -> Option<[Add; 2]> {
        let lo = self.add(words[0], "s_add_u32")?;
        let hi = self.add(words[1], "s_addc_u32")?;
        if self.read_at(hi.site, Reg::Scc)? != lo.carry {
            return None;
        }
        roles
            .values
            .extend([lo.result, lo.carry, hi.result, hi.carry]);
        roles.reads.insert((hi.site, Reg::Scc, lo.carry));
        roles.carries.insert((hi.site, lo.carry));
        Some([lo, hi])
    }

    // Record precisely the base operand. The caller must certify the other
    // explicit operand separately, including on an ADDC that also reads SCC.
    fn base_operand(&self, add: &Add, name: &str, roles: &mut Roles) -> Option<(Value, Operand)> {
        [(0, 1), (1, 0)].into_iter().find_map(|(base, other)| {
            let (r, v) = self.sgpr(add.site, add.src[base])?;
            (self.inst(v)?.0.name() == name).then(|| {
                roles.reads.insert((add.site, r, v));
                (v, add.src[other])
            })
        })
    }

    /// The shifted register pair and GETPC must each come from a single
    /// instruction, with neither high/low substitution nor special registers.
    fn pair_result(&self, values: [Value; 2], name: &str) -> Option<Site> {
        let &(site, _) = self.insts.get(&values[0])?;
        let inst = self.at(site);
        let Fields::Sop {
            dst:
                Some(Operand::Scalar {
                    code: c @ 0..=104,
                    dwords: 2,
                }),
            ..
        } = inst.fields
        else {
            return None;
        };
        (inst.name() == name
            && inst.write(Reg::S(c)) == Some(values[0])
            && inst.write(Reg::S(c + 1)) == Some(values[1]))
        .then_some(site)
    }

    pub(super) fn paired(
        &self,
        sink: Site,
        code: u8,
        soffset: Operand,
        dynamic: Value,
    ) -> Option<Paired> {
        if code > 104
            || matches!(soffset, Operand::Scalar { code: c, .. } if c == code || c == code+1)
        {
            return None;
        }
        let mut roles = Roles::default();
        let words = [
            self.read_at(sink, Reg::S(code))?,
            self.read_at(sink, Reg::S(code + 1))?,
        ];
        roles.reads.extend([
            (sink, Reg::S(code), words[0]),
            (sink, Reg::S(code + 1), words[1]),
        ]);
        let [lo, hi] = self.add_pair(words, &mut roles)?;
        let (fixed_lo, shift_lo) = self.base_operand(&lo, "s_add_u32", &mut roles)?;
        let (fixed_hi, shift_hi) = self.base_operand(&hi, "s_addc_u32", &mut roles)?;
        let (_, shift_lo) = self.sgpr(lo.site, shift_lo)?;
        let (_, shift_hi) = self.sgpr(hi.site, shift_hi)?;
        if dynamic != shift_lo {
            return None;
        }
        let shift_site = self.pair_result([shift_lo, shift_hi], "s_lshl_b64")?;
        let Fields::Sop { src, .. } = &self.at(shift_site).fields else {
            return None;
        };
        let &[
            Operand::Scalar {
                code: c @ 0..=104,
                dwords: 2,
            },
            count,
        ] = src.as_slice()
        else {
            return None;
        };
        let index = self.read_at(shift_site, Reg::S(c))?;
        let high = self.read_at(shift_site, Reg::S(c + 1))?;
        let &(sign, _) = self.insts.get(&high)?;
        let sign_inst = self.at(sign);
        let Fields::Sop {
            dst: Some(Operand::Scalar { code: d, dwords: 1 }),
            src,
            ..
        } = &sign_inst.fields
        else {
            return None;
        };
        let &[value, bits] = src.as_slice() else {
            return None;
        };
        if sign_inst.name() != "s_ashr_i32"
            || *d != c + 1
            || self.sgpr(sign, value)?.1 != index
            || scalar_constant(bits)? & 31 != 31
        {
            return None;
        }
        let dynamic_low_site = lo.site;
        let [lo, hi] = self.add_pair([fixed_lo, fixed_hi], &mut roles)?;
        if !self.before(hi.site, dynamic_low_site) {
            return None;
        }
        let (pc_lo, low_adjustment) = self.base_operand(&lo, "s_getpc_b64", &mut roles)?;
        let (pc_hi, high_adjustment) = self.base_operand(&hi, "s_getpc_b64", &mut roles)?;
        let pc = self.at(self.pair_result([pc_lo, pc_hi], "s_getpc_b64")?);
        if scalar_constant(high_adjustment)? != 0 {
            return None;
        }
        let base = u32::try_from(pc.machine.offset)
            .ok()?
            .checked_add(4)?
            .checked_add(scalar_constant(low_adjustment)?)?;
        roles.values.extend([pc_lo, pc_hi]);
        Some(Paired {
            base,
            shift_site,
            count,
            index,
            roles,
        })
    }
}

/// The high input is the sign extension of this exact low input. Refuse
/// shifted-out bits and any result wider than the tracker's numeric term.
pub(super) fn displacement(index: u32, count: u32) -> Option<u32> {
    let input = index as i32 as i64 as u64;
    let shift = count & 63;
    if input > (u64::MAX >> shift) {
        return None;
    }
    u32::try_from(input << shift).ok()
}

impl Roles {
    pub(super) fn extend(&mut self, other: Self) {
        self.values.extend(other.values);
        self.reads.extend(other.reads);
        self.carries.extend(other.carries);
    }

    pub(super) fn valid(&self, defs: &Definitions, st: &Structured, res: &Resources) -> bool {
        let protected = |v: &Value| self.values.contains(v);
        let mut completed = Set::default();
        if !self
            .carries
            .iter()
            .all(|&(_, carry)| completed.insert(carry))
        {
            return false;
        }
        for (b, block) in defs.p.blocks.iter().enumerate() {
            if defs.rank[b] == usize::MAX {
                continue;
            }
            if block.phis.iter().flat_map(|p| &p.args).any(protected) {
                return false;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if inst
                    .reads
                    .iter()
                    .any(|&(r, v)| protected(&v) && !self.reads.contains(&((b, i), r, v)))
                {
                    return false;
                }
            }
        }
        if res
            .scalar
            .iter()
            .any(|(_, a)| a.dynamic.as_ref().is_some_and(protected))
            || res
                .descriptors
                .iter()
                .any(|d| self.descriptor_use(&d.source))
            || res.waterfall.reads.iter().any(protected)
            || st
                .restored
                .iter()
                .chain(&st.highs)
                .any(|(a, b)| protected(a) || protected(b))
            || st
                .masks
                .keys()
                .chain(st.other.keys())
                .chain(st.scc_votes.keys())
                .chain(&st.active)
                .any(protected)
            || st
                .mask_inputs
                .keys()
                .any(|(a, b)| protected(a) || protected(b))
        {
            return false;
        }
        let mut pending: Vec<_> = st
            .exits
            .iter()
            .filter_map(|e| match e {
                LaneExit::Branch { when, .. } => Some(when),
                _ => None,
            })
            .chain(st.guards.iter().flatten().flatten())
            .chain(&st.reach)
            .chain(st.discards.values())
            .chain(st.masks.values())
            .chain(st.other.values())
            .chain(st.mask_inputs.values().flatten())
            .chain(st.scc_votes.values())
            .copied()
            .collect();
        let mut seen = Set::default();
        while let Some(b) = pending.pop() {
            if !seen.insert(b) {
                continue;
            }
            if let Some((a, lo, hi)) = st.bools.node(b) {
                match a {
                    Atom::Any(inner) => pending.push(inner),
                    Atom::PairPhi(a, b) | Atom::OpaquePair(a, b)
                        if protected(&a) || protected(&b) =>
                    {
                        return false;
                    }
                    Atom::Lane(v)
                    | Atom::Uniform(v)
                    | Atom::Phi(v)
                    | Atom::Quad(v)
                    | Atom::Opaque(v)
                        if protected(&v) =>
                    {
                        return false;
                    }
                    _ => {}
                }
                pending.extend([lo, hi]);
            }
        }
        true
    }

    fn descriptor_use(&self, source: &Source) -> bool {
        let args = match source {
            Source::Built { address, .. } => {
                return address.dynamic.is_some_and(|v| self.values.contains(&v));
            }
            Source::Select { scc, .. } | Source::Lanes { mask: scc, .. }
                if self.values.contains(scc) =>
            {
                return true;
            }
            Source::Select { args, .. } | Source::Lanes { args, .. } | Source::Phi { args, .. } => {
                args
            }
            Source::LanePhi { args, .. } => {
                return args.iter().flatten().any(|s| self.descriptor_use(s));
            }
            _ => return false,
        };
        args.iter().any(|s| self.descriptor_use(s))
    }
}
