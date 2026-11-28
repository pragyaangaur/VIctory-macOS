//! Integer equality waterfalls that perform one independent register move per lane.

use super::{Tracer, Waterfall};
use crate::ShaderError;
use crate::gfx10::{Fields, Operand, Valu};
use crate::ir::{EXEC, Inst, M0, Reg, VCC};

impl Tracer<'_> {
    pub(super) fn indexed_waterfalls(&self, out: &mut Waterfall) -> Result<(), ShaderError> {
        let start = out.reads.len();
        let mut escapes = None;
        for (b, block) in self.p.blocks.iter().enumerate() {
            let Some(proof) = self.index_waterfall(b) else {
                continue;
            };
            // Only EXEC may return to this header through its own backedge.
            // Every other scalar result (including both SCC writes) is private.
            let external = escapes.get_or_insert_with(|| self.escaping_scalar_values());
            if block
                .insts
                .iter()
                .flat_map(|i| &i.writes)
                .any(|&(r, v)| !matches!(r, Reg::V(_)) && external[v.0 as usize])
            {
                continue;
            }
            out.reads.extend(proof.reads);
            out.compares.extend(proof.compares);
        }
        // A newly supported waterfall can expose a separate relative move to lowering.
        if out.reads.len() != start
            && let Some(inst) = self
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| !i.relative_in_file(self.p.vgprs))
        {
            return Err(ShaderError::Unsupported {
                offset: inst.machine.offset,
                what: "an indexed-waterfall program with a relative VGPR operand not proved inside its register file",
            });
        }
        Ok(())
    }

    fn index_waterfall(&self, b: usize) -> Option<Waterfall> {
        let [first, cmp, select, copy, access, retry, _branch] = &self.p.blocks[b].insts[..] else {
            return None;
        };
        if first.name() != "v_readfirstlane_b32"
            || cmp.name() != "v_cmp_eq_u32"
            || select.name() != "s_and_saveexec_b64"
            || copy.name() != "s_mov_b32"
            || retry.name() != "s_xor_b64"
        {
            return None;
        }
        let (index, key) = plain_unary(first)?;
        let index_reg = vector(index)?;
        let key_reg = scalar(key, 1)?;
        let value = first.write(Reg::S(key_reg))?;
        let compare = plain(cmp)?;
        let mask = compare.dst?;
        let (saved, mask_src) = sop(select)?;
        let exec = Operand::Scalar {
            code: EXEC,
            dwords: 2,
        };
        let m0 = Operand::Scalar {
            code: M0,
            dwords: 1,
        };
        if key_reg >= VCC
            || scalar(mask, 2)? > VCC
            || scalar(saved, 2)? > VCC
            || compare.src != [key, index]
            || cmp.read(Reg::S(key_reg))? != value
            || cmp.read(Reg::V(index_reg))? != first.read(Reg::V(index_reg))?
            || mask_src != [mask]
            || sop(copy)? != (m0, &[key][..])
            || copy.read(Reg::S(key_reg))? != value
            || sop(retry)? != (exec, &[exec, saved][..])
            || access.read(Reg::S(M0))? != copy.write(Reg::S(M0))?
            || !self.stable_move(access, index_reg)
        {
            return None;
        }
        // Exact select/retry makes each lane run once, at its own unchanged index.
        // The only effect is on that lane's VGPRs, so visit order is unobservable.
        self.waterfall(b, &[4], &[value]).ok()
    }

    fn stable_move(&self, inst: &Inst, index: u8) -> bool {
        let Some((src, dst)) = plain_unary(inst).and_then(|(s, d)| Some((vector(s)?, vector(d)?)))
        else {
            return false;
        };
        matches!(inst.name(), "v_movrels_b32" | "v_movreld_b32")
            && [src, dst, index]
                .iter()
                .all(|&r| u16::from(r) < self.p.vgprs)
            && inst.write(Reg::V(index)).is_none()
    }

    fn escaping_scalar_values(&self) -> Vec<bool> {
        let mut external = vec![false; self.defs.len()];
        for (b, block) in self.p.blocks.iter().enumerate() {
            for &(reg, v) in block.insts.iter().flat_map(|i| &i.reads) {
                if !matches!(reg, Reg::V(_)) && self.defs[v.0 as usize].is_some_and(|(d, _)| d != b)
                {
                    external[v.0 as usize] = true;
                }
            }
            for phi in &block.phis {
                if matches!(phi.reg, Reg::V(_)) {
                    continue;
                }
                for (&v, &pred) in phi.args.iter().zip(&block.preds) {
                    if !(pred == b
                        && matches!(phi.reg, Reg::S(126 | 127))
                        && self.defs[v.0 as usize].is_some_and(|(d, _)| d == b))
                    {
                        external[v.0 as usize] = true;
                    }
                }
            }
        }
        external
    }
}

fn scalar(op: Operand, width: u8) -> Option<u8> {
    match op {
        Operand::Scalar { code, dwords } if dwords == width => Some(code),
        _ => None,
    }
}

fn vector(op: Operand) -> Option<u8> {
    match op {
        Operand::Vector { reg, dwords: 1 } => Some(reg),
        _ => None,
    }
}

fn sop(inst: &Inst) -> Option<(Operand, &[Operand])> {
    let Fields::Sop {
        dst: Some(dst),
        src,
        ..
    } = &inst.fields
    else {
        return None;
    };
    Some((*dst, src))
}

fn plain(inst: &Inst) -> Option<&Valu> {
    let Fields::Valu(f) = &inst.fields else {
        return None;
    };
    (f.abs == 0
        && f.neg == 0
        && f.neg_hi == 0
        && f.opsel == 0
        && f.opsel_hi == 0
        && !f.clamp
        && f.omod == 0
        && f.sdwa.is_none()
        && f.dpp.is_none()
        && f.carry.is_none())
    .then_some(f)
}

fn plain_unary(inst: &Inst) -> Option<(Operand, Operand)> {
    let f = plain(inst)?;
    let [src] = f.src[..] else { return None };
    Some((src, f.dst?))
}
