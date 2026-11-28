//! Paired LDS phases in one ticketed guest wave. The caller has already
//! proved all reachable branch decisions and cap edges common to both halves.
use super::*;

/// Only unguarded definitions in the entry block establish lane addresses.
/// A phi or an instruction that preserves an old VGPR never supplies a fact.
fn entry_value(inst: &Inst, facts: &Map<Value, [u32; 64]>) -> Option<(Value, [u32; 64])> {
    use crate::gfx10::Encoding;
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    if !matches!(inst.machine.encoding, Encoding::Vop1 | Encoding::Vop2)
        || v.sdwa.is_some()
        || v.dpp.is_some()
    {
        return None;
    }
    let operand = |op: &Operand| match *op {
        Operand::Const(c) => Some([crate::resource::inline(c)?; 64]),
        Operand::Literal(k) => Some([k; 64]),
        Operand::Vector { reg, dwords: 1 } => facts.get(&inst.read(Reg::V(reg))?).copied(),
        _ => None,
    };
    let a = operand(v.src.first()?)?;
    let values = match (inst.name(), v.src.as_slice()) {
        ("v_mov_b32", [_]) => a,
        (op, [_, b]) => {
            let b = operand(b)?;
            match op {
                "v_add_nc_u32" => std::array::from_fn(|l| a[l].wrapping_add(b[l])),
                "v_lshlrev_b32" => std::array::from_fn(|l| b[l].wrapping_shl(a[l])),
                "v_xor_b32" => std::array::from_fn(|l| b[l] ^ a[l]),
                _ => return None,
            }
        }
        _ => return None,
    };
    let [(Reg::V(_), value)] = inst.writes.as_slice() else {
        return None;
    };
    Some((*value, values))
}

pub(super) fn group_size(threads: [u32; 3]) -> Result<u32, Stop> {
    threads
        .into_iter()
        .try_fold(1, u32::checked_mul)
        .ok_or_else(|| unsupported(0, "overflow in a literal LDS group size"))
}

pub(super) fn prove(
    x: &Parts,
    order: &[usize],
    lds: u32,
    scalar: &Set<Site>,
) -> Result<Set<Site>, Stop> {
    let mut phases = Set::default();
    if lds == 0 {
        return Ok(phases);
    }
    let StageIo::Cs(cs) = x.native.io else {
        unreachable!()
    };
    if group_size(cs.threads)? != 64 {
        return Err(unsupported(
            0,
            "paired literal LDS outside one complete wave64",
        ));
    }
    // Keep declaration-only programs on their original refusal path.
    if !order.iter().any(|&b| {
        x.p.blocks[b]
            .insts
            .iter()
            .any(|i| matches!(i.fields, Fields::Ds { .. }))
    }) {
        return Err(unsupported(
            0,
            "literal scalar publication without one ticketed wave64 and zero guest LDS",
        ));
    }
    if scalar.iter().any(|&(b, i)| {
        !matches!(
            x.p.blocks[b].insts[i].name(),
            "s_buffer_load_dword" | "s_buffer_load_dwordx2" | "s_buffer_load_dwordx4"
        )
    }) {
        return Err(unsupported(0, "a wide scalar publication with guest LDS"));
    }
    if cs.thread_id_vgprs() > 3 {
        return Err(unsupported(
            0,
            "a reserved thread ID layout in paired literal LDS",
        ));
    }
    let mut facts = Map::default();
    for &(v, input) in &x.io.inputs {
        if let Input::ThreadIdInGroup(c) = input {
            let [sx, sy, _] = cs.threads;
            let lanes = std::array::from_fn(|l| match c {
                0 => l as u32 % sx,
                1 => (l as u32 / sx) % sy,
                _ => l as u32 / (sx * sy),
            });
            facts.insert(v, lanes);
        }
    }
    let mut initialized = Set::default();
    for &b in order {
        for (i, inst) in x.p.blocks[b].insts.iter().enumerate() {
            let fail = |why| unsupported(inst.machine.offset, why);
            let unguarded_entry = b == 0 && x.st.guards[b][i].is_none();
            let name = inst.name();
            let safe = |op: &Operand| *op != Operand::Const(254);
            let Fields::Ds {
                addr: Some(Operand::Vector { reg, dwords: 1 }),
                offset0,
                offset1,
                gds: false,
                ..
            } = inst.fields
            else {
                // Enumerate the non-LDS operations allowed alongside these phases.
                let allowed = match &inst.fields {
                    Fields::Sop { src, .. } => {
                        src.iter().all(safe)
                            && matches!(
                                name,
                                "s_mov_b32"
                                    | "s_mov_b64"
                                    | "s_add_u32"
                                    | "s_sub_u32"
                                    | "s_cmp_lg_u32"
                            )
                    }
                    Fields::Sopp { .. } => matches!(
                        name,
                        "s_waitcnt" | "s_cbranch_scc1" | "s_endpgm" | "s_branch" | "s_nop"
                    ),
                    Fields::Valu(v) => {
                        v.src.iter().all(safe)
                            && matches!(
                                name,
                                "v_mov_b32" | "v_add_nc_u32" | "v_lshlrev_b32" | "v_xor_b32"
                            )
                    }
                    Fields::Smem { soffset, .. } => {
                        safe(soffset)
                            && matches!(
                                name,
                                "s_load_dwordx4"
                                    | "s_buffer_load_dword"
                                    | "s_buffer_load_dwordx2"
                                    | "s_buffer_load_dwordx4"
                            )
                    }
                    Fields::Buffer {
                        lds: false,
                        soffset,
                        ..
                    } => safe(soffset) && name == "buffer_store_dword",
                    _ => false,
                };
                if !allowed {
                    return Err(fail("an unsupported operation in paired literal LDS"));
                }
                if unguarded_entry && let Some((v, lanes)) = entry_value(inst, &facts) {
                    facts.insert(v, lanes);
                }
                continue;
            };
            let write = match name {
                "ds_write2_b32" => true,
                "ds_read2_b32" => false,
                _ => return Err(fail("an unsupported operation in paired literal LDS")),
            };
            let bases = inst
                .read(Reg::V(reg))
                .and_then(|v| facts.get(&v))
                .ok_or_else(|| fail("a paired LDS address without an exact entry value"))?;
            let mut words = Set::default();
            for &base in bases {
                for offset in [offset0, offset1] {
                    let at = base.wrapping_add(4 * u32::from(offset));
                    if at % 4 != 0 || at / 4 >= lds {
                        return Err(fail("an unaligned or out-of-range paired LDS address"));
                    }
                    if write && !words.insert(at) {
                        return Err(fail("overlapping writes within one paired LDS instruction"));
                    }
                    if !write && !initialized.contains(&at) {
                        return Err(fail(
                            "a paired LDS read without a dominating initialization",
                        ));
                    }
                }
            }
            if write && unguarded_entry {
                initialized.extend(words);
            }
            // A fence before each dynamic phase also separates the last visit
            // from the next. Lower::run emits it before either guarded store.
            phases.insert((b, i));
        }
    }
    Ok(phases)
}

#[cfg(test)]
mod tests;
