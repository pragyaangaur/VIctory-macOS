//! Publication assumes both halves enter the same instruction visit. Prove
//! this by induction from dispatch entry: each branch picks one common edge,
//! so every header counter sees the same resets, increments and cap exits.
//! Scalar equality is local to the visit; a load can change on every trip.
use super::*;

pub(super) struct Proof {
    pub step: Vec<bool>,
    pub common: Set<Value>,
    pub sites: Set<Site>,
    pub guest_barriers: Set<Site>,
}

fn memory_sources(x: &Parts) -> (Map<Site, Address>, Set<Site>) {
    let addresses: Map<_, _> = x.res.scalar.iter().copied().collect();
    let handles: Set<_> = x
        .res
        .descriptors
        .iter()
        .flat_map(|d| d.source.leaves())
        .filter_map(|s| match s {
            Source::Loaded { load, .. }
                if addresses
                    .get(load)
                    .and_then(|a| x.native.layout.get(a.param))
                    .is_some_and(|p| {
                        matches!(
                            p.root,
                            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
                        )
                    }) =>
            {
                Some(*load)
            }
            _ => None,
        })
        .collect();
    (addresses, handles)
}

/// A folded graph must have valid numeric consumers even when it needs no
/// scalar publication or descriptor handles.
pub(super) fn folded_roles(x: &Parts, carries: &Set<(Site, Value)>) -> Result<(), ShaderError> {
    let (addresses, handles) = memory_sources(x);
    let order = crate::ir::reverse_postorder(&x.p.blocks);
    symbolic_addresses(x, &order, &addresses, &handles, carries).map(drop)
}

pub(super) fn prove(
    x: &Parts,
    group: u32,
    lds: u32,
    bad: &HashMap<Value, Why>,
) -> Result<Option<Proof>, Stop> {
    let (addresses, handles) = memory_sources(x);
    let order = crate::ir::reverse_postorder(&x.p.blocks);
    let memory = order.iter().flat_map(|&b| {
        x.p.blocks[b]
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| matches!(inst.fields, Fields::Smem { .. }))
            .map(move |(i, inst)| ((b, i), inst))
    });
    let immutable = |site: &Site| {
        addresses
            .get(site)
            .is_some_and(|a| matches!(a.param, CODE | USER_DATA))
    };
    let code_buffers: Set<_> = x
        .res
        .descriptors
        .iter()
        .filter(|d| matches!(d.source, Source::Built { address, .. } if address.param == CODE))
        .map(|d| d.site)
        .collect();
    let facts = super::retry::immutable::prove(x.p, x.st, x.res, x.native);
    if !facts.carries.is_empty() {
        folded_code(x.p, x.res, x.native, &facts.reads).map_err(Stop::Fail)?;
    }
    if memory
        .clone()
        .all(|(s, _)| immutable(&s) || handles.contains(&s) || code_buffers.contains(&s))
    {
        // A descriptor role on one operand does not make every use of its
        // storage a handle. Check the legacy exit before granting that exemption.
        if x.host_wave32 && x.native.wave64 && (!handles.is_empty() || !facts.carries.is_empty()) {
            symbolic_addresses(x, &order, &addresses, &handles, &facts.carries)
                .map_err(Stop::Fail)?;
        }
        return Ok(None);
    }
    if !x.host_wave32
        || !x.native.wave64
        || group != 64
        || !halves::uses_lane_tickets(group, 32, lds, true)
    {
        return Err(unsupported(
            0,
            "literal scalar publication without one ticketed wave64 and zero guest LDS",
        ));
    }
    if x.res.waterfall != Default::default() {
        return Err(unsupported(
            0,
            "a resource waterfall in literal scalar publication",
        ));
    }
    // Table CBVs already use the runtime's raw SRV representation.
    let buffers: Map<_, _> = predicate::fixed_scalar_buffers(x, |kind| {
        matches!(kind, ResourceType::Srv | ResourceType::Cbv)
    })
    .filter_map(|(site, offset, load, first)| {
        let inst = &x.p.blocks[site.0].insts[site.1];
        let def = &x.p.blocks[load.0].insts[load.1];
        direct_descriptor(inst, def, first).then_some((site, offset))
    })
    .collect();
    let mut sites = Set::default();
    for (s @ (b, i), inst) in memory.clone() {
        let Fields::Smem { cache, .. } = inst.fields else {
            unreachable!()
        };
        if cache != Default::default() || x.st.guards[b][i].is_some() {
            return Err(unsupported(
                inst.machine.offset,
                "a cached or predicated literal scalar memory access",
            ));
        }
        if (handles.contains(&s) || immutable(&s)) && smem_dwords(inst.name()).is_some() {
            continue;
        }
        if !buffers.contains_key(&s)
            || !(matches!(
                inst.name(),
                "s_buffer_load_dword" | "s_buffer_load_dwordx2" | "s_buffer_load_dwordx4"
            ) || wide_buffer_read(inst))
        {
            return Err(unsupported(
                inst.machine.offset,
                "literal scalar publication from an unsupported source or width",
            ));
        }
        sites.insert(s);
    }

    let mut rank = vec![usize::MAX; x.p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        rank[b] = n;
    }
    let computed = &facts.reads;
    // Reject at the producer: ordinary vector uses and lane reads do not
    // require the original scalar to be in the common set.
    for &((b, i), address) in &x.res.scalar {
        if address.param == CODE && rank[b] != usize::MAX {
            let inst = &x.p.blocks[b].insts[i];
            if !computed.contains_key(&(b, i)) && !bounded_immutable_read(x.native, inst, address) {
                return Err(unsupported(
                    inst.machine.offset,
                    "a code read without a proved byte range in literal publication",
                ));
            }
        }
    }
    let idom = crate::ir::idoms(&x.p.blocks, &order, &rank);
    let mut definitions = Map::default();
    for &b in &order {
        let block = &x.p.blocks[b];
        definitions.extend(block.phis.iter().map(|phi| (phi.value, b)));
        definitions.extend(
            block
                .insts
                .iter()
                .flat_map(|inst| inst.writes.iter().map(|w| (w.1, b))),
        );
    }
    let dominates = |def: usize, mut pred: usize| {
        while rank[pred] > rank[def] {
            pred = idom[pred];
        }
        def == pred
    };
    let entry: Set<_> = x
        .res
        .params
        .iter()
        .filter_map(|&(v, p, d)| {
            let rp = x.native.layout.get(p)?;
            (d < rp.sgprs
                && matches!(
                    rp.root,
                    RootType::Table
                        | RootType::Constants
                        | RootType::ResourceHeap
                        | RootType::SamplerHeap
                ))
            .then_some(v)
        })
        .chain(
            x.io.inputs
                .iter()
                .filter_map(|&(v, input)| matches!(input, Input::GroupId(_)).then_some(v)),
        )
        .collect();
    let entry =
        x.p.inputs
            .iter()
            .filter_map(|&(reg, v)| (matches!(reg, Reg::S(_)) && entry.contains(&v)).then_some(v))
            .collect::<Set<_>>();
    // Only backedge phi operands can use a hypothesis. On the first trip
    // every forward operand must already have been produced in this pass.
    let mut previous: Option<Set<Value>> = None;
    let scratch = barriers::scalar_scratch(x.p, x.st);
    let mut common = loop {
        let mut known = entry.clone();
        for &b in &order {
            let block = &x.p.blocks[b];
            for phi in &block.phis {
                if !matches!(phi.reg, Reg::S(_) | Reg::Scc)
                    || bad.contains_key(&phi.value)
                    || b == 0
                    || phi.args.len() != block.preds.len()
                    || !block.preds.iter().any(|&p| rank[p] < rank[b])
                {
                    continue;
                }
                if block.preds.iter().zip(&phi.args).all(|(&p, v)| {
                    rank[p] == usize::MAX
                        || (definitions
                            .get(v)
                            .map_or_else(|| entry.contains(v), |&def| dominates(def, p))
                            && if rank[p] < rank[b] {
                                known.contains(v)
                            } else {
                                previous.as_ref().is_none_or(|set| set.contains(v))
                            })
                }) {
                    known.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let site = (b, i);
                let common = if sites.contains(&site) {
                    buffers[&site].is_none_or(|v| known.contains(&v))
                } else if immutable(&site) && matches!(inst.fields, Fields::Smem { .. }) {
                    inst.reads.iter().all(|&(_, v)| known.contains(&v))
                        && (computed.contains_key(&site)
                            || bounded_immutable_read(x.native, inst, addresses[&site]))
                } else {
                    match inst.name() {
                        "v_readfirstlane_b32" => {
                            matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none() && matches!(v.src.first(), Some(Operand::Vector { .. })))
                        } // literal lowering always ballots and exchanges
                        "v_readlane_b32" => {
                            halves::uniform_selector(inst, &known)
                                && inst
                                    .writes
                                    .first()
                                    .and_then(|w| x.st.restored.get(&w.1))
                                    .is_none_or(|v| known.contains(v))
                        }
                        _ => {
                            deterministic(inst, |v| scratch.contains(&v))
                                && inst.reads.iter().all(|&(_, v)| known.contains(&v))
                        }
                    }
                };
                if common {
                    known.extend(
                        inst.writes
                            .iter()
                            .filter_map(|&(r, v)| matches!(r, Reg::S(_) | Reg::Scc).then_some(v)),
                    );
                }
            }
        }
        if previous.as_ref() == Some(&known) {
            break known;
        }
        previous = Some(known);
    };
    for (site, inst) in memory {
        if (sites.contains(&site) && buffers[&site].is_some_and(|v| !common.contains(&v)))
            || (handles.contains(&site) && !inst.reads.iter().all(|&(_, v)| common.contains(&v)))
        {
            return Err(unsupported(
                inst.machine.offset,
                "a literal scalar memory address without common initialized operands",
            ));
        }
    }

    let symbolic =
        symbolic_addresses(x, &order, &addresses, &handles, &facts.carries).map_err(Stop::Fail)?;
    common.retain(|v| !symbolic.contains(v));
    let step = check_control(x, &order, &rank, &common)?;
    let guest_barriers = literal_lds::prove(x, &order, lds, &sites)?;
    Ok(Some(Proof {
        guest_barriers,
        step,
        common,
        sites,
    }))
}

/// A table/heap address and the PC have relative host representations. Equality
/// across halves suffices for checked resource addressing, but says nothing
/// about their guest bits. A tracked descriptor also has a handle but no
/// numeric storage words. Follow both through copies, ALU results and phis.
fn symbolic_addresses(
    x: &Parts,
    order: &[usize],
    addresses: &Map<Site, Address>,
    handles: &Set<Site>,
    carries: &Set<(Site, Value)>,
) -> Result<Set<Value>, ShaderError> {
    let mut symbolic: Set<_> = x
        .res
        .params
        .iter()
        .filter_map(|&(v, p, _)| {
            matches!(
                x.native.layout.get(p)?.root,
                RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap
            )
            .then_some(v)
        })
        .collect();
    for &(b, i) in handles {
        symbolic.extend(x.p.blocks[b].insts[i].writes.iter().map(|w| w.1));
    }
    loop {
        let before = symbolic.len();
        for &b in order {
            let block = &x.p.blocks[b];
            for phi in &block.phis {
                if phi.args.iter().any(|v| symbolic.contains(v)) {
                    symbolic.insert(phi.value);
                }
            }
            for inst in &block.insts {
                if inst.name() == "s_getpc_b64"
                    || matches!(inst.fields, Fields::Sop { .. } | Fields::Valu(_))
                        && inst.reads.iter().any(|(_, v)| symbolic.contains(v))
                {
                    symbolic.extend(inst.writes.iter().map(|w| w.1));
                }
            }
        }
        if before == symbolic.len() {
            break;
        }
    }

    const WHY: Why = "a relocated address or descriptor word read as numeric data without guest-exact bits in literal publication";
    let refusal = |offset| ShaderError::Unsupported { offset, what: WHY };
    let exact_term = |a: &Address| a.dynamic.is_none_or(|v| !symbolic.contains(&v));
    let mut roles = Map::<Site, Vec<Role>>::default();
    for d in &x.res.descriptors {
        let leaves = d.source.leaves();
        if leaves
            .iter()
            .any(|s| matches!(s, Source::Built { address, .. } if !exact_term(address)))
        {
            return Err(refusal(x.p.blocks[d.site.0].insts[d.site.1].machine.offset));
        }
        if !leaves
            .iter()
            .any(|s| matches!(s, Source::Built { address, .. } if address.param == MEMORY))
        {
            roles.entry(d.site).or_default().push(d.role);
        }
    }
    for &b in order {
        for (i, inst) in x.p.blocks[b].insts.iter().enumerate() {
            let site = (b, i);
            let address = addresses.get(&site);
            let relocated = |r: Reg| inst.read(r).is_some_and(|v| symbolic.contains(&v));
            if address.is_some_and(|a| !exact_term(a))
                || inst
                    .writes
                    .iter()
                    .any(|(_, v)| x.st.restored.get(v).is_some_and(|v| symbolic.contains(v)))
            {
                return Err(refusal(inst.machine.offset));
            }
            // A predicated memory result can keep the old VGPR in inactive lanes.
            // Never erase its symbolic provenance at an overlapping address/result.
            if inst.writes.iter().any(|&(r, v)| {
                matches!(r, Reg::S(106..)) && symbolic.contains(&v)
                    || matches!(r, Reg::V(_))
                        && !matches!(inst.fields, Fields::Valu(_))
                        && relocated(r)
            }) {
                return Err(refusal(inst.machine.offset));
            }
            let reads = inst.reads.iter().filter(|(_, v)| symbolic.contains(v));
            if reads.clone().next().is_none() {
                continue;
            }
            let descriptor = |role| roles.get(&site).is_some_and(|r| r.contains(&role));
            let contains = |op: Operand, r: Reg| match (op, r) {
                (Operand::Scalar { code, dwords }, Reg::S(n))
                | (Operand::Vector { reg: code, dwords }, Reg::V(n)) => {
                    n.checked_sub(code).is_some_and(|k| k < dwords)
                }
                _ => false,
            };
            let exact = |op: Operand| !reads.clone().any(|&(r, _)| contains(op, r));
            let base = address.is_some_and(|a| a.param != MEMORY);
            let valid = match &inst.fields {
                // The resource tracker must independently recognize the result as
                // an address. A carry or a selector that depends on guest address
                // bits is itself a numeric input, including for address formation.
                Fields::Sop { .. } | Fields::Valu(_) => {
                    (deterministic(inst, |_| false)
                        || matches!(
                            inst.name(),
                            "s_bitset1_b32"
                                | "s_cmov_b32"
                                | "s_cmov_b64"
                                | "s_lshr_b64"
                                | "s_bfe_u32"
                                | "s_max_u32"
                                | "v_mov_b32"
                                | "v_add_nc_u32"
                                | "v_readfirstlane_b32"
                        ))
                        && (!relocated(Reg::Scc)
                            || inst
                                .read(Reg::Scc)
                                .is_some_and(|v| carries.contains(&(site, v))))
                }
                Fields::Smem {
                    base: op, soffset, ..
                } => {
                    inst.name().contains("_load_")
                        && (base || descriptor(Role::Resource))
                        && exact(*soffset)
                        && reads.clone().all(|&(r, _)| contains(*op, r))
                }
                Fields::Buffer { rsrc, soffset, .. } => {
                    descriptor(Role::Resource)
                        && exact(*soffset)
                        && reads.clone().all(|&(r, _)| contains(*rsrc, r))
                }
                Fields::Image { rsrc, sampler, .. } => reads.clone().all(|&(r, _)| {
                    descriptor(Role::Resource) && contains(*rsrc, r)
                        || descriptor(Role::Sampler) && sampler.is_some_and(|s| contains(s, r))
                }),
                Fields::Flat {
                    addr, saddr, data, ..
                } if base => {
                    // With an SGPR base, the VGPR is a numeric offset.
                    let op = saddr
                        .filter(|s| !matches!(s, Operand::Scalar { code: NULL, .. }))
                        .or(*addr);
                    data.is_none_or(exact)
                        && reads
                            .clone()
                            .all(|&(r, _)| op.is_some_and(|o| contains(o, r)))
                }
                _ => false,
            };
            if !valid {
                return Err(refusal(inst.machine.offset));
            }
        }
    }
    Ok(symbolic)
}

fn check_control(
    x: &Parts,
    order: &[usize],
    rank: &[usize],
    common: &Set<Value>,
) -> Result<Vec<bool>, Stop> {
    // Check *every* branch, including reconvergent ones. A common PC then
    // implies a common predecessor on every visit, including all cap edges.
    let mut step = vec![false; rank.len()];
    let mut masks = Vec::new();
    for &b in order {
        let block = &x.p.blocks[b];
        // Lower::enter_loop resets on each forward edge and count_iteration
        // advances the selected header. A counter at dispatch entry would
        // have no incoming edge supplying its initial zero.
        if (b == 0 && block.preds.iter().any(|&q| rank[q] != usize::MAX))
            || block
                .phis
                .iter()
                .any(|phi| phi.args.len() != block.preds.len())
        {
            return Err(unsupported(
                block.offset,
                "a literal publication block without an initialized entry",
            ));
        }
        masks.extend(x.st.guards[b].iter().flatten().copied());
        let valid = match x.st.exits[b] {
            LaneExit::End | LaneExit::Jump(_) => true,
            LaneExit::Indirect => false,
            LaneExit::Branch { when, .. } => {
                x.st.bools.support(when).iter().all(|atom| match atom {
                    Atom::Uniform(v) => common.contains(v),
                    // The nested lane predicate and ballot insertion point must
                    // also succeed in Lower::atom with this exact common set.
                    Atom::Any(bit) => {
                        masks.push(*bit);
                        true
                    }
                    _ => false,
                })
            }
        };
        if !valid {
            return Err(unsupported(
                block
                    .insts
                    .last()
                    .map_or(block.offset, |i| i.machine.offset),
                "a literal scalar publication branch without common group control flow",
            ));
        }
        step[b] = true;
    }
    // Validate the lane predicates inside actual votes and EXEC guards.
    // Mask joins can be cyclic. The SSA builder orders their incoming bits
    // by the same predecessors as the ordinary phis.
    let mut checked = Set::default();
    while let Some(mask) = masks.pop() {
        if !checked.insert(mask) {
            continue;
        }
        for atom in x.st.bools.support(mask) {
            let valid = match atom {
                Atom::Lane(_) | Atom::Constant(_) => true,
                Atom::Any(bit) => {
                    masks.push(bit);
                    true
                }
                Atom::Uniform(v) => common.contains(&v),
                Atom::OpaquePair(lo, hi) => common.contains(&lo) && common.contains(&hi),
                Atom::PairPhi(lo, hi) => x.st.mask_inputs.get(&(lo, hi)).is_some_and(|args| {
                    masks.extend(args);
                    !args.is_empty()
                }),
                _ => false,
            };
            if !valid {
                return Err(unsupported(
                    0,
                    "an unproved literal scalar publication mask",
                ));
            }
        }
    }
    Ok(step)
}

/// A direct table handle has no added base displacement. A nonnegative signed
/// 21-bit immediate plus even the last x16 byte fits in u32. Each four-dword
/// raw load independently bounds its dwords against the same view as the SMEM.
fn wide_buffer_read(inst: &Inst) -> bool {
    let Fields::Smem {
        data: Some(Operand::Scalar { code, dwords }),
        base: Operand::Scalar {
            code: base,
            dwords: 4,
        },
        soffset: Operand::Scalar {
            code: NULL,
            dwords: 1,
        },
        offset,
        ..
    } = inst.fields
    else {
        return false;
    };
    matches!(
        (inst.name(), dwords),
        ("s_buffer_load_dwordx8", 8) | ("s_buffer_load_dwordx16", 16)
    ) && (0..(1 << 20)).contains(&offset)
        && offset % 4 == 0
        && base.is_multiple_of(4)
        && base <= VCC - 4
        && code.is_multiple_of(4)
        && code <= VCC - dwords
        && inst.writes.len() == usize::from(dwords)
        && (0..dwords).all(|k| inst.write(Reg::S(code + k)).is_some())
}

/// Direct SSA dwords of the exact tracked resource entry.
fn direct_descriptor(inst: &Inst, def: &Inst, first: u8) -> bool {
    let Fields::Smem {
        base: Operand::Scalar { code, dwords: 4 },
        ..
    } = inst.fields
    else {
        return false;
    };
    let Fields::Smem {
        data: Some(Operand::Scalar { code: dst, dwords }),
        ..
    } = def.fields
    else {
        return false;
    };
    u16::from(first) + 4 <= u16::from(dwords)
        && (0..4).all(|k| {
            def.write(Reg::S(dst + first + k))
                .is_some_and(|v| inst.read(Reg::S(code + k)) == Some(v))
        })
}

/// A fold admits a graph the old literal path could refuse. Check every
/// remaining CODE producer before either publication plan can lower it.
pub(super) fn folded_code(
    p: &Program,
    res: &Resources,
    native: &NativeProgram,
    computed: &Map<Site, Set<u32>>,
) -> Result<(), ShaderError> {
    let refusal = |inst: &Inst| ShaderError::Unsupported {
        offset: inst.machine.offset,
        what: "a code read without a proved byte range after literal edge folding",
    };
    for &(site @ (b, i), address) in &res.scalar {
        let inst = &p.blocks[b].insts[i];
        if address.param == CODE
            && !computed.contains_key(&site)
            && !bounded_immutable_read(native, inst, address)
        {
            return Err(refusal(inst));
        }
    }
    for d in &res.descriptors {
        if d.source
            .leaves()
            .iter()
            .any(|s| matches!(s, Source::Built { address, .. } if address.param == CODE))
        {
            let inst = &p.blocks[d.site.0].insts[d.site.1];
            if !bounded_code_buffer(native, inst, &d.source) {
                return Err(refusal(inst));
            }
        }
    }
    Ok(())
}

fn bounded_code_buffer(native: &NativeProgram, inst: &Inst, source: &Source) -> bool {
    let Source::Built { address, .. } = source else {
        return false;
    };
    let Fields::Smem {
        offset,
        soffset: Operand::Scalar { code: NULL, .. },
        data: Some(Operand::Scalar { dwords, .. }),
        ..
    } = inst.fields
    else {
        return false;
    };
    // A fixed descriptor base is present in the emitted table. A positive
    // displacement stays above it, including when the table starts mid-code.
    inst.name().starts_with("s_buffer_load_dword")
        && dwords > 0
        && address.dynamic.is_none()
        && u32::try_from(offset)
            .ok()
            .and_then(|d| address.offset.checked_add(d))
            .is_some_and(|start| {
                start.is_multiple_of(4)
                    && start
                        .checked_add(4 * u32::from(dwords))
                        .is_some_and(|end| end as usize <= native.code.len())
            })
}

fn bounded_immutable_read(native: &NativeProgram, inst: &Inst, a: Address) -> bool {
    let bytes = match (inst.name(), &inst.fields) {
        ("global_load_ubyte" | "global_load_sbyte", _) => 1,
        ("global_load_ushort" | "global_load_sshort", _) => 2,
        (
            name,
            Fields::Flat {
                dst: Some(Operand::Vector { dwords, .. }),
                ..
            },
        ) if name.starts_with("global_load_dword") => 4 * u32::from(*dwords),
        (name, _) => smem_dwords(name).unwrap_or(0) * 4,
    };
    // code_dword rounds down to a word. Subword extraction can read any
    // byte in that word, but cannot span two words.
    let aligned = if bytes < 4 {
        a.offset % 4 + bytes <= 4
    } else {
        a.offset.is_multiple_of(4)
    };
    bytes > 0
        && a.dynamic.is_none()
        && aligned
        && match a.param {
            CODE => a
                .offset
                .checked_add(bytes)
                .is_some_and(|end| end as usize <= native.code.len()),
            USER_DATA if matches!(inst.fields, Fields::Smem { .. }) => {
                (a.offset / 4..a.offset / 4 + bytes / 4).all(|d| {
                    native.layout.iter().any(|p| {
                        matches!(p.root, RootType::Table | RootType::Constants)
                            && d.checked_sub(p.user_data.into())
                                .is_some_and(|n| n < p.sgprs)
                    })
                })
            }
            _ => false,
        }
}

/// Only instructions with implemented deterministic scalar transfers. Special
/// operands (e.g. a hardware register) are not immediates or initialized SGPRs.
fn deterministic(inst: &Inst, scalar_scratch: impl Fn(Value) -> bool) -> bool {
    let Fields::Sop { src, .. } = &inst.fields else {
        return false;
    };
    src.iter().all(|&op| match op {
        Operand::Scalar { code, dwords } => {
            code < VCC || code == NULL
                || (dwords == 1 && (code == VCC || code == VCC + 1)
                    && inst.read(Reg::S(code)).is_some_and(&scalar_scratch))
        }
        _ => scalar_constant(op).is_some(),
    }) && matches!(
        inst.name(),
        "s_mov_b32"
            | "s_mov_b64"
            | "s_movk_i32"
            | "s_getpc_b64"
            | "s_add_i32"
            | "s_add_u32"
            | "s_addc_u32"
            | "s_addk_i32"
            | "s_sub_i32"
            | "s_sub_u32"
            | "s_mul_i32"
            | "s_lshl_b32"
            | "s_lshl_b64"
            | "s_lshr_b32"
            | "s_ashr_i32"
            | "s_and_b32"
            | "s_and_b64"
            | "s_andn2_b64"
            | "s_or_b32"
            | "s_or_b64"
            | "s_brev_b32"
            | "s_cselect_b32"
            | "s_cselect_b64"
            | "s_cmp_eq_u32"
            | "s_cmp_ge_u32"
            | "s_cmp_gt_i32"
            | "s_cmp_gt_u32"
            | "s_cmp_le_u32"
            | "s_cmp_lg_u32"
            | "s_cmp_lt_u32"
    )
}

#[cfg(test)]
mod tests;
