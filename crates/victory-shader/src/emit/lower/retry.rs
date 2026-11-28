//! The lowering's entry: a program in the per-lane form, and a 64-thread
//! console wave64 on a 32-lane host that form refuses at a cross-half gate
//! again with whole-wave control flow (#435).

use super::*;

pub(super) mod immutable;

#[cfg(any(test, all(windows, feature = "host")))]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_pruned(parts, None, false, false)
}

#[cfg(any(test, all(windows, feature = "host")))]
pub(crate) fn lower_pruned(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    literal_halves: bool,
) -> Result<Lowered, ShaderError> {
    lower_limited(parts, keep, per_vertex_center, literal_halves, false)
}

/// [`lower_pruned`]; `limits`: a compute program that can leave the lanes
/// past a dispatch's count of threads does so at once
/// ([`Lower::leave_past_the_count`]; never in literal halves).
pub(crate) fn lower_limited(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    literal_halves: bool,
    limits: bool,
) -> Result<Lowered, ShaderError> {
    // Retry refusals from the legacy cross-half gates with whole-wave control flow.
    let original = lower_with_structure(parts, keep, per_vertex_center, limits);
    if !literal_halves
        || !parts.host_wave32
        || !parts.native.wave64
        || !matches!(&parts.native.io, StageIo::Cs(cs) if cs.threads.iter().product::<u32>() == 64)
        || !matches!(
            &original,
            Err(ShaderError::Unsupported {
                what: halves::MASK_ACROSS
                    | LANE_MASK
                    | MASK_CYCLE
                    | MASK_ELSEWHERE
                    | halves::LDS_ACROSS
                    | halves::REDUCTION_ACROSS
                    | halves::FIRST_ACROSS
                    | halves::READ_ACROSS
                    | halves::WRITE_ACROSS,
                ..
            })
        )
    {
        return original;
    }
    lower_literal(parts, keep, per_vertex_center)
}

/// The whole-wave translation of a 64-thread wave64 compute program on a
/// 32-lane host, tried when the per-lane halves translation refuses it.
pub(crate) fn lower_literal(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
) -> Result<Lowered, ShaderError> {
    let native = parts.native;
    // Register-relative move bounds and resource paths must use the same
    // control flow as lowering. The per-lane move bounds do not apply here.
    let mut p = crate::ir::build(&native.patched_code(), true, native.vgprs)?;
    let mut simplified = false;
    let (st, res, carries) = loop {
        let st = crate::structure::split_literal(&p)?;
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let facts = immutable::prove(&p, &st, &res, native);
        if !facts.jumps.is_empty()
            && let Some(folded) = crate::ir::folded(&p, &facts.jumps)?
        {
            p = folded;
            simplified = true;
            continue;
        }
        if simplified || !facts.carries.is_empty() {
            literal_publish::folded_code(&p, &res, native, &facts.reads)?;
        }
        break (st, res, facts.carries);
    };
    if !res.waterfall.reads.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a resource waterfall in literal wave64 halves",
        });
    }
    let io = crate::io::raise(&p, native)?;
    let x = Parts {
        p: &p,
        st: &st,
        res: &res,
        io: &io,
        ..parts.clone()
    };
    if simplified || !carries.is_empty() {
        literal_publish::folded_roles(&x, &carries)?;
    }
    // A group in literal halves takes one path and cannot drop lanes
    // past a dispatch's thread count.
    lower_with_structure(&x, keep, per_vertex_center, false)
}

fn lower_with_structure(
    parts: &Parts,
    keep: Option<&KeptOutputs>,
    per_vertex_center: bool,
    limits: bool,
) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    // Divergent sites where the halves meet (`lower::meet`): a pass that
    // finds new ones goes again with them laid out.
    let mut plan: Set<MeetSite> = Set::default();
    loop {
        let found = std::cell::RefCell::new(Set::default());
        let r =
            Lower::new(parts, &bad, keep, per_vertex_center, &plan, &found).and_then(|l| l.run(limits));
        let found = found.into_inner();
        if !found.is_empty() {
            plan.extend(found);
            continue;
        }
        match r {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests;
