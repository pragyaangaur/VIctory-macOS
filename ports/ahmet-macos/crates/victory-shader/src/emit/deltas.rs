//! Non-provoking vertex values on the host (pragyaangaur/VIctory-macOS#203, PR #300).
//!
//! The console keeps each interpolated attribute of a triangle as its value at
//! the provoking vertex `P0` and the differences `P10 = P1 - P0` and
//! `P20 = P2 - P0`, and a pixel program can read any of the three
//! (`v_interp_mov_f32`) or weigh them (`v_interp_p1`/`p2`). On D3DMetal 4.0b3,
//! `dx.op.attributeAtVertex` produces a no-op pipeline. For vertex+pixel
//! triangle pipelines, the pixel shader reads `P10` and `P20` as flat inputs
//! ([`VERTEX_DELTAS`], indexed by the attribute's row) placed from
//! [`VERTEX_DELTA_TOP`] downward, and a pass-through geometry shader
//! ([`vertex_delta_gs`]) computes `v1 - v0` and `v2 - v0` per triangle from the
//! vertex shader's outputs.

use super::bitcode::{Attr, Bin, Term};
use super::container::SigElement;
use super::shader::{Geometry, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The semantics of P10 and P20 inputs; the semantic index is the attribute's row.
pub const VERTEX_DELTAS: [&str; 2] = ["VICTORY_P10", "VICTORY_P20"];

/// The row of the first such input; subsequent delta inputs go downward from here.
pub const VERTEX_DELTA_TOP: u32 = 31;

/// D3D12 limit on total scalar components emitted per geometry shader invocation.
const MAX_SCALARS: u32 = 1024;

/// The rows of a container's signature part that carry a register.
fn elements(
    container: &[u8],
    part: FourCC,
    what: &'static str,
) -> Result<Vec<SigElement>, ShaderError> {
    let c = Container::parse(container)?;
    let part = c.part(part).ok_or(ShaderError::Malformed(what))?;
    let mut elements = super::signature(part.data)?;
    elements.retain(|e| e.row.is_some());
    Ok(elements)
}

/// Whether the pixel shader container `ps` declares `VICTORY_P10` or
/// `VICTORY_P20` inputs and therefore needs [`vertex_delta_gs`] before it.
pub fn needs_vertex_deltas(ps: &[u8]) -> bool {
    elements(ps, FourCC::ISG1, "").is_ok_and(|inputs| {
        inputs
            .iter()
            .any(|e| VERTEX_DELTAS.contains(&e.name.as_str()))
    })
}

/// Builds the pass-through geometry shader that supplies `ps`'s `VICTORY_P10`
/// and `VICTORY_P20` inputs from `vs`'s triangle vertex outputs.
pub fn vertex_delta_gs(vs: &[u8], ps: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let mut outputs = elements(vs, FourCC::OSG1, "a vertex shader without outputs")?;
    for e in &mut outputs {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1 << e.cols) - 1;
    }
    let refuse = |what| ShaderError::Unsupported { offset: 0, what };
    // Each delta input paired with the vertex output (element index, first column)
    // and non-provoking corner (1 or 2) it is computed from.
    let mut deltas = Vec::new();
    for d in elements(ps, FourCC::ISG1, "a pixel shader without inputs")? {
        let Some(vertex) = VERTEX_DELTAS.iter().position(|&n| n == d.name) else {
            continue;
        };
        let row = d.row.expect("kept with a row");
        if outputs.iter().any(|o| o.row == Some(row)) {
            return Err(refuse(
                "vertex outputs in the rows of non-provoking vertex values",
            ));
        }
        let from = outputs
            .iter()
            .position(|o| {
                o.row == Some(d.index) && o.col <= d.col && d.col + d.cols <= o.col + o.cols
            })
            .ok_or(refuse("a non-provoking vertex value of no vertex output"))?;
        if outputs[from].comp != 9 {
            return Err(refuse("a non-provoking vertex value of an integer output"));
        }
        let first = d.col - outputs[from].col;
        deltas.push((
            SigElement {
                interp: 1,
                usage: (1 << d.cols) - 1,
                ..d
            },
            vertex as u32 + 1,
            from,
            first,
        ));
    }
    if deltas.is_empty() {
        return Err(ShaderError::Malformed(
            "a pixel shader that reads no non-provoking vertex value",
        ));
    }
    let scalars: u32 = outputs
        .iter()
        .map(|e| u32::from(e.cols))
        .chain(deltas.iter().map(|(d, ..)| u32::from(d.cols)))
        .sum();
    if scalars * 3 > MAX_SCALARS {
        return Err(refuse(
            "vertex delta geometry shader exceeds 1024 output scalars",
        ));
    }
    let mut s = Shader::new(false);
    s.geometry = Some(Geometry { max_vertices: 3 });
    let b = s.entry;
    let load = |s: &mut Shader, id: usize, e: &SigElement, col: u8, k: u32| {
        let (ty, suffix) = if e.comp == 9 {
            (s.t.f32, ".f32")
        } else {
            (s.t.i32, ".i32")
        };
        let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
        s.op(
            b,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        )
    };
    // Compute P10 = v1 - v0 and P20 = v2 - v0 once per triangle.
    let mut values = Vec::new();
    for (d, vertex, from, first) in &deltas {
        let e = &outputs[*from];
        let mut cols = Vec::new();
        for col in 0..d.cols {
            let at = first + col;
            let (pk, p0) = (
                load(&mut s, *from, e, at, *vertex),
                load(&mut s, *from, e, at, 0),
            );
            cols.push(s.m.bin(b, Bin::Sub, pk, p0));
        }
        values.push(cols);
    }
    for k in 0..3u32 {
        for (id, e) in outputs.iter().enumerate() {
            let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
            for col in 0..e.cols {
                let v = load(&mut s, id, e, col, k);
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                s.op(
                    b,
                    &format!("storeOutput{suffix}"),
                    Attr::NoUnwind,
                    s.t.void,
                    5,
                    &args,
                );
            }
        }
        for (j, cols) in values.iter().enumerate() {
            for (col, &v) in cols.iter().enumerate() {
                let id = s.m.i32((outputs.len() + j) as u32);
                let args = [id, s.m.i32(0), s.m.i8(col as u8), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
        }
        let stream = [s.m.i8(0)];
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
    }
    let stream = [s.m.i8(0)];
    s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    s.inputs = outputs.clone();
    s.outputs = outputs;
    s.outputs.extend(deltas.into_iter().map(|(d, ..)| d));
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::shader::Shader;

    fn elem(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    #[test]
    fn vertex_delta_gs_emits_flat_p10_and_p20_outputs() {
        let mut vs = Shader::new(false);
        let b = vs.entry;
        let outs = vec![
            elem("SV_Position", 3, 4, 0, 4),
            elem("TEXCOORD", 0, 2, 1, 2),
        ];
        for (id, e) in outs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [vs.m.i32(id as u32), vs.m.i32(0), vs.m.i8(col), vs.m.undef(vs.t.i32)];
                let v = vs.op(b, "loadInput.f32", Attr::ReadNone, vs.t.f32, 4, &args);
                let args = [vs.m.i32(id as u32), vs.m.i32(0), vs.m.i8(col), v];
                vs.op(b, "storeOutput.f32", Attr::NoUnwind, vs.t.void, 5, &args);
            }
        }
        vs.inputs = vec![elem("POSITION", 0, 0, 0, 4), elem("TEXCOORD", 0, 0, 1, 2)];
        vs.outputs = outs;
        vs.m.terminate(b, Term::Ret);
        let vs_dxil = vs.finish();

        let mut ps = Shader::new(true);
        let pb = ps.entry;
        ps.inputs = vec![
            elem("TEXCOORD", 0, 1, 1, 2),
            SigElement {
                name: VERTEX_DELTAS[0].into(),
                index: 1,
                kind: 0,
                comp: 9,
                interp: 1,
                row: Some(VERTEX_DELTA_TOP),
                col: 0,
                cols: 2,
                usage: 0b11,
                half: false,
            },
            SigElement {
                name: VERTEX_DELTAS[1].into(),
                index: 1,
                kind: 0,
                comp: 9,
                interp: 1,
                row: Some(VERTEX_DELTA_TOP - 1),
                col: 0,
                cols: 2,
                usage: 0b11,
                half: false,
            },
        ];
        ps.m.terminate(pb, Term::Ret);
        let ps_dxil = ps.finish();

        assert!(needs_vertex_deltas(&ps_dxil));
        assert!(!needs_vertex_deltas(&vs_dxil));
        let gs_dxil = vertex_delta_gs(&vs_dxil, &ps_dxil).expect("vertex_delta_gs");
        let gs_outs = elements(&gs_dxil, FourCC::OSG1, "gs outputs").unwrap();
        assert!(gs_outs.iter().any(|e| e.name == "VICTORY_P10" && e.index == 1 && e.row == Some(31)));
        assert!(gs_outs.iter().any(|e| e.name == "VICTORY_P20" && e.index == 1 && e.row == Some(30)));
    }
}
