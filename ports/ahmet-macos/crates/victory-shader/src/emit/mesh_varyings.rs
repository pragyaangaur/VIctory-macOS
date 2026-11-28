use super::{Attr, Bin, Lower, StageIo, Term, V, Why, pred};

pub(super) struct Staging {
    scratch: V,
    /// Every original mesh return branches here before the group barrier.
    pub exit: usize,
    written: Vec<bool>,
}

impl Lower<'_> {
    pub(super) fn begin_mesh_varyings(&mut self) {
        let (Some(layout), StageIo::Ms(mesh)) = (self.varyings, &self.x.native.io) else {
            return;
        };
        let dwords = layout.shared_dwords(mesh);
        let array = self.s.m.array(dwords, self.s.t.i32);
        let scratch = self.s.m.global("mesh_varyings", array, 3);
        self.s.extra_lds_dwords = dwords;
        self.mesh_varyings = Some(Staging {
            scratch,
            exit: self.s.m.new_block(),
            written: vec![false; layout.fields.len()],
        });

        // Stage writes are to lane-owned slots. Initialize only those slots,
        // so a lane with a skipped export still leaves defined scratch data.
        let lane = self.mesh_thread();
        let z = self.s.m.i32(0);
        let vertices = self.s.m.i32(mesh.max_vertices);
        let in_range = self.s.m.cmp(self.cur, pred::IULT, lane, vertices);
        let active = self.begin_guarded(Some(in_range));
        for field in 0..layout.fields.len() {
            self.write_mesh_word(field as u32 * mesh.max_vertices, lane, z);
        }
        self.end_guarded(active, &[]);
        let primitives = self.s.m.i32(mesh.max_primitives);
        let in_range = self.s.m.cmp(self.cur, pred::IULT, lane, primitives);
        let active = self.begin_guarded(Some(in_range));
        self.stage_mesh_indices(lane, z);
        self.end_guarded(active, &[]);
    }

    fn write_mesh_word(&mut self, offset: u32, index: V, bits: V) {
        let Some(stage) = &self.mesh_varyings else {
            return;
        };
        let c = self.cur;
        let offset = self.s.m.i32(offset);
        let index = self.s.m.bin(c, Bin::Add, offset, index);
        let addr = self.s.m.element(c, stage.scratch, index);
        self.s.m.store(c, addr, bits);
    }

    pub(super) fn stage_mesh_varying(&mut self, lane: V, source: (usize, u8), value: V) {
        let (Some(layout), StageIo::Ms(mesh)) = (self.varyings, &self.x.native.io) else {
            return;
        };
        for (index, field) in layout.fields.iter().enumerate() {
            if field.mesh == source
                && let Some(stage) = &mut self.mesh_varyings
            {
                stage.written[index] = true;
                let bits = self.as_int(value);
                self.write_mesh_word(index as u32 * mesh.max_vertices, lane, bits);
            }
        }
    }

    pub(super) fn stage_mesh_indices(&mut self, primitive: V, packed: V) {
        let (Some(layout), StageIo::Ms(mesh)) = (self.varyings, &self.x.native.io) else {
            return;
        };
        self.write_mesh_word(
            layout.fields.len() as u32 * mesh.max_vertices,
            primitive,
            packed,
        );
    }

    pub(super) fn end_mesh_varyings(&mut self) -> Result<(), Why> {
        let (Some(layout), Some(stage), StageIo::Ms(mesh)) =
            (self.varyings, &self.mesh_varyings, &self.x.native.io)
        else {
            return Ok(());
        };
        if stage.written.iter().any(|written| !written) {
            return Err("the mesh shader never writes a linked varying");
        }
        let scratch = stage.scratch;
        self.cur = stage.exit;
        let shared_and_group = self.s.m.i32(9);
        self.s.op(
            self.cur,
            "barrier",
            Attr::NoDuplicate,
            self.s.t.void,
            80,
            &[shared_and_group],
        );
        let lane = self.mesh_thread();
        let limit = self.s.m.i32(mesh.max_primitives);
        let in_range = self.s.m.cmp(self.cur, pred::IULT, lane, limit);
        let active = self.begin_guarded(Some(in_range));
        let c = self.cur;
        let indices = self.s.m.i32(layout.fields.len() as u32 * mesh.max_vertices);
        let index = self.s.m.bin(c, Bin::Add, indices, lane);
        let ptr = self.s.m.element(c, scratch, index);
        let packed = self.s.m.load(c, ptr);
        let vertices = [0, 10, 20].map(|shift| {
            let shift = self.s.m.i32(shift);
            let mask = self.s.m.i32(0x1ff);
            let index = self.s.m.bin(c, Bin::LShr, packed, shift);
            self.s.m.bin(c, Bin::And, index, mask)
        });
        let mut dwords = Vec::with_capacity(layout.dwords);
        let mut halves = Vec::new();
        let i32t = self.s.t.i32;
        for (idx, field) in layout.fields.iter().enumerate() {
            let start = self.s.m.i32(idx as u32 * mesh.max_vertices);
            for &vertex in &vertices {
                let index = self.s.m.bin(c, Bin::Add, start, vertex);
                let ptr = self.s.m.element(c, scratch, index);
                let bits = self.s.m.load(c, ptr);
                if field.half {
                    let f = self.as_float(bits);
                    let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    halves.push(h);
                } else {
                    dwords.push(bits);
                }
            }
        }
        let shift16 = self.s.m.i32(16);
        for pair in halves.chunks(2) {
            let packed = match pair {
                [lo, hi] => {
                    let hi = self.s.m.bin(c, Bin::Shl, *hi, shift16);
                    self.s.m.bin(c, Bin::Or, *lo, hi)
                }
                [lo] => *lo,
                _ => unreachable!(),
            };
            dwords.push(packed);
        }
        // Define all components before storing a vector. Metal coalesces
        // the stores of one signature row at their first instruction.
        let first = self.s.prims.len() - layout.rows();
        for (row, values) in dwords.chunks(4).enumerate() {
            let id = first + row;
            self.s.prims[id].usage = (1 << values.len()) - 1;
            for (column, &value) in values.iter().enumerate() {
                let id = self.s.m.i32(id as u32);
                let row = self.s.m.i32(0);
                let column = self.s.m.i8(column as u8);
                self.s.op(
                    c,
                    "storePrimitiveOutput.i32",
                    Attr::NoUnwind,
                    self.s.t.void,
                    172,
                    &[id, row, column, value, lane],
                );
            }
        }
        self.end_guarded(active, &[]);
        self.s.m.terminate(self.cur, Term::Ret);
        Ok(())
    }
}
