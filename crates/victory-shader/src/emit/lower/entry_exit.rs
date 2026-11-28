//! Lanes outside a loop's entry predicate leave it at its first exit test
//! ([`Options::loop_entry_exit`](crate::emit::Options)).
//!
//! In a wave64 program predicated for 32-lane halves (`predicate`), an
//! early exit (`v_cmpx` then `s_cbranch_execz` to the end) no longer ends
//! the lanes it takes out of EXEC: they run on with the block's `active`
//! predicate false, every side effect guarded by it. The console runs no
//! pass of a later loop for them (the loop's mask starts as EXEC), but here
//! they follow the loop's own exit test, which the structurisation
//! simplified for the lanes that reach it in EXEC: a lane whose stale
//! counter never meets the test runs to the loop limit, and its wave with
//! it (`CS_VisHair`, `b6df66af105d8b2d`: about 0.3 s a frame).
//!
//! The exit test of such a loop is rewritten so that lanes outside the
//! entry predicate take the exit: `stay && active`. Only where that changes
//! nothing but how long those lanes run:
//! - every block of the loop runs under the very predicate the exit test
//!   does (no linearised or lockstep region in it), so every vector write
//!   and side effect there keeps those lanes' old values;
//! - the loop writes no lane operation (a barrier, a wave or quad
//!   operation, or one taking implicit derivatives from the quad, such as
//!   an implicit-LOD sample, [`super::super::shader::lane_op`]) that such
//!   lanes take part in: the 64-bit atomics' locks and `umax` reduction involve only
//!   the lanes of the instruction's guard;
//! - nothing in it acts for the wave as a whole or for other lanes
//!   (`s_barrier`, scalar memory writes, messages, `v_writelane_b32`);
//! - no value the loop writes to a scalar register is read after it: a lane
//!   leaving early would carry an earlier pass's.

use super::*;

/// A loop exit test noted while lowering, rewritten at the end if its
/// loop allows ([`Lower::leave_loops_at_entry`]).
pub(super) struct Candidate {
    /// The DXIL block the test ends.
    at: usize,
    /// The IR block of the test.
    b: usize,
    /// The loop's header.
    header: usize,
    /// The IR block the exit goes to.
    out: usize,
    /// The test (taken when true) and its DXIL targets.
    cond: V,
    targets: (usize, usize),
    /// Whether staying in the loop is the taken side.
    stay_taken: bool,
    /// The predicate the test runs under.
    active: V,
}

impl Lower<'_> {
    /// Notes the branch ending IR block `b` (to `taken` where `cond`, DXIL
    /// targets `targets`) if it leaves a loop: one side goes back to the
    /// header, directly or through a latch that only jumps there, and the
    /// other out of the loop.
    pub(super) fn note_loop_exit(
        &mut self,
        b: usize,
        cond: V,
        (taken, not_taken): (usize, usize),
        targets: (usize, usize),
    ) {
        if self.x.st.literal || self.pred.is_empty() {
            return;
        }
        let Some(active) = self.pred_active(b) else {
            return;
        };
        let header = |l: &Self, to: usize| -> Option<usize> {
            let back = |h: usize| l.counters.contains_key(&h) && l.rpo[h] <= l.rpo[b];
            if back(to) {
                return Some(to);
            }
            match l.x.st.exits[to] {
                LaneExit::Jump(h) if back(h) && l.rpo[h] <= l.rpo[to] => Some(h),
                _ => None,
            }
        };
        let (stay_taken, h, out) = match (header(self, taken), header(self, not_taken)) {
            (Some(h), None) => (true, h, not_taken),
            (None, Some(h)) => (false, h, taken),
            _ => return,
        };
        self.entry_exits.push(Candidate {
            at: self.cur,
            b,
            header: h,
            out,
            cond,
            targets,
            stay_taken,
            active,
        });
    }

    /// Rewrites the noted loop exit tests whose loops allow it (module
    /// notes); none if `late`, lane operations written after the blocks.
    pub(super) fn leave_loops_at_entry(&mut self, late: bool) {
        for c in std::mem::take(&mut self.entry_exits) {
            if late {
                break;
            }
            let p = self.x.p;
            let body = natural_loop(&p.blocks, &self.rpo, c.header, |q| self.block[q].is_some());
            let plain = |l: &Self, x: usize| {
                !l.pred.exits.contains_key(&x)
                    && !l.pred.dead_blocks.contains(&x)
                    && !l.pred.skipped_ends.contains(&x)
                    && !l.pred.in_lockstep_loop.contains(&x)
                    && !l.pred.lockstep_headers.contains_key(&x)
                    && (x == c.header
                        || !(l.pred.skip_joins.contains_key(&x)
                            || l.pred.diamond_joins.contains_key(&x)
                            || l.pred.lockstep_exits.contains_key(&x)
                            || l.pred.skip_bodies.contains_key(&x)
                            || l.pred.early_conts.contains_key(&x)
                            || l.pred.in_diamond.contains(&x)))
            };
            if !body.contains(&c.b)
                || body.contains(&c.out)
                || body.iter().any(|&x| {
                    !plain(self, x)
                        || self.block_active[x] != Some(c.active)
                        || self.lane_ops_in[x] != 0
                })
                || refusal(p, &body).is_some()
            {
                continue;
            }
            let new = if c.stay_taken {
                self.s.m.bin(c.at, Bin::And, c.cond, c.active)
            } else {
                // `cond || !active`, without an `xor` of a compare (one in
                // a loop exit hung the AMD driver).
                let yes = self.s.m.bool(true);
                self.s.m.select(c.at, c.active, c.cond, yes)
            };
            self.s
                .m
                .terminate(c.at, Term::CondBr(new, c.targets.0, c.targets.1));
            self.loop_entry_exits += 1;
        }
    }
}

/// The blocks of the loop at `header` (among those `live` says are lowered):
/// the header and every block reaching one of its back edges' sources
/// without passing it, in increasing order.
pub(super) fn natural_loop(
    blocks: &[crate::ir::Block],
    rpo: &[usize],
    header: usize,
    live: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut seen = vec![false; blocks.len()];
    seen[header] = true;
    let mut stack: Vec<usize> = blocks[header]
        .preds
        .iter()
        .copied()
        .filter(|&q| live(q) && rpo[q] >= rpo[header])
        .collect();
    while let Some(x) = stack.pop() {
        if std::mem::replace(&mut seen[x], true) {
            continue;
        }
        stack.extend(blocks[x].preds.iter().copied().filter(|&q| live(q)));
    }
    (0..blocks.len()).filter(|&x| seen[x]).collect()
}

/// Why lanes outside the entry predicate of the loop made of `body` must
/// not leave it early (module notes), if they must not.
pub(super) fn refusal(p: &Program, body: &[usize]) -> Option<&'static str> {
    let inside = |x: usize| body.binary_search(&x).is_ok();
    let mut scalars: Set<Value> = Set::default();
    for &x in body {
        let block = &p.blocks[x];
        for phi in &block.phis {
            if !matches!(phi.reg, Reg::V(_)) {
                scalars.insert(phi.value);
            }
        }
        for inst in &block.insts {
            let n = inst.name();
            if n == "s_barrier" {
                return Some("a barrier");
            }
            if n.starts_with("s_store")
                || n.starts_with("s_buffer_store")
                || n.starts_with("s_scratch_store")
                || n.starts_with("s_") && n.contains("atomic")
                || n.starts_with("s_dcache_wb")
            {
                return Some("a scalar memory write");
            }
            if n.starts_with("s_sendmsg") || n.starts_with("s_ttrace") {
                return Some("a message");
            }
            if n == "v_writelane_b32" {
                return Some("a write to another lane");
            }
            scalars.extend(
                inst.writes
                    .iter()
                    .filter(|w| !matches!(w.0, Reg::V(_)))
                    .map(|w| w.1),
            );
        }
    }
    for (x, block) in p.blocks.iter().enumerate() {
        if inside(x) {
            continue;
        }
        let read = block.phis.iter().flat_map(|phi| phi.args.iter().copied());
        let read = read.chain(block.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1)));
        if read.into_iter().any(|v| scalars.contains(&v)) {
            return Some("a scalar written in the loop read after it");
        }
    }
    None
}

/// Where [`fixture`] puts something that keeps lanes from leaving early.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::emit) enum Spoiler {
    /// Nothing: the loop allows it.
    None,
    /// A value the loop writes to an SGPR is read after it.
    ScalarAfter,
    /// A DPP read of the neighbouring lane in the loop (into `v7`): a quad
    /// or wave operation.
    NeighbourInside,
    /// An implicit-LOD `image_sample` in the loop (into `v8`, at `v6`): its
    /// derivatives come from the other lanes of the quad.
    SampleInside,
}

/// The shape of `CS_VisHair` (`b6df66af105d8b2d`) in small, a wave64
/// program of 64 lanes: thread `l` keeps `n = (l & 7) + 1` and `v15 =
/// f32(n)`; `if (l >= 40) return;` (`v_cmpx`, `s_cbranch_execz` to the
/// end), then `v_readfirstlane_b32` (a read across the halves, so the early
/// exit is predicated); then a loop with the console's own mask (`s[20:21]`
/// from EXEC, a latch giving EXEC back, `v_cmpx_ge_f32 exec, 0, v15` and
/// `s_andn2`/`s_cbranch_scc1`) counting its passes in `v6` while `v15`
/// goes down by 1. The lanes past 40 run no pass on the console and keep a
/// positive `v15`. Stores `v6` and the first lane's id (`v7`) at `l * 32`.
#[cfg(test)]
pub(in crate::emit) fn fixture(spoiler: Spoiler) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    ];
    if spoiler == Spoiler::SampleInside {
        w.extend([
            0xF40C_0C02,
            0xFA00_0000, // s_load_dwordx8 s[48:55], s[4:5], 0x0
            0xF408_0802,
            0xFA00_0050, // s_load_dwordx4 s[32:35], s[4:5], 0x50
            0x7E04_0280, // v_mov_b32 v2, 0
        ]);
    }
    w.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7E0E_0280, // v_mov_b32 v7, 0
        0x360A_0087, // v_and_b32 v5, 7, v0
        0x4A0A_0A81, // v_add_nc_u32 v5, 1, v5
        0x7E1E_0D05, // v_cvt_f32_u32 v15, v5
        0x7DA8_00A8, // v_cmpx_gt_u32 40, v0
    ]);
    let execz = w.len();
    w.extend([
        0,           // s_cbranch_execz end (patched)
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E0E_020C, // v_mov_b32 v7, s12
        0xBE98_047E, // s_mov_b64 s[24:25], exec
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0xBF82_0001, // s_branch body
    ]);
    let latch = w.len();
    w.extend([
        0x87FE_1416, // latch: s_and_b64 exec, s[22:23], s[20:21]
        0xBE96_047E, // body: s_mov_b64 s[22:23], exec
        0x4A0C_0C81, // v_add_nc_u32 v6, 1, v6
        0x061E_1EF3, // v_add_f32 v15, -1.0, v15
    ]);
    if spoiler == Spoiler::NeighbourInside {
        w.extend([0x7E0E_02FA, 0xFF00_B106]); // v_mov_b32 v7, v6 quad_perm:[1,0,3,2] (DPP)
    }
    if spoiler == Spoiler::SampleInside {
        w.extend([
            0x7E02_0D06, // v_cvt_f32_u32 v1, v6
            0xF080_0108,
            0x010C_0801, // image_sample v8, v[1:2], s[48:55], s[32:35]
        ]);
    }
    w.extend([
        0x7C2C_1E80, // v_cmpx_ge_f32 0, v15
        0x8A94_7E14, // s_andn2_b64 s[20:21], s[20:21], exec
    ]);
    let back = latch as i32 - (w.len() as i32 + 1);
    w.push(0xBF85_0000 | (back as u32 & 0xFFFF)); // s_cbranch_scc1 latch
    w.push(0xBEFE_0418); // s_mov_b64 exec, s[24:25]
    if spoiler == Spoiler::ScalarAfter {
        w.push(0xBE9A_0314); // s_mov_b32 s26, s20
    }
    // buffer_store_dword v6/v7, v16, s[40:43], 0 offen offset:0/4
    w.extend([0xE070_1000, 0x800A_0610, 0xE070_1004, 0x800A_0710]);
    let end = w.len();
    w[execz] = 0xBF88_0000 | ((end - execz - 1) as u32 & 0xFFFF);
    w.push(0xBF81_0000); // end: s_endpgm
    w
}

/// [`fixture`]'s console results, per lane: the passes and the first lane.
#[cfg(all(test, windows))]
pub(in crate::emit) fn console_fixture(l: u32) -> [u32; 2] {
    if l < 40 { [(l & 7) + 1, 0] } else { [0, 0] }
}

#[cfg(test)]
mod tests {
    use super::{Spoiler, fixture, natural_loop, refusal};
    use crate::emit::lower::{self, Parts};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{gfx10, io, ir, resource, structure};

    fn native(words: &[u32]) -> NativeProgram {
        let range = |kind, table_offset_dwords, dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords,
            user_data: 0,
            slot: 0,
        };
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 32,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                range(ResourceType::Srv, 0, 8),
                range(ResourceType::Srv, 8, 4),
                range(ResourceType::Uav, 16, 4),
                range(ResourceType::Sampler, 20, 4),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// Loop exits rewritten in halves, with the switch `on`.
    fn rewritten(words: &[u32], on: bool) -> usize {
        let native = native(words);
        let p = ir::build(&native.code, true, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
            atomic_locks: true,
            atomic_lock_rounds: true,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: true,
            uav_size_level0: false,
            halves_rendezvous: true,
            host_wave32: true,
            loop_entry_exit: on,
        };
        let l = lower::lower_limited(&parts, None, false, false, false).expect("lowers in halves");
        assert_eq!(l.wave_size, Some(32));
        l.loop_entry_exits
    }

    #[test]
    fn the_fixture_is_what_its_comments_say() {
        for spoiler in [
            Spoiler::None,
            Spoiler::ScalarAfter,
            Spoiler::NeighbourInside,
            Spoiler::SampleInside,
        ] {
            let code: Vec<u8> = fixture(spoiler)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let all = gfx10::decode(&code)
                .expect("decodes")
                .iter()
                .map(|i| i.text(true))
                .collect::<Vec<_>>()
                .join("\n");
            for want in [
                "v_cmpx_gt_u32 exec, 40, v0",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s12, v0",
                "s_and_b64 exec, s[22:23], s[20:21]",
                "v_add_f32 v15, -1.0, v15",
                "v_cmpx_ge_f32 exec, 0, v15",
                "s_andn2_b64 s[20:21], s[20:21], exec",
                "s_cbranch_scc1",
                "s_mov_b64 exec, s[24:25]",
                "buffer_store_dword v7, v16, s[40:43], 0 offen offset:4",
            ] {
                assert!(all.contains(want), "{want:?} not in:\n{all}");
            }
            assert_eq!(
                all.contains("s_mov_b32 s26, s20"),
                spoiler == Spoiler::ScalarAfter
            );
            assert_eq!(
                all.contains("quad_perm:[1,0,3,2]"),
                spoiler == Spoiler::NeighbourInside
            );
            assert_eq!(
                all.contains("image_sample v8, v[1:2], s[48:55], s[32:35]"),
                spoiler == Spoiler::SampleInside,
                "{all}"
            );
        }
    }

    /// The loop is its header (the body after the `s_branch`) and the
    /// latch, without the early exit's end or the stores.
    #[test]
    fn the_loop_is_the_header_and_its_latch() {
        let code: Vec<u8> = fixture(Spoiler::None)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p = ir::build(&code, true, 32).unwrap();
        let order = ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            rpo[b] = n;
        }
        let header = (0..p.blocks.len())
            .find(|&b| p.blocks[b].preds.iter().any(|&q| rpo[q] >= rpo[b]))
            .expect("a loop");
        let body = natural_loop(&p.blocks, &rpo, header, |_| true);
        let offsets: Vec<usize> = body.iter().map(|&b| p.blocks[b].offset).collect();
        // The latch's `s_and_b64` and the body after it.
        let latch = 18 * 4;
        assert_eq!(offsets, [latch, latch + 4], "{body:?}");
        assert_eq!(refusal(&p, &body), None);
    }

    #[test]
    fn what_the_loop_writes_to_scalars_and_reads_after_it_refuses() {
        for (spoiler, want) in [
            (Spoiler::None, None),
            (
                Spoiler::ScalarAfter,
                Some("a scalar written in the loop read after it"),
            ),
        ] {
            let code: Vec<u8> = fixture(spoiler)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let p = ir::build(&code, true, 32).unwrap();
            let order = ir::reverse_postorder(&p.blocks);
            let mut rpo = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                rpo[b] = n;
            }
            let header = (0..p.blocks.len())
                .find(|&b| p.blocks[b].preds.iter().any(|&q| rpo[q] >= rpo[b]))
                .expect("a loop");
            let body = natural_loop(&p.blocks, &rpo, header, |_| true);
            assert_eq!(refusal(&p, &body), want, "{spoiler:?}");
        }
    }

    /// A barrier, a scalar memory write or a write to another lane in the
    /// loop refuses it.
    #[test]
    fn what_acts_for_other_lanes_in_the_loop_refuses() {
        for (word, want) in [
            (0xBF8A_0000, "a barrier"),               // s_barrier
            (0xD761_0006, "a write to another lane"), // v_writelane_b32 v6, s0, 0
        ] {
            let mut w = fixture(Spoiler::None);
            // In the body, after its `s_mov_b64 s[22:23], exec`.
            let at = 5 + 7 + 6 + 2;
            w.insert(at, word);
            if word >> 16 == 0xD761 {
                w.insert(at + 1, 0x0001_0000);
            }
            // The early exit and the back branch span the new words.
            let n = 1 + u32::from(word >> 16 == 0xD761);
            let execz = w.iter().position(|&x| x >> 16 == 0xBF88).unwrap();
            w[execz] += n;
            let scc1 = w.iter().rposition(|&x| x >> 16 == 0xBF85).unwrap();
            w[scc1] = 0xBF85_0000 | (w[scc1].wrapping_sub(n) & 0xFFFF);
            let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = ir::build(&code, true, 32).unwrap();
            let order = ir::reverse_postorder(&p.blocks);
            let mut rpo = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                rpo[b] = n;
            }
            let header = (0..p.blocks.len())
                .find(|&b| p.blocks[b].preds.iter().any(|&q| rpo[q] >= rpo[b]))
                .expect("a loop");
            let body = natural_loop(&p.blocks, &rpo, header, |_| true);
            assert_eq!(refusal(&p, &body), Some(want));
        }
    }

    /// With the switch, the fixture's one loop exit is rewritten; without
    /// it, or where the loop does not allow it, none is.
    #[test]
    fn the_loop_exit_is_rewritten_only_with_the_switch_and_where_exact() {
        let plain = fixture(Spoiler::None);
        assert_eq!(rewritten(&plain, false), 0);
        assert_eq!(rewritten(&plain, true), 1);
        assert_eq!(rewritten(&fixture(Spoiler::ScalarAfter), true), 0);
        assert_eq!(rewritten(&fixture(Spoiler::NeighbourInside), true), 0);
        assert_eq!(rewritten(&fixture(Spoiler::SampleInside), true), 0);
    }
}
