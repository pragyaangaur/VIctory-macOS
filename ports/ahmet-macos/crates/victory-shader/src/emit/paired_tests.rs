//! The paired Wave64 form ([`super::paired`]) on every platform, without `dxil.dll` or a GPU.
//!
//! A console wave64 compute program on a host of 32-lane waves (Apple's GPUs, NVIDIA's) in
//! the paired form runs each console wave as one host wave of 32 threads, each thread two
//! console lanes (n and n + 32). These tests take the six synthetic programs the old
//! paired-lane branch ran on the GPU (its `wave64_tests`) and check what can be checked where
//! the translation is made, on the Air: that the form takes each of them, as a group of half
//! the console's threads that needs only 32-lane waves (`PSV0`); what the emitted module holds
//! (instruction counts, [`super::census`]); that `VICTORY_WAVE_PAIRS` decides when [`super::emit`]
//! picks it, and that off (the default) or where the form refuses a program, `emit` gives
//! exactly what it gave before, byte for byte.
//!
//! What the programs compute on the console is noted at each; checking the host's results
//! against it needs the host GPU (see `STATUS.md`: the Pro run).

use super::{HostShader, Pairs, TEST_PAIRS, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

// Operands as the hardware numbers them: SGPR n is n, VCC 106 (its high dword 107), EXEC 126,
// an inline integer n 128 + n (-1: 193), a literal (the next dword) 255, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

const fn ds(op: u32, vdst: u32, addr: u32, data: u32, offset: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18 | offset, vdst << 24 | data << 8 | addr]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// A compute program (the thread index in v0) of `threads` threads in a row with 4 root
/// constants in s0..s3, a raw UAV's descriptor in s[8:11] and `lds` bytes of LDS, then `body`;
/// wave64 when `wave64` is set.
fn native_with(body: &[u32], wave64: bool, threads: u32, lds: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | lds.div_ceil(512) << 15, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 OR theirs over the wave inside `if (t < s0) { ... }` and store the result to
/// dword 0 of the raw UAV. `skip`: the branch past the body when no thread is below s0:
/// `s_cbranch_execz`, or `s_cbranch_scc0` on the SCC `s_and_saveexec` sets.
/// The console: the OR of `1 << (t >> 1)` over t < s0, nothing stored for s0 = 0.
fn wave_or_in_a_lane_branch(skip: u32, threads: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C,          // s_mov_b64 exec, s[12:13]
    ]);
    native_with(&words, true, threads, 0)
}

/// One console wave with 256 bytes of LDS. The even lanes write their dword in one arm of a
/// per-lane branch; the odd lanes theirs in the other arm, then, in that arm and with no
/// barrier, read their mirror lane's (63 - n, an even lane's); the even lanes read theirs (an
/// odd lane's) after the join. Then lanes 5 and 40 are read (`v_readlane_b32`), the lanes
/// holding more than 250 are counted (`s_bcnt1_i32_b64`) and counted below each lane
/// (`v_mbcnt_lo/hi`): each of these takes lanes of both halves.
/// The console: lane n's dword is 7n + 100 (even n) or 7n + 200, lane n reads 63 - n's.
fn lds_and_lanes_across_halves() -> NativeProgram {
    let mut w = vec![vop2(0x1A, 1, k(2), 0)]; // v_lshlrev_b32 v1, 2, v0: this lane's dword
    w.extend(vop3(0x169, 2, v(0), k(7))); // v_mul_lo_u32 v2, v0, 7
    w.extend([
        vop2(0x26, 5, k(63), 0), // v_sub_nc_u32 v5, 63, v0: the mirror lane
        vop2(0x1A, 5, k(2), 5),  // v_lshlrev_b32 v5, 2, v5: its dword
        vop2(0x1B, 3, k(1), 0),  // v_and_b32 v3, 1, v0
        vopc(0xC2, k(0), 3),     // v_cmp_eq_u32 vcc, 0, v3: the even lanes
        sop1(0x24, 12, VCC),     // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 4),           // s_cbranch_execz past the even lanes' arm
        vop2(0x25, 4, LIT, 2),   // v_add_nc_u32 v4, 100, v2
        100,
    ]);
    w.extend(ds(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend([
        sop2(0x15, EXEC, 12, EXEC), // s_andn2_b64 exec, s[12:13], exec: the odd lanes
        sopp(0x08, 6),              // s_cbranch_execz past their arm
        vop2(0x25, 4, LIT, 2),      // v_add_nc_u32 v4, 200, v2
        200,
    ]);
    w.extend(ds(0x0D, 0, 1, 4, 0)); // ds_write_b32 v1, v4
    w.extend(ds(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an even lane's, from the other arm
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        sop1(0x24, 12, VCC),  // s_and_saveexec_b64 s[12:13], vcc: the even lanes again
    ]);
    w.extend(ds(0x36, 6, 5, 0, 0)); // ds_read_b32 v6, v5: an odd lane's
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w.extend(vop3(0x360, 14, v(6), k(5))); // v_readlane_b32 s14, v6, 5
    w.extend(vop3(0x360, 15, v(6), k(40))); // v_readlane_b32 s15, v6, 40
    w.extend([
        sop2(0x00, 14, 14, 15), // s_add_u32 s14, s14, s15
        vopc(0xC1, LIT, 6),     // v_cmp_lt_u32 vcc, 250, v6
        250,
        sop1(0x10, 16, VCC), // s_bcnt1_i32_b64 s16, vcc
    ]);
    w.extend(vop3(0x365, 7, VCC, k(0))); // v_mbcnt_lo_u32_b32 v7, vcc_lo, 0
    w.extend(vop3(0x366, 7, VCC + 1, v(7))); // v_mbcnt_hi_u32_b32 v7, vcc_hi, v7
    w.extend([
        vop1(0x01, 9, 14),  // v_mov_b32 v9, s14
        vop1(0x01, 10, 16), // v_mov_b32 v10, s16
    ]);
    w.extend(stores([6, 9, 10, 7]));
    native_with(&w, true, 64, 256)
}

/// Two console waves of 64 in a group of 128 with 512 bytes of LDS. In each, lane 0 zeroes
/// the wave's dword; every lane loops (lane & 3) + 1 + wave times, adding its lane number to
/// its sum and 1 to the wave's dword each round (`ds_add_u32`; the loop's branch is the
/// wave's); after the loop each lane reads the wave's total, with no barrier; lanes 40-63
/// read their first lane's sum (`v_readfirstlane_b32`), then lanes 20-63; past the barrier,
/// each lane reads the other wave's total.
fn loops_apart_meeting_at_the_barrier() -> NativeProgram {
    let mut w = vec![
        vop2(0x16, 1, k(6), 0),  // v_lshrrev_b32 v1, 6, v0: the console wave
        vop2(0x1B, 2, k(63), 0), // v_and_b32 v2, 63, v0: the lane in it
        vop2(0x1A, 11, k(2), 1), // v_lshlrev_b32 v11, 2, v1: the wave's dword
        vop1(0x01, 12, k(0)),    // v_mov_b32 v12, 0
        sop1(0x04, 22, EXEC),    // s_mov_b64 s[22:23], exec
        vopc(0xD2, k(0), 2),     // v_cmpx_eq_u32 exec, 0, v2: lane 0
    ];
    w.extend(ds(0x0D, 0, 11, 12, 0)); // ds_write_b32 v11, v12
    w.extend([
        sop1(0x04, EXEC, 22),   // s_mov_b64 exec, s[22:23]
        vop2(0x1B, 3, k(3), 2), // v_and_b32 v3, 3, v2
        vop2(0x25, 3, k(1), 3), // v_add_nc_u32 v3, 1, v3
        vop2(0x25, 3, v(1), 3), // v_add_nc_u32 v3, v1, v3: the rounds
        vop1(0x01, 4, k(0)),    // v_mov_b32 v4, 0
        vop1(0x01, 13, k(1)),   // v_mov_b32 v13, 1
        sop1(0x04, 20, EXEC),   // s_mov_b64 s[20:21], exec
        vop2(0x25, 4, v(2), 4), // loop: v_add_nc_u32 v4, v2, v4
    ]);
    w.extend(ds(0x00, 0, 11, 13, 0)); // ds_add_u32 v11, v13
    w.extend([
        vop2(0x25, 3, 193, 3),       // v_add_nc_u32 v3, -1, v3
        vopc(0xC5, k(0), 3),         // v_cmp_ne_u32 vcc, 0, v3
        sop2(0x0F, EXEC, EXEC, VCC), // s_and_b64 exec, exec, vcc
        sopp(0x09, -7),              // s_cbranch_execnz loop
        sop1(0x04, EXEC, 20),        // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(ds(0x36, 5, 11, 0, 0)); // ds_read_b32 v5, v11: the wave's total
    w.extend([
        sop1(0x04, 24, EXEC),          // s_mov_b64 s[24:25], exec
        vopc(0xD1, k(39), 2),          // v_cmpx_lt_u32 exec, 39, v2: lanes 40-63
        vop1(0x02, 6, v(4)),           // v_readfirstlane_b32 s6, v4
        sop1(0x04, EXEC, 24),          // s_mov_b64 exec, s[24:25]
        vopc(0xD1, k(19), 2),          // v_cmpx_lt_u32 exec, 19, v2: lanes 20-63
        vop1(0x02, 7, v(4)),           // v_readfirstlane_b32 s7, v4
        sop1(0x04, EXEC, 24),          // s_mov_b64 exec, s[24:25]
        vop1(0x01, 6, 6),              // v_mov_b32 v6, s6
        vop1(0x01, 9, 7),              // v_mov_b32 v9, s7
        0xD400_0000 | 0x346 << 16 | 6, // v_lshl_add_u32 v6, v9, 16, v6
        v(9) | k(16) << 9 | v(6) << 18,
        sopp(0x0A, 0),            // s_barrier
        vop2(0x1D, 14, k(4), 11), // v_xor_b32 v14, 4, v11: the other wave's dword
    ]);
    w.extend(ds(0x36, 7, 14, 0, 0)); // ds_read_b32 v7, v14
    w.extend(stores([4, 5, 6, 7]));
    native_with(&w, true, 128, 512)
}

/// One console wave: `ds_bpermute_b32` from the mirror lane; a DPP row shift whose row mask
/// leaves rows 0 and 3 alone (rows 2 and 3 are lanes 32-63); a `ds_swizzle_b32` swapping the
/// two 16-lane rows of each 32 lanes; `v_writelane_b32` of s0 into lane 45.
fn permute_shift_and_write_lanes() -> NativeProgram {
    let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
    w.extend([
        vop2(0x25, 1, k(1), 1),  // v_add_nc_u32 v1, 1, v1
        vop2(0x26, 2, k(63), 0), // v_sub_nc_u32 v2, 63, v0
        vop2(0x1A, 2, k(2), 2),  // v_lshlrev_b32 v2, 2, v2: the mirror lane's address
    ]);
    w.extend(ds(0xB3, 3, 2, 1, 0)); // ds_bpermute_b32 v3, v2, v1
    w.extend([
        vop1(0x01, 4, LIT), // v_mov_b32 v4, 1000
        1000,
        vop1(0x01, 4, 0xFA), // v_mov_b32 v4, v1 row_shr:1 row_mask:0x6 bank_mask:0xf
        0x6 << 28 | 0xF << 24 | 0x111 << 8 | 1,
    ]);
    w.extend(ds(0x35, 5, 1, 0, 0x401F)); // ds_swizzle_b32 v5, v1 offset:swizzle(BITMASK_PERM, xor 16)
    w.push(vop1(0x01, 6, v(0))); // v_mov_b32 v6, v0
    w.extend(vop3(0x361, 6, 0, k(45))); // v_writelane_b32 v6, s0, 45
    w.extend(stores([3, 4, 5, 6]));
    native_with(&w, true, 64, 0)
}

/// One console wave of 64 threads, thread t holding the bit `1 << (t >> 1)`: the threads
/// below s0 OR theirs over the wave inside `if (t < s0) { ... }` into s7, which holds 0x77
/// before, and every thread stores s7 after the branch: the wave's OR, or 0x77 when no thread
/// took the branch. A wave-wide value used past a per-lane branch, as the title's tile
/// classification and light clustering programs use theirs: the forms in two host waves of 32
/// cannot take it exactly, and the approximate form gives each half-wave its own OR.
fn wave_value_past_a_lane_branch() -> NativeProgram {
    let mut w = vec![
        sop1(0x03, 7, LIT), // s_mov_b32 s7, 0x77
        0x77,
        vop1(0x01, 5, k(1)),    // v_mov_b32 v5, 1
        vop2(0x16, 6, k(1), 0), // v_lshrrev_b32 v6, 1, v0
        vop2(0x1A, 7, v(6), 5), // v_lshlrev_b32 v7, v6, v5
        vopc(0xC4, 0, 0),       // v_cmp_gt_u32 vcc, s0, v0
        sop1(0x24, 12, VCC),    // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 20),         // s_cbranch_execz past the OR
    ];
    w.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    w.extend([
        sop1(0x04, EXEC, 12), // s_mov_b64 exec, s[12:13]
        vop1(0x01, 8, 7),     // v_mov_b32 v8, s7
    ]);
    w.extend(stores([8, 7, 6, 0]));
    native_with(&w, true, 64, 0)
}

/// 64 threads in one console wave: thread t holds the 16-bit float with bits 0x3C00 + 37 t;
/// all of them store the wave's sum (`WaveActiveSum` of halves) to dword 0. The console adds
/// in its balanced tree (lanes 2k and 2k + 1, then pairs, ... then the halves), each addition
/// rounded to 16 bits.
fn sum_of_halves() -> NativeProgram {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(
        super::wave::WAVE_OR_V7_TO_S7
            .iter()
            .map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }),
    );
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    native_with(&words, true, 64, 0)
}

/// The six programs the old branch ran on the GPU in the paired form, by name.
fn synthetic() -> Vec<(&'static str, NativeProgram)> {
    vec![
        ("LDS and lanes across halves", lds_and_lanes_across_halves()),
        ("loops apart, meeting at the barrier", loops_apart_meeting_at_the_barrier()),
        ("permute, shift, write lanes", permute_shift_and_write_lanes()),
        ("wave OR in a lane branch (execz)", wave_or_in_a_lane_branch(0xBF88_0000, 64)),
        ("wave OR in a lane branch (scc0)", wave_or_in_a_lane_branch(0xBF84_0000, 64)),
        ("wave value past a lane branch", wave_value_past_a_lane_branch()),
        ("sum of halves", sum_of_halves()),
    ]
}

/// What [`super::emit`] and [`super::paired`] take besides the container.
struct Stages {
    p: ir::Program,
    st: structure::Structured,
    io: io::Io,
    res: resource::Resources,
}

fn stages(native: &NativeProgram) -> Stages {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    Stages { p, st, io, res }
}

/// `f` on a host of 32-lane waves with `VICTORY_WAVE_PAIRS` as `mode`, for this thread only.
fn on_32_lanes<R>(mode: Pairs, f: impl FnOnce() -> R) -> R {
    let wave = lower::TEST_HOST_WAVE.replace(Some(32));
    let pairs = TEST_PAIRS.replace(Some(mode));
    let r = f();
    TEST_PAIRS.set(pairs);
    lower::TEST_HOST_WAVE.set(wave);
    r
}

/// `native` in the paired form on a host of 32-lane waves.
fn paired(native: &NativeProgram) -> Result<(HostShader, Option<u32>), ShaderError> {
    let s = stages(native);
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("container");
    on_32_lanes(Pairs::Off, || super::paired(&c, native, &s.p, &s.io, &s.res))
}

/// What [`super::emit`] gives `native` on a host of 32-lane waves with `mode`.
fn emitted(native: &NativeProgram, mode: Pairs) -> Result<HostShader, ShaderError> {
    let s = stages(native);
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("container");
    on_32_lanes(mode, || super::emit(&c, native, &s.p, &s.st, &s.io, &s.res))
}

/// The emitted module's bytes, or the error's text: what two translations are compared by.
fn bytes(r: &Result<HostShader, ShaderError>) -> Result<&[u8], String> {
    r.as_ref().map(|s| s.dxil.as_slice()).map_err(|e| e.to_string())
}

/// A compute shader's thread group and its required wave size (minimum, maximum) from `PSV0`
/// (runtime info: wave sizes at 16 and 20, the group at 36).
fn group_and_wave(shader: &HostShader) -> ([u32; 3], (u32, u32)) {
    let c = Container::parse(&shader.dxil).expect("the emitted container");
    let psv = c.part(FourCC::PSV0).expect("PSV0").data;
    let at = |o: usize| u32::from_le_bytes(psv[4 + o..8 + o].try_into().unwrap());
    ([at(36), at(40), at(44)], (at(16), at(20)))
}

/// The instruction counts of the module `f` emits last as `shader`, by [`super::census::KINDS`].
fn census_of(f: impl FnOnce() -> HostShader) -> std::collections::HashMap<&'static str, u32> {
    let was = super::bitcode::census::FORCE.replace(true);
    let shader = f();
    super::bitcode::census::FORCE.set(was);
    let c = super::census::take(&shader.dxil).expect("the census of the emitted module");
    super::census::KINDS.iter().copied().zip(c.counts).collect()
}

#[test]
fn the_switch_is_off_unless_it_names_a_mode() {
    use super::parse_pairs;
    assert_eq!(parse_pairs(None), Pairs::Off);
    assert_eq!(parse_pairs(Some("0")), Pairs::Off);
    assert_eq!(parse_pairs(Some("")), Pairs::Off);
    assert_eq!(parse_pairs(Some("yes")), Pairs::Off);
    assert_eq!(parse_pairs(Some("last")), Pairs::Last);
    assert_eq!(parse_pairs(Some("1")), Pairs::BeforeApproximate);
    assert_eq!(parse_pairs(Some(" 1 ")), Pairs::BeforeApproximate);
    assert_eq!(parse_pairs(Some("first")), Pairs::First);
    assert_eq!(parse_pairs(Some("all")), Pairs::All);
}

/// Each synthetic program takes the paired form: a group of half the console's threads
/// (one host wave of 32 for each console wave of 64) that asks for 32-lane waves exactly,
/// which a 32-lane host accepts; the same bytes each time.
#[test]
fn every_synthetic_program_takes_the_paired_form_in_32_lane_waves() {
    for (name, native) in synthetic() {
        let (shader, wave) = paired(&native).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(wave, Some(32), "{name}");
        let StageIo::Cs(cs) = native.io else { unreachable!() };
        let (group, waves) = group_and_wave(&shader);
        assert_eq!(group, [cs.threads[0] / 2, 1, 1], "{name}: the host's group");
        assert_eq!(waves, (32, 32), "{name}: the required wave size");
        let (again, _) = paired(&native).expect("again");
        assert!(again.dxil == shader.dxil, "{name}: two translations differ");
    }
}

/// What the paired form emits for each program: the lanes' work twice (one copy per console
/// lane: each LDS access, each store), cross-lane work as host wave operations, and no group
/// barrier but the console's own; LDS accesses are ordered by fences.
#[test]
fn the_paired_form_runs_each_lane_and_orders_lds_by_fences() {
    let count = |native: &NativeProgram| census_of(|| paired(native).expect("paired").0);
    // two LDS writes and two reads per console lane, each lane's own; lane reads, ballots and
    // counts below a lane as wave operations; a fence before each write and before the first
    // read after a write in its block
    let lds = count(&lds_and_lanes_across_halves());
    assert_eq!((lds["lds_store"], lds["lds_load"]), (4, 4), "{lds:?}");
    assert_eq!(lds["buffer_store"], 8, "4 dwords for each of 2 lanes: {lds:?}");
    assert!(lds["wave_op"] > 0, "{lds:?}");
    // (the even arm's write; the odd arm's write and its read after it; the read after the
    // join: four fences, LDS only (mode 8), no thread waiting for another)
    assert_eq!(lds["barrier"], 4, "fences between the lanes' accesses: {lds:?}");
    // two console waves: lane 0's write in each lane copy, plus the LDS counter `wave_base`
    // numbers the host waves with (zeroed, then one group barrier on entry);
    // four reads; three fences, the entry barrier and the console's own `s_barrier`
    let loops = count(&loops_apart_meeting_at_the_barrier());
    assert_eq!((loops["lds_store"], loops["lds_load"]), (3, 4), "{loops:?}");
    assert_eq!(loops["barrier"], 5, "{loops:?}");
    assert!(loops["wave_op"] > 0, "{loops:?}");
    assert_eq!(loops["buffer_store"], 8, "{loops:?}");
    // permutes and lane writes read and write other host lanes: wave operations, no LDS
    let permute = count(&permute_shift_and_write_lanes());
    assert_eq!((permute["lds_store"], permute["lds_load"]), (0, 0), "{permute:?}");
    assert!(permute["wave_op"] > 0, "{permute:?}");
    // the OR over 64 lanes: two host-wave ORs joined; no LDS exchange, no group barrier
    let or = count(&wave_value_past_a_lane_branch());
    assert_eq!((or["lds_store"], or["lds_load"], or["barrier"]), (0, 0, 0), "{or:?}");
    assert!(or["wave_op"] >= 2, "{or:?}");
}

/// Off (the default), [`super::emit`] never gives the paired form: for every synthetic program
/// its translation differs from the paired one, and a program needing the paired form gets
/// the forms it got before.
#[test]
fn off_by_default_emit_gives_the_forms_it_gave_before() {
    for (name, native) in synthetic() {
        let (pairs, _) = paired(&native).expect("paired");
        let off = emitted(&native, Pairs::Off);
        assert!(bytes(&off) != Ok(pairs.dxil.as_slice()), "{name}: off, emit gave the paired form");
    }
}

/// Where the paired form refuses a program, [`super::emit`] goes on exactly as without it,
/// whatever the switch says: a thread group that does not fill its console waves (32 threads
/// of a wave64 program), and a program of 32-lane waves, which the host's waves already fit.
#[test]
fn a_program_the_paired_form_refuses_gets_the_form_it_got_before() {
    let half_group = wave_or_in_a_lane_branch(0xBF88_0000, 32);
    let e = paired(&half_group).err().expect("refused");
    assert!(e.to_string().contains("does not fill its console waves"), "{e}");
    let wave32 = {
        let mut n = permute_shift_and_write_lanes();
        n.wave64 = false;
        n
    };
    let e = paired(&wave32).err().expect("refused");
    assert!(e.to_string().contains("already fit"), "{e}");
    for native in [half_group, wave32] {
        let off = emitted(&native, Pairs::Off);
        for mode in [Pairs::BeforeApproximate, Pairs::First, Pairs::All] {
            let on = emitted(&native, mode);
            assert!(bytes(&on) == bytes(&off), "{mode:?}: not what emit gives off");
        }
    }
}

/// Where `VICTORY_WAVE_PAIRS` puts the paired form, program by program: `true` where
/// [`super::emit`] gives the paired form, `false` where it gives exactly what it gives with
/// the switch off.
/// * `last` preserves every already usable form. Only the refused permute changes here.
/// * `1` keeps every exact form new-base has and takes the paired form before the approximate
///   ones and as the last resort: the `scc0` wave OR and the wave value past a branch went to
///   an approximate form (each half-wave its own OR), the permute was refused.
/// * `first` also goes before the exact retries: the `execz` wave OR, which needs its
///   early-out left out to translate as two host waves.
/// * `all` takes it for every program, also those the first form already gives as two host
///   waves of 32 that exchange through LDS (the LDS, loop and 16-bit sum programs).
#[test]
fn the_switch_decides_where_emit_takes_the_paired_form() {
    let table: [(&str, NativeProgram, [bool; 4]); 7] = [
        ("LDS and lanes across halves", lds_and_lanes_across_halves(), [false, false, false, true]),
        ("loops apart", loops_apart_meeting_at_the_barrier(), [false, false, false, true]),
        ("permute, shift, write lanes", permute_shift_and_write_lanes(), [true, true, true, true]),
        ("wave OR (execz)", wave_or_in_a_lane_branch(0xBF88_0000, 64), [false, false, true, true]),
        ("wave OR (scc0)", wave_or_in_a_lane_branch(0xBF84_0000, 64), [false, true, true, true]),
        ("wave value past a lane branch", wave_value_past_a_lane_branch(), [false, true, true, true]),
        ("sum of halves", sum_of_halves(), [false, false, false, true]),
    ];
    for (name, native, want) in table {
        let (pairs, _) = paired(&native).expect("paired");
        let off = emitted(&native, Pairs::Off);
        assert!(bytes(&off) != Ok(pairs.dxil.as_slice()), "{name}: off, the paired form");
        for (mode, paired_here) in [Pairs::Last, Pairs::BeforeApproximate, Pairs::First, Pairs::All].into_iter().zip(want) {
            let on = emitted(&native, mode);
            if paired_here {
                assert!(bytes(&on) == Ok(pairs.dxil.as_slice()), "{name}, {mode:?}: not the paired form");
            } else {
                assert!(bytes(&on) == bytes(&off), "{name}, {mode:?}: not what emit gives off");
            }
        }
    }
}

/// A program no form takes gets the paired form as the last resort when the switch is on: the
/// permute and the lane write of `permute_shift_and_write_lanes` cross the halves of the
/// console wave, which two host waves of 32 cannot do (refused: "a lane permute across two
/// host waves"), and one host wave per console wave can. Off, it stays refused.
#[test]
fn a_program_no_form_takes_gets_the_paired_form_last() {
    let native = permute_shift_and_write_lanes();
    let off = emitted(&native, Pairs::Off);
    assert!(matches!(&off, Err(ShaderError::Unsupported { .. })), "{:?}", bytes(&off));
    let (pairs, _) = paired(&native).expect("paired");
    for mode in [Pairs::Last, Pairs::BeforeApproximate, Pairs::First, Pairs::All] {
        let on = emitted(&native, mode);
        assert!(bytes(&on) == Ok(pairs.dxil.as_slice()), "{mode:?}: {:?}", bytes(&on).err());
    }
}

/// What one dispatch of a synthetic program should leave on the console: the root constants
/// s0..s3, the raw UAV's dword count (zeroed before), and every dword after.
#[derive(Clone)]
struct Expected {
    constants: [u32; 4],
    dwords: usize,
    want: Vec<u32>,
}

/// The console's results for the programs the old branch checked on the GPU (its
/// `wave64_tests`), one entry per dispatch: what a run on the host GPU must reproduce.
fn console_results() -> Vec<(&'static str, NativeProgram, Vec<Expected>)> {
    let one = |constants, dwords, want| Expected { constants, dwords, want };
    // lane n's dword is 7n + 100 (even n) or 7n + 200, and lane n reads 63 - n's
    let lds = {
        let dword = |n: u32| 7 * n + if n.is_multiple_of(2) { 100 } else { 200 };
        let read: Vec<u32> = (0..64).map(|n| dword(63 - n)).collect();
        let big = (0..64).filter(|&n| read[n] > 250).fold(0u64, |m, n| m | 1 << n);
        let want = (0..64)
            .flat_map(|n| [read[n], read[5] + read[40], big.count_ones(), (big & ((1 << n) - 1)).count_ones()])
            .collect();
        vec![one([0; 4], 256, want)]
    };
    let loops = {
        let rounds = |wave: u32, lane: u32| (lane & 3) + 1 + wave;
        let total = |wave: u32| (0..64).map(|lane| rounds(wave, lane)).sum::<u32>();
        let first = |wave: u32, lane: u32| lane * rounds(wave, lane);
        let want = (0..128)
            .flat_map(|t| {
                let (wave, lane) = (t / 64, t % 64);
                [first(wave, lane), total(wave), first(wave, 20) << 16 | first(wave, 40), total(1 - wave)]
            })
            .collect();
        vec![one([0; 4], 512, want)]
    };
    let permute = {
        let s0 = 0xABCD_1234;
        let x = |n: u32| 3 * n + 1;
        let want = (0..64)
            .flat_map(|n| {
                let shifted = if matches!(n >> 4, 1 | 2) && n & 15 != 0 { x(n - 1) } else { 1000 };
                [x(63 - n), shifted, x(n & 32 | (n & 31) ^ 16), if n == 45 { s0 } else { n }]
            })
            .collect();
        vec![one([s0, 0, 0, 0], 256, want)]
    };
    let or = || {
        [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1), (0, 0)]
            .map(|(below, or)| one([below, 0, 0, 0], 1, vec![or]))
            .to_vec()
    };
    let past = [64u32, 40, 33, 32, 10, 1, 0]
        .map(|s0| {
            let or = (0..64u32).filter(|&t| t < s0).fold(0u32, |m, t| m | 1 << (t >> 1));
            let s7 = if s0 == 0 { 0x77 } else { or };
            one([s0, 0, 0, 0], 256, (0..64u32).flat_map(|t| [s7, 1 << (t >> 1), t >> 1, t]).collect())
        })
        .to_vec();
    // the console's balanced tree of 16-bit additions, each rounded to nearest (ties to even)
    let tree = {
        let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
        let narrow = |f: f32| {
            let b = f.to_bits();
            let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
            let rest = b & 0x1FFF;
            (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
        };
        let mut sums: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
        while sums.len() > 1 {
            sums = sums.chunks(2).map(|c| narrow(wide(c[0]) + wide(c[1]))).collect();
        }
        u32::from(sums[0])
    };
    vec![
        ("lds_and_lanes_across_halves", lds_and_lanes_across_halves(), lds),
        ("loops_apart_meeting_at_the_barrier", loops_apart_meeting_at_the_barrier(), loops),
        ("permute_shift_and_write_lanes", permute_shift_and_write_lanes(), permute),
        ("wave_or_in_a_lane_branch_execz", wave_or_in_a_lane_branch(0xBF88_0000, 64), or()),
        ("wave_or_in_a_lane_branch_scc0", wave_or_in_a_lane_branch(0xBF84_0000, 64), or()),
        ("wave_value_past_a_lane_branch", wave_value_past_a_lane_branch(), past),
        ("sum_of_halves", sum_of_halves(), vec![one([0; 4], 1, vec![tree])]),
    ]
}

/// For a run on the host GPU (the Pro): writes each synthetic program's paired-form container
/// (`<name>.dxil`, one group; `<name>.bindings`: its `HostShader::user_data`, where the root
/// constants s0..s3 and the raw UAV's table go) and what the console leaves
/// (`<name>.expected`: per dispatch a line `constants <s0> <s1> <s2> <s3> dwords <n>`, then
/// the dwords in hex, eight a line) into the directory `PAIRED_DUMP_DIR` names.
/// `PAIRED_DUMP_DIR=<dir> cargo test --locked -p victory-shader --lib dump_for_the_host_gpu -- --ignored`
#[test]
#[ignore = "writes files for a GPU run elsewhere; needs PAIRED_DUMP_DIR"]
fn dump_for_the_host_gpu() {
    use std::fmt::Write as _;
    let dir = std::env::var_os("PAIRED_DUMP_DIR").expect("PAIRED_DUMP_DIR");
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("the directory");
    for (name, native, runs) in console_results() {
        let (shader, _) = paired(&native).unwrap_or_else(|e| panic!("{name}: {e}"));
        std::fs::write(dir.join(format!("{name}.dxil")), &shader.dxil).expect("write");
        let bindings: String = shader.user_data.iter().map(|u| format!("{u:?}\n")).collect();
        std::fs::write(dir.join(format!("{name}.bindings")), bindings).expect("write");
        let mut text = String::new();
        for run in runs {
            let [a, b, c, d] = run.constants;
            assert_eq!(run.want.len(), run.dwords, "{name}");
            writeln!(text, "constants {a:#x} {b:#x} {c:#x} {d:#x} dwords {}", run.dwords).unwrap();
            for row in run.want.chunks(8) {
                let row: Vec<String> = row.iter().map(|w| format!("{w:08x}")).collect();
                writeln!(text, "{}", row.join(" ")).unwrap();
            }
        }
        std::fs::write(dir.join(format!("{name}.expected")), text).expect("write");
    }
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Wave32 D3D12 hardware device and the DXIL validator"]
fn paired_dispatches_match_the_console() {
    use super::host::{Dispatch, Host, Root, View};
    use super::validator::Validator;

    let host = Host::new(false).expect("Wave32 hardware");
    let mut validator = Validator::load(Validator::default_path().as_deref())
        .expect("DXIL validator");
    let mut values = 0;
    for (name, native, cases) in console_results() {
        let (mut shader, _) = paired(&native).unwrap_or_else(|error| panic!("{name}: {error}"));
        validator.validate(&mut shader.dxil).unwrap_or_else(|error| panic!("{name}: {error}"));
        let (root, pipeline) = host.compute_pipeline(&shader).unwrap_or_else(|error| panic!("{name}: {error}"));
        for case in cases {
            let mut constants = vec![0; shader.user_data_dwords().max(1) as usize];
            for slot in &shader.user_data {
                if let (RootType::Constants, 0, index @ 0..=3) = (slot.root, slot.slot, slot.dword) {
                    constants[slot.index as usize] = case.constants[index as usize];
                }
            }
            let buffers = [View::RawUav(case.dwords as u32)];
            let request = Dispatch {
                roots: vec![(0, Root::Constants(constants))], views: &buffers, samplers: &[], groups: [1, 1, 1],
            };
            let got = host.dispatch(&root, &pipeline, &request).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(got.len(), 1, "{name}: expected exactly one UAV");
            assert_eq!(got[0].len(), case.want.len(), "{name}: buffer length");
            if let Some(index) = got[0].iter().zip(&case.want).position(|(got, want)| got != want) {
                panic!("{name} {:?}: dword {index}, got {:#x}, expected {:#x}", case.constants, got[0][index], case.want[index]);
            }
            values += case.want.len();
        }
        eprintln!("{name}: validator and hardware readback passed");
    }
    eprintln!("Verified {values} dwords on the hardware device");
}
