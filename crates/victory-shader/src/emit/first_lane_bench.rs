//! Matched GPU completion-time workloads for the embedded CODE first-lane proof.

use super::halves_tests::{
    PROLOGUE, S_ENDPGM, compute_container, lower_halves_limited, native, store,
};
use super::host::{Dispatch, Host, Root, Stride, View};
use super::{Options, translate_with};
use crate::native::{NativeProgram, StageIo};

const THREADS: u32 = 128;
const GROUPS: u32 = 8192;
const WARMUP: usize = 3;
const SAMPLES: usize = 15;
const READS: u32 = 8;

#[derive(Clone, Copy, Debug)]
enum Workload {
    Embedded,
    EightReads,
    ScalarControl,
    VaryingSource,
}

fn program(workload: Workload) -> NativeProgram {
    let mut words = PROLOGUE.to_vec();
    words.extend([
        0x8F02_8701, // s_lshl_b32 s2, s1 (group id X), 7
        0x4A22_0002, // v_add_nc_u32 v17, s2, v0 (global id)
        0x3420_2282, // v_lshlrev_b32 v16, 2, v17 (output byte address)
    ]);
    let pc = words.len();
    words.extend([
        0xBEA0_1F00, // s_getpc_b64 s[32:33]
        0xF404_0390, // s_load_dwordx2 s[14:15], s[32:33], embedded data
        0xFA00_0000,
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]);
    if matches!(workload, Workload::EightReads) {
        words.extend([
            0x7E02_0300, // v_mov_b32 v1, v0
            0x7E08_0280, // v_mov_b32 v4, 0
        ]);
        // Each read feeds a scalar branch and each taken edge contributes
        // to the result. The source grows by the loaded dword at each site.
        for site in 1..=READS {
            words.extend([
                0x4A08_080F, // v_add_nc_u32 v4, s15, v4
                0x7E18_0504, // v_readfirstlane_b32 s12, v4
                0xBF06_FF0C, // s_cmp_eq_u32 s12, literal
                16 * site,
                0xBF84_0001, // s_cbranch_scc0 next
                0x4A02_020C, // v_add_nc_u32 v1, s12, v1
            ]);
        }
    } else {
        words.extend([
            0x7E08_020F, // v_mov_b32 v4, s15
            match workload {
                Workload::ScalarControl => 0xBE8C_030F, // s_mov_b32 s12, s15
                Workload::VaryingSource => 0x7E18_0500, // v_readfirstlane_b32 s12, v0
                _ => 0x7E18_0504,                       // v_readfirstlane_b32 s12, v4
            },
            0x7E02_020C, // v_mov_b32 v1, s12
            0xBF06_900C, // s_cmp_eq_u32 s12, 16
            0xBF84_0001, // s_cbranch_scc0 past
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
        ]);
    }
    words.extend(store(1, 0));
    words.push(S_ENDPGM);
    words[pc + 2] |= ((words.len() - pc - 1) * 4) as u32;
    words.extend([0xABCD_0123, 16]);
    let mut program = native(&words, true, THREADS, 0);
    let StageIo::Cs(cs) = &mut program.io else {
        unreachable!("compute fixture");
    };
    cs.rsrc2 |= 1 << 7;
    program
}

fn expected(workload: Workload, lane: u32) -> u32 {
    match workload {
        Workload::Embedded | Workload::ScalarControl => 17,
        Workload::EightReads => lane + 16 * (1..=READS).sum::<u32>(),
        Workload::VaryingSource => lane / 64 * 64,
    }
}

#[test]
#[ignore = "hardware GPU completion benchmark; run serially with the D3DMetal Wine runtime"]
fn embedded_code_first_lane_gpu_completion() {
    assert!(
        super::host::under_wine(),
        "requires the D3DMetal Wine runtime"
    );
    let hosts = Host::each_hardware();
    let (adapter, host) = hosts.first().expect("no hardware D3D12 device with SM 6.6");
    for workload in [
        Workload::Embedded,
        Workload::EightReads,
        Workload::ScalarControl,
        Workload::VaryingSource,
    ] {
        let program = program(workload);
        let bytes = compute_container(&program);
        let lowered =
            lower_halves_limited(&program, true, true, true, false).expect("lowering statistics");
        let translation_start = std::time::Instant::now();
        let shader = translate_with(
            &bytes,
            Options {
                host_wave32: true,
                narrow_waves: true,
                halves_rendezvous: true,
                ..Default::default()
            },
        )
        .expect("production translation");
        let translation_us = translation_start.elapsed().as_micros();
        let (root, pso) = host
            .compute_pipeline(&shader)
            .expect("GPU pipeline creation");
        eprintln!(
            "setup {workload:?} adapter={adapter:?} wave_size={:?} half_mask={} group_barriers={} dxil_bytes={} translation_us={translation_us}",
            lowered.wave_size,
            lowered.has_half_wave_mask,
            lowered.barriers,
            shader.dxil.len(),
        );
        let bindings = Dispatch {
            roots: vec![(
                0,
                Root::Constants(vec![0; shader.user_data_dwords() as usize]),
            )],
            views: &[
                View::Texture(1, 1, &[[0.0; 4]]),
                View::Buffer {
                    data: &[0; 4],
                    stride: Stride::Structured(16),
                },
                View::RawUav(GROUPS * THREADS),
            ],
            samplers: &[],
            groups: [GROUPS, 1, 1],
        };
        for sample in 0..WARMUP + SAMPLES {
            let (outputs, duration) = host
                .dispatch_timed(&root, &pso, &bindings)
                .unwrap_or_else(|error| panic!("{workload:?} sample {sample}: {error}"));
            assert_eq!(outputs.len(), 1, "{workload:?} UAV count");
            let output = &outputs[0];
            assert_eq!(output.len(), (GROUPS * THREADS) as usize);
            for (thread, &word) in output.iter().enumerate() {
                assert_eq!(
                    word,
                    expected(workload, thread as u32 % THREADS),
                    "{workload:?} sample {sample}, thread {thread}",
                );
            }
            if sample >= WARMUP {
                eprintln!(
                    "sample {workload:?} {} {}",
                    sample - WARMUP,
                    duration.as_nanos(),
                );
            }
        }
    }
}
