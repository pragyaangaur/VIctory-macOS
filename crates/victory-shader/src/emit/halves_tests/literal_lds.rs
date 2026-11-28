use super::*;

/// Each lane initializes an adjacent pair. Every visit reads its partner's pair,
/// overwrites its own, then lanes 0 and 32 successively overwrite pair zero.
pub(in crate::emit) fn paired(threads: [u32; 3], bytes: u32) -> NativeProgram {
    let valu = |op: u32, dst: u32, src: u32, v: u32| op | dst << 17 | v << 9 | src;
    let add = |dst, src, v| valu(0x4a00_0000, dst, src, v);
    let shift = |dst, v, bits| valu(0x3400_0000, dst, 128 + bits, v);
    let read = |dst: u32, addr| [0xd8dc_0100, dst << 24 | addr, S_WAITCNT_LGKM];
    let write = |addr: u32| [0xd838_0100, 7 << 16 | 6 << 8 | addr, S_WAITCNT_LGKM];
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xf408_0b02,
        0xfa00_0020,
        S_WAITCNT_LGKM, // input table descriptor
        shift(8, 1, threads[0].trailing_zeros()),
        add(8, 256, 8), // flat logical lane = x + y * size_x
        shift(16, 8, 5),
        shift(2, 8, 3),
        valu(0x3a00_0000, 4, 160, 8),
        shift(4, 4, 3), // partner byte address
        0x7e0a_0280,    // v5 = pair zero's byte address
        add(6, 130, 8),
        add(7, 180, 8),
    ]);
    w.extend(write(2));
    w.extend([0xbe94_0383, 0xbe98_0380]); // three visits, byte offset zero
    let top = w.len();
    w.extend([
        0xf424_0796,
        0x3000_0000,
        S_WAITCNT_LGKM, // two new scalar words each visit
    ]);
    w.extend(read(10, 4));
    w.extend([add(6, 30, 10), add(7, 31, 11)]);
    w.extend(write(2));
    w.push(0xbefe_0481); // EXEC = bit 0
    w.extend(write(2));
    w.extend([0xbefe_0480, 0xbeff_0381]); // EXEC = bit 32
    w.extend(write(4));
    w.push(0xbefe_04c1); // restore every lane
    w.extend(read(14, 5));
    for (i, reg) in [10, 11, 6, 7, 14, 15].into_iter().enumerate() {
        w.extend(store(reg, 4 * i as u32));
    }
    w.extend([0x8018_8818, 0x8094_8114, 0xbf07_8014]); // offset += 8; --trips != 0
    w.push(0xbf85_0000 | (top.wrapping_sub(w.len() + 1) as u32 & 0xffff));
    w.push(S_ENDPGM);
    let mut n = native(&w, true, 64, bytes);
    n.used[1].kind = ResourceType::Cbv;
    let StageIo::Cs(cs) = &mut n.io else {
        unreachable!()
    };
    cs.threads = threads;
    cs.rsrc2 |= 1 << 11; // the guest requests both x and y
    n
}

/// Walk the written CFG including both arms of every per-lane store. A lane
/// can skip either component, but cannot skip a phase fence. Publication
/// fences are absent from this trace.
fn check_phases(out: &lower::Lowered) {
    use super::super::bitcode::lds_tests::GuestEvent::*;
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    let mut pending = vec![(0, None)];
    let mut barriers = [None; 6];
    let mut components: [HashSet<_>; 6] = Default::default();
    while let Some((b, mut phase)) = pending.pop() {
        if !seen.insert((b, phase)) {
            continue;
        }
        for (i, event) in out.guest_lds.events[b].iter().enumerate() {
            match event {
                Fence => {
                    let next = match phase {
                        None => 0,
                        Some(5) => 1,
                        Some(n) => n + 1,
                    };
                    let expected = barriers[next].get_or_insert((b, i));
                    assert_eq!(*expected, (b, i), "a path missed a guest phase fence");
                    phase = Some(next);
                }
                Load | Store => {
                    let n = phase.expect("a guest phase fence before the access");
                    assert_eq!(matches!(event, Load), matches!(n, 1 | 5));
                    components[n].insert((b, i));
                }
            }
        }
        let succ = &out.barrier_cfg.0[b];
        if succ.is_empty() {
            assert_eq!(phase, Some(5), "exit before the final LDS read");
        }
        pending.extend(succ.iter().map(|&s| (s, phase)));
    }
    assert_eq!(
        barriers.into_iter().flatten().collect::<HashSet<_>>().len(),
        6
    );
    assert_eq!(components.map(|c| c.len()), [2; 6]);
}

#[test]
fn literal_lds_pairs_have_ordered_phases() {
    for threads in [[8, 8, 1], [64, 1, 1]] {
        for bytes in [512, 32_256] {
            let out = lower_halves(&paired(threads, bytes), true, true, true).unwrap();
            if bytes == 512 {
                eprintln!(
                    "paired LDS {threads:?}, cap {}, {} bytes, DXBC {:02x?}",
                    super::super::loop_limit(),
                    out.dxil.len(),
                    &out.dxil[4..20]
                );
            }
            assert_eq!(out.wave_size, Some(32));
            assert_eq!(out.split_barriers, 6);
            assert_eq!(live_sources(&out), [(None, 0), (None, 1)]);
            let mut globals = out.guest_lds.globals.clone();
            globals.sort();
            assert_eq!(
                globals,
                [
                    ("half_words".into(), 2),
                    ("lds".into(), bytes / 4),
                    ("wave_ticket".into(), 1)
                ]
            );
            check_phases(&out);
        }
    }
}

const RECORDS: [[u32; 2]; 3] = [[7, 19], [3, 23], [17, 31]];
const SENTINEL: u32 = 0xa5c3_9b17;

fn serial(visits: usize) -> Vec<u32> {
    let mut output = [[SENTINEL; 8]; 64];
    let mut slots: [[u32; 2]; 64] = std::array::from_fn(|l| [l as u32 + 2, l as u32 + 52]);
    for record in RECORDS.into_iter().take(visits) {
        let partner: [[u32; 2]; 64] = std::array::from_fn(|l| slots[l ^ 32]);
        let own: [[u32; 2]; 64] = partner.map(|p| [p[0] + record[0], p[1] + record[1]]);
        slots = own;
        for (mask, address_xor) in [(1u64, 0), (1u64 << 32, 32)] {
            for (l, value) in own.iter().enumerate() {
                if mask & (1 << l) != 0 {
                    slots[l ^ address_xor] = *value;
                }
            }
        }
        for (l, row) in output.iter_mut().enumerate() {
            row[..2].copy_from_slice(&partner[l]);
            row[2..4].copy_from_slice(&own[l]);
            row[4..6].copy_from_slice(&slots[0]);
        }
    }
    output.concat()
}

#[test]
fn literal_lds_serial_opposite_writers() {
    let one = serial(1);
    assert_eq!(&one[..8], &[34, 84, 41, 103, 9, 71, SENTINEL, SENTINEL]);
    let two = serial(2);
    assert_eq!(
        &two[32 * 8..33 * 8],
        &[9, 71, 12, 94, 12, 94, SENTINEL, SENTINEL]
    );
    assert_ne!(two, serial(3));
}

#[cfg(windows)]
#[test]
fn literal_lds_pairs_match_gpu() {
    if !in_bounded_gpu_child("emit::halves_tests::literal_lds::literal_lds_pairs_match_gpu") {
        return;
    }
    let input = RECORDS.concat();
    let start = vec![SENTINEL; 64 * 8];
    let views = [
        View::Texture(1, 1, &[[0.; 4]]),
        View::Buffer {
            data: &input,
            stride: Stride::Raw,
        },
        View::RawUavData(&start),
    ];
    let limit = super::super::loop_limit();
    let want = serial(limit as usize);
    for threads in [[8, 8, 1], [64, 1, 1]] {
        let out = lower_halves(&paired(threads, 512), true, true, true).unwrap();
        eprintln!(
            "paired LDS {threads:?}, cap {limit}, {} bytes, DXBC {:02x?}",
            out.dxil.len(),
            &out.dxil[4..20]
        );
        let (errors, ran) =
            compare_dispatches(&[("publication wave32", shader(out))], 1, 8, &want, &views);
        assert!(ran > 0, "no physical wave32 device");
        assert!(
            errors.is_empty(),
            "{threads:?} cap {limit}: {}",
            errors.join("\n")
        );
        eprintln!("matched {} words (128 untouched) {want:?}", want.len());
    }
}
