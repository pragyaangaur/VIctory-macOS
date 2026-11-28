use super::*;

pub(in crate::emit) fn fixture(
    width: u32,
    kind: ResourceType,
    mask: &[u32],
    sparse: bool,
) -> NativeProgram {
    let op = 0xf420_0016 | width.trailing_zeros() << 18; // table s[44:47]
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xf408_0b02,
        0xfa00_0020,
        S_WAITCNT_LGKM,
        0x3420_0087, // v16 = 128 * lane
        0xf420_0516,
        0xfa00_0000,
        S_WAITCNT_LGKM, // s20 = trips
    ]);
    let used = |k: u32| !sparse || k == width - 1;
    for k in (0..width).filter(|&k| used(k)) {
        w.push(0xbe80_0380 | (64 + k) << 16); // initialize each accumulator
    }
    w.push(0xbf06_8014); // trips == 0
    let skip = w.len();
    w.push(0);
    let top = w.len();
    w.extend(mask);
    w.extend([0x87fe_6a7e, 0xbf06_8214]); // exec &= vcc; remaining trips == 2
    let select = w.len();
    w.push(0);
    let record = |w: &mut Vec<u32>, byte: u32, tag: u32| {
        w.extend([op | 80 << 6, 0xfa00_0000 | byte, S_WAITCNT_LGKM]); // s[80..]
        for k in (0..width).filter(|&k| used(k)) {
            let word = 80 + k;
            w.push(0x8000_0000 | word << 16 | (128 + tag + k) << 8 | word);
        }
    };
    record(&mut w, 4, 1);
    let join = w.len();
    w.push(0);
    w[select] = 0xbf85_0000 | (w.len() - select - 1) as u32;
    record(&mut w, 84, 17);
    w[join] = 0xbf82_0000 | (w.len() - join - 1) as u32;
    for k in (0..width).filter(|&k| used(k)) {
        let total = 64 + k;
        w.push(0x8000_0000 | total << 16 | (80 + k) << 8 | total);
    }
    w.push(0x7e0c_0214); // active lanes record the remaining trips
    w.extend(store(6, 120));
    w.extend([0xbefe_04c1, 0x8094_8114, 0xbf07_8014]); // full EXEC, --trips != 0
    w.push(0xbf85_0000 | ((top as i32 - w.len() as i32 - 1) as u32 & 0xffff));
    w[skip] = 0xbf85_0000 | (w.len() - skip - 1) as u32;
    for k in (0..width).filter(|&k| used(k)) {
        w.push(0x7e0c_0200 | (64 + k)); // store the corresponding SGPR
        w.extend(store(6, 4 * k));
    }
    w.push(S_ENDPGM);
    let mut n = native(&w, true, 64, 0);
    n.used[1].kind = kind;
    n
}

#[test]
fn literal_publication_wide_dataflow() {
    for kind in [ResourceType::Srv, ResourceType::Cbv] {
        for width in [8, 16] {
            for (mask, _) in MASKS {
                for sparse in [false, true] {
                    let n = fixture(width, kind, mask, sparse);
                    let out = lower_halves(&n, true, true, true).unwrap();
                    assert_eq!(out.wave_size, Some(32));
                    let mut chunks = out.raw_load_args.clone();
                    chunks.sort_unstable();
                    let expected_chunks = std::iter::once((Some(0), Some(1)))
                        .chain([4, 84].into_iter().flat_map(|base| {
                            (0..width / 4).map(move |j| (Some(base + 16 * i64::from(j)), Some(15)))
                        }))
                        .collect::<Vec<_>>();
                    assert_eq!(chunks, expected_chunks);
                    let exchanges = &out.live_scalar_exchanges;
                    assert_eq!(exchanges.len(), (1 + 2 * width) as usize);
                    assert_eq!(exchanges[0].source, (Some(0), 0));
                    assert!(exchanges[0].lower_read);
                    let mut records = exchanges[1..]
                        .chunks_exact(width as usize)
                        .collect::<Vec<_>>();
                    records.sort_by_key(|run| run[0].source.0);
                    for (record, (run, base)) in records.into_iter().zip([4, 84]).enumerate() {
                        for (k, x) in (0..width).zip(run) {
                            assert_eq!(x.source, (Some(base + i64::from(k / 4) * 16), k % 4));
                            let used = !sparse || k == width - 1;
                            assert_eq!(x.lower_read, used);
                            assert_eq!(
                                x.addend,
                                used.then_some(1 + 16 * record as i64 + i64::from(k))
                            );
                        }
                    }
                }
            }
        }
    }
}

const SEEDS: [u32; 16] = [3, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];
const RECORD: [u32; 16] = [
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
];

fn expected_words(width: usize, sparse: bool, mask: u64, trips: u32, cap: u32) -> Vec<u32> {
    let mut rows = [[MARK; 32]; 64];
    let visits = trips.min(cap);
    for (lane, row) in rows.iter_mut().enumerate() {
        for k in 0..width {
            if !sparse || k == width - 1 {
                row[k] = (0..visits)
                    .map(|visit| {
                        if trips - visit == 2 {
                            RECORD[k] + 17 + k as u32
                        } else {
                            SEEDS[k] + 1 + k as u32
                        }
                    })
                    .sum();
            }
        }
        if visits != 0 && mask & (1 << lane) != 0 {
            row[30] = trips - visits + 1;
        }
    }
    rows.concat()
}

#[test]
fn literal_publication_wide_expected_words() {
    let out = expected_words(16, true, 1 << 40, 3, 2);
    assert_eq!(out.len(), 2048);
    assert_eq!(out[15], 270); // 71 + 16 on visit one, 151 + 32 on visit two
    assert_eq!(out[40 * 32 + 30], 2);
    assert_eq!(out.iter().filter(|&&x| x == MARK).count(), 1983);
}

#[cfg(windows)]
#[test]
fn literal_publication_wide_matches_gpu() {
    if !in_bounded_gpu_child(
        concat!(module_path!(), "::literal_publication_wide_matches_gpu")
            .trim_start_matches("victory_shader::"),
    ) {
        return;
    }
    let cap = crate::emit::loop_limit();
    for kind in [ResourceType::Srv, ResourceType::Cbv] {
        for width in [8, 16] {
            for (mask, bits) in MASKS {
                for sparse in [false, true] {
                    let out = lower_halves(&fixture(width, kind, mask, sparse), true, true, true)
                        .unwrap();
                    eprintln!(
                        "cap {cap} {kind:?} x{width} mask {bits:016x} sparse {sparse}: {} bytes, checksum {:02x?}",
                        out.dxil.len(),
                        &out.dxil[4..20]
                    );
                    let forms = [("publication wave32", shader(out))];
                    for trips in [0, 3] {
                        let mut input = vec![trips];
                        input.extend(SEEDS);
                        input.resize(21, MARK);
                        input.extend_from_slice(&RECORD[..width as usize]);
                        let start = vec![MARK; 64 * 32];
                        let views = [
                            View::Texture(1, 1, &[[0.0; 4]]),
                            View::Buffer {
                                data: &input,
                                stride: Stride::Raw,
                            },
                            View::RawUavData(&start),
                        ];
                        let want = expected_words(width as usize, sparse, bits, trips, cap);
                        let (errors, ran) = compare_dispatches(&forms, 1, 32, &want, &views);
                        assert!(ran > 0, "no physical wave32 device");
                        assert!(
                            errors.is_empty(),
                            "{kind:?} x{width}, mask {bits:x}, trips {trips}, sparse {sparse}, cap {cap}: {}",
                            errors.join("\n")
                        );
                    }
                }
            }
        }
    }
}
