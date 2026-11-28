use super::*;

pub(in crate::emit) fn program(mask: &[u32]) -> Vec<u32> {
    publication(mask, None)
}

fn publication(mask: &[u32], cbv_width: Option<u32>) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xf408_0b02,
        0xfa00_0020,
        S_WAITCNT_LGKM, // table buffer descriptor s[44:47]
        0x3420_0085,    // v16 = 32 * lane
    ]);
    if cbv_width.is_some() {
        w.push(0xbe9c_0382); // enter the inner loop twice
    }
    let entry = w.len();
    w.extend([
        0xf420_0516,
        0xfa00_0000,
        S_WAITCNT_LGKM, // s20 = trips
    ]);
    let width_op = cbv_width.map_or(1, u32::trailing_zeros) << 18;
    if cbv_width.is_some() {
        w.extend([
            0xf420_0896 | width_op,
            0xfa00_0010,
            S_WAITCNT_LGKM, // s[34..] = initial markers, immediate byte offset 16
            0xbe98_03a0,    // s24 = first visit record's byte offset, 32
        ]);
    } else {
        w.extend([
            0xbe98_0388, // s24 = first pair's byte offset, 8
            0xbe9a_0380, // s26 = sum of visit markers
        ]);
    }
    w.push(0xbf06_8014); // trips == 0
    let zero = w.len();
    w.push(0);
    let top = w.len();
    w.extend(mask);
    w.extend([
        0x87fe_6a7e, // exec &= vcc
        0xf420_0796 | width_op,
        0x3000_0000,
        S_WAITCNT_LGKM, // visit record at s24, even if EXEC is empty
    ]);
    if let Some(width) = cbv_width {
        for k in 0..width {
            let total = 34 + k;
            let word = 30 + k;
            w.push(0x8000_0000 | total << 16 | word << 8 | total); // componentwise sum
        }
    } else {
        w.push(0x801a_1f1a); // s26 += marker
    }
    w.push(0xbf06_801e); // A zero width skips this reservation.
    let skip_width = w.len();
    w.push(0);
    let empty = w.len();
    w.push(0);
    w.extend([
        0xbe8c_147e, // s12 = first active lane
        0xbe8d_107e, // s13 = active count
        0x930d_1e0d, // total = count * width
        0xd765_000a,
        0x0001_007e, // v10 = rank in EXEC
        0xd766_000a,
        0x0002_147f,
        0x7d84_000c, // vcc = lane == leader
        0xbe96_246a, // save mask and elect leader
        0x7e0c_020d, // v6 = total
        0xe0c8_401c,
        0x800a_0600, // reserve total, returning old counter
        0xbf8c_3f70, // wait for VMEM result
        0x7e1e_0506, // s15 = leader's returned base
        0xbefe_0416, // restore work mask
        0xd569_000a,
        0x0200_3d0a, // v10 *= width
        0x4a14_140f, // v10 += base
    ]);
    w.extend(store(10, 4));
    w[skip_width] = 0xbf85_0000 | (w.len() - skip_width - 1) as u32;
    w[empty] = 0xbf88_0000 | (w.len() - empty - 1) as u32;
    w.extend([
        0xbefe_04c1, // restore full EXEC
        if cbv_width.is_some() {
            0x8018_9018
        } else {
            0x8018_8818
        }, // step to next record
        0x8094_8114, // --trips
        0xbf07_8014, // trips != 0
    ]);
    w.push(0xbf85_0000 | ((top as i32 - w.len() as i32 - 1) as u32 & 0xffff));
    w[zero] = 0xbf85_0000 | (w.len() - zero - 1) as u32;
    if let Some(width) = cbv_width {
        for (k, slot) in (0..width).zip(MARK_SLOTS) {
            w.push(0x7e0c_0200 | (34 + k)); // v6 = sum for this component
            w.extend(store(6, slot as u32 * 4));
        }
        w.extend([0x809c_811c, 0xbf07_801c]); // --s28 != 0
        w.push(0xbf85_0000 | ((entry as i32 - w.len() as i32 - 1) as u32 & 0xffff));
    } else {
        w.push(0x7e0c_021a); // v6 = markers visited (also under empty EXEC)
        w.extend(store(6, 0));
    }
    w.push(S_ENDPGM);
    w
}

#[test]
fn literal_publication_captures_each_visit() {
    let words = program(&MASK_LANES_40_41);
    let lower =
        |words: &[u32]| lower_halves(&native(words, true, 64, 0), true, true, true).unwrap();
    let split = lower(&words);
    assert_eq!(
        live_sources(&split),
        [(Some(0), 0), (None, 0), (None, 1)],
        "live lower-half words"
    );
    let mut fixed = words.clone();
    fixed[5..7].fill(0xbf80_0000);
    let count = fixed.iter().position(|&w| w == 0xf420_0516).unwrap();
    fixed[count..count + 2].copy_from_slice(&[0xbe94_0383, 0xbf80_0000]);
    let pair = fixed.iter().position(|&w| w == 0xf424_0796).unwrap();
    fixed[pair..pair + 2].copy_from_slice(&[0xbe9e_0383, 0xbe9f_0382]);
    assert_eq!(
        split.barriers,
        lower(&fixed).barriers + 6,
        "three scalar dword exchanges"
    );
}

fn fixed_table_cbv(mask: &[u32], width: u32) -> NativeProgram {
    let mut n = native(&publication(mask, Some(width)), true, 64, 0);
    n.used[1].kind = ResourceType::Cbv;
    n
}

#[test]
fn literal_publication_fixed_table_cbv() {
    for width in [1, 2, 4] {
        for (lanes, _) in MASKS {
            let n = fixed_table_cbv(lanes, width);
            let out = lower_halves(&n, true, true, true).unwrap();
            let components = std::iter::once((Some(0), 0))
                .chain((0..width).map(|k| (Some(16), k)))
                .chain((0..width).map(|k| (None, k)));
            assert_eq!(live_sources(&out), components.collect::<Vec<_>>());
            let mask = Some(i64::from((1 << width) - 1));
            assert_eq!(
                out.raw_load_args,
                [(Some(0), Some(1)), (Some(16), mask), (None, mask)]
            );
            assert_eq!(out.wave_size, Some(32));
        }
    }
}

const MASKS: [(&[u32], u64); 5] = [
    (&[0x7d84_0100], u64::MAX), // v_cmp_eq_u32 vcc, v0, v0
    (&MASK_LANES_8_9, 3 << 8),
    (&MASK_LANES_40_41, 3 << 40),
    (&MASK_LANES_12_44, 1 << 12 | 1 << 44),
    (&MASK_EMPTY, 0),
];

const MARK: u32 = 0x1357_9bdf;
const PAIRS: [(u32, u32); 3] = [(3, 19), (0, 31), (5, 43)];

const CBV_INITIAL: [u32; 4] = [13, 23, 37, 47];
const CBV_RECORDS: [[u32; 4]; 3] = [[3, 19, 59, 67], [7, 31, 73, 89], [5, 43, 97, 113]];
const MARK_SLOTS: [usize; 4] = [0, 2, 3, 4];

fn expected(
    active: u64,
    trips: usize,
    entries: usize,
    limit: usize,
    cbv_width: Option<usize>,
) -> Vec<u32> {
    let mut rows = [[MARK; 8]; 64];
    let mut counter = 11;
    for _ in 0..entries.min(limit) {
        let mut totals = if cbv_width.is_some() {
            CBV_INITIAL
        } else {
            [0; 4]
        };
        for i in 0..trips.min(limit) {
            let (width, record) = if cbv_width.is_some() {
                let record = CBV_RECORDS[i];
                (record[0], record)
            } else {
                let (width, marker) = PAIRS[i];
                (width, [width, marker, 0, 0])
            };
            for (total, value) in totals.iter_mut().zip(record) {
                *total += value;
            }
            for (lane, row) in rows.iter_mut().enumerate() {
                if width > 0 && active & (1 << lane) != 0 {
                    row[1] = counter;
                    counter += width;
                }
            }
        }
        for row in &mut rows {
            if let Some(width) = cbv_width {
                for (slot, total) in MARK_SLOTS.into_iter().zip(totals).take(width) {
                    row[slot] = total;
                }
            } else {
                row[0] = totals[1];
            }
        }
    }
    rows[0][7] = counter;
    rows.concat()
}

#[test]
fn literal_publication_trace_is_per_visit() {
    // Each tuple is one allowed observation by each half on that visit.
    // A reservation runs on the nonzero branch. Only one common snapshot
    // per visit follows one legal wave trace throughout.
    let reads = [(0, 3), (3, 0), (0, 5)];
    let trace = |select: fn(usize, (u32, u32)) -> (u32, u32)| {
        let mut trace = [Vec::new(), Vec::new()];
        for (i, &pair) in reads.iter().enumerate() {
            let (lo, hi) = select(i, pair);
            trace[0].push(lo != 0);
            trace[1].push(hi != 0);
        }
        trace
    };
    let wanted = vec![false, true, false];
    assert_eq!(trace(|_, (lo, _)| (lo, lo)), [wanted.clone(), wanted]);
    for [lo, hi] in [
        trace(|_, pair| pair),
        trace(|i, (lo, hi)| (lo, if i == 0 { lo } else { hi })),
    ] {
        assert_ne!(lo, hi);
    }
    let upper = expected(3 << 40, 1, 1, 8, None);
    assert_eq!(
        [upper[40 * 8 + 1], upper[41 * 8 + 1], upper[7]],
        [11, 14, 17]
    );
    let empty = expected(0, 3, 2, 1, None);
    assert_eq!((empty[7], empty[0], empty[1]), (11, 19, MARK));
    assert_eq!(expected(3 << 40, 3, 2, 2, None)[7], 23);
    let cbv = expected(3 << 40, 3, 2, 2, Some(4));
    assert_eq!(cbv[..8], [23, MARK, 73, 169, 203, MARK, MARK, 51]);
    assert_eq!(cbv[40 * 8 + 1], 37);
    assert_eq!(cbv[41 * 8 + 1], 44);
}

#[cfg(windows)]
#[test]
fn literal_publication_matches_gpu() {
    if !in_bounded_gpu_child("emit::halves_tests::literal_publish::literal_publication_matches_gpu")
    {
        return;
    }
    for (mask, bits) in MASKS {
        let mut words = program(mask);
        let entry = words.iter().position(|&w| w == 0xf420_0516).unwrap();
        words.insert(entry, 0xbe9c_0382); // two separate entries into the inner loop
        words.pop();
        words.extend([0x809c_811c, 0xbf07_801c]); // --s28 != 0
        words.push(0xbf85_0000 | ((entry as i32 - words.len() as i32) as u32 & 0xffff));
        words.push(S_ENDPGM);
        let n = native(&words, true, 64, 0);
        let forms = [(
            "publication wave32",
            shader(lower_halves(&n, true, true, true).unwrap()),
        )];
        for trips in [0, 1, 3] {
            let mut input = [0u32; 8];
            input[0] = trips as u32;
            for (pair, (width, marker)) in input[2..].as_chunks_mut::<2>().0.iter_mut().zip(PAIRS) {
                pair.copy_from_slice(&[width, marker]);
            }
            let mut start = vec![MARK; 64 * 8];
            start[7] = 11;
            let views = [
                View::Texture(1, 1, &[[0.0; 4]]),
                View::Buffer {
                    data: &input,
                    stride: Stride::Raw,
                },
                View::RawUavData(&start),
            ];
            let want = expected(bits, trips, 2, super::super::loop_limit() as usize, None);
            let (errors, ran) = compare_dispatches(&forms, 1, 8, &want, &views);
            assert!(ran > 0, "no physical wave32 device");
            assert!(
                errors.is_empty(),
                "mask {bits:#x}, trips {trips}: {}",
                errors.join("\n")
            );
        }
    }
}

#[cfg(windows)]
#[test]
fn literal_publication_fixed_table_cbv_matches_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::literal_publish::literal_publication_fixed_table_cbv_matches_gpu",
    ) {
        return;
    }
    let limit = super::super::loop_limit() as usize;
    for width in [1, 2, 4] {
        for (mask, bits) in MASKS {
            let n = fixed_table_cbv(mask, width);
            let out = lower_halves(&n, true, true, true).unwrap();
            eprintln!(
                "cap {limit} CBV x{width} mask {bits:#018x}: {} bytes, checksum {:02x?}",
                out.dxil.len(),
                &out.dxil[4..20]
            );
            let forms = [("publication wave32", shader(out))];
            for trips in [0, 1, 3] {
                let mut input = vec![trips as u32, 0, 0, 0];
                input.extend(CBV_INITIAL);
                input.extend(CBV_RECORDS.into_iter().flatten());
                let mut start = vec![MARK; 64 * 8];
                start[7] = 11;
                // The guest declares a CBV, but the runtime binds a raw SRV.
                let views = [
                    View::Texture(1, 1, &[[0.0; 4]]),
                    View::Buffer {
                        data: &input,
                        stride: Stride::Raw,
                    },
                    View::RawUavData(&start),
                ];
                let want = expected(bits, trips, 2, limit, Some(width as usize));
                let (errors, ran) = compare_dispatches(&forms, 1, 8, &want, &views);
                assert!(ran > 0, "no physical wave32 device");
                assert!(
                    errors.is_empty(),
                    "CBV x{width}, mask {bits:#x}, trips {trips}, cap {limit}: {}",
                    errors.join("\n")
                );
            }
        }
    }
}

/// Exercise the final literal IR directly, including rejected geometries
/// which the normal retry dispatcher already excludes.
fn refused(
    n: &NativeProgram,
    change: impl FnOnce(&mut ir::Program, &mut structure::Structured, &mut resource::Resources),
    reason: &str,
) {
    let mut p = ir::build(&n.code, true, n.vgprs).unwrap();
    let mut st = structure::split_literal(&p).unwrap();
    let mut res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    change(&mut p, &mut st, &mut res);
    let io = io::raise(&p, n).unwrap();
    let err = lower::lower(&parts(n, &p, &st, &res, &io, true, true))
        .err()
        .expect("unsafe publication accepted");
    assert!(err.to_string().contains(reason), "{err}");
}

#[test]
fn literal_publication_requires_full_wave_and_exact_source() {
    let words = program(&MASK_LANES_40_41);
    let geometry = "one ticketed wave64 and zero guest LDS";
    for (threads, lds) in [(48, 0), (128, 0), (64, 512)] {
        refused(&native(&words, true, threads, lds), |_, _, _| {}, geometry);
    }
    let n = native(&words, true, 64, 0);
    super::super::with_thread_ticket(false, || refused(&n, |_, _, _| {}, geometry));
    let pair = words.iter().position(|&w| w == 0xf424_0796).unwrap();
    let mut ordinary = words.clone();
    ordinary[pair] = 0xf404_0782; // s_load_dwordx2 s[30:31], s[4:5], s24
    refused(
        &native(&ordinary, true, 64, 0),
        |_, _, _| {},
        "unsupported source or width",
    );
    let mut descriptor_data = words.clone();
    let out = descriptor_data
        .iter()
        .position(|&w| w == 0x7e0c_021a)
        .unwrap();
    descriptor_data[out] = 0x7e0c_022c; // v6 = numeric word zero of the SRV
    refused(
        &native(&descriptor_data, true, 64, 0),
        |_, _, _| {},
        "descriptor word read as numeric data",
    );
    for bit in [14, 16] {
        let mut changed = words.clone();
        changed[pair] |= 1 << bit;
        refused(
            &native(&changed, true, 64, 0),
            |_, _, _| {},
            "cached or predicated",
        );
    }
    let mut wide = words.clone();
    wide[pair] = 0xf42c_0796; // x8
    refused(
        &native(&wide, true, 64, 0),
        |_, _, _| {},
        "unsupported source or width",
    );
    let mut interior = n.clone();
    interior.used[1].table_offset_dwords -= 1;
    refused(&interior, |_, _, _| {}, "unsupported source or width");
    let mut uav = n.clone();
    uav.used[1].kind = ResourceType::Uav;
    refused(&uav, |_, _, _| {}, "unsupported source or width");
    for (which, reason) in [
        (0, "a resource waterfall"),
        (1, "cached or predicated"),
        (2, "unsupported source or width"),
    ] {
        refused(
            &n,
            |p, st, res| {
                let desc = res
                    .descriptors
                    .iter_mut()
                    .find(|d| p.blocks[d.site.0].insts[d.site.1].name() == "s_buffer_load_dwordx2")
                    .unwrap();
                let (b, i) = desc.site;
                match which {
                    0 => res.waterfall.loads.push(desc.site),
                    1 => st.guards[b][i] = Some(structure::Bool::FALSE),
                    _ => desc.source = resource::Source::Constant(vec![0; 4]),
                }
            },
            reason,
        );
    }
}

#[test]
fn literal_publication_requires_initialized_phi_and_common_edges() {
    let words = program(&MASK_LANES_40_41);
    let init = words.iter().position(|&w| w == 0xbe98_0388).unwrap();
    for source in [0xbf80_0000, 0xb918_0004] {
        // missing phi entry, hardware state as entry offset
        let mut changed = words.clone();
        changed[init] = source;
        refused(
            &native(&changed, true, 64, 0),
            |_, _, _| {},
            "without common initialized operands",
        );
    }

    let unknown = 0xb909_0004; // hardware state may differ by half
    let compare = 0xbf06_8009;
    for (tested, branch) in [
        (0xbf06_8014, None),              // one half can leave before the loop
        (0xbf07_8014, None),              // divergent counter histories at the latch
        (0xbf06_801e, Some(0xbf85_0001)), // two edges join after a single NOP
    ] {
        let mut changed = words.clone();
        let i = changed.iter().position(|&w| w == tested).unwrap();
        // Replace a same-size compare pair; keep branch destinations intact.
        changed[i] = compare;
        changed[init + 1] = unknown; // overwrites the marker sum initialization
        if let Some(branch) = branch {
            changed[i + 1] = branch;
            changed[i + 2] = 0xbf80_0000;
        }
        refused(
            &native(&changed, true, 64, 0),
            |_, _, _| {},
            "branch without common group control flow",
        );
    }
}

#[test]
fn literal_publication_cbv_requires_all_descriptor_and_destination_dwords() {
    let mut words = program(&MASK_LANES_40_41);
    let pair = words.iter().position(|&w| w == 0xf424_0796).unwrap();
    words[pair] = 0xf428_0796;
    let mut n = native(&words, true, 64, 0);
    n.used[1].kind = ResourceType::Cbv;
    for (reg, op, reason) in [
        (47, "s_load_dwordx4", "unsupported source or width"),
        (33, "s_buffer_load_dwordx4", "guarded or incomplete"),
    ] {
        refused(
            &n,
            |p, _, _| {
                let reg = ir::Reg::S(reg);
                let def = p
                    .blocks
                    .iter_mut()
                    .flat_map(|b| &mut b.insts)
                    .find(|i| i.name() == op && i.write(reg).is_some())
                    .unwrap();
                def.writes.retain(|&(r, _)| r != reg);
            },
            reason,
        );
    }
}

#[test]
fn literal_publication_four_dwords_include_unused_components() {
    let mut words = program(&MASK_LANES_40_41);
    let pair = words.iter().position(|&w| w == 0xf424_0796).unwrap();
    words[pair] = 0xf428_0796;
    let unused = words.clone();
    // Give the additional components real consumers without changing the CFG.
    words[pair + 3] = 0x801a_201f; // s26 = s31 + s32
    let add = words.iter().position(|&w| w == 0x8018_8818).unwrap();
    words.splice(add..add, [0x801a_211a]); // s26 += s33
    // One added instruction between each forward skip and the latch.
    let back = words
        .iter_mut()
        .find(|w| **w & 0xffff_8000 == 0xbf85_8000)
        .unwrap();
    *back -= 1;
    let zero = words.iter_mut().find(|w| **w >> 16 == 0xbf85).unwrap();
    *zero += 1;
    for kind in [ResourceType::Srv, ResourceType::Cbv] {
        let lower = |words: &[u32]| {
            let mut n = native(words, true, 64, 0);
            n.used[1].kind = kind;
            lower_halves(&n, true, true, true).unwrap()
        };
        let used = lower(&words);
        assert_eq!(
            live_sources(&used),
            [(Some(0), 0), (None, 0), (None, 1), (None, 2), (None, 3)]
        );
        assert_eq!(
            used.barriers,
            lower(&unused).barriers,
            "unused dwords must still publish"
        );
    }
}

pub(in crate::emit) mod wide;
