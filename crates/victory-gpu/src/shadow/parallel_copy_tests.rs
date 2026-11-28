use super::*;
use crate::backend::HostGpu;

const COPY_LEN: usize = 2 * PAGE;
const INITIAL: u8 = 0x19;

#[derive(Debug)]
enum Layout {
    Adjacent,
    Overlap,
    Mirrored,
    SharesPage,
}

struct GuestMemory {
    base: usize,
    mirror: Option<usize>,
}

impl GuestMemory {
    fn pool() -> Self {
        Self {
            base: crate::watch::tests::pool(5),
            mirror: None,
        }
    }

    fn mirrored() -> Self {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };
        // SAFETY: one unnamed pagefile allocation, viewed twice for this fixture.
        let (base, mirror) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                COPY_LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, COPY_LEN).Value as usize;
            let mirror = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, COPY_LEN).Value as usize;
            assert_ne!(CloseHandle(section), 0);
            (base, mirror)
        };
        assert!(base != 0 && mirror != 0 && base != mirror);
        for offset in (0..COPY_LEN).step_by(PAGE) {
            crate::watch::title_memory_aliased(mirror + offset, &[base + offset]);
        }
        Self {
            base,
            mirror: Some(mirror),
        }
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        if let Some(mirror) = self.mirror {
            for address in [self.base, mirror] {
                crate::watch::title_memory_remapped(address, COPY_LEN);
                assert_ne!(
                    // SAFETY: the copy trackers are gone and these views belong to the fixture.
                    unsafe {
                        UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: address as *mut _,
                        })
                    },
                    0
                );
            }
        } else {
            crate::watch::tests::free(self.base);
        }
    }
}

fn fill_title(at: usize, len: usize, byte: u8) {
    let _writing = crate::watch::writing(at, len);
    // SAFETY: a writable range of this fixture's guest memory.
    unsafe { (at as *mut u8).write_bytes(byte, len) };
}

fn exercise(layout: Layout) -> (bool, bool) {
    let host = host().expect("the marked-copy fixture requires a D3D12 host");
    let _quiet = host.quiet();
    let memory = match layout {
        Layout::Mirrored => GuestMemory::mirrored(),
        _ => GuestMemory::pool(),
    };
    let copy_len = match layout {
        Layout::SharesPage => COPY_LEN + PAGE / 2,
        _ => COPY_LEN,
    };
    let second_offset = match layout {
        Layout::Adjacent | Layout::SharesPage => copy_len,
        Layout::Overlap => PAGE,
        Layout::Mirrored => 0,
    };
    let physical_offsets = [0, second_offset];
    let starts = [
        memory.base,
        memory.mirror.unwrap_or(memory.base + second_offset),
    ];
    let len = second_offset + copy_len;
    fill_title(memory.base, len, INITIAL);
    let copies = starts.map(|at| {
        let (buffer, data) = host
            .shared_buffer(copy_len as u64)
            .expect("a mapped host buffer");
        // SAFETY: a live host buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu, at as u64, copy_len as u64);
        assert_eq!(copy.update(at, copy_len), copy_len);
        copy.writing(at, copy_len, true);
        copy
    });
    let marked = copies.each_ref().map(|copy| {
        copy.marks
            .expect("fault tracking and marks must be enabled")
    });
    assert_ne!(marked[0], marked[1]);
    let disjoint = ranges_disjoint(&copies.each_ref().map(|copy| copy.range()));
    assert_eq!(disjoint, !matches!(layout, Layout::Overlap));
    if matches!(layout, Layout::Mirrored) {
        assert!(
            starts
                .into_iter()
                .all(|at| crate::watch::aliased(at, COPY_LEN))
        );
    }

    let mut expected = vec![INITIAL; len];
    for (i, &at) in starts.iter().enumerate() {
        // For SharesPage, leave the shared page initially clean so only a
        // worker's own copy-back can mark the other tracker changed there.
        let offset = match (&layout, i) {
            (Layout::Overlap, 0) | (Layout::SharesPage, 1) => PAGE + 300,
            _ => 300,
        };
        fill_title(at + offset, 9, 0x61 + i as u8);
        let physical = physical_offsets[i] + offset;
        expected[physical..physical + 9].fill(0x61 + i as u8);
    }
    let first_patch_page = match layout {
        Layout::Overlap => PAGE,
        Layout::SharesPage => COPY_LEN,
        _ => 0,
    };
    let patches = [
        [(first_patch_page + 16, 0xa1), (first_patch_page + 32, 0xa2)],
        [(16, 0xb1), (64, 0xb2)],
    ];
    for i in [0, 1] {
        for (offset, byte) in patches[i] {
            copies[i].write_host((starts[i] + offset) as u64, &[byte; 11]);
        }
    }
    let mut order = [0, 1];
    order.sort_unstable_by_key(|&i| marked[i]);
    for i in order {
        for (offset, byte) in patches[i] {
            let physical = physical_offsets[i] + offset;
            expected[physical..physical + 11].fill(byte);
        }
    }

    let bits = bitmap(false).expect("a mapped write-mark bitmap").data as *mut u32;
    let marks = marked
        .into_iter()
        .zip(starts)
        .flat_map(|(first, at)| first..first + pages(at as u64, copy_len as u64));
    for mark in marks.clone() {
        // SAFETY: mapped bitmap, only this test submits work in this process.
        unsafe { *bits.add(mark as usize / 32) |= 1 << (mark % 32) };
    }
    MARKED_COPY_BACK_PARALLEL.with(|taken| taken.set(None));
    let started = std::time::Instant::now();
    copy_back_marks(&[], false);
    let elapsed_ns = started.elapsed().as_nanos();
    let parallel =
        MARKED_COPY_BACK_PARALLEL.with(|taken| taken.take().expect("marked copy-back ran"));
    for mark in marks {
        assert_eq!(
            // SAFETY: the same mapped bitmap.
            unsafe { *bits.add(mark as usize / 32) } & (1 << (mark % 32)),
            0
        );
    }
    // SAFETY: waits above finished, and the guest mapping is readable for this length.
    let actual = unsafe { std::slice::from_raw_parts(memory.base as *const u8, len) };
    let mismatch = actual
        .iter()
        .zip(&expected)
        .position(|(actual, expected)| actual != expected);
    if let Some(mirror) = memory.mirror {
        // SAFETY: a second view of the same complete allocation.
        let second = unsafe { std::slice::from_raw_parts(mirror as *const u8, COPY_LEN) };
        assert!(second == actual, "both views must see the same title bytes");
    }
    println!(
        "marked copy-back fixture: layout={layout:?} disjoint_va={disjoint} known_alias={} parallel_branch={parallel} exact_bytes={} elapsed_ns={elapsed_ns}",
        memory.mirror.is_some(),
        mismatch.is_none()
    );
    if let Some(i) = mismatch {
        println!(
            "first fixture mismatch at offset {i}: expected {:02x}, observed {:02x}",
            expected[i], actual[i]
        );
    }
    (parallel, mismatch.is_none())
}

#[test]
#[ignore = "requires D3D12 and VICTORY_FAULT_TRACKING=1"]
fn marked_copy_back_preserves_disjoint_and_overlapping_bytes() {
    assert_eq!(
        exercise(Layout::Adjacent),
        (parallel_copies_enabled(), true)
    );
    assert_eq!(exercise(Layout::Overlap), (false, true));
}

#[test]
#[ignore = "requires D3D12 and VICTORY_FAULT_TRACKING=1"]
fn marked_copy_back_serializes_mirrored_title_views() {
    assert_eq!(exercise(Layout::Mirrored), (false, true));
}

#[test]
#[ignore = "requires D3D12 and VICTORY_FAULT_TRACKING=1"]
fn marked_copy_back_serializes_subranges_of_one_tracked_page() {
    assert_eq!(exercise(Layout::SharesPage), (false, true));
}

/// One physical section with two title addresses, acquired through the
/// registered-buffer lookup used by descriptor binding.
#[test]
#[ignore = "requires D3D12 and times title buffer shadow refreshes"]
fn benchmark_mirrored_registered_buffers() {
    const SAMPLES: usize = 9;
    const PASSES: usize = 100;
    let _quiet = host().expect("D3D12 host for buffer copies").quiet();
    let memory = GuestMemory::mirrored();
    let second = memory.mirror.unwrap();
    let starts = [memory.base, second];
    let copies = starts.map(|at| {
        register(at, at as u64, COPY_LEN as u64);
        find(at as u64, COPY_LEN as u64).expect("registered title buffer shadow")
    });
    let [a, b] = &copies;
    let allocated: u64 = copies
        .iter()
        .map(|copy| {
            // SAFETY: a live host buffer held by the shadow.
            unsafe { copy.buffer.GetDesc().Width }
        })
        .sum();
    assert_ne!(a.gpu(starts[0] as u64), b.gpu(starts[1] as u64));
    assert_eq!(allocated, 2 * COPY_LEN as u64);
    assert!(crate::watch::aliased(starts[0], COPY_LEN));
    for sample in 0..=SAMPLES {
        let mut bytes = 0;
        let mut refreshed = 0;
        let mut elapsed = std::time::Duration::ZERO;
        for pass in 0..PASSES {
            let written = (pass + sample * PASSES) as u8;
            fill_title(starts[pass % starts.len()], COPY_LEN, written);
            let started = std::time::Instant::now();
            for (&at, copy) in starts.iter().zip(&copies) {
                let copied = copy.update(at, COPY_LEN);
                refreshed += usize::from(copied != 0);
                bytes += copied;
            }
            elapsed += started.elapsed();
            let expected = [u32::from_ne_bytes([written; 4]); 2];
            for (&at, copy) in starts.iter().zip(&copies) {
                assert_eq!(copy.read_host(at as u64, 2), expected);
                assert_eq!(copy.read_host((at + COPY_LEN - 8) as u64, 2), expected);
            }
        }
        assert_eq!(refreshed, PASSES * starts.len());
        if sample > 0 {
            eprintln!(
                "physical_alias sample={sample} passes={PASSES} physical_bytes={COPY_LEN} allocated_bytes={allocated} refreshed={refreshed} refreshed_bytes={bytes} elapsed_ns={}",
                elapsed.as_nanos()
            );
        }
    }
    // A GPU-equivalent byte write through one copy must preserve a
    // neighboring CPU write through its other title alias during copy-back.
    a.writing(starts[0], PAGE, false);
    fill_title(starts[1] + 3, 1, 0xd8);
    a.write_host(starts[0] as u64 + 2, &[0x67]);
    assert_eq!(a.copy_back(starts[0], PAGE), PAGE);
    b.update(starts[1], PAGE);
    let value = b.read_host(starts[1] as u64, 1)[0];
    assert_eq!(value.to_ne_bytes()[2..], [0x67, 0xd8]);
    for at in starts {
        unregister(at, at as u64);
    }
}

#[test]
#[ignore = "requires a D3D12 device and the HLSL compiler"]
fn registered_physical_aliases_read_back_gpu_and_cpu_writes() {
    use crate::compute::{self, Param};

    let host = host().expect("D3D12 host for physical aliases");
    let memory = GuestMemory::mirrored();
    let starts = [memory.base, memory.mirror.unwrap()];
    fill_title(starts[0], COPY_LEN, INITIAL);
    let copies = starts.map(|at| {
        register(at, at as u64, COPY_LEN as u64);
        find(at as u64, COPY_LEN as u64).expect("registered title buffer shadow")
    });
    let [a, b] = &copies;
    let (write_root, write_pipeline) = compute::pipeline(
        "RWByteAddressBuffer b : register(u0);\n\
         [numthreads(1,1,1)] void main() {\n\
             uint prior = b.Load(0);\n\
             b.Store(0, (prior & 0xffff00ffu) | 0x00006700u);\n\
          }",
        windows::core::s!("physical alias writer"),
        0,
        &[Param::Uav { register: 0 }],
    )
    .expect("compile physical alias writer");
    let writer = host.new_list().expect("record physical alias writer");
    a.refresh(&writer, starts[0] as u64, PAGE as u64);
    a.gpu_writes(&writer, starts[0] as u64, PAGE as u64, false);
    // SAFETY: open list and live pipeline and guest shadow.
    unsafe {
        writer.list.SetComputeRootSignature(&write_root);
        writer.list.SetPipelineState(&write_pipeline);
        writer
            .list
            .SetComputeRootUnorderedAccessView(0, a.gpu(starts[0] as u64));
        writer.list.Dispatch(1, 1, 1);
    }
    writer.close();
    host.execute_sync(&[&writer], &[]);
    fill_title(starts[1] + 3, 1, 0xd8);

    let (read_root, read_pipeline) = compute::pipeline(
        "ByteAddressBuffer a : register(t0), b : register(t1);\n\
         RWByteAddressBuffer outbuf : register(u0);\n\
         [numthreads(1,1,1)] void main() { outbuf.Store2(0, uint2(a.Load(0), b.Load(0))); }",
        windows::core::s!("physical alias reader"),
        0,
        &[
            Param::Srv {
                register: 0,
                space: 0,
            },
            Param::Srv {
                register: 1,
                space: 0,
            },
            Param::Uav { register: 0 },
        ],
    )
    .expect("compile physical alias reader");
    let reader = host.new_list().expect("record physical alias reader");
    a.refresh(&reader, starts[0] as u64, PAGE as u64);
    b.refresh(&reader, starts[1] as u64, PAGE as u64);
    let (output, data) = host.shared_buffer(8).expect("readback buffer");
    // SAFETY: open list and live pipeline, guest shadows and output buffer.
    unsafe {
        reader.list.SetComputeRootSignature(&read_root);
        reader.list.SetPipelineState(&read_pipeline);
        reader
            .list
            .SetComputeRootShaderResourceView(0, a.gpu(starts[0] as u64));
        reader
            .list
            .SetComputeRootShaderResourceView(1, b.gpu(starts[1] as u64));
        reader
            .list
            .SetComputeRootUnorderedAccessView(2, output.GetGPUVirtualAddress());
        reader.list.Dispatch(1, 1, 1);
    }
    reader.close();
    host.execute_sync(&[&reader], &[]);
    // SAFETY: both dwords are in the mapped buffer; the queue has completed.
    let readback = unsafe { data.cast::<[u32; 2]>().read_volatile() };
    assert_eq!(readback, [0xd819_6719; 2]);
    for at in starts {
        unregister(at, at as u64);
    }
}
