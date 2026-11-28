use super::residents::{Census, IMAGE_RANGES, request};
use super::{Map, Sample, TRAPS, enabled};
use crate::LoadedImage;
use std::fmt;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, ReadProcessMemory};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const NAMES: [&str; 3] = ["BUILDING", "OBJECT", "pool-256"];
const STRIDES: [u32; 3] = [224, 272, 256];
const SLOTS: u64 = 1200;
const ROW_BYTES: usize = 1024;

fn region(at: usize, len: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let end = at.checked_add(len)?;
    // SAFETY: VirtualQuery writes only the supplied fixed-size record.
    let mut info = unsafe { std::mem::zeroed::<MEMORY_BASIC_INFORMATION>() };
    // SAFETY: the output points to a writable, fully sized record.
    let got = unsafe { VirtualQuery(at as _, &mut info, size_of_val(&info)) };
    if at == 0
        || got != size_of_val(&info)
        || info.State != MEM_COMMIT
        || info.Protect & PAGE_GUARD != 0
        || !matches!(
            info.Protect & 0xff,
            PAGE_READONLY
                | PAGE_READWRITE
                | PAGE_WRITECOPY
                | PAGE_EXECUTE_READ
                | PAGE_EXECUTE_READWRITE
                | PAGE_EXECUTE_WRITECOPY
        )
        || end > (info.BaseAddress as usize).checked_add(info.RegionSize)?
    {
        return None;
    }
    Some(info)
}

fn read<const N: usize>(at: usize) -> Option<[u8; N]> {
    let mut bytes = [0; N];
    read_into(at, &mut bytes).then_some(bytes)
}

fn read_into(mut at: usize, mut bytes: &mut [u8]) -> bool {
    if at == 0 || at.checked_add(bytes.len()).is_none() {
        return false;
    }
    let mut allocation = None;
    while !bytes.is_empty() {
        let Some(info) = region(at, 1) else {
            return false;
        };
        if allocation
            .replace(info.AllocationBase)
            .is_some_and(|a| a != info.AllocationBase)
        {
            return false;
        }
        let region_end = info.BaseAddress as usize + info.RegionSize;
        let len = bytes.len().min(region_end - at);
        let mut count = 0;
        // SAFETY: the OS validates the source again and the destination contains len writable bytes.
        let ok = unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                at as _,
                bytes.as_mut_ptr().cast(),
                len,
                &mut count,
            )
        };
        if ok == 0 || count != len {
            return false;
        }
        at += len;
        bytes = &mut bytes[len..];
    }
    true
}

impl Map {
    fn pointer(self, index: usize) -> Option<usize> {
        Some(usize::from_le_bytes(read(self.slots[index])?))
    }

    fn sample(self, index: usize, pointer: usize) -> Result<Sample, &'static str> {
        if pointer == 0 || !pointer.is_multiple_of(8) {
            return Err("null-or-misaligned-header");
        }
        let first = read(pointer).ok_or("unreadable-header")?;
        let a = Sample::parse(pointer, STRIDES[index], &first)?;
        let second = read(pointer).ok_or("unreadable-header")?;
        let b = Sample::parse(pointer, STRIDES[index], &second)?;
        if a != b || self.pointer(index) != Some(pointer) {
            return Err("header-or-global-changed-during-read");
        }
        Ok(b)
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "header={:#x} backing={:#x} metadata={:#x} stride={} capacity={} used={} head={:#018x} tag={}",
            self.pointer,
            self.backing,
            self.metadata,
            self.stride,
            self.capacity,
            self.used,
            self.head,
            self.head >> 32
        )?;
        if self.backing == 0 || self.metadata == 0 {
            write!(f, " live_header=unavailable-null-storage")?;
        }
        Ok(())
    }
}

struct Registration {
    map: Map,
    start: Instant,
    observed: bool,
}

// The sampler works from its own map and never takes this mutex.
static REGISTRATION: Mutex<Option<Registration>> = Mutex::new(None);
static BASE: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct Fault {
    ms: u128,
    site_rva: usize,
    candidate: &'static str,
    pointer: usize,
    rax: u64,
    request: Option<request::Request>,
    sample: Result<Sample, &'static str>,
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active = if self.sample.is_ok() {
            self.candidate
        } else {
            "unknown"
        };
        write!(
            f,
            "pool observation elapsed_ms={} site_rva={:#x} active={active} rcx={:#x} fault_rax={:#018x} ",
            self.ms, self.site_rva, self.pointer, self.rax
        )?;
        match self.sample {
            Ok(sample) => {
                write!(f, "{sample}")?;
                if self.candidate == "BUILDING" {
                    if let Some(r) = self.request {
                        write!(
                            f,
                            " batch_source={:#x} key={:#x} override={} descriptor={:#x} task={:#x} list={:#x} index={} end={} list_count={} output_array={:#x} owner_id={:#x}",
                            r.source,
                            r.key,
                            r.override_byte,
                            r.descriptor,
                            r.task,
                            r.list,
                            r.index,
                            r.end,
                            r.count,
                            r.array,
                            r.id
                        )?;
                        if let Some(o) = r.matched_owner {
                            write!(
                                f,
                                " matched_owner={:#x} registry_record={:#x} output_count={}",
                                o.pointer, o.record, o.count
                            )?;
                        } else {
                            write!(f, " matched_owner=unavailable")?;
                        }
                    } else {
                        write!(f, " batch=unavailable")?;
                    }
                }
                Ok(())
            }
            Err(reason) => write!(
                f,
                "global_candidate={} unavailable={reason}",
                self.candidate
            ),
        }
    }
}

/// Only called for naturally dispatched first-chance breakpoints.
pub(crate) fn first_fault(at: usize, context: &CONTEXT) -> Option<Fault> {
    let base = BASE.load(Ordering::Acquire);
    let site_rva = at.checked_sub(base)?;
    if base == 0 || !TRAPS.contains(&site_rva) {
        return None;
    }
    // Only registration, teardown or another matching fault can own the lock.
    let mut entry = REGISTRATION.try_lock().ok()?;
    let reg = entry.as_mut()?;
    if !reg.map.traps.contains(&at) || std::mem::replace(&mut reg.observed, true) {
        return None;
    }
    let pointer = context.Rcx as usize;
    let mut fault = Fault {
        ms: reg.start.elapsed().as_millis(),
        site_rva,
        pointer,
        rax: context.Rax,
        candidate: "unknown",
        request: None,
        sample: Err("exception-address-context-mismatch"),
    };
    if context.Rip == at as u64 || context.Rip.checked_sub(1) == Some(at as u64) {
        let globals = reg
            .map
            .slots
            .map(|slot| read(slot).map(usize::from_le_bytes));
        if let Some(i) = reg.map.select(at, pointer, globals) {
            fault.candidate = NAMES[i];
            fault.sample = reg.map.sample(i, pointer).and_then(|s| {
                if s.metadata == 0 {
                    Err("null-metadata")
                } else {
                    Ok(s)
                }
            });
            if i == 0
                && fault.sample.is_ok_and(|s| {
                    s.backing != 0
                        && s.used == s.capacity
                        && s.head == context.Rax
                        && (s.head as u32) == super::EMPTY
                })
            {
                fault.request = request::capture(
                    base,
                    [
                        context.Rsp as usize,
                        context.Rdi as usize,
                        context.R14 as usize,
                        context.R15 as usize,
                        context.R12 as usize,
                        context.Rbp as u32 as usize,
                    ],
                    &mut read_into,
                );
            }
        } else {
            fault.sample = Err("no-unique-readable-global");
        }
    }
    Some(fault)
}

fn history(
    mut file: File,
    map: Map,
    start: Instant,
    sidecar: Option<(File, usize)>,
    mut stop_after_sample: impl FnMut() -> bool,
) -> std::io::Result<()> {
    writeln!(
        file,
        "Most recent 1200 samples. Read 1024-byte slots. Discard short slots or slots without matching initial seq and final complete values. Sort complete slots by seq after wrap. Tag deltas count published pushes modulo 2^32 for an apparently continuous lifetime. Pop estimates include in-flight skew. Samples cannot detect every wrap, replacement or failed attempt."
    )?;
    let data_start = file.stream_position()?;
    let mut previous = [None; 3];
    let mut residents = sidecar.map(|(file, base)| (file, base, Census::new()));
    for seq in 0..u64::MAX {
        let mut row = [b' '; ROW_BYTES];
        {
            const FOOTER_BYTES: usize = b"complete=00000000000000000000\n".len();
            let (mut out, mut footer) = row.split_at_mut(ROW_BYTES - FOOTER_BYTES);
            writeln!(footer, "complete={seq:020}")?;
            writeln!(out, "seq={seq} elapsed_ms={}", start.elapsed().as_millis())?;
            for i in 0..3 {
                let pointer = map.pointer(i);
                let sample = pointer
                    .ok_or("unreadable-global")
                    .and_then(|p| map.sample(i, p));
                write!(out, "  {} ", NAMES[i])?;
                match sample {
                    Ok(s) => {
                        write!(out, "{s} ")?;
                        match s.activity(previous[i]) {
                            Ok((r, Some(p))) => {
                                writeln!(out, "release_tag_delta_mod32={r} pop_estimate={p}")?
                            }
                            Ok((r, None)) => writeln!(
                                out,
                                "raw_tag_delta_mod32={r} pop_estimate_unavailable=tag-decreased-wrap-or-reset"
                            )?,
                            Err(reason) => writeln!(out, "delta_unavailable={reason}")?,
                        }
                    }
                    Err(reason) => writeln!(out, "unavailable={reason}")?,
                }
                previous[i] = sample.ok();
            }
        }
        file.seek(SeekFrom::Start(
            data_start + (seq % SLOTS) * ROW_BYTES as u64,
        ))?;
        file.write_all(&row)?;
        if let Some((file, base, census)) = &mut residents {
            let ms = start.elapsed().as_millis() as u64;
            let first = previous[0];
            census.capture(*base, first, read_into, || {
                map.sample(0, first?.pointer).ok()
            });
            if let Err(e) = census.publish(file, seq, ms) {
                eprintln!("BUILDING resident history unavailable: {e}");
                residents = None;
            }
        }
        if stop_after_sample() {
            break;
        }
    }
    Ok(())
}

/// Keeps the observation registered until the title is done.
pub struct PoolObserver {
    stop: Sender<()>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for PoolObserver {
    fn drop(&mut self) {
        let mut entry = REGISTRATION.lock().unwrap_or_else(|e| e.into_inner());
        BASE.store(0, Ordering::Release);
        *entry = None;
        drop(entry);
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Starts the opted-in observer for the identified original main image.
pub fn start_pool_observation(image: &LoadedImage, path: &Path) -> Option<PoolObserver> {
    let flag = std::env::var("VICTORY_POOL_OBSERVE").ok();
    if flag.as_deref() != Some("1") {
        return None;
    }
    if !enabled(
        flag.as_deref(),
        image.original_hash.to_hex().as_str(),
        image.info().is_dll(),
    ) {
        eprintln!("pool observation unavailable: unrecognized original main image");
        return None;
    }
    let result = (|| -> anyhow::Result<_> {
        let base = image.base();
        let map = Map::new(base, image.info().size_of_image as usize)
            .ok_or_else(|| anyhow::anyhow!("pool ranges outside image"))?;
        for (at, len) in map
            .slots
            .map(|at| (at, 8))
            .into_iter()
            .chain(map.traps.map(|at| (at, 1)))
        {
            anyhow::ensure!(
                region(at, len).is_some_and(|r| r.AllocationBase as usize == base),
                "pool range unavailable in main image"
            );
        }
        let mut entry = REGISTRATION.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(entry.is_none(), "observer already registered");
        victory_core::ensure_outside_repo(path)?;
        let file = File::create(path)?;
        let sidecar = (|| -> anyhow::Result<_> {
            let size = image.info().size_of_image as usize;
            for (rva, len) in IMAGE_RANGES {
                anyhow::ensure!(
                    rva + len <= size
                        && region(base + rva, len)
                            .is_some_and(|r| r.AllocationBase as usize == base),
                    "resident range outside main image"
                );
            }
            let dest = path.with_file_name("pool-residents.bin");
            victory_core::ensure_outside_repo(&dest)?;
            Ok((File::create(dest)?, base))
        })()
        .map_err(|e| eprintln!("BUILDING resident history unavailable: {e}"))
        .ok();
        let start = Instant::now();
        let (stop, stopped) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("pool-observation".into())
            .spawn(move || {
                let stop_after_sample = || {
                    !matches!(
                        stopped.recv_timeout(Duration::from_secs(1)),
                        Err(mpsc::RecvTimeoutError::Timeout)
                    )
                };
                if let Err(e) = history(file, map, start, sidecar, stop_after_sample) {
                    eprintln!("pool history unavailable: {e}");
                }
            })?;
        *entry = Some(Registration {
            map,
            start,
            observed: false,
        });
        BASE.store(base, Ordering::Release);
        Ok(PoolObserver {
            stop,
            worker: Some(worker),
        })
    })();
    match result {
        Ok(guard) => Some(guard),
        Err(e) => {
            eprintln!("pool observation unavailable: {e}");
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pool::tests::header;
    use windows_sys::Win32::Foundation::EXCEPTION_BREAKPOINT;
    use windows_sys::Win32::System::Diagnostics::Debug::{EXCEPTION_POINTERS, EXCEPTION_RECORD};
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree, VirtualProtect,
    };

    #[test]
    fn protected_reads_require_every_byte_and_preserve_guard_pages() {
        // SAFETY: reserve two new pages and release the allocation after the reads.
        unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                8192,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            let at = p as usize;
            let mut old = 0;
            // A single backing allocation may have adjacent readable regions with different protections.
            assert_ne!(
                VirtualProtect((at + 4096) as _, 4096, PAGE_READONLY, &mut old),
                0
            );
            let mut bulk = vec![9; 8192];
            assert!(read_into(at, &mut bulk));
            assert_eq!(bulk, vec![0; 8192]);
            for protection in [PAGE_NOACCESS, PAGE_READWRITE | PAGE_GUARD] {
                assert_ne!(
                    VirtualProtect((at + 4096) as _, 4096, protection, &mut old),
                    0
                );
                assert_eq!(read::<8>(at + 4088), Some([0; 8]));
                assert_eq!(read::<8>(at + 4092), None);
                assert_eq!(read::<8>(at + 4096), None);
                assert!(!read_into(at, &mut bulk));
                let mut page: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                VirtualQuery((at + 4096) as _, &mut page, size_of_val(&page));
                assert_eq!(page.Protect, protection);
            }
            assert_ne!(VirtualFree(p, 0, MEM_RELEASE), 0);
            assert_eq!(read::<8>(usize::MAX - 3), None);
        }
    }

    pub(crate) fn check_dispatch(
        handler: unsafe extern "system" fn(*mut EXCEPTION_POINTERS) -> i32,
    ) {
        const CHILD: &str = "VICTORY_TEST_POOL_REPORT";
        let Some(log) = std::env::var_os(CHILD) else {
            let dir = tempfile::tempdir().unwrap();
            for (file, identity, expected) in [
                (
                    "readable.log",
                    "active=BUILDING",
                    "stride=224 capacity=17 used=10",
                ),
                (
                    "unreadable.log",
                    "active=unknown",
                    "global_candidate=BUILDING unavailable=unreadable-header",
                ),
                (
                    "wrong-stride.log",
                    "active=unknown",
                    "global_candidate=BUILDING unavailable=invalid-header",
                ),
                ("null-backing.log", "active=BUILDING", "backing=0x0"),
                (
                    "null-metadata.log",
                    "active=unknown",
                    "global_candidate=BUILDING unavailable=null-metadata",
                ),
            ] {
                let log = dir.path().join(file);
                let child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "crash::tests::first_pool_fault_preserves_dispatch",
                        "--nocapture",
                    ])
                    .env(CHILD, &log)
                    .output()
                    .unwrap();
                assert!(
                    child.status.success(),
                    "{}",
                    String::from_utf8_lossy(&child.stderr)
                );
                let text = std::fs::read_to_string(log).unwrap();
                assert_eq!(text.matches("pool observation").count(), 1);
                assert!(
                    text.contains("site_rva=0x5149aec")
                        && text.contains(identity)
                        && text.contains(expected),
                    "{file}: {text}"
                );
                assert!(text.contains("fault_rax=0x000000000007fffe"));
            }
            return;
        };
        let unreadable = Path::new(&log).ends_with("unreadable.log");
        let mut fields = header();
        if Path::new(&log).ends_with("wrong-stride.log") {
            fields[8..12].copy_from_slice(&272_u32.to_le_bytes());
        } else if Path::new(&log).ends_with("null-backing.log") {
            fields[..8].fill(0);
        } else if Path::new(&log).ends_with("null-metadata.log") {
            fields[40..48].fill(0);
        }
        crate::log_crashes_to(log.into());
        #[repr(align(8))]
        struct Header([u8; 52]);
        let bytes = Header(fields);
        let ptr = bytes.0.as_ptr() as usize;
        let slots = [
            AtomicUsize::new(if unreadable { 8 } else { ptr }),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        ];
        let map = Map {
            slots: std::array::from_fn(|i| std::ptr::from_ref(&slots[i]) as usize),
            traps: [0x10000000 + TRAPS[0], 0x10000000 + TRAPS[1]],
        };
        *REGISTRATION.lock().unwrap() = Some(Registration {
            map,
            start: Instant::now(),
            observed: false,
        });
        BASE.store(0x10000000, Ordering::Release);
        // SAFETY: zeroed synthetic exception records and context contain no owning references.
        let (mut context, mut record): (CONTEXT, EXCEPTION_RECORD) = unsafe { std::mem::zeroed() };
        context.Rip = map.traps[0] as u64;
        context.Rcx = slots[0].load(Ordering::Relaxed) as u64;
        context.Rax = 0x7fffe;
        context.EFlags = 0x246;
        record.ExceptionAddress = map.traps[0] as _;
        record.ExceptionCode = EXCEPTION_BREAKPOINT;
        record.ExceptionFlags = 1;
        let original_record =
            read::<{ size_of::<EXCEPTION_RECORD>() }>(std::ptr::from_ref(&record) as usize)
                .unwrap();
        let mut info = EXCEPTION_POINTERS {
            ExceptionRecord: &mut record,
            ContextRecord: &mut context,
        };
        for _ in 0..2 {
            let original_context =
                read::<{ size_of::<CONTEXT>() }>(info.ContextRecord as usize).unwrap();
            // SAFETY: synthetic records stay live, with a zero stack pointer ending the ordinary stack walk.
            assert_eq!(unsafe { handler(&mut info) }, 0);
            assert_eq!(
                read::<{ size_of::<CONTEXT>() }>(info.ContextRecord as usize).unwrap(),
                original_context
            );
            assert_eq!(
                read::<{ size_of::<EXCEPTION_RECORD>() }>(info.ExceptionRecord as usize).unwrap(),
                original_record
            );
            slots[0].store(ptr, Ordering::Relaxed);
            context.Rcx = ptr as u64;
        }
        // SAFETY: the same live synthetic records exercise an unhandled redelivery.
        unsafe { crate::report_unhandled(&info) };
        let (stop, stopped) = mpsc::channel();
        let worker = std::thread::spawn(move || stopped.recv().unwrap());
        drop(PoolObserver {
            stop,
            worker: Some(worker),
        });
        assert!(REGISTRATION.lock().unwrap().is_none());
        assert!(first_fault(map.traps[0], &context).is_none());
    }

    #[test]
    fn a_replaced_global_invalidates_the_previously_selected_header() {
        #[repr(align(8))]
        struct Header([u8; 52]);
        let original = Header(header());
        let pointer = original.0.as_ptr() as usize;
        let slot = AtomicUsize::new(pointer);
        let empty = 0_usize;
        let map = Map {
            slots: [
                std::ptr::from_ref(&slot) as usize,
                std::ptr::from_ref(&empty) as usize,
                std::ptr::from_ref(&empty) as usize,
            ],
            traps: [1, 2],
        };
        let globals = map.slots.map(|at| read(at).map(usize::from_le_bytes));
        let i = map.select(1, pointer, globals).unwrap();
        slot.store(0, Ordering::Relaxed);
        let sample = map.sample(i, pointer);
        assert_eq!(sample, Err("header-or-global-changed-during-read"));
        let output = Fault {
            ms: 3,
            site_rva: TRAPS[0],
            candidate: NAMES[i],
            pointer,
            rax: 75,
            request: None,
            sample,
        }
        .to_string();
        assert!(output.contains("active=unknown"));
        assert!(output.contains(
            "global_candidate=BUILDING unavailable=header-or-global-changed-during-read"
        ));
    }

    #[test]
    fn history_wrap_rejects_an_interrupted_slot_replacement() {
        fn sequence(slot: &[u8]) -> Option<u64> {
            if slot.len() != ROW_BYTES {
                return None;
            }
            let text = std::str::from_utf8(slot).ok()?;
            let first = text
                .strip_prefix("seq=")?
                .split_once(' ')?
                .0
                .parse::<u64>()
                .ok()?;
            let (_, footer) = text.strip_suffix('\n')?.rsplit_once("complete=")?;
            (footer.len() == 20 && footer.parse::<u64>().ok()? == first).then_some(first)
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.log");
        #[repr(align(8))]
        struct Header([u8; 52]);
        let mut fields = Header(header());
        let slots = [fields.0.as_mut_ptr() as usize, 0, 0];
        let map = Map {
            slots: std::array::from_fn(|i| std::ptr::from_ref(&slots[i]) as usize),
            traps: [1, 2],
        };
        let mut count = 0;
        let mut first = Vec::new();
        history(
            File::create(&path).unwrap(),
            map,
            Instant::now(),
            None,
            || {
                count += 1;
                if count == 1 {
                    let bytes = std::fs::read(&path).unwrap();
                    first = bytes[bytes.len() - ROW_BYTES..].to_vec();
                    fields.0[16..20].copy_from_slice(&11_u32.to_le_bytes());
                    std::hint::black_box(&fields);
                }
                count == SLOTS + 1
            },
        )
        .unwrap();
        let bytes = std::fs::read(path).unwrap();
        let header_end = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
        let data = &bytes[header_end..];
        assert_eq!(data.len(), SLOTS as usize * ROW_BYTES);
        assert_eq!(sequence(&first), Some(0));
        for (index, slot) in data.as_chunks::<ROW_BYTES>().0.iter().enumerate() {
            assert_eq!(
                sequence(slot),
                Some(if index == 0 { SLOTS } else { index as u64 })
            );
        }
        let new = &data[..ROW_BYTES];
        assert!(std::str::from_utf8(new).unwrap().contains("used=11"));
        first[..40].copy_from_slice(&new[..40]);
        assert!(std::str::from_utf8(&first).unwrap().contains("used=10"));
        assert_eq!(sequence(&first), None);
        assert_eq!(sequence(&new[..ROW_BYTES - 1]), None);
    }

    #[test]
    fn history_writes_one_complete_sample_and_stops_on_disconnect() {
        let mut bytes = header();
        bytes[8..12].copy_from_slice(&272_u32.to_le_bytes());
        #[repr(align(8))]
        struct Header([u8; 52]);
        let bytes = Header(bytes);
        let slots = [0, bytes.0.as_ptr() as usize, 0];
        let map = Map {
            slots: std::array::from_fn(|i| std::ptr::from_ref(&slots[i]) as usize),
            traps: [1, 2],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.log");
        let sidecar = dir.path().join("residents.bin");
        let (stop, stopped) = mpsc::channel::<()>();
        drop(stop);
        history(
            File::create(&path).unwrap(),
            map,
            Instant::now(),
            Some((File::create(&sidecar).unwrap(), 0x10000)),
            || stopped.recv().is_err(),
        )
        .unwrap();
        let frame = std::fs::read(sidecar).unwrap();
        assert_eq!(&frame[..8], b"BRES0002");
        assert_eq!(&frame[8..16], &0_u64.to_le_bytes());
        assert_eq!(&frame[frame.len() - 8..], &u64::MAX.to_le_bytes());
        assert_eq!(frame[24], 0);
        let text = std::fs::read_to_string(path).unwrap();
        assert_eq!(text.matches("seq=").count(), 1);
        assert!(text.contains("OBJECT header=") && text.contains("stride=272 capacity=17 used=10"));
        assert!(text.contains("BUILDING unavailable=") && text.contains("pool-256 unavailable="));
        assert!(text.contains("delta_unavailable=no-contiguous-sample"));
        assert!(text.ends_with("complete=00000000000000000000\n"));
    }
}
