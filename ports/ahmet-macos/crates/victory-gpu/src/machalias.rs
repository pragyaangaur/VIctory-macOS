//! Title memory as the card's own memory on macOS (`VICTORY_ALIAS=1`, Wine on Darwin only).
//!
//! D3DMetal does not implement `OpenExistingHeapFromAddress[1]` / `OpenExistingHeapFromFileMapping`
//! (E_NOTIMPL / E_FAIL, measured with GPTK 4.0b3), so [`crate::inplace`]'s heaps over title memory
//! cannot be made there. The other way round works: a buffer D3DMetal makes itself in the PC's
//! memory ([`crate::host::Host::shared_buffer`], custom heap, write-back pages, pool L0) and keeps
//! mapped is the very memory the GPU reads (Apple silicon has one memory). Pointing the title's
//! addresses at that buffer's pages with `mach_vm_remap(copy = FALSE, VM_FLAGS_FIXED |
//! VM_FLAGS_OVERWRITE)` makes title memory and the buffer one and the same: what the title writes
//! the GPU reads, and what shaders write the title reads, with no copy, refresh or copy-back.
//!
//! Measured in a probe under CrossOver 26.3 / GPTK 4.0b3 / macOS 27.3 (`~/Code/gta-push/uma-probe`),
//! 64 MiB and 4 GiB: coherent both ways (copies and a compute shader through a root UAV), 8 of 8
//! rounds; Wine's `VirtualQuery` and `VirtualProtect` (with a write fault into a vectored handler)
//! keep working on the alias; no sync at submit (a tiny submit costs the same after every page was
//! written); CPU speed through the alias equals plain memory.
//!
//! When: at [`title_memory_aliasable`], called by the runtime right after the title's
//! `XMemVirtualAlloc` committed fresh memory and before the title has its address, so nothing can
//! write it between our copy and the remap (the pages are zero; nothing is copied). A title buffer
//! or window inside such memory is then placed on the alias by [`crate::inplace::place_for`].
//!
//! The Mach calls are raw traps from this PE code: Wine on macOS does not filter `syscall`, and
//! Rosetta 2 translates x86-64 Mach traps (class 1: `rax = 0x0100_0000 | n`). `mach_vm_remap` is a
//! MIG request to the task port; it must go through `mach_msg2_trap` (-47) with
//! `MACH64_SEND_KOBJECT_CALL`: the old `mach_msg_trap` (-31) gets the process killed on macOS 27
//! (EXC_GUARD "mach_msg_trap() called with msgh_id 4813. The trap is not allowed on this platform").
//! Message layouts follow `<mach/mach_vm.h>`'s generated code (`#pragma pack(4)`) and
//! `osfmk/mach/message.h` (`mach_msg2`).
//!
//! shortcut: the title freeing or decommitting part of an aliased allocation lets the whole
//! allocation's alias go (its shadows are let go, the table forgets it); the buffers stay alive
//! while lists hold them. The title's heaps are allocated once and kept, so this is rare.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// `VICTORY_ALIAS`: 0 off (the default); 1 alias committed title allocations of at least
/// `VICTORY_ALIAS_MIN_KB` (1024) under Wine on macOS, and place title buffers and windows in them.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var("VICTORY_ALIAS").is_ok_and(|v| v.trim() == "1");
        let on = asked && crate::is_darwin();
        if asked && !on {
            tracing::warn!("VICTORY_ALIAS=1 needs Wine on macOS: off");
        }
        on
    })
}

fn env_mb(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Smallest allocation aliased, in bytes (`VICTORY_ALIAS_MIN_KB`, 1024).
fn min_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_mb("VICTORY_ALIAS_MIN_KB", 1024) << 10)
}

/// Most title memory aliased at once (`VICTORY_ALIAS_MB`, 24576).
fn most_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_mb("VICTORY_ALIAS_MB", 24576) << 20)
}

/// Largest single buffer (`VICTORY_ALIAS_CHUNK_MB`, 4096); bigger allocations are cut into
/// chunks of this size from their start, and a range across two chunks gets a copy as before.
fn chunk_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| (env_mb("VICTORY_ALIAS_CHUNK_MB", 4096) << 20).max(1 << 20))
}

/// One buffer the title's addresses `[start, start + len)` show.
struct Chunk {
    len: u64,
    buffer: ID3D12Resource,
    /// The buffer's GPU address (of `start`).
    gpu: u64,
}

/// Aliased allocations by start: their length and chunks by start.
struct Table {
    allocations: BTreeMap<u64, (u64, BTreeMap<u64, Chunk>)>,
}

static TABLE: Mutex<Table> = Mutex::new(Table { allocations: BTreeMap::new() });

/// Allocations aliased, bytes aliased, allocations turned down (too small, over the cap, host
/// refused), remap failures, placements handed out, allocations let go.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn table() -> std::sync::MutexGuard<'static, Table> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title's `XMemVirtualAlloc` just committed `[at, at + len)` read-write, fresh (all zero),
/// and has not handed the address out yet: point it at buffers of the card's.
pub fn title_memory_aliasable(at: usize, len: usize) {
    if !enabled() || at == 0 {
        return;
    }
    let (at, len) = (at as u64, len as u64);
    if len < min_bytes() || at % (16 << 10) != 0 || len % (16 << 10) != 0 {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    if STATS[1].load(Ordering::Relaxed) + len > most_bytes() {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Some(host) = crate::host::host() else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    };
    // An allocation still noted here overlaps the new one: it was freed unseen (the host's
    // `NtFreeVirtualMemory`), so what is placed over it shows its old buffers. Let it go first.
    title_memory_alias_freed(at as usize, len as usize, true);
    let mut chunks = BTreeMap::new();
    let mut offset = 0;
    while offset < len {
        let part = (len - offset).min(chunk_bytes());
        let Some((buffer, data)) = host.shared_buffer(part) else {
            STATS[2].fetch_add(1, Ordering::Relaxed);
            return; // the chunks made so far are dropped; the memory stays plain
        };
        // SAFETY: `data` is the mapping of a live buffer of `part` bytes (page aligned: Metal's
        // allocations are), and `[at + offset, + part)` is title memory just committed whose
        // address nobody else has yet; the remap replaces those pages with the buffer's.
        let r = unsafe { remap_shared(at + offset, data as u64, part) };
        if r.msg != 0 || r.kr != 0 || r.address != at + offset {
            if STATS[3].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(
                    at = format_args!("{:#x}", at + offset),
                    part,
                    msg = format_args!("{:#x}", r.msg),
                    kr = r.kr,
                    "title memory not aliased onto the card's buffer: copies as before"
                );
            }
            // Parts already remapped stay pointed at their buffers' pages (the pages live as long
            // as either mapping does), so the memory is still correct; it is just not placed.
            return;
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        crate::host::name(&buffer, || format!("title memory {:#x}+{part:#x} aliased", at + offset));
        chunks.insert(at + offset, Chunk { len: part, buffer, gpu });
        offset += part;
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(len, Ordering::Relaxed);
    if STATS[0].load(Ordering::Relaxed) <= 32 {
        tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, chunks = chunks.len(), "title memory aliased onto the card's buffer (no copies for it)");
    }
    table().allocations.insert(at, (len, chunks));
}

/// Buffers of allocations let go while some of their pages may still be mapped at title
/// addresses (a decommit of part of an allocation): kept for good, so those pages never belong
/// to a buffer D3DMetal has freed.
static KEPT: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// The title frees (`release`: the whole allocation at `at`, whose mappings go with it) or
/// decommits `[at, at + len)`: every aliased allocation it touches is let go.
pub fn title_memory_alias_freed(at: usize, len: usize, release: bool) {
    if !enabled() {
        return;
    }
    let (at, len) = (at as u64, len.max(1) as u64);
    let gone: Vec<(u64, u64, BTreeMap<u64, Chunk>)> = {
        let mut t = table();
        let hit: Vec<u64> = t
            .allocations
            .range(..at + len)
            .filter(|(start, (bytes, _))| **start + *bytes > at)
            .map(|(start, _)| *start)
            .collect();
        hit.into_iter().filter_map(|s| t.allocations.remove(&s).map(|(bytes, chunks)| (s, bytes, chunks))).collect()
    };
    for (start, bytes, chunks) in gone {
        STATS[1].fetch_sub(bytes, Ordering::Relaxed);
        STATS[5].fetch_add(1, Ordering::Relaxed);
        crate::inplace::title_memory_freeing(start as usize, bytes as usize);
        let whole = release && start == at;
        if !whole {
            KEPT.lock().unwrap_or_else(|e| e.into_inner()).extend(chunks.into_values().map(|c| c.buffer));
        }
        tracing::info!(at = format_args!("{start:#x}"), mb = bytes >> 20, whole, "aliased title memory let go");
    }
}

/// The buffer that `[start, start + len)` shows if it lies in one chunk of an aliased allocation:
/// the buffer, the GPU address of `start`, and `start`'s offset in the buffer.
pub(crate) fn placed(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if !enabled() {
        return None;
    }
    if let Some(p) = placed_private(start, len) {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return Some(p);
    }
    if let Some(p) = placed_in_view(start, len) {
        POOL[6].fetch_add(1, Ordering::Relaxed);
        return Some(p);
    }
    placed_scattered(start, len)
}

fn placed_private(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    let t = table();
    let (a_start, (a_len, chunks)) = t.allocations.range(..=start).next_back()?;
    if start + len.max(1) > a_start + a_len {
        return None;
    }
    let (c_start, chunk) = chunks.range(..=start).next_back()?;
    if start + len.max(1) > c_start + chunk.len {
        return None;
    }
    let bias = start - c_start;
    Some((chunk.buffer.clone(), chunk.gpu + bias, bias))
}

/// `[start, start + len)` as one run of an aliased section's bytes (title views of consecutive
/// pages of it, [`crate::inplace::cover`]) inside one of its chunks: that chunk's buffer at the
/// run's offset.
fn placed_in_view(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if !pool_enabled() {
        return None;
    }
    let crate::inplace::Cover::Run(section, offset) = crate::inplace::cover(start as usize, len as usize) else {
        return None;
    };
    let s = sections();
    let aliased = s.get(&section)?;
    let (c_off, chunk) = aliased.chunks.range(..=offset).next_back()?;
    if offset + len.max(1) > c_off + chunk.len {
        return None;
    }
    let bias = offset - c_off;
    Some((chunk.buffer.clone(), chunk.gpu + bias, bias))
}

/// Whether `[start, start + len)` lies in aliased memory (an aliased allocation, or one run of an
/// aliased section's bytes in one chunk).
pub(crate) fn covers(start: u64, len: u64) -> bool {
    enabled()
        && ({
            let t = table();
            t.allocations.range(..=start).next_back().is_some_and(|(s, (l, _))| start + len.max(1) <= s + l)
        } || placed_in_view(start, len).is_some()
            || (tiles_enabled() && scattered_plan(start, len).is_some()))
}

// The title's physical pages (`VICTORY_ALIAS_POOL`).
//
// The runtime backs the title's physical pages with pagefile sections and maps them as views
// (`xmem`: `XMemMapPhysicalPages`, the GPU's page-mapping copies, DirectStorage's view of a whole
// section). With the switch, each section is backed by buffers of the card's from its creation, and
// every view of it the runtime maps is pointed at those buffers' pages right after Wine maps it
// (one `mach_vm_remap` a view), so all views of a page show the very bytes the GPU reads. Wine
// still believes it mapped the section (its `VirtualQuery`, `VirtualProtect` and unmapping work as
// before); the section's own file pages are never touched. A title buffer over one run of a
// section's pages is then that section's buffer at the run's offset ([`placed`]).

/// `VICTORY_ALIAS_POOL`: 1 back the title's physical pages (sections) by the card's buffers and
/// alias every view of them (needs `VICTORY_ALIAS=1`); 0 (the default) not.
pub fn pool_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enabled() && std::env::var("VICTORY_ALIAS_POOL").is_ok_and(|v| v.trim() == "1"))
}

struct SectionChunk {
    len: u64,
    buffer: ID3D12Resource,
    gpu: u64,
    /// The buffer's mapping (where views are pointed).
    data: u64,
    /// With `VICTORY_ALIAS_TILES`: the heap the buffer is placed over (all of it), which reserved
    /// buffers' tiles are mapped onto.
    heap: Option<ID3D12Heap>,
}

struct AliasedSection {
    len: u64,
    /// Chunks by offset in the section.
    chunks: BTreeMap<u64, SectionChunk>,
}

static SECTIONS: Mutex<BTreeMap<usize, AliasedSection>> = Mutex::new(BTreeMap::new());

/// Heaps of sections let go while reserved buffers may still map them.
static HEAPS_KEPT: Mutex<Vec<ID3D12Heap>> = Mutex::new(Vec::new());

fn sections() -> std::sync::MutexGuard<'static, BTreeMap<usize, AliasedSection>> {
    SECTIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sections aliased, their bytes, sections turned down, views pointed at their buffers, the
/// microseconds that took, view remap failures, placements in views.
/// Then (`VICTORY_ALIAS_TILES`): placements over scattered pages, their tiles, the microseconds
/// mapping them took (made, mapped, waited for), placements turned down.
/// Then (`VICTORY_ALIAS_HOLES`): those of the placements with pages not mapped, and those pages.
static POOL: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// `VICTORY_ALIAS_TILES`: 1 a title buffer over scattered pages of aliased sections is a reserved
/// buffer whose 64 KiB tiles are mapped onto the sections' heaps page by page (needs
/// `VICTORY_ALIAS_POOL=1`); 0 (the default) such buffers are copied.
pub fn tiles_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pool_enabled() && std::env::var("VICTORY_ALIAS_TILES").is_ok_and(|v| v.trim() == "1"))
}

/// A heap of `bytes` in the PC's memory (write-back, pool L0, as `Host::shared_buffer`'s) with a
/// buffer placed over all of it and mapped: the heap, the buffer and its mapping.
fn heap_buffer(device: &ID3D12Device, bytes: u64) -> windows::core::Result<(ID3D12Heap, ID3D12Resource, *mut u8)> {
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let desc = D3D12_HEAP_DESC { SizeInBytes: bytes, Properties: props, Alignment: 64 << 10, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
    let mut heap: Option<ID3D12Heap> = None;
    let mut buffer: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation, placement over the whole new heap, and mapping of a CPU-visible buffer.
    unsafe {
        device.CreateHeap(&desc, &mut heap)?;
        let heap = heap.ok_or_else(windows::core::Error::empty)?;
        device.CreatePlacedResource(&heap, 0, &buffer_desc(bytes), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)?;
        let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
        buffer.Map(0, None, Some(&mut data))?;
        Ok((heap, buffer, data.cast()))
    }
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// Most pages one reserved buffer covers (1 GiB).
const MOST_TILES: usize = 16384;

/// `VICTORY_ALIAS_HOLES`: 1 a title buffer or window over aliased sections' pages with pages the
/// title has not mapped among them (reserved, nothing committed there) is a reserved buffer too,
/// those pages' tiles mapped to nothing (they read zeros, as a copy reads them; shader writes there
/// are dropped, as the copy-back leaves them). Mapping anything there later lets the buffer go
/// (`crate::title_memory_remapped`). Needs `VICTORY_ALIAS_TILES=1`; 0 (the default) they are
/// copied. (Probe: pragyaangaur/VIctory-macOS#266.)
pub fn holes_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| tiles_enabled() && std::env::var("VICTORY_ALIAS_HOLES").is_ok_and(|v| v.trim() == "1"))
}

/// Whether nothing is committed in `[at, at + len)` (reserved or free, as Windows says now).
fn nothing_committed(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory as mem;
    let end = at + len;
    let mut p = at;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(p as *const _, &mut info, size_of_val(&info)) } == 0 || info.State == mem::MEM_COMMIT {
            return false;
        }
        let reach = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if reach <= p {
            return false;
        }
        p = reach;
    }
    true
}

/// A plan of tiles: the first page's address, the number of pages, the tile runs (chunk key:
/// section and chunk offset; first tile; first heap page; tiles), and the runs of tiles over pages
/// not mapped (first tile, tiles; only with [`holes_enabled`]).
type TilePlan = (u64, usize, Vec<((usize, u64), u32, u32, u32)>, Vec<(u32, u32)>);

/// Where `[start, start + len)`'s pages are, if every one is shown by a title view of an aliased
/// section with a heap (with [`holes_enabled`]: or is not mapped at all, and one at least is).
fn scattered_plan(start: u64, len: u64) -> Option<TilePlan> {
    const K64: u64 = 64 << 10;
    let page0 = start & !(K64 - 1);
    let n = (start + len.max(1) - page0).div_ceil(K64) as usize;
    if n > MOST_TILES {
        return None;
    }
    let shown: Vec<Option<(usize, u64)>> = if holes_enabled() {
        crate::inplace::pages_shown_or_holes(page0 as usize, n)?
    } else {
        crate::inplace::pages_shown(page0 as usize, n)?.into_iter().map(Some).collect()
    };
    let mut holes: Vec<(u32, u32)> = Vec::new();
    for (k, page) in shown.iter().enumerate() {
        if page.is_none() {
            match holes.last_mut() {
                Some((first, count)) if *first + *count == k as u32 => *count += 1,
                _ => holes.push((k as u32, 1)),
            }
        }
    }
    if !holes.is_empty() {
        let unmapped: u32 = holes.iter().map(|h| h.1).sum();
        if unmapped as usize == n || !holes.iter().all(|&(first, count)| nothing_committed(page0 as usize + first as usize * K64 as usize, count as usize * K64 as usize)) {
            return None;
        }
    }
    let s = sections();
    let mut runs: Vec<((usize, u64), u32, u32, u32)> = Vec::new();
    for (k, page) in shown.into_iter().enumerate() {
        let Some((section, offset)) = page else {
            continue;
        };
        let aliased = s.get(&section)?;
        let (&c_off, chunk) = aliased.chunks.range(..=offset).next_back()?;
        if chunk.heap.is_none() || offset + K64 > c_off + chunk.len {
            return None;
        }
        let page = ((offset - c_off) / K64) as u32;
        match runs.last_mut() {
            Some((key, first, heap_first, count)) if *key == (section, c_off) && *first + *count == k as u32 && *heap_first + *count == page => *count += 1,
            _ => runs.push(((section, c_off), k as u32, page, 1)),
        }
    }
    Some((page0, n, runs, holes))
}

/// Our queue for tile mappings, its fence and event, and the last value signalled.
struct TileQueue {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    value: u64,
}

// SAFETY: D3D12 objects and an event handle are free-threaded; the mutex orders their use.
unsafe impl Send for TileQueue {}

static TILE_QUEUE: Mutex<Option<TileQueue>> = Mutex::new(None);

/// A reserved buffer over `[start, start + len)`'s scattered pages, its tiles mapped onto the
/// sections' heaps and the mapping finished before it is handed out (so lists on any queue may use
/// it): the buffer, the GPU address of `start`, and `start`'s offset in it.
fn placed_scattered(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if !tiles_enabled() {
        return None;
    }
    let (page0, n, runs, holes) = scattered_plan(start, len)?;
    let began = std::time::Instant::now();
    let host = crate::host::host()?;
    let device = &host.device;
    let turned_down = || {
        POOL[10].fetch_add(1, Ordering::Relaxed);
        None
    };
    let mut reserved: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    if unsafe { device.CreateReservedResource(&buffer_desc(n as u64 * (64 << 10)), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved) }.is_err() {
        return turned_down();
    }
    let reserved = reserved?;
    // Runs by heap: one UpdateTileMappings a heap.
    let mut by_heap: BTreeMap<(usize, u64), Vec<(u32, u32, u32)>> = BTreeMap::new();
    for (key, first, heap_first, count) in runs {
        by_heap.entry(key).or_default().push((first, heap_first, count));
    }
    let heaps: Vec<(ID3D12Heap, Vec<(u32, u32, u32)>)> = {
        let s = sections();
        let mut out = Vec::new();
        for (key, runs) in by_heap {
            let heap = s.get(&key.0).and_then(|a| a.chunks.get(&key.1)).and_then(|c| c.heap.clone());
            let Some(heap) = heap else {
                return turned_down();
            };
            out.push((heap, runs));
        }
        out
    };
    let mut tq = TILE_QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if tq.is_none() {
        // SAFETY: plain creation.
        let made = unsafe {
            device.CreateCommandQueue::<ID3D12CommandQueue>(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).and_then(|queue| {
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
                Ok(TileQueue { queue, fence, event, value: 0 })
            })
        };
        match made {
            Ok(q) => *tq = Some(q),
            Err(e) => {
                tracing::warn!(error = %e, "no queue for tile mappings: scattered pages are copied");
                return turned_down();
            }
        }
    }
    let q = tq.as_mut()?;
    for (heap, runs) in &heaps {
        let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = runs.iter().map(|r| D3D12_TILED_RESOURCE_COORDINATE { X: r.0, ..Default::default() }).collect();
        let sizes: Vec<D3D12_TILE_REGION_SIZE> = runs.iter().map(|r| D3D12_TILE_REGION_SIZE { NumTiles: r.2, ..Default::default() }).collect();
        let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; runs.len()];
        let starts: Vec<u32> = runs.iter().map(|r| r.1).collect();
        let counts: Vec<u32> = runs.iter().map(|r| r.2).collect();
        // SAFETY: a live reserved buffer and heap; `runs.len()` regions and ranges in arrays of that
        // length, living through the call; each region lies in the buffer, each range in the heap.
        unsafe {
            q.queue.UpdateTileMappings(
                &reserved,
                runs.len() as u32,
                Some(coords.as_ptr()),
                Some(sizes.as_ptr()),
                heap,
                runs.len() as u32,
                Some(flags.as_ptr()),
                Some(starts.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }
    if !holes.is_empty() {
        // A reserved buffer's tiles start mapped to nothing; said once more, as the probe that
        // found NULL tiles read zeros on D3DMetal did.
        let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = holes.iter().map(|h| D3D12_TILED_RESOURCE_COORDINATE { X: h.0, ..Default::default() }).collect();
        let sizes: Vec<D3D12_TILE_REGION_SIZE> = holes.iter().map(|h| D3D12_TILE_REGION_SIZE { NumTiles: h.1, ..Default::default() }).collect();
        let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; holes.len()];
        let counts: Vec<u32> = holes.iter().map(|h| h.1).collect();
        // SAFETY: a live reserved buffer; `holes.len()` regions and ranges in arrays of that
        // length, living through the call; NULL ranges take no heap nor heap offsets.
        unsafe {
            q.queue.UpdateTileMappings(
                &reserved,
                holes.len() as u32,
                Some(coords.as_ptr()),
                Some(sizes.as_ptr()),
                None::<&ID3D12Heap>,
                holes.len() as u32,
                Some(flags.as_ptr()),
                None,
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        POOL[11].fetch_add(1, Ordering::Relaxed);
        POOL[12].fetch_add(holes.iter().map(|h| h.1 as u64).sum(), Ordering::Relaxed);
    }
    q.value += 1;
    // SAFETY: our queue, fence and event.
    let waited = unsafe {
        q.queue.Signal(&q.fence, q.value).is_ok()
            && q.fence.SetEventOnCompletion(q.value, q.event).is_ok()
            && windows::Win32::System::Threading::WaitForSingleObject(q.event, 5_000).0 == 0
    };
    drop(tq);
    if !waited {
        tracing::warn!("a tile mapping did not finish in 5 s: that buffer is copied");
        return turned_down();
    }
    crate::host::name(&reserved, || format!("title memory {start:#x}+{len:#x} over scattered aliased pages"));
    POOL[7].fetch_add(1, Ordering::Relaxed);
    POOL[8].fetch_add(n as u64, Ordering::Relaxed);
    POOL[9].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { reserved.GetGPUVirtualAddress() };
    let bias = start - page0;
    Some((reserved, gpu + bias, bias))
}

/// The runtime made a section of `bytes` (fresh, all zero) for physical pages, and has mapped no
/// view of it yet: back it by buffers of the card's.
pub fn title_section_made(section: usize, bytes: u64) {
    if !pool_enabled() {
        return;
    }
    // A handle number that came round again: the old section is gone (its views with it).
    title_section_gone(section);
    if bytes % (16 << 10) != 0 || STATS[1].load(Ordering::Relaxed) + POOL[1].load(Ordering::Relaxed) + bytes > most_bytes() {
        POOL[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Some(host) = crate::host::host() else {
        POOL[2].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(section, mb = bytes >> 20, "a section of physical pages made before the host: not aliased");
        return;
    };
    let mut chunks = BTreeMap::new();
    let mut offset = 0;
    while offset < bytes {
        let part = (bytes - offset).min(chunk_bytes());
        let made = if tiles_enabled() {
            match heap_buffer(&host.device, part) {
                Ok((heap, buffer, data)) => Some((buffer, data, Some(heap))),
                Err(e) => {
                    tracing::warn!(error = %e, part, "no heap for aliased physical pages");
                    None
                }
            }
        } else {
            host.shared_buffer(part).map(|(buffer, data)| (buffer, data, None))
        };
        let Some((buffer, data, heap)) = made else {
            POOL[2].fetch_add(1, Ordering::Relaxed);
            return;
        };
        // A fresh section reads zero. Fresh pages from the kernel are, but a buffer D3DMetal hands
        // out again from its own pool might not be (pragyaangaur/VIctory-macOS#207 probes for
        // this): one word a MiB is looked at, and the buffer cleared if any is not zero.
        // SAFETY: `data` maps `part` bytes of a live buffer of ours.
        unsafe {
            let words = data.cast::<u64>();
            if (0..part as usize).step_by(1 << 20).any(|at| words.add(at / 8).read_volatile() != 0) {
                tracing::warn!(section, part, "a buffer for physical pages did not read zero: cleared");
                std::ptr::write_bytes(data, 0, part as usize);
            }
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        crate::host::name(&buffer, || format!("physical pages, section {section:#x}+{offset:#x} aliased"));
        chunks.insert(offset, SectionChunk { len: part, buffer, gpu, data: data as u64, heap });
        offset += part;
    }
    POOL[0].fetch_add(1, Ordering::Relaxed);
    POOL[1].fetch_add(bytes, Ordering::Relaxed);
    tracing::info!(section, mb = bytes >> 20, chunks = chunks.len(), "physical pages backed by the card's buffers (VICTORY_ALIAS_POOL)");
    sections().insert(section, AliasedSection { len: bytes, chunks });
}

/// Wine just mapped `[at, at + len)` as a view of `section` from `offset` on (nothing has used the
/// view yet): point it at the section's buffers. False if the section is aliased and a part could
/// not be pointed (that part then shows the section's own pages, which nothing else shows: a
/// warning says so).
pub fn title_section_view(at: usize, len: usize, section: usize, offset: u64) -> bool {
    if !pool_enabled() || len == 0 {
        return true;
    }
    let s = sections();
    let Some(aliased) = s.get(&section) else {
        return true;
    };
    let (at, len) = (at as u64, len as u64);
    if offset + len > aliased.len || at % (16 << 10) != 0 || offset % (16 << 10) != 0 {
        POOL[5].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(at = format_args!("{at:#x}"), len, offset, "a view of aliased physical pages outside its section or not aligned: not pointed at the card's buffer");
        return false;
    }
    let began = std::time::Instant::now();
    let mut ok = true;
    for (&c_off, chunk) in aliased.chunks.range(..offset + len) {
        let from = offset.max(c_off);
        let to = (offset + len).min(c_off + chunk.len);
        if from >= to {
            continue;
        }
        let target = at + (from - offset);
        // SAFETY: `[target, + to - from)` is a view Wine just mapped for the runtime, unused yet;
        // the source is the mapping of a live buffer the table keeps, inside it.
        let r = unsafe { remap_shared(target, chunk.data + (from - c_off), to - from) };
        if r.msg != 0 || r.kr != 0 || r.address != target {
            ok = false;
            if POOL[5].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::error!(
                    at = format_args!("{target:#x}"),
                    bytes = to - from,
                    msg = format_args!("{:#x}", r.msg),
                    kr = r.kr,
                    "a view of aliased physical pages not pointed at the card's buffer: it shows stale bytes"
                );
            }
        }
    }
    POOL[3].fetch_add(1, Ordering::Relaxed);
    POOL[4].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    ok
}

/// The runtime closed `section` (no view of it is left): its buffers go when the last list or
/// placement holding them does.
pub fn title_section_gone(section: usize) {
    if !pool_enabled() {
        return;
    }
    if let Some(gone) = sections().remove(&section) {
        // Reserved buffers may still have tiles on these heaps: kept for good (sections live as
        // long as the title, so this is rare).
        KEPT.lock().unwrap_or_else(|e| e.into_inner()).extend(gone.chunks.values().filter(|c| c.heap.is_some()).map(|c| c.buffer.clone()));
        HEAPS_KEPT.lock().unwrap_or_else(|e| e.into_inner()).extend(gone.chunks.values().filter_map(|c| c.heap.clone()));
        POOL[0].fetch_sub(1, Ordering::Relaxed);
        POOL[1].fetch_sub(gone.len, Ordering::Relaxed);
        tracing::info!(section, mb = gone.len >> 20, "aliased physical pages let go");
    }
}

/// The line of what is aliased, for [`crate::inplace`]'s ten-second report.
pub(crate) fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let mut line = format!(
        "title memory aliased onto the card's buffers (VICTORY_ALIAS): {} allocations {} MB; turned down {}; remap failures {}; placements {}; let go {}",
        s[0],
        s[1] >> 20,
        s[2],
        s[3],
        s[4],
        s[5]
    );
    if pool_enabled() {
        let p: Vec<u64> = POOL.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        line += &format!(
            " | physical pages (VICTORY_ALIAS_POOL): {} sections {} MB, turned down {}; {} views pointed at them in {} ms ({:.1} us each), failures {}; placements in views {}",
            p[0],
            p[1] >> 20,
            p[2],
            p[3],
            p[4] / 1000,
            p[4] as f64 / p[3].max(1) as f64,
            p[5],
            p[6]
        );
        if tiles_enabled() {
            line += &format!(
                "; over scattered pages (VICTORY_ALIAS_TILES): {} placements {} tiles in {} ms ({:.0} us each), turned down {}",
                p[7],
                p[8],
                p[9] / 1000,
                p[9] as f64 / p[7].max(1) as f64,
                p[10]
            );
        }
        if holes_enabled() {
            line += &format!("; with pages not mapped (VICTORY_ALIAS_HOLES): {} placements, {} tiles to nothing", p[11], p[12]);
        }
    }
    Some(line)
}

// Shader writes into aliased memory.

/// `VICTORY_ALIAS_NOTIFY`: shader writes into aliased title memory are told to the write watching
/// and the texture passes, as the copy-back told them for the bytes that changed. In place nobody
/// knows which bytes changed, so a declared range is told whole, once per range between two
/// submissions' starts. (After skakwy/victory-mac `zerocopy.rs`.)
/// - unset or `1` (the default): [`crate::watch::gpu_wrote`] (guarded blocks there and at the
///   pages' other views count as written, with no protection change) and the textures over those
///   bytes are looked at again. (After Marvisak/VIctory 29decb37.)
/// - `open`: [`crate::watch::writing`], as before: the guarded blocks are opened (two protection
///   changes each, one now and one at the next check). 353 against 300 ms p50 with `0` (pp4).
/// - `guards`: `gpu_wrote` alone, no texture told (for measuring; textures may go stale).
/// - `0`: nothing told (for measuring: what shaders write is not seen by copies of it elsewhere).
fn notify() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_ALIAS_NOTIFY").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("open") => 2,
        Ok("guards") => 3,
        _ => 1,
    })
}

static DECLARED: Mutex<std::collections::BTreeSet<(usize, usize)>> = Mutex::new(std::collections::BTreeSet::new());

/// Shaders of a list being recorded may write aliased title memory `[address, address + len)`.
pub(crate) fn declared_in_place(address: usize, len: usize) {
    let mode = notify();
    if mode == 0 || len == 0 {
        return;
    }
    if !DECLARED.lock().unwrap_or_else(|e| e.into_inner()).insert((address, len)) {
        return;
    }
    match mode {
        2 => crate::watch::writing(address, len),
        3 => crate::watch::gpu_wrote(address, len),
        _ => {
            crate::watch::gpu_wrote(address, len);
            // A placed texture there changes with no descriptor written.
            crate::descriptor::host::remapped(address, len);
        }
    }
}

/// A submission's preparing begins: declared ranges are told again from now on.
pub(crate) fn submission_starts() {
    if enabled() {
        DECLARED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

// The Mach calls.

use core::arch::asm;

const MACH: u64 = 0x0100_0000;
const TRAP_REPLY_PORT: u64 = 26;
const TRAP_TASK_SELF: u64 = 28;
const TRAP_MSG2: u64 = 47;
/// `mach_msg2` option bits (osfmk/mach/message.h): a kernel-object call must carry this one and
/// be a combined send and receive.
const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
const VM_FLAGS_OVERWRITE: i32 = 0x4000;

/// A Mach trap: six arguments in registers, a seventh and eighth read by the kernel from the user
/// stack at `rsp + 8` and `rsp + 16` (x86-64 XNU's munger, which Rosetta follows).
unsafe fn trap(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
    let ret: u64;
    // SAFETY: the caller picks a trap whose arguments are valid; syscall clobbers rcx and r11.
    unsafe {
        asm!(
            "sub rsp, 32",
            "mov qword ptr [rsp], 0",
            "mov qword ptr [rsp + 8], {a7}",
            "mov qword ptr [rsp + 16], {a8}",
            "syscall",
            "add rsp, 32",
            a7 = in(reg) a7, a8 = in(reg) a8,
            inlateout("rax") MACH | n => ret,
            in("rdi") a[0], in("rsi") a[1], in("rdx") a[2], in("r10") a[3], in("r8") a[4], in("r9") a[5],
            lateout("rcx") _, lateout("r11") _,
        );
    }
    ret
}

/// The task's port and a reply port, made once (a reply port per call would leak names).
fn ports() -> (u32, u32) {
    static PORTS: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    // SAFETY: argument-less traps.
    *PORTS.get_or_init(|| unsafe { (trap(TRAP_TASK_SELF, [0; 6], 0, 0) as u32, trap(TRAP_REPLY_PORT, [0; 6], 0, 0) as u32) })
}

/// One `mach_vm_remap` at a time: they share the reply port.
static REMAP: Mutex<()> = Mutex::new(());

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct Header {
    bits: u32,
    size: u32,
    remote: u32,
    local: u32,
    voucher: u32,
    id: i32,
}

/// `mach_vm_remap`'s request (msgh_id 4813).
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct RemapRequest {
    head: Header,
    descriptors: u32,
    // mach_msg_port_descriptor_t: name, pad1, (pad2:16 | disposition:8 | type:8)
    src_task: u32,
    pad1: u32,
    disp_type: u32,
    ndr: [u8; 8],
    target_address: u64,
    size: u64,
    mask: u64,
    flags: i32,
    src_address: u64,
    copy: u32,
    inheritance: u32,
}

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct RemapReply {
    head: Header,
    ndr: [u8; 8],
    ret: i32,
    target_address: u64,
    cur_protection: i32,
    max_protection: i32,
    trailer: [u8; 128],
}

struct Remapped {
    /// `mach_msg2_trap`'s own result (0: sent and received).
    msg: u32,
    /// The routine's `kern_return_t` (0: KERN_SUCCESS).
    kr: i32,
    address: u64,
}

/// Maps `[src, src + size)` of this task again at `target`, replacing what is there, sharing the
/// pages (`copy = FALSE`).
///
/// # Safety
/// Replaces whatever is mapped at `target`; both ranges page aligned and owned by the caller.
unsafe fn remap_shared(target: u64, src: u64, size: u64) -> Remapped {
    let _one = REMAP.lock().unwrap_or_else(|e| e.into_inner());
    let (task, reply) = ports();
    let req = RemapRequest {
        head: Header {
            // MACH_MSGH_BITS(COPY_SEND = 19, MAKE_SEND_ONCE = 21) | MACH_MSGH_BITS_COMPLEX
            bits: 19 | (21 << 8) | 0x8000_0000,
            size: size_of::<RemapRequest>() as u32,
            remote: task,
            local: reply,
            voucher: 0,
            id: 4813,
        },
        descriptors: 1,
        src_task: task,
        pad1: 0,
        disp_type: 19 << 16, // disposition COPY_SEND, type PORT_DESCRIPTOR (0)
        ndr: [0, 0, 0, 0, 1, 0, 0, 0],
        target_address: target,
        size,
        mask: 0,
        flags: VM_FLAGS_OVERWRITE, // | VM_FLAGS_FIXED (0)
        src_address: src,
        copy: 0,
        inheritance: 1, // VM_INHERIT_COPY, the default
    };
    let mut msg = [0u8; size_of::<RemapReply>()];
    // SAFETY: plain bytes of the right sizes.
    unsafe { std::ptr::copy_nonoverlapping((&raw const req).cast::<u8>(), msg.as_mut_ptr(), size_of::<RemapRequest>()) };
    let pack = |lo: u32, hi: u32| (hi as u64) << 32 | lo as u64;
    // mach_msg2(data, SEND | RCV | KOBJECT_CALL, bits | send_size << 32, remote | local << 32,
    //           voucher | id << 32, descriptors | rcv_name << 32, rcv_size | priority << 32, timeout)
    let args = [
        msg.as_mut_ptr() as u64,
        3 | MACH64_SEND_KOBJECT_CALL,
        pack(req.head.bits, size_of::<RemapRequest>() as u32),
        pack(task, reply),
        pack(0, 4813),
        pack(1, reply),
    ];
    // SAFETY: `msg` holds the request and room for the reply and a trailer.
    let r = unsafe { trap(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
    // SAFETY: the kernel wrote a reply of at most `msg.len()` bytes (zeros if it did not).
    let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
    Remapped { msg: r, kr: rep.ret, address: rep.target_address }
}

/// A check of the whole path without the title: `VICTORY_ALIAS=1 victory alias-check` under Wine on
/// macOS (a CrossOver bottle with the D3DMetal backend). Panics with what failed.
pub mod check {
    use super::*;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory as mem;

    fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
        r.unwrap()
    }

    /// Records one copy, runs it on a queue of its own and waits.
    fn copy(device: &ID3D12Device, dst: &ID3D12Resource, dst_off: u64, src: &ID3D12Resource, src_off: u64, n: u64) {
        // SAFETY: objects of our own, one run, waited for.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.CopyBufferRegion(dst, dst_off, src, src_off, n);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0, "GPU copy timed out");
        }
    }

    /// Aliases 64 MiB the way the title's allocation is, places a "title buffer" in it through
    /// `crate::inplace::place_for`, and checks 8 rounds each way that CPU and GPU see one memory.
    pub fn run() {
        assert!(enabled(), "run with VICTORY_ALIAS=1 under Wine on macOS");
        const LEN: usize = 64 << 20;
        const AT: usize = 1 << 20;
        let device = crate::host::host().expect("host device").device.clone();
        // SAFETY: memory of the test's own, LEN bytes; every access lies inside it.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), LEN, mem::MEM_RESERVE | mem::MEM_COMMIT | 0x20_0000, mem::PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            title_memory_aliasable(base as usize, LEN);
            assert!(covers(base as u64 + AT as u64, 4096), "allocation not aliased: {:?}", report());
            // What a title buffer at base + 1 MiB gets from the in-place path.
            let placed = crate::inplace::place_for(base as u64 + AT as u64, 4096, false).expect("placed");
            assert_eq!(placed.bias, AT as u64);
            assert!(placed.heap.is_none());
            for round in 0..8u8 {
                // CPU writes at the title's address; the GPU reads them through the placed buffer.
                let fill: Vec<u8> = (0..4096).map(|i| (i as u8).wrapping_mul(31) ^ round).collect();
                std::ptr::copy_nonoverlapping(fill.as_ptr(), base.add(AT), 4096);
                let back = committed(&device, D3D12_HEAP_TYPE_READBACK, 4096, D3D12_RESOURCE_STATE_COPY_DEST);
                copy(&device, &back, 0, &placed.buffer, placed.bias, 4096);
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                assert_eq!(std::slice::from_raw_parts(p.cast::<u8>(), 4096), &fill[..], "GPU did not see the CPU's write, round {round}");
                back.Unmap(0, None);
                // The GPU writes the placed buffer; the CPU reads it at the title's address.
                let new: Vec<u8> = (0..4096).map(|i| (i as u8) ^ round ^ 0xa5).collect();
                let up = committed(&device, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
                up.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(new.as_ptr(), p.cast::<u8>(), 4096);
                up.Unmap(0, None);
                copy(&device, &placed.buffer, placed.bias + 65536, &up, 0, 4096);
                assert_eq!(std::slice::from_raw_parts(base.add(AT + 65536), 4096), &new[..], "CPU did not see the GPU's write, round {round}");
            }
            // The title's VirtualProtect still works on aliased memory.
            let mut old = 0;
            assert_ne!(mem::VirtualProtect(base.add(AT).cast(), 65536, mem::PAGE_READONLY, &mut old), 0);
            assert_ne!(mem::VirtualProtect(base.add(AT).cast(), 65536, mem::PAGE_READWRITE, &mut old), 0);
            title_memory_alias_freed(base as usize, 0, true);
            assert!(!covers(base as u64 + AT as u64, 4096));
            assert_ne!(mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE), 0);
            assert!(device.GetDeviceRemovedReason().is_ok());
            if pool_enabled() {
                pool(&device);
            }
            if std::env::var_os("VICTORY_ALIAS_CHECK_TILES").is_some() {
                tiles(&device);
            }
            println!("{}", report().unwrap_or_default());
        }
    }

    /// For pages scattered over the title's address space (`VICTORY_ALIAS_CHECK_TILES=1`): a heap of
    /// the card's in the PC's memory, a buffer placed over all of it and kept mapped (the CPU's
    /// side, where views would point), and a reserved buffer whose 64 KiB tiles are mapped onto
    /// pages of that heap out of order. Checks that the GPU sees CPU writes through the reserved
    /// buffer and the CPU sees GPU writes into it, and what one tile mapping update costs.
    unsafe fn tiles(device: &ID3D12Device) {
        const K64: u64 = 64 << 10;
        const PAGES: u64 = 256;
        // SAFETY (whole body): objects of the check's own; accesses lie in the mapped buffer.
        unsafe {
            let props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let mut heap: Option<ID3D12Heap> = None;
            let hd = D3D12_HEAP_DESC { SizeInBytes: PAGES * K64, Properties: props, Alignment: K64, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            device.CreateHeap(&hd, &mut heap).expect("CreateHeap custom write-back L0");
            let heap = heap.unwrap();
            let desc = |size: u64, layout| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: layout,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut whole: Option<ID3D12Resource> = None;
            device.CreatePlacedResource(&heap, 0, &desc(PAGES * K64, D3D12_TEXTURE_LAYOUT_ROW_MAJOR), D3D12_RESOURCE_STATE_COMMON, None, &mut whole).expect("placed buffer over the heap");
            let whole = whole.unwrap();
            let mut p = std::ptr::null_mut();
            whole.Map(0, None, Some(&mut p)).expect("Map of a placed buffer in a custom heap");
            let base = p.cast::<u8>();
            println!("tiles check: heap {} MiB, placed buffer mapped at {:p}", (PAGES * K64) >> 20, base);
            let mut reserved: Option<ID3D12Resource> = None;
            device.CreateReservedResource(&desc(8 * K64, D3D12_TEXTURE_LAYOUT_ROW_MAJOR), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("reserved buffer");
            let reserved = reserved.unwrap();
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let mut value = 0;
            let mut wait = |queue: &ID3D12CommandQueue| {
                value += 1;
                queue.Signal(&fence, value).unwrap();
                fence.SetEventOnCompletion(value, event).unwrap();
                assert_eq!(WaitForSingleObject(event, 10_000).0, 0, "queue timed out");
            };
            // Tiles 0..8 of the reserved buffer onto heap pages 7, 3, 7, 200, 5, 6, 100, 1.
            let map = |queue: &ID3D12CommandQueue, tile: u32, page: u32| {
                let at = D3D12_TILED_RESOURCE_COORDINATE { X: tile, ..Default::default() };
                let size = D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() };
                let flags = D3D12_TILE_RANGE_FLAG_NONE;
                let count = 1u32;
                queue.UpdateTileMappings(&reserved, 1, Some(&at), Some(&size), &heap, 1, Some(&flags), Some(&page), Some(&count), D3D12_TILE_MAPPING_FLAG_NONE);
            };
            let pages = [7u32, 3, 7, 200, 5, 6, 100, 1];
            for (t, &pg) in pages.iter().enumerate() {
                map(&queue, t as u32, pg);
            }
            wait(&queue);
            for round in 0..8u8 {
                for (t, &pg) in pages.iter().enumerate() {
                    if t == 2 {
                        continue; // page 7 again
                    }
                    std::ptr::write_bytes(base.add((pg as u64 * K64) as usize), round.wrapping_mul(16) ^ t as u8, 4096);
                }
                let back = committed(device, D3D12_HEAP_TYPE_READBACK, 8 * K64, D3D12_RESOURCE_STATE_COPY_DEST);
                copy(device, &back, 0, &reserved, 0, 8 * K64);
                let mut q = std::ptr::null_mut();
                back.Map(0, None, Some(&mut q)).unwrap();
                for (t, &_pg) in pages.iter().enumerate() {
                    let want = round.wrapping_mul(16) ^ if t == 2 { 0 } else { t as u8 };
                    let got = *q.cast::<u8>().add(t * K64 as usize);
                    assert_eq!(got, want, "GPU read of reserved tile {t} wrong, round {round}");
                }
                back.Unmap(0, None);
                // The GPU writes reserved tile 3; the CPU sees heap page 200.
                let up = committed(device, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
                up.Map(0, None, Some(&mut q)).unwrap();
                std::ptr::write_bytes(q.cast::<u8>(), 0xa0 ^ round, 4096);
                up.Unmap(0, None);
                copy(device, &reserved, 3 * K64 + 8192, &up, 0, 4096);
                assert_eq!(*base.add((200 * K64 + 8192) as usize), 0xa0 ^ round, "CPU did not see the GPU's write through a tile, round {round}");
            }
            // A Wine view pointed at heap page 5 through the mapping; the GPU reads it via tile 4.
            let view = mem::VirtualAlloc(std::ptr::null(), K64 as usize, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE).cast::<u8>();
            let r = remap_shared(view as u64, base as u64 + 5 * K64, K64);
            assert!(r.msg == 0 && r.kr == 0, "remap onto the placed buffer's mapping: {:#x} {}", r.msg, r.kr);
            std::ptr::write_bytes(view, 0x5e, 4096);
            let back = committed(device, D3D12_HEAP_TYPE_READBACK, K64, D3D12_RESOURCE_STATE_COPY_DEST);
            copy(device, &back, 0, &reserved, 4 * K64, K64);
            let mut q = std::ptr::null_mut();
            back.Map(0, None, Some(&mut q)).unwrap();
            assert_eq!(*q.cast::<u8>(), 0x5e, "GPU did not see a write through a view pointed at the heap");
            back.Unmap(0, None);
            // Cost: remapping one tile and waiting, and 64 tiles in one call (one range each).
            let n = 500;
            let t = std::time::Instant::now();
            for i in 0..n {
                map(&queue, (i % 8) as u32, 10 + (i % 64) as u32);
            }
            let calls = t.elapsed();
            wait(&queue);
            println!(
                "tiles check: {n} single-tile UpdateTileMappings {:.1} us a call, all done {:.1} ms after the first",
                calls.as_secs_f64() * 1e6 / n as f64,
                t.elapsed().as_secs_f64() * 1e3
            );
            let t = std::time::Instant::now();
            for _ in 0..20 {
                wait(&queue);
            }
            println!("tiles check: an empty signal + wait {:.1} us", t.elapsed().as_secs_f64() * 1e6 / 20.0);
            assert!(device.GetDeviceRemovedReason().is_ok(), "device removed");
            println!("tiles check: passed");
        }
    }

    /// The physical-page path (`VICTORY_ALIAS_POOL=1`): a section backed by the card's buffer, views
    /// of it in placeholders as `xmem` maps them, two views of one page, a run placed through
    /// `place_for`, both directions of coherence, `VirtualQuery` / `VirtualProtect` on a view, and
    /// what one 64 KiB view costs with and without being pointed at the buffer.
    unsafe fn pool(device: &ID3D12Device) {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const K64: usize = 64 << 10;
        const PAGES: usize = 64;
        // SAFETY (whole body): a section and a placeholder of the check's own; every access lies
        // in a view mapped just before.
        unsafe {
            let section = mem::CreateFileMappingW(-1isize as _, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, (PAGES * K64) as u32, std::ptr::null()) as usize;
            assert_ne!(section, 0);
            crate::inplace::title_memory_section(section, (PAGES * K64) as u64, true);
            assert!(sections().contains_key(&section), "section not aliased: {:?}", report());
            let base = mem::VirtualAlloc2(GetCurrentProcess(), std::ptr::null(), 16 * K64, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0) as usize;
            assert_ne!(base, 0);
            // As xmem's `map`: split the placeholder, map the view, point it at the buffer, tell the census.
            let map = |at: usize, page: usize, n: usize, alias: bool| {
                mem::VirtualFree(at as *mut _, n * K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
                let view = mem::MapViewOfFile3(section as _, GetCurrentProcess(), at as *const _, (page * K64) as u64, n * K64, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
                assert_eq!(view.Value as usize, at, "MapViewOfFile3: {}", std::io::Error::last_os_error());
                if alias {
                    assert!(title_section_view(at, n * K64, section, (page * K64) as u64));
                }
                crate::inplace::title_memory_view(at, n * K64, section, (page * K64) as u64);
            };
            let unmap = |at: usize, n: usize| {
                crate::inplace::views_gone(at, n * K64);
                assert_ne!(mem::UnmapViewOfFile2(GetCurrentProcess(), mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ }, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            };
            // Page 3 at slot 0 and at slot 5; pages 10..13 as one run at slots 1..4.
            map(base, 3, 1, true);
            map(base + 5 * K64, 3, 1, true);
            map(base + K64, 10, 4, true);
            let slot = |k: usize| (base + k * K64) as *mut u8;
            for round in 0..8u8 {
                let fill: Vec<u8> = (0..K64).map(|i| (i as u8).wrapping_mul(7) ^ round).collect();
                std::ptr::copy_nonoverlapping(fill.as_ptr(), slot(0), K64);
                assert_eq!(std::slice::from_raw_parts(slot(5), K64), &fill[..], "the other view of a page differs, round {round}");
            }
            // A title buffer over the run is the section's buffer at page 10.
            let placed = crate::inplace::place_for((base + K64 + 4096) as u64, 2 * K64 as u64, false).expect("run placed");
            assert_eq!(placed.bias, (10 * K64 + 4096) as u64);
            assert!(covers((base + K64) as u64, 4 * K64 as u64));
            assert!(tiles_enabled() || !covers(base as u64, 2 * K64 as u64), "page 3 then page 10 is not one run");
            for round in 0..8u8 {
                let fill: Vec<u8> = (0..4096).map(|i| (i as u8).wrapping_mul(13) ^ round).collect();
                std::ptr::copy_nonoverlapping(fill.as_ptr(), slot(1).add(4096), 4096);
                let back = committed(device, D3D12_HEAP_TYPE_READBACK, 4096, D3D12_RESOURCE_STATE_COPY_DEST);
                copy(device, &back, 0, &placed.buffer, placed.bias, 4096);
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                assert_eq!(std::slice::from_raw_parts(p.cast::<u8>(), 4096), &fill[..], "GPU did not see a write through a view, round {round}");
                back.Unmap(0, None);
                let new: Vec<u8> = (0..4096).map(|i| (i as u8) ^ round ^ 0x3c).collect();
                let up = committed(device, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
                up.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(new.as_ptr(), p.cast::<u8>(), 4096);
                up.Unmap(0, None);
                // Into page 12 (slot 3) through the buffer.
                copy(device, &placed.buffer, placed.bias + 2 * K64 as u64 - 4096, &up, 0, 4096);
                assert_eq!(std::slice::from_raw_parts(slot(3), 4096), &new[..], "a view did not see the GPU's write, round {round}");
            }
            // Scattered pages (page 3, then 10, 11 at slots 0..3): a reserved buffer over them.
            if tiles_enabled() {
                assert!(covers(base as u64, 3 * K64 as u64), "scattered pages not coverable: {:?}", report());
                let placed = crate::inplace::place_for((base + 100) as u64, (3 * K64 - 200) as u64, false).expect("scattered placed");
                assert_eq!(placed.bias, 100);
                for round in 0..4u8 {
                    for k in 0..3 {
                        std::ptr::write_bytes(slot(k).add(4096), 0x11 * (k as u8 + 1) ^ round, 4096);
                    }
                    let back = committed(device, D3D12_HEAP_TYPE_READBACK, 3 * K64 as u64, D3D12_RESOURCE_STATE_COPY_DEST);
                    copy(device, &back, 0, &placed.buffer, 0, 3 * K64 as u64);
                    let mut p = std::ptr::null_mut();
                    back.Map(0, None, Some(&mut p)).unwrap();
                    for k in 0..3 {
                        assert_eq!(*p.cast::<u8>().add(k * K64 + 4096), 0x11 * (k as u8 + 1) ^ round, "GPU read of scattered page {k} wrong, round {round}");
                    }
                    back.Unmap(0, None);
                    // The GPU writes page 10 through the reserved buffer; the view at slot 1 sees it.
                    let up = committed(device, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
                    up.Map(0, None, Some(&mut p)).unwrap();
                    std::ptr::write_bytes(p.cast::<u8>(), 0xc3 ^ round, 4096);
                    up.Unmap(0, None);
                    copy(device, &placed.buffer, K64 as u64 + 8192, &up, 0, 4096);
                    assert_eq!(*slot(1).add(8192), 0xc3 ^ round, "a view did not see the GPU's write through a tile, round {round}");
                }
                println!("pool check: scattered pages through a reserved buffer: passed ({:?})", report());
            }
            // Pages not mapped among them (page 13, page 3, nothing, nothing at slots 4..8): with
            // VICTORY_ALIAS_HOLES a reserved buffer whose last two tiles read zeros.
            if holes_enabled() {
                assert!(!covers((base + 6 * K64) as u64, 2 * K64 as u64), "pages with nothing mapped covered");
                let placed = crate::inplace::place_for((base + 4 * K64) as u64, 4 * K64 as u64, false).expect("pages with holes placed");
                std::ptr::write_bytes(slot(4), 0x4d, K64);
                std::ptr::write_bytes(slot(5), 0x5d, K64);
                let back = committed(device, D3D12_HEAP_TYPE_READBACK, 4 * K64 as u64, D3D12_RESOURCE_STATE_COPY_DEST);
                copy(device, &back, 0, &placed.buffer, placed.bias, 4 * K64 as u64);
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                let got = std::slice::from_raw_parts(p.cast::<u8>(), 4 * K64);
                assert!(got[..K64].iter().all(|&b| b == 0x4d) && got[K64..2 * K64].iter().all(|&b| b == 0x5d), "GPU read of mapped pages beside holes wrong");
                assert!(got[2 * K64..].iter().all(|&b| b == 0), "a tile over a page not mapped did not read zeros");
                back.Unmap(0, None);
                // The GPU writes page 3 (tile 1) and a hole; the views of page 3 see the first.
                let up = committed(device, D3D12_HEAP_TYPE_UPLOAD, 2 * K64 as u64, D3D12_RESOURCE_STATE_GENERIC_READ);
                up.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::write_bytes(p.cast::<u8>(), 0xe7, 2 * K64);
                up.Unmap(0, None);
                copy(device, &placed.buffer, placed.bias + K64 as u64, &up, 0, 2 * K64 as u64);
                assert_eq!(*slot(0).add(4096), 0xe7, "a view did not see the GPU's write through a tile beside a hole");
                assert!(device.GetDeviceRemovedReason().is_ok(), "device removed after a write into a tile mapped to nothing");
                // Something committed in the hole: not placed.
                let committed_hole = mem::VirtualAlloc(std::ptr::null(), 2 * K64, mem::MEM_RESERVE, mem::PAGE_NOACCESS) as usize;
                assert!(super::nothing_committed(committed_hole, 2 * K64));
                mem::VirtualAlloc(committed_hole as *const _, 4096, mem::MEM_COMMIT, mem::PAGE_READWRITE);
                assert!(!super::nothing_committed(committed_hole, 2 * K64));
                mem::VirtualFree(committed_hole as *mut _, 0, mem::MEM_RELEASE);
                println!("pool check: pages with holes through a reserved buffer: passed ({:?})", report());
            }
            // Wine still sees a mapped, committed, read-write view; guards work.
            let mut info: mem::MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(mem::VirtualQuery(slot(1) as *const _, &mut info, size_of_val(&info)), 0);
            assert!(info.State == mem::MEM_COMMIT && info.Type == mem::MEM_MAPPED && info.Protect == mem::PAGE_READWRITE, "VirtualQuery: state {:#x} type {:#x} protect {:#x}", info.State, info.Type, info.Protect);
            let mut old = 0;
            assert_ne!(mem::VirtualProtect(slot(1).cast(), K64, mem::PAGE_READONLY, &mut old), 0);
            assert_ne!(mem::VirtualProtect(slot(1).cast(), K64, mem::PAGE_READWRITE, &mut old), 0);
            // Remapping a slot: what the page held is still there in the new view.
            let held = std::slice::from_raw_parts(slot(3), 4096).to_vec();
            unmap(base + K64, 4);
            map(base + 8 * K64, 12, 1, true);
            assert_eq!(std::slice::from_raw_parts(slot(8), 4096), &held[..], "a page remapped elsewhere lost its bytes");
            // Cost of one 64 KiB view: map + unmap, with and without the remap.
            for alias in [false, true, false, true] {
                let n = 2000;
                let t = std::time::Instant::now();
                for i in 0..n {
                    map(base + 10 * K64, 20 + i % 40, 1, alias);
                    unmap(base + 10 * K64, 1);
                }
                println!("pool check: map + unmap of one 64 KiB view{}: {:.1} us", if alias { " + remap" } else { "" }, t.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            let p: Vec<u64> = POOL.iter().map(|s| s.load(Ordering::Relaxed)).collect();
            println!("pool check: {} views pointed, {:.1} us each on average, failures {}", p[3], p[4] as f64 / p[3].max(1) as f64, p[5]);
            assert_eq!(p[5], 0);
            for at in [base, base + 5 * K64, base + 8 * K64] {
                unmap(at, 1);
            }
            drop(placed);
            title_section_gone(section);
            assert!(device.GetDeviceRemovedReason().is_ok());
            windows_sys::Win32::Foundation::CloseHandle(section as _);
            println!("pool check: passed");
        }
    }
}
