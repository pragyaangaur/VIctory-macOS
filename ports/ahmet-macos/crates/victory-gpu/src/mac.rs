//! What running under Wine on macOS (CrossOver, D3DMetal, Rosetta 2) needs
//! that a Windows PC does not. Kept in one place so the layer stays thin
//! when the base tree moves; the rest of the crate calls in with one-line
//! hooks.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS, ID3D12Device, ID3D12Fence,
};
use windows::Win32::System::Threading::WaitForSingleObject;
use windows::core::{Interface, PCWSTR, w};

/// The OS Wine runs on ("Darwin", "Linux"), if running under Wine.
pub(crate) fn wine_host() -> Option<&'static str> {
    static HOST: OnceLock<Option<String>> = OnceLock::new();
    HOST.get_or_init(|| {
        use std::ffi::CStr;
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        type HostVersion = unsafe extern "C" fn(*mut *const i8, *mut *const i8);
        // SAFETY: ntdll is always loaded; wine_get_host_version is a Wine export.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            let f = GetProcAddress(ntdll, c"wine_get_host_version".as_ptr().cast())?;
            let f: HostVersion = std::mem::transmute(f);
            let (mut sys, mut rel) = (std::ptr::null(), std::ptr::null());
            f(&mut sys, &mut rel);
            (!sys.is_null()).then(|| CStr::from_ptr(sys).to_string_lossy().into_owned())
        }
    })
    .as_deref()
}

/// Whether we run under Wine on macOS (D3DMetal, Rosetta 2).
pub fn is_darwin() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wine_host().is_some_and(|s| s.eq_ignore_ascii_case("Darwin")))
}

/// A switch that defaults one way on the Mac and the base tree's way
/// elsewhere: `name=0` off, any other value on, unset `mac` on Darwin and
/// `elsewhere` otherwise.
pub(crate) fn switch(name: &str, mac: bool, elsewhere: bool) -> bool {
    match std::env::var_os(name) {
        Some(v) => v != "0",
        None if is_darwin() => mac,
        None => elsewhere,
    }
}

/// Whether this run is a scripted background test (`VICTORY_BACKGROUND=1`):
/// window past the desktop's corner and never activated, audio muted, the
/// real keyboard not read. Agents' runs only; never set in a play copy.
pub fn background() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACKGROUND").is_some_and(|v| v != "0"))
}

/// In a background run: where a top-level window goes so that it covers
/// nothing of the work of whoever sits at the screen: past the bottom right
/// corner of the desktop (all monitors). At a fixed place inside a large
/// desktop macOS moved it back onto the screen, where it was taken for the
/// play copy (2026-11-25). `None` outside such a run, or with
/// `VICTORY_BACKGROUND_WINDOW=visible`.
pub fn background_window() -> Option<(i32, i32)> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    };
    if !background() || std::env::var("VICTORY_BACKGROUND_WINDOW").is_ok_and(|v| v.trim() == "visible") {
        return None;
    }
    // SAFETY: plain queries.
    unsafe {
        Some((
            GetSystemMetrics(SM_XVIRTUALSCREEN) + GetSystemMetrics(SM_CXVIRTUALSCREEN) + 64,
            GetSystemMetrics(SM_YVIRTUALSCREEN) + GetSystemMetrics(SM_CYVIRTUALSCREEN) + 64,
        ))
    }
}

/// In a background run: puts a top-level window just shown behind every
/// other window, without activating it.
pub fn background_behind(hwnd: *mut std::ffi::c_void) {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GA_ROOT, GetAncestor, HWND_BOTTOM, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetWindowPos,
    };
    if !background() || hwnd.is_null() {
        return;
    }
    let hwnd = HWND(hwnd);
    // SAFETY: plain calls on a window of ours.
    unsafe {
        if GetAncestor(hwnd, GA_ROOT) == hwnd {
            let _ = SetWindowPos(hwnd, Some(HWND_BOTTOM), 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);
        }
    }
}

/// The branch and commit this was built from (`build.rs`).
pub(crate) const BUILD: &str = env!("VICTORY_BUILD");

/// What the window caption says after the title's own text: which build, from
/// which folder, whether the keyboard is read, and how it runs. Several builds
/// run side by side on one machine (Ahmet's play copies and agents' tests), and
/// a window has to say which one it is (`xcl-mac/AGENTS.md`).
pub(crate) fn caption(fps: f64, ms: f64, draws: f64) -> (String, String) {
    let folder = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "?".into());
    let bg = background();
    let prefix = if bg { "[background test run] " } else { "" };
    let kb = if bg { "keyboard off" } else { "keyboard on" };
    let rate = if ms >= 1000.0 {
        format!("{:.1} s/frame", ms / 1000.0)
    } else {
        format!("{fps:.0} fps {ms:.1} ms/frame")
    };
    let gb = working_set_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    static FIRST: OnceLock<std::time::Instant> = OnceLock::new();
    let up = FIRST.get_or_init(std::time::Instant::now).elapsed().as_secs();
    let suffix = format!(
        " | VIctory {BUILD} | {folder}, {kb} | {rate} | {draws:.0} draws | {gb:.1} GB | up {}:{:02}",
        up / 60,
        up % 60
    );
    (prefix.to_owned(), suffix)
}

fn working_set_bytes() -> usize {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a plain query into a zeroed struct of ours.
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 { c.WorkingSetSize } else { 0 }
    }
}

/// Waits until `fence` reaches `value` (after `SetEventOnCompletion(value,
/// event)`), waking every second to read the fence itself. Under D3DMetal a
/// submission can complete without setting its event
/// (`pragyaangaur/VIctory-macOS#50`), and an `INFINITE` wait then hangs the
/// run. Reading the fence on every wake also keeps a late event from ending a
/// later wait early. Off the Mac this is the plain infinite wait.
pub(crate) fn wait_fence(fence: &ID3D12Fence, value: u64, event: HANDLE) {
    if !is_darwin() {
        // SAFETY: the caller's event.
        unsafe { WaitForSingleObject(event, windows::Win32::System::Threading::INFINITE) };
        return;
    }
    static MISSED: AtomicU64 = AtomicU64::new(0);
    let mut seconds = 0u32;
    loop {
        // SAFETY: the caller's event and fence.
        let woke = unsafe { WaitForSingleObject(event, 1000) };
        // SAFETY: as above.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed >= value {
            if woke != WAIT_OBJECT_0 {
                let n = MISSED.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(n, value, completed, "a host submission finished without setting its event");
                }
            }
            return;
        }
        if woke != WAIT_OBJECT_0 {
            seconds += 1;
            if seconds.is_power_of_two() {
                tracing::warn!(seconds, value, completed, "still waiting for the host GPU");
            }
        }
    }
}

static HOST_MAX_WAVE: AtomicU32 = AtomicU32::new(64);

/// Notes the host GPU's widest wave (32 on Apple GPUs).
pub(crate) fn set_host_max_wave(lanes: u32) {
    HOST_MAX_WAVE.store(lanes, Ordering::Relaxed);
}

/// The host GPU's widest wave.
pub(crate) fn host_max_wave() -> u32 {
    HOST_MAX_WAVE.load(Ordering::Relaxed)
}

/// Reads a file without `File::metadata()` (`GetFileInformationByHandle`,
/// `NtQueryVolumeInformationFile`), which deadlocks under CrossOver when
/// Wine's mount manager completes an ioctl inside its virtual memory lock
/// while the title suspends threads.
pub(crate) fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        match file.read(&mut chunk) {
            Ok(0) => return Ok(out),
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Loads `dxcompiler.dll` from `VICTORY_DXC_DIR`, beside our executable, or
/// the DLL search path.
fn load_dxc() -> windows::core::Result<windows::Win32::Foundation::HMODULE> {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::System::LibraryLoader::LoadLibraryW;
    let try_dir = |dir: &std::path::Path| {
        let p = dir.join("dxcompiler.dll");
        let wide: Vec<u16> = p.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: NUL-terminated path.
        unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) }
    };
    if let Some(dir) = std::env::var_os("VICTORY_DXC_DIR")
        && let Ok(m) = try_dir(std::path::Path::new(&dir))
    {
        return Ok(m);
    }
    if let Some(dir) = std::env::current_exe().ok().as_deref().and_then(std::path::Path::parent)
        && let Ok(m) = try_dir(dir)
    {
        return Ok(m);
    }
    // SAFETY: static wide string.
    unsafe { LoadLibraryW(w!("dxcompiler.dll")) }
}

/// Whether one of our own HLSL programs should go to DXC first: under Wine,
/// `d3dcompiler_47` (vkd3d-shader) ignores `[RootSignature]` and makes DXBC
/// that `CreateRootSignature` refuses.
pub(crate) fn dxc_first(source: &str) -> bool {
    wine_host().is_some() && source.contains("[RootSignature")
}

/// Compiles HLSL with `dxcompiler.dll` (SM 5.x targets as 6.0), for when
/// `D3DCompile` refuses or drops the embedded root signature.
pub(crate) fn dxc_compile(source: &str, entry: &str, target: &str) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Dxc::*;
    use windows::Win32::System::LibraryLoader::GetProcAddress;
    type Create = unsafe extern "system" fn(
        *const windows::core::GUID,
        *const windows::core::GUID,
        *mut *mut std::ffi::c_void,
    ) -> windows::core::HRESULT;
    let t6 = match target {
        "cs_5_0" | "cs_5_1" => "cs_6_0",
        "vs_5_0" | "vs_5_1" => "vs_6_0",
        "ps_5_0" | "ps_5_1" => "ps_6_0",
        other => other,
    };
    let entry_w: Vec<u16> = entry.encode_utf16().chain([0]).collect();
    let target_w: Vec<u16> = t6.encode_utf16().chain([0]).collect();
    // SAFETY: dxcompiler.dll's documented COM factory; buffers live through the calls.
    unsafe {
        let library = load_dxc()?;
        let create: Create = std::mem::transmute(
            GetProcAddress(library, windows::core::s!("DxcCreateInstance"))
                .ok_or_else(windows::core::Error::from_thread)?,
        );
        let mut raw = std::ptr::null_mut();
        create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw).ok()?;
        let compiler = IDxcCompiler3::from_raw(raw);
        let buffer = DxcBuffer { Ptr: source.as_ptr().cast(), Size: source.len(), Encoding: DXC_CP_UTF8.0 };
        let arguments = [w!("-T"), PCWSTR(target_w.as_ptr()), w!("-E"), PCWSTR(entry_w.as_ptr()), w!("-O3")];
        let result: IDxcResult = compiler.Compile(&buffer, Some(&arguments), None)?;
        let status = result.GetStatus()?;
        if status.is_err() {
            if let Ok(errors) = result.GetErrorBuffer() {
                let bytes = std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize());
                tracing::warn!(target = t6, entry, text = %String::from_utf8_lossy(bytes).trim(), "DXC refused a shader of ours");
            }
            return Err(windows::core::Error::from_hresult(status));
        }
        let blob = result.GetResult()?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec())
    }
}

/// The wave width a DXIL container's `PSV0` part fixes, if it fixes one.
pub(crate) fn wave_lanes(dxil: &[u8]) -> Option<u32> {
    let u32_at = |b: &[u8], at: usize| b.get(at..at + 4).map(|v| u32::from_le_bytes(v.try_into().expect("4 bytes")) as usize);
    let info = (0..u32_at(dxil, 28)?).find_map(|i| {
        let at = u32_at(dxil, 32 + 4 * i)?;
        (dxil.get(at..at + 4)? == b"PSV0").then(|| dxil.get(at + 8..at + 8 + u32_at(dxil, at + 4)?)).flatten()
    })?;
    let (min, max) = (u32_at(info, 4 + 16)? as u32, u32_at(info, 4 + 20)? as u32);
    (min != 0 && min == max).then_some(min)
}

/// D3DMetal can abort in the converter on a double-precision wave operation.
/// Respect the adapter's reported lack of FP64 before creating that pipeline.
pub(crate) fn refuses_compute_doubles(device: &ID3D12Device, dxil: &[u8]) -> bool {
    use victory_shader::{Container, FourCC};

    // Only with the paired Wave64 form on (`VICTORY_WAVE_PAIRS`), which is
    // what newly sends these programs to Apple's converter: without it the
    // pipelines made before stay made.
    static PAIRS: OnceLock<bool> = OnceLock::new();
    if !is_darwin() || !*PAIRS.get_or_init(|| crate::switch::on("VICTORY_WAVE_PAIRS")) {
        return false;
    }
    let Ok(container) = Container::parse(dxil) else { return false };
    let Some(features) = container.part(FourCC::SFI0) else { return false };
    let Some(bits) = features.data.get(..8).and_then(|bytes| bytes.try_into().ok()) else { return false };
    // D3D_SHADER_REQUIRES_DOUBLES in the container's feature mask.
    if u64::from_le_bytes(bits) & 1 == 0 {
        return false;
    }
    static NO_DOUBLES: OnceLock<bool> = OnceLock::new();
    *NO_DOUBLES.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: the feature query writes exactly the provided options structure.
        let ok = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                std::mem::size_of_val(&options) as u32,
            )
        };
        ok.is_ok() && !options.DoublePrecisionFloatShaderOps.as_bool()
    })
}

/// Diagnostics: every frame's time, logged every five seconds in the form the
/// Mac lanes' tools read (`framelog: frame times: ... each: <ms> <ms> ...`,
/// `xcl-mac/lanes/gpu-overlap/frames.py`), so runs of this tree and of the
/// older one are measured the same way. Called at the end of each present.
#[cfg(feature = "diagnostics")]
pub(crate) fn presented() {
    use std::sync::Mutex;
    struct Frames {
        last: std::time::Instant,
        began: std::time::Instant,
        each: Vec<u32>,
        drawn: u64,
    }
    static FRAMES: Mutex<Option<Frames>> = Mutex::new(None);
    let now = std::time::Instant::now();
    let mut f = FRAMES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(f) = f.as_mut() else {
        *f = Some(Frames { last: now, began: now, each: Vec::new(), drawn: crate::draw::drawn() });
        return;
    };
    f.each.push(now.duration_since(f.last).as_millis().min(u128::from(u32::MAX)) as u32);
    f.last = now;
    let span = now.duration_since(f.began).as_secs_f64();
    if span >= 5.0 {
        let mut sorted = f.each.clone();
        sorted.sort_unstable();
        let each: Vec<String> = f.each.iter().map(u32::to_string).collect();
        tracing::info!(
            target: "victory_gpu::framelog",
            "frame times: {} frames in {span:.1} s: shortest {} ms, median {}, longest {}; {} draws a frame; each: {}",
            f.each.len(),
            sorted[0],
            sorted[sorted.len() / 2],
            sorted[sorted.len() - 1],
            (crate::draw::drawn() - f.drawn) / f.each.len() as u64,
            each.join(" ")
        );
        // The line `xcl-mac/tools/phases.py` and the shared harness read draws from.
        let mean = f.each.iter().map(|&m| f64::from(m)).sum::<f64>() / f.each.len() as f64;
        tracing::info!(
            target: "victory_gpu::framelog",
            "usual frame (the mean of {}): {mean:.0} ms | {} draws |",
            f.each.len(),
            (crate::draw::drawn() - f.drawn) / f.each.len() as u64
        );
        f.each.clear();
        f.began = now;
        f.drawn = crate::draw::drawn();
    }
}

/// Mac layer: the bytes to make a CPU-mapped buffer (upload or shared) of
/// `size` with. D3DMetal reads some copies out of such buffers on the CPU and
/// reads every row at its full pitch, the last one too, where D3D12 asks only
/// for the last row's own bytes: 23 KB past the end of a 2.6 MB buffer, into
/// unmapped memory, and the process died (run nb-av-d, 2026-11-26). The
/// overrun is under one row's pitch, which is at most the buffer's size; a
/// buffer gets that much more, up to 256 KiB. `VICTORY_MAPPED_PAD=0`: exact.
pub(crate) fn mapped_buffer_bytes(size: u64) -> u64 {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| switch("VICTORY_MAPPED_PAD", true, false)) {
        return size;
    }
    size + size.min(256 << 10).next_multiple_of(4096)
}

/// Whether every draw and dispatch sets its pipeline again, even when the last
/// one it set is the same (see `draw.rs`): on by default on Darwin
/// (`VICTORY_PIPELINE_EVERY_DRAW=0`: only when it changes, as the base does).
pub(crate) fn pipeline_every_draw() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_PIPELINE_EVERY_DRAW", true, false))
}

/// Whether pipelines whose pixel program needs barycentrics are refused: on
/// by default on Darwin (`VICTORY_REFUSE_BARYCENTRICS=0`: made). D3DMetal 4.0b3
/// makes no-op pipelines of them, and binding one killed 17 of 19 runs at
/// world load (a vertex slot count of 0xFFFF read in SetRenderVertexBuffers);
/// with them refused, 2 of 2 ran clean (nb-bary-x, -y). Their draws draw
/// nothing either way.
pub(crate) fn refuse_barycentrics() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_REFUSE_BARYCENTRICS", true, false))
}

/// Whether pixel programs read per-vertex inputs without `attributeAtVertex`
/// (`victory_shader::emit::set_per_vertex_solved`), so hair, doors and the
/// rest of those draws are drawn. Off for now (`VICTORY_PER_VERTEX_SOLVED=1`:
/// on, Darwin only): Ahmet's release build of b867f881 died silently ~1 s
/// into pipeline preload with it on (2026-11-26), though all 12,055 such
/// pixel programs pass Apple's converter offline.
pub(crate) fn per_vertex_solved() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::mac::is_darwin() && std::env::var_os("VICTORY_PER_VERTEX_SOLVED").is_some_and(|v| v == "1"))
}

/// The feature bits of a DXIL container's `SFI0` part (0 without one).
pub(crate) fn feature_bits(dxil: &[u8]) -> u64 {
    let u32_at = |b: &[u8], at: usize| b.get(at..at + 4).map(|v| u32::from_le_bytes(v.try_into().expect("4 bytes")) as usize);
    (|| {
        let parts = u32_at(dxil, 28)?;
        (0..parts).find_map(|i| {
            let at = u32_at(dxil, 32 + 4 * i)?;
            (dxil.get(at..at + 4)? == b"SFI0")
                .then(|| Some(u64::from_le_bytes(dxil.get(at + 8..at + 16)?.try_into().ok()?)))
                .flatten()
        })
    })()
    .unwrap_or(0)
}

/// Whether a mesh pipeline's pixel program reads its per-vertex values from
/// flat per-primitive outputs the mesh program writes (Darwin with
/// `VICTORY_MAC_PRIMITIVE_VARYINGS=1`; off by default): no
/// `attributeAtVertex`, which D3DMetal 4.0b3 cannot run.
pub(crate) fn primitive_varyings() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| is_darwin() && crate::switch::on("VICTORY_MAC_PRIMITIVE_VARYINGS"))
}

/// Whether a vertex+pixel triangle pipeline whose pixel program reads
/// non-provoking per-vertex values uses a pass-through geometry shader
/// (`VICTORY_P10`/`VICTORY_P20`, pragyaangaur/VIctory-macOS#300) instead of
/// `attributeAtVertex` (default on on Darwin; `VICTORY_VERTEX_DELTAS=0`: off).
pub(crate) fn vertex_deltas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_VERTEX_DELTAS", true, false))
}

/// Whether over-32-KiB mesh pipelines (e.g. the 71 `MS_VS_PedTransform*`
/// character shaders) are re-translated paired with their pixel shader to
/// prune unread outputs so they fit Metal's 32 KiB limit instead of turning
/// into D3DMetal no-ops (`pragyaangaur/VIctory-macOS#161`). Default on on
/// Darwin (`VICTORY_LINK_MESH_OUTPUTS=0`: off).
pub(crate) fn link_mesh_outputs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_LINK_MESH_OUTPUTS", true, false))
}

/// Whether a mesh+pixel pipeline whose `attributeAtVertex` pixel shader still
/// exceeds Metal's 32 KiB output budget after packed `VICTORY_VERTEX_DATA`
/// linking falls back to hardware center interpolation (no `attributeAtVertex`,
/// no extra primitive outputs; recovers heavy vehicle/boat/ped shaders).
/// Default on on Darwin (`VICTORY_PER_VERTEX_CENTER=0`: off).
pub(crate) fn per_vertex_center() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_PER_VERTEX_CENTER", true, false))
}

/// Whether a mesh pipeline whose outputs still exceed Metal's 32 KiB limit
/// (by packed or aligned Metal layout, `pragyaangaur/VIctory-macOS#282`) is
/// refused instead of hanging D3DMetal. Default on on Darwin
/// (`VICTORY_MESH_OVER_LIMIT=1`: allow).
pub(crate) fn refuse_over_limit_mesh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        match std::env::var_os("VICTORY_MESH_OVER_LIMIT") {
            Some(v) => v == "0",
            None => is_darwin(),
        }
    })
}
