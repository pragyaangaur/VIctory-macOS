//! A sampling profiler for named threads of this process:
//! `VICTORY_PROFILE=<name>[,<name>...]` samples every thread whose name
//! contains one of them (`victory-submit`, the texture workers) and writes
//! each distinct stack with its count to `VICTORY_PROFILE_FILE` (default
//! `victory-profile.txt` in the working directory).
//!
//! Under Rosetta, macOS's own `sample` sees translated ARM code and cannot
//! unwind through Wine's PE frames, so where the host thread preparing
//! submissions spends a world frame was known only from the timers the
//! frame report prints. Here each sample suspends the thread, reads its
//! x86-64 context and unwinds it by the images' unwind data, as the
//! scheduled stack dumps do; the addresses are placed offline (`module
//! base size` lines, then per 10 s window and thread `count addr addr ...`,
//! innermost first). The file is written again after each window.
//!
//! `VICTORY_PROFILE_AT=<start s>,<seconds>` (default `60,600`) picks the
//! span from the profiler's start; `VICTORY_PROFILE_MS=<n>` (default 2)
//! the interval. A sample stops its thread for the context read and
//! unwind (tens of microseconds plus the server round trips under Wine),
//! so the profiled thread runs a little slower.

use std::collections::HashMap;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, INFINITE, OpenThread,
    SetWaitableTimer, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
    TIMER_ALL_ACCESS, WaitForSingleObject,
};

use crate::stacks::{frames_of, modules, name_of, other_threads};

/// What `VICTORY_PROFILE*` asked for.
struct Ask {
    names: Vec<String>,
    start: std::time::Duration,
    length: std::time::Duration,
    every_ms: u32,
    file: std::path::PathBuf,
}

impl Ask {
    fn from_env() -> Option<Ask> {
        let names: Vec<String> = std::env::var("VICTORY_PROFILE")
            .ok()?
            .split(',')
            .map(|n| n.trim().to_owned())
            .filter(|n| !n.is_empty())
            .collect();
        if names.is_empty() {
            return None;
        }
        let (start, length) = window(std::env::var("VICTORY_PROFILE_AT").ok().as_deref());
        let every_ms = std::env::var("VICTORY_PROFILE_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&ms| ms > 0)
            .unwrap_or(2);
        let file = std::env::var_os("VICTORY_PROFILE_FILE")
            .map(Into::into)
            .unwrap_or_else(|| "victory-profile.txt".into());
        Some(Ask { names, start, length, every_ms, file })
    }
}

/// `<start s>,<seconds>` (either may be left out), defaulting to 60 and 600.
fn window(value: Option<&str>) -> (std::time::Duration, std::time::Duration) {
    let secs = |s: Option<&str>, default: f64| {
        s.and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s >= 0.0)
            .and_then(|s| std::time::Duration::try_from_secs_f64(s).ok())
            .unwrap_or(std::time::Duration::from_secs_f64(default))
    };
    let mut parts = value.unwrap_or("").splitn(2, ',');
    (secs(parts.next(), 60.0), secs(parts.next(), 600.0))
}

/// Starts the profiler thread if `VICTORY_PROFILE` names threads.
pub fn profile_on_request() {
    let Some(ask) = Ask::from_env() else {
        return;
    };
    let began = std::time::Instant::now();
    let _ = std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(move || run(&ask, began));
}

/// A thread being sampled.
struct Target {
    id: u32,
    name: String,
    handle: HANDLE,
}

/// Samples are kept per [`WINDOW`] of the profile, and the file is written
/// again at the end of each: a run that ends first (the benchmark exits at
/// a frame count) still leaves what it took, and the windows can be matched
/// to the frame report's lines (the log says when the profile started).
const WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// Per window, per thread name, per stack: how many samples.
type Stacks = HashMap<(usize, usize, Vec<u64>), u32>;

fn run(ask: &Ask, began: std::time::Instant) {
    std::thread::sleep(ask.start.saturating_sub(began.elapsed()));
    let Some(timer) = periodic(ask.every_ms) else {
        return;
    };
    tracing::info!(
        file = %ask.file.display(),
        seconds = ask.length.as_secs_f64(),
        every_ms = ask.every_ms,
        "profile started (VICTORY_PROFILE): windows of {} s from now",
        WINDOW.as_secs()
    );
    let mut targets: Vec<Target> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(8192);
    let mut stacks: Stacks = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    // Per window, per thread name: samples; and ticks per window.
    let mut samples: Vec<Vec<u64>> = Vec::new();
    let mut ticks: Vec<u64> = Vec::new();
    let mut scanned = std::time::Instant::now() - std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    let stop = start + ask.length;
    let mut written = 0usize;
    while std::time::Instant::now() < stop {
        // Threads come and go: look again every second.
        if scanned.elapsed() >= std::time::Duration::from_secs(1) {
            scanned = std::time::Instant::now();
            rescan(&ask.names, &mut targets, &mut ids);
        }
        // SAFETY: our periodic timer.
        unsafe { WaitForSingleObject(timer, INFINITE) };
        let window = (start.elapsed().as_nanos() / WINDOW.as_nanos()) as usize;
        if window > written {
            write(ask, &ticks, &names, &samples, &stacks);
            written = window;
        }
        if ticks.len() <= window {
            ticks.resize(window + 1, 0);
            samples.resize_with(window + 1, Vec::new);
        }
        ticks[window] += 1;
        for t in &targets {
            let (frames, _) = frames_of(t.handle);
            if frames.is_empty() {
                continue;
            }
            let idx = match names.iter().position(|n| *n == t.name) {
                Some(i) => i,
                None => {
                    names.push(t.name.clone());
                    names.len() - 1
                }
            };
            let counts = &mut samples[window];
            if counts.len() <= idx {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += 1;
            *stacks.entry((window, idx, frames)).or_insert(0) += 1;
        }
    }
    for t in targets {
        // SAFETY: opened in `rescan`.
        unsafe { CloseHandle(t.handle) };
    }
    // SAFETY: made in `periodic`.
    unsafe { CloseHandle(timer) };
    write(ask, &ticks, &names, &samples, &stacks);
}

/// Opens the threads whose names match that are not open yet, and closes
/// the ones gone.
fn rescan(wanted: &[String], targets: &mut Vec<Target>, ids: &mut Vec<u32>) {
    if !other_threads(ids, ids.capacity()) {
        return;
    }
    targets.retain(|t| {
        let alive = ids.contains(&t.id);
        if !alive {
            // SAFETY: opened below on an earlier scan.
            unsafe { CloseHandle(t.handle) };
        }
        alive
    });
    for &id in ids.iter() {
        if targets.iter().any(|t| t.id == id) {
            continue;
        }
        // SAFETY: opens a thread of our own process.
        let handle =
            unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id) };
        if handle.is_null() {
            continue;
        }
        let name = name_of(handle);
        if wanted.iter().any(|w| name.contains(w.as_str())) {
            targets.push(Target { id, name, handle });
        } else {
            // SAFETY: opened above.
            unsafe { CloseHandle(handle) };
        }
    }
}

/// A high-resolution periodic timer firing every `ms` (one made and waited
/// on: `sleep` makes and closes a timer a call, a server round trip each
/// under Wine).
fn periodic(ms: u32) -> Option<HANDLE> {
    // SAFETY: no name, default security; the handle is checked.
    let timer = unsafe {
        let t = CreateWaitableTimerExW(
            std::ptr::null(),
            std::ptr::null(),
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_ALL_ACCESS,
        );
        if t.is_null() { CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS) } else { t }
    };
    if timer.is_null() {
        return None;
    }
    let due: i64 = -(i64::from(ms) * 10_000);
    // SAFETY: our timer; no completion routine.
    if unsafe { SetWaitableTimer(timer, &due, ms as i32, None, std::ptr::null(), 0) } == 0 {
        // SAFETY: made above.
        unsafe { CloseHandle(timer) };
        return None;
    }
    Some(timer)
}

fn write(ask: &Ask, ticks: &[u64], names: &[String], samples: &[Vec<u64>], stacks: &Stacks) {
    use std::io::Write;
    let Ok(file) = std::fs::File::create(&ask.file) else {
        tracing::warn!(file = %ask.file.display(), "the profile could not be written");
        return;
    };
    modules(&file);
    let mut out = std::io::BufWriter::new(&file);
    let _ = writeln!(out, "every {} ms, windows of {} s", ask.every_ms, WINDOW.as_secs());
    let mut sorted: Vec<_> = stacks.iter().collect();
    sorted.sort_by(|a, b| (a.0.0, a.0.1).cmp(&(b.0.0, b.0.1)).then(b.1.cmp(a.1)));
    let mut current = (usize::MAX, usize::MAX);
    for ((window, idx, frames), count) in sorted {
        if (*window, *idx) != current {
            if *window != current.0 {
                let _ = writeln!(out, "window {window} ticks {}", ticks.get(*window).copied().unwrap_or(0));
            }
            current = (*window, *idx);
            let n = samples.get(*window).and_then(|c| c.get(*idx)).copied().unwrap_or(0);
            let _ = writeln!(out, "thread {} samples {n}", names[*idx]);
        }
        let _ = write!(out, "{count}");
        for pc in frames {
            let _ = write!(out, " {pc:x}");
        }
        let _ = writeln!(out);
    }
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_default_and_parse() {
        let s = std::time::Duration::from_secs;
        assert_eq!(window(None), (s(60), s(600)));
        assert_eq!(window(Some("30,5")), (s(30), s(5)));
        assert_eq!(window(Some("x,-1")), (s(60), s(600)));
        assert_eq!(window(Some("10")), (s(10), s(600)));
    }

    /// A busy named thread is found by name and sampled; the file lists it
    /// with stacks whose counts add up to its samples.
    #[test]
    fn samples_a_named_thread() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("profile.txt");
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let busy = std::thread::Builder::new()
            .name("profiled-spinner".into())
            .spawn(move || {
                let mut x = 0u64;
                while !s.load(std::sync::atomic::Ordering::Relaxed) {
                    x = std::hint::black_box(x.wrapping_mul(31).wrapping_add(7));
                }
                x
            })
            .unwrap();
        let ask = Ask {
            names: vec!["profiled-spinner".into()],
            start: std::time::Duration::ZERO,
            length: std::time::Duration::from_millis(300),
            every_ms: 2,
            file: file.clone(),
        };
        run(&ask, std::time::Instant::now());
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        busy.join().unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        let samples: u64 = text
            .lines()
            .filter(|l| l.starts_with("thread profiled-spinner"))
            .map(|l| l.rsplit(' ').next().unwrap().parse::<u64>().unwrap())
            .sum();
        assert!(samples > 10, "{text}");
        let counted: u64 = text
            .lines()
            .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
            .map(|l| l.split(' ').next().unwrap().parse::<u64>().unwrap())
            .sum();
        assert_eq!(counted, samples, "{text}");
    }
}
