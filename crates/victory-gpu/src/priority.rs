//! `VICTORY_PREP_PRIORITY=<n>`: the Windows thread priority (`-2` to `2`,
//! or `15` for time-critical) of the threads that prepare host submissions
//! while the host GPU waits for them: `victory-submit`, `victory-flight`,
//! the texture and copy workers, and a title thread for as long as it
//! drains the submission queue (`queue::drain_submit`, the title's present).
//!
//! The title gives its own threads the console's priorities, and VIctory's
//! threads ran at normal. Wine on macOS turns thread priorities into Mach
//! thread policies, so with more threads runnable than performance cores,
//! one that prepares while the GPU idles could wait behind the title's or
//! run on an efficiency core. Off (unset) by default.

/// The asked-for priority, `None` when unset or not one of the values above.
pub(crate) fn wanted() -> Option<i32> {
    static WANTED: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
    *WANTED.get_or_init(|| {
        let value = parse(std::env::var("VICTORY_PREP_PRIORITY").ok().as_deref());
        if let Some(n) = value {
            tracing::info!(priority = n, "threads preparing host submissions raised (VICTORY_PREP_PRIORITY)");
        }
        value
    })
}

fn parse(value: Option<&str>) -> Option<i32> {
    let n = value?.trim().parse::<i32>().ok()?;
    ((-2..=2).contains(&n) || n == 15).then_some(n)
}

/// Gives the calling thread the asked-for priority for good (threads of
/// VIctory's own, at their start).
pub(crate) fn raise_this_thread() {
    #[cfg(windows)]
    if let Some(n) = wanted() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
        // SAFETY: the pseudo-handle of the calling thread.
        unsafe { SetThreadPriority(GetCurrentThread(), n) };
    }
}

/// The calling thread at the asked-for priority until dropped, then at
/// the priority it had (a title thread draining the queue).
pub(crate) struct Raised(#[cfg_attr(not(windows), allow(dead_code))] Option<i32>);

impl Raised {
    pub(crate) fn now() -> Self {
        #[cfg(windows)]
        if let Some(n) = wanted() {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, SetThreadPriority};
            /// `GetThreadPriority`'s failure (`THREAD_PRIORITY_ERROR_RETURN`).
            const FAILED: i32 = i32::MAX;
            // SAFETY: the pseudo-handle of the calling thread.
            let was = unsafe { GetThreadPriority(GetCurrentThread()) };
            if was != FAILED && was < n {
                // SAFETY: as above.
                unsafe { SetThreadPriority(GetCurrentThread(), n) };
                return Raised(Some(was));
            }
        }
        Raised(None)
    }
}

impl Drop for Raised {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(was) = self.0 {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
            // SAFETY: the pseudo-handle of the calling thread.
            unsafe { SetThreadPriority(GetCurrentThread(), was) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn priorities_are_windows_values() {
        assert_eq!(parse(None), None);
        assert_eq!(parse(Some("2")), Some(2));
        assert_eq!(parse(Some(" -1 ")), Some(-1));
        assert_eq!(parse(Some("15")), Some(15));
        assert_eq!(parse(Some("3")), None);
        assert_eq!(parse(Some("high")), None);
    }
}
