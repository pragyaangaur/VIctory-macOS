//! Pages the title writes in frame after frame, left open (opt-in,
//! `VICTORY_HOT_PAGES=1`).
//!
//! Guarding a tracked page costs a `VirtualProtect` when a tracker takes it,
//! and the title's first write to it after that costs a fault and another
//! `VirtualProtect` to open it again. Under Wine on Rosetta each costs a
//! Wine call under its one memory lock. On a 16 GB M1 Pro in the world that
//! was 545 ms of protecting (2,295 calls for 3,981 pages) and 235 ms in the
//! guard handler (1,480 pages opened) per frame, while 625 of the 1,166
//! tracked pages written in a frame had been written the frame before too
//! (#554). For a page written in every frame, the guard saves nothing: the
//! tracker takes the page again anyway.
//!
//! [`Hot`] counts, per page, the frames in a row in which a guarded write
//! opened it. From [`STREAK`] frames on, the page is hot for [`HOLD`]
//! frames: a take leaves it open and changed, so the trackers holding it
//! copy it at every use without a guard or a fault. After the hold the
//! page is guarded again and has to earn its streak anew, so a page that
//! went quiet costs at most [`HOLD`] frames of copies.
//!
//! Only pages that no host buffer copy holds are left open: a buffer copy's
//! copy-back of shader writes needs the page's bytes from the title's first
//! write after the take, which only the guard's fault saves
//! (`crate::watch::Tracked::saved`).
//!
//! Plain std and no Windows calls, so its tests run everywhere.

use std::collections::BTreeMap;

/// Frames in a row a page must be written in before it is left open.
pub(crate) const STREAK: u32 = 3;
/// Frames a hot page stays open before it is guarded again.
pub(crate) const HOLD: u64 = 30;

/// Whether hot pages are left open: `VICTORY_HOT_PAGES=1` (off by default).
#[cfg(windows)]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| crate::switch::on("VICTORY_HOT_PAGES"));
    #[cfg(test)]
    if FORCED.with(std::cell::Cell::get) {
        return true;
    }
    *ON
}

#[cfg(all(windows, test))]
thread_local! {
    /// The switch, on for one test's thread (its takes, its faults and its
    /// frames) and no other test's.
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Turns the switch on for this thread (tests).
#[cfg(all(windows, test))]
pub(crate) fn force_on_this_thread() {
    FORCED.with(|f| f.set(true));
}

#[derive(Clone, Copy)]
struct Page {
    /// The last frame a guarded write opened the page in.
    last: u64,
    /// Frames in a row, up to `last`, a guarded write opened it in.
    streak: u32,
    /// The page is hot (left open) while the frame is before this.
    until: u64,
}

/// Write streaks of tracked pages, by page number.
#[derive(Default)]
pub(crate) struct Hot {
    frame: u64,
    pages: BTreeMap<usize, Page>,
}

impl Hot {
    pub(crate) const fn new() -> Self {
        Self {
            frame: 0,
            pages: BTreeMap::new(),
        }
    }

    /// A guarded write opened page `page` in the current frame.
    pub(crate) fn written(&mut self, page: usize) {
        let frame = self.frame;
        let p = self.pages.entry(page).or_insert(Page {
            last: frame,
            streak: 0,
            until: 0,
        });
        if p.streak > 0 && p.last == frame {
            return;
        }
        p.streak = if p.streak > 0 && p.last + 1 == frame {
            p.streak + 1
        } else {
            1
        };
        p.last = frame;
        if p.streak >= STREAK {
            p.until = frame + HOLD;
        }
    }

    /// Whether page `page` is to be left open now.
    pub(crate) fn is_hot(&self, page: usize) -> bool {
        self.pages.get(&page).is_some_and(|p| self.frame < p.until)
    }

    /// A frame was presented. Forgets pages neither hot nor written in the
    /// frame that just ended, so the map holds only pages written lately.
    pub(crate) fn frame_done(&mut self) {
        let ended = self.frame;
        self.frame += 1;
        let now = self.frame;
        self.pages.retain(|_, p| now < p.until || p.last >= ended);
    }

    /// Pages hot now.
    #[cfg(test)]
    fn hot_count(&self) -> usize {
        self.pages.values().filter(|p| self.frame < p.until).count()
    }
}

#[cfg(test)]
mod tests {
    use super::{HOLD, Hot, STREAK};

    /// `frames` frames, each with a guarded write to `page` unless the page
    /// is left open then (hot pages take no faults).
    fn write_every_frame(hot: &mut Hot, page: usize, frames: u64) {
        for _ in 0..frames {
            if !hot.is_hot(page) {
                hot.written(page);
            }
            hot.frame_done();
        }
    }

    #[test]
    fn a_page_written_in_streak_frames_in_a_row_is_left_open() {
        let mut hot = Hot::new();
        for _ in 0..STREAK - 1 {
            hot.written(7);
            hot.frame_done();
            assert!(!hot.is_hot(7), "not before {STREAK} frames in a row");
        }
        hot.written(7);
        assert!(hot.is_hot(7), "hot from the frame of its {STREAK}th write");
        hot.frame_done();
        assert!(hot.is_hot(7));
        assert_eq!(hot.hot_count(), 1);
    }

    #[test]
    fn several_writes_in_one_frame_count_once() {
        let mut hot = Hot::new();
        for _ in 0..10 {
            hot.written(3);
        }
        assert!(!hot.is_hot(3));
        hot.frame_done();
        hot.written(3);
        assert!(!hot.is_hot(3));
    }

    #[test]
    fn a_frame_without_a_write_breaks_the_streak() {
        let mut hot = Hot::new();
        for _ in 0..STREAK - 1 {
            hot.written(1);
            hot.frame_done();
        }
        hot.frame_done();
        hot.written(1);
        assert!(!hot.is_hot(1), "streak restarted");
        for _ in 0..STREAK - 2 {
            hot.frame_done();
            hot.written(1);
        }
        assert!(!hot.is_hot(1));
        hot.frame_done();
        hot.written(1);
        assert!(hot.is_hot(1));
    }

    #[test]
    fn a_hot_page_is_guarded_again_after_the_hold_and_relearns() {
        let mut hot = Hot::new();
        write_every_frame(&mut hot, 9, u64::from(STREAK));
        assert!(hot.is_hot(9));
        // Written in every frame of the hold, unseen: no faults while open.
        for _ in 1..HOLD {
            assert!(hot.is_hot(9));
            hot.frame_done();
        }
        assert!(!hot.is_hot(9), "guarded again after {HOLD} frames");
        // Still written every frame: back to hot after a new streak.
        write_every_frame(&mut hot, 9, u64::from(STREAK) - 1);
        assert!(!hot.is_hot(9));
        hot.written(9);
        assert!(hot.is_hot(9));
    }

    #[test]
    fn quiet_pages_are_forgotten() {
        let mut hot = Hot::new();
        hot.written(1);
        write_every_frame(&mut hot, 2, u64::from(STREAK));
        hot.frame_done();
        assert_eq!(hot.pages.len(), 1, "only the hot page is kept");
        for _ in 0..HOLD {
            hot.frame_done();
        }
        assert!(hot.pages.is_empty());
        assert_eq!(hot.hot_count(), 0);
    }

    #[test]
    fn steady_writes_fault_in_few_frames() {
        // A page written in every one of 1,000 frames faults only in the
        // frames it is guarded: STREAK of every STREAK + HOLD - 1.
        let mut hot = Hot::new();
        let mut faults = 0;
        for _ in 0..1000 {
            if !hot.is_hot(5) {
                hot.written(5);
                faults += 1;
            }
            hot.frame_done();
        }
        let cycle = u64::from(STREAK) + HOLD - 1;
        assert!(
            faults <= 1000 / cycle * u64::from(STREAK) + u64::from(STREAK),
            "{faults} faults"
        );
        assert!(faults < 150, "{faults} faults instead of 1,000");
    }
}
