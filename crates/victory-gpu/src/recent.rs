//! Scratch memory kept by what recent work needed: the texture staging
//! chunks ([`Peak`]) and the snapshot buffers ([`Pool`]) give back what the
//! last [`WINDOW`] did not use, and keep all it did (a pool kept at its
//! peak for good held 512 MB after one heavy submission).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How long what was used is kept after its last use: ~300 frames at 30
/// fps (a few frames of today's 2-3 s world frames). Steady play, where
/// each frame needs what the last did, never sees a release, nor does a
/// load that recurs within it, and a menu or a loading screen gives its
/// memory back that long after the load. Remaking is the price of a load that comes back
/// after longer (upload buffers cost ~5 ms per 12.7 MB, fresh snapshots a
/// page fault per 4 KiB).
pub(crate) const WINDOW: Duration = Duration::from_secs(10);

/// The most any submission needed of a pool in the last [`WINDOW`].
pub(crate) struct Peak {
    /// (when a submission began, what it needed), oldest first, the needs
    /// falling: one no larger than a later need is dropped, for the later
    /// one outlasts it in the window.
    seen: VecDeque<(Instant, usize)>,
    /// When the submission being prepared began.
    began: Option<Instant>,
}

impl Peak {
    pub(crate) const fn new() -> Self {
        Peak {
            seen: VecDeque::new(),
            began: None,
        }
    }

    /// A submission begins at `now`; the last one, which began when this
    /// was last called, needed `needed`. Returns the most any submission
    /// needed in the last [`WINDOW`], for the pool to keep (the last
    /// submission's need counts from when it began, not from now: a pool
    /// that sat idle must not take a load long gone for a recent one).
    pub(crate) fn begin(&mut self, now: Instant, needed: usize) -> usize {
        if let Some(began) = self.began.replace(now) {
            while self.seen.back().is_some_and(|&(_, n)| n <= needed) {
                self.seen.pop_back();
            }
            self.seen.push_back((began, needed));
        }
        while self
            .seen
            .front()
            .is_some_and(|&(at, _)| now.saturating_duration_since(at) > WINDOW)
        {
            self.seen.pop_front();
        }
        self.seen.front().map_or(0, |&(_, n)| n)
    }
}

/// Snapshot buffers kept for reuse (already touched, so no page faults),
/// each with when it was returned.
pub(crate) struct Pool {
    /// Smallest first, and of equal sizes the last returned first: a buffer
    /// that serves one submission serves the next, and the others of its
    /// size, which would rotate through and never age, go unused.
    free: Vec<(Vec<u8>, Instant)>,
}

impl Pool {
    pub(crate) const fn new() -> Self {
        Pool { free: Vec::new() }
    }

    /// The smallest free buffer with room for `n` bytes, if any.
    pub(crate) fn take(&mut self, n: usize) -> Option<Vec<u8>> {
        let i = self.free.partition_point(|(v, _)| v.capacity() < n);
        (i < self.free.len()).then(|| self.free.remove(i).0)
    }

    /// Takes `v` back, emptied, at `now`. The pool keeps at most `cap`
    /// bytes, the largest buffers.
    pub(crate) fn give(&mut self, mut v: Vec<u8>, now: Instant, cap: usize) {
        if v.capacity() <= cap {
            v.clear();
            let i = self
                .free
                .partition_point(|(f, _)| f.capacity() < v.capacity());
            self.free.insert(i, (v, now));
        }
        let mut total = 0;
        if let Some(i) = self.free.iter().rposition(|(f, _)| {
            total += f.capacity();
            total > cap
        }) {
            self.free.drain(..=i);
        }
    }

    /// Releases the largest buffer that was not returned in the last
    /// [`WINDOW`], if any, to be dropped by the caller outside the pool's
    /// lock: one a call (once a frame), so that freeing a peak's buffers
    /// does not fall on one frame.
    pub(crate) fn trim(&mut self, now: Instant) -> Option<Vec<u8>> {
        let i = self
            .free
            .iter()
            .rposition(|&(_, at)| now.saturating_duration_since(at) > WINDOW)?;
        Some(self.free.remove(i).0)
    }

    /// Bytes held (tests).
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.free.iter().map(|(v, _)| v.capacity()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Duration = Duration::from_millis(33);
    const MIB: usize = 1 << 20;

    /// Chunks as `crate::texture::recycle` keeps them, counted: what each
    /// submission needs is made if the pool does not hold it (a model of
    /// that function's arithmetic; the policy is `Peak`'s).
    struct Chunks {
        held: usize,
        last: usize,
        peak: Peak,
        made: usize,
    }

    impl Chunks {
        fn submit(&mut self, now: Instant, need: usize) {
            const KEPT: usize = 16;
            self.held = self.held.min(self.peak.begin(now, self.last).min(KEPT));
            self.made += need.saturating_sub(self.held);
            (self.held, self.last) = (self.held.max(need), need);
        }
    }

    #[test]
    fn chunks_follow_the_load_and_steady_state_makes_none() {
        let mut c = Chunks {
            held: 0,
            last: 0,
            peak: Peak::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |c: &mut Chunks, n, need| {
            for _ in 0..n {
                now += FRAME;
                c.submit(now, need);
            }
        };
        run(&mut c, 300, 3);
        assert_eq!((c.held, c.made), (3, 3));
        run(&mut c, 30, 16);
        assert_eq!((c.held, c.made), (16, 16));
        // The spike's chunks stay while it is in the window.
        let window = (WINDOW.as_millis() / FRAME.as_millis()) as usize;
        run(&mut c, window - 5, 3);
        assert_eq!(c.held, 16);
        run(&mut c, 10, 3);
        assert_eq!(c.held, 3);
        let made = c.made;
        run(&mut c, 300, 3);
        assert_eq!((c.held, c.made), (3, made), "steady state makes no chunk");
    }

    #[test]
    fn a_need_counts_from_when_its_submission_began() {
        let mut p = Peak::new();
        let t = Instant::now();
        assert_eq!(p.begin(t, 0), 0);
        assert_eq!(p.begin(t + FRAME, 16), 16);
        // A pool idle for longer than the window does not take the load
        // before it for a recent one.
        assert_eq!(p.begin(t + 2 * WINDOW, 3), 0);
        assert_eq!(p.begin(t + 2 * WINDOW + FRAME, 3), 3);
    }

    #[test]
    fn the_peak_falls_in_steps_as_larger_needs_leave_the_window() {
        let mut p = Peak::new();
        let t = Instant::now();
        let at = |s: u64| t + Duration::from_secs(s);
        p.begin(at(0), 0);
        for (s, need) in [(3, 8), (6, 5), (9, 2)] {
            p.begin(at(s), need);
        }
        // The needs began at 0 s (8), 3 s (5) and 6 s (2).
        assert_eq!(p.begin(at(10), 0), 8);
        assert_eq!(p.begin(at(11), 0), 5);
        assert_eq!(p.begin(at(14), 0), 2);
        assert_eq!(p.begin(at(17), 0), 0);
    }

    #[test]
    fn the_pool_takes_the_smallest_that_fits_and_the_last_returned_of_a_size() {
        let mut pool = Pool::new();
        let t = Instant::now();
        let buffer = |n: usize| {
            let mut v = Vec::with_capacity(n);
            v.push(1);
            v
        };
        let (a, b) = (buffer(100), buffer(100));
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        pool.give(buffer(1000), t, usize::MAX);
        pool.give(a, t, usize::MAX);
        pool.give(b, t, usize::MAX);
        assert!(pool.take(2000).is_none());
        let v = pool.take(50).unwrap();
        assert!(v.is_empty() && v.capacity() >= 100);
        assert_eq!(v.as_ptr(), pb, "the last returned");
        pool.give(v, t, usize::MAX);
        let (x, y) = (pool.take(50).unwrap(), pool.take(50).unwrap());
        assert_eq!((x.as_ptr(), y.as_ptr()), (pb, pa));
        assert!(pool.take(500).unwrap().capacity() >= 1000);
    }

    #[test]
    fn oversized_return_preserves_buffers_that_fit_the_current_cap() {
        let mut pool = Pool::new();
        let now = Instant::now();
        pool.give(Vec::with_capacity(32), now, 128);
        pool.give(Vec::with_capacity(64), now, 128);
        pool.give(Vec::with_capacity(256), now, 64);
        assert_eq!(pool.bytes(), 64);
        assert!(pool.take(64).is_some());
    }

    #[test]
    fn the_pool_keeps_the_largest_within_its_cap() {
        let mut pool = Pool::new();
        let t = Instant::now();
        for n in [10, 40, 20, 30] {
            pool.give(Vec::with_capacity(n), t, 70);
        }
        assert!(pool.bytes() <= 70);
        assert!(pool.take(40).is_some() && pool.take(30).is_some() && pool.take(10).is_none());
    }

    #[test]
    fn the_pool_releases_the_largest_unused_first_and_one_at_a_time() {
        let mut pool = Pool::new();
        let t = Instant::now();
        for n in [4, 64, 16, 32] {
            pool.give(Vec::with_capacity(n * MIB), t, usize::MAX);
        }
        // The 4 and 16 are used each frame; the 32 and 64 are not.
        let mut now = t;
        let mut freed = Vec::new();
        for _ in 0..400 {
            now += FRAME;
            let (a, b) = (pool.take(4 * MIB).unwrap(), pool.take(16 * MIB).unwrap());
            pool.give(a, now, usize::MAX);
            pool.give(b, now, usize::MAX);
            freed.extend(pool.trim(now).map(|v| v.capacity() / MIB));
        }
        assert_eq!(freed, [64, 32]);
        assert_eq!(pool.bytes(), 20 * MIB);
    }

    /// Snapshot buffers as `crate::shadow` takes and returns them: each
    /// submission takes `sizes` (made if the pool has none), then returns
    /// them all, and the frame trims.
    struct Snapshots {
        pool: Pool,
        made: usize,
    }

    impl Snapshots {
        fn frame(&mut self, now: Instant, sizes: &[usize]) {
            let taken: Vec<Vec<u8>> = sizes
                .iter()
                .map(|&n| {
                    self.pool.take(n).unwrap_or_else(|| {
                        self.made += 1;
                        Vec::with_capacity(n)
                    })
                })
                .collect();
            for v in taken {
                self.pool.give(v, now, 512 * MIB);
            }
            drop(self.pool.trim(now));
        }
    }

    #[test]
    fn snapshots_follow_the_load_and_steady_state_makes_none() {
        let steady: Vec<usize> = [64, 48, 32, 16, 8, 4, 2, 2]
            .iter()
            .map(|n| n * MIB)
            .collect();
        let spike: Vec<usize> = steady
            .iter()
            .copied()
            .chain([128 * MIB, 96 * MIB, 64 * MIB])
            .collect();
        let mut s = Snapshots {
            pool: Pool::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |s: &mut Snapshots, n, sizes: &[usize]| {
            for _ in 0..n {
                now += FRAME;
                s.frame(now, sizes);
            }
        };
        run(&mut s, 300, &steady);
        let (steady_bytes, made) = (s.pool.bytes(), s.made);
        assert_eq!(made, steady.len());
        run(&mut s, 30, &spike);
        assert_eq!(
            (s.pool.bytes(), s.made),
            (steady_bytes + 288 * MIB, made + 3)
        );
        run(&mut s, 400, &steady);
        assert_eq!(
            s.pool.bytes(),
            steady_bytes,
            "the spike's buffers went back"
        );
        assert_eq!(s.made, made + 3, "and steady state made none");
    }
}
