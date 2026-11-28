use std::cell::{Cell, UnsafeCell};
use std::sync::{Barrier, MutexGuard, TryLockError};
use std::time::Instant;

use super::*;

#[derive(Clone, Copy, Default)]
struct Wait {
    acquisitions: u64,
    blocked: u64,
    blocked_ns: u128,
}

thread_local! {
    static WAIT: Cell<Option<Wait>> = const { Cell::new(None) };
}

pub(super) fn lock(made: &Mutex<Vec<u64>>) -> Option<MutexGuard<'_, Vec<u64>>> {
    WAIT.with(|counter| {
        let mut wait = counter.get()?;
        wait.acquisitions += 1;
        let guard = match made.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                let started = Instant::now();
                let guard = made.lock().unwrap_or_else(|error| error.into_inner());
                wait.blocked += 1;
                wait.blocked_ns += started.elapsed().as_nanos();
                guard
            }
        };
        counter.set(Some(wait));
        Some(guard)
    })
}

#[test]
#[ignore = "times real host descriptor recording with a D3D12 device"]
fn benchmark_recording_contention() {
    const BATCH: u32 = 1024;
    const WAVES: usize = 12;
    const SAMPLES: usize = 7;
    let probe = std::env::var_os("VICTORY_TEST_MIRROR_WAIT").is_some_and(|value| value == "1");
    let raw = [39, 1, 0x1688, 0, 0, 0, 4, 0, 1, 0];
    let resources = [(); 2].map(|()| crate::resource::test_buffer(256));
    let records = resources.map(|resource| {
        let (address, len) = crate::resource::buffer_bytes(resource).expect("title buffer bytes");
        crate::shadow::register(resource as usize, address as u64, len);
        let mut record = [0u32; 4];
        crate::descriptor::create_srv(std::ptr::null(), resource, &raw, record.as_mut_ptr() as u64);
        assert_eq!(record[0], VIEW_TAG);
        record
    });
    for (workers, shared) in [(1, false), (4, false), (4, true)] {
        for changing in [false, true] {
            let count = if shared { BATCH } else { BATCH * workers };
            let heap = HostHeap::new(false, count).expect("D3D12 descriptor heap");
            let slots: Vec<_> = (0..count).map(|_| UnsafeCell::new(records[0])).collect();
            let base = slots.as_ptr() as usize;
            let handle = move |index: u32| (base + index as usize * size_of::<[u32; 4]>()) as u64;
            for index in 0..count {
                assert!(heap.mirror(index, handle(index), true).is_some());
            }
            for sample in 0..=SAMPLES {
                let ready = &Barrier::new(workers as usize + 1);
                let completed = &Barrier::new(workers as usize + 1);
                let workers_done = std::thread::scope(|scope| {
                    let heap = &heap;
                    let mut threads = Vec::new();
                    for worker in 0..workers {
                        threads.push(scope.spawn(move || {
                            WAIT.set(probe.then(Wait::default));
                            let start = if shared { 0 } else { worker * BATCH };
                            for _ in 0..WAVES {
                                ready.wait();
                                for index in start..start + BATCH {
                                    assert!(heap.mirror(index, handle(index), true).is_some());
                                }
                                completed.wait();
                            }
                            WAIT.take().unwrap_or_default()
                        }));
                    }
                    let started = Instant::now();
                    for wave in 0..WAVES {
                        if changing {
                            let record = records[(wave + 1) % records.len()];
                            for index in 0..count {
                                // The workers access the records only between these barriers.
                                // SAFETY: the volatile write stays in the owned slot storage.
                                unsafe { (handle(index) as *mut [u32; 4]).write_volatile(record) };
                            }
                        }
                        ready.wait();
                        completed.wait();
                    }
                    let elapsed = started.elapsed();
                    let wait = threads
                        .into_iter()
                        .fold(Wait::default(), |mut total, worker| {
                            let wait = worker.join().expect("recorder thread");
                            total.acquisitions += wait.acquisitions;
                            total.blocked += wait.blocked;
                            total.blocked_ns += wait.blocked_ns;
                            total
                        });
                    (elapsed, wait)
                });
                let (elapsed, wait) = workers_done;
                if sample != 0 {
                    eprintln!(
                        "recording workers={workers} shared={shared} changing={changing} sample={sample} calls={} unique_slots={count} wall_ns={} acquisitions={} blocked={} blocked_ns={}",
                        u64::from(workers * BATCH) * WAVES as u64,
                        elapsed.as_nanos(),
                        wait.acquisitions,
                        wait.blocked,
                        wait.blocked_ns,
                    );
                }
            }
            let [_, number, generation, _] = records[0];
            let key = u64::from(generation) << 32 | u64::from(number);
            for index in 0..count {
                assert_eq!(heap.ready[index as usize].load(Ordering::Acquire), key);
                assert!(heap.kept(index));
            }
        }
    }
    resources.into_iter().for_each(|resource| {
        crate::object::release(resource);
    });
}
