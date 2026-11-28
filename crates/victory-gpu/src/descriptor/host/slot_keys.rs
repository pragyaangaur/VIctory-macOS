use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

const STRIPES: usize = 64;

/// Host publication keys. Recorders can publish independent stripes in
/// parallel. Batch publication and eviction exclude all recorders with one
/// lock so their descriptor copies and retention updates stay batched.
pub(super) struct SlotKeys {
    locks: PublicationLocks,
    keys: Box<[AtomicU64]>,
}

enum PublicationLocks {
    Single(Mutex<()>),
    Striped(Box<Stripes>),
}

struct Stripes {
    bulk: RwLock<()>,
    slots: [Mutex<()>; STRIPES],
}

pub(super) enum SlotGuard<'a> {
    Single {
        _lock: MutexGuard<'a, ()>,
    },
    Bulk {
        _bulk: RwLockWriteGuard<'a, ()>,
    },
    Striped {
        _stripe: MutexGuard<'a, ()>,
        _bulk: RwLockReadGuard<'a, ()>,
    },
}

impl SlotKeys {
    pub(super) fn new(count: u32) -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| crate::switch::on("VICTORY_SLOT_STRIPES"));
        let locks = if enabled {
            PublicationLocks::Striped(Box::new(Stripes {
                bulk: RwLock::new(()),
                slots: [const { Mutex::new(()) }; STRIPES],
            }))
        } else {
            PublicationLocks::Single(Mutex::new(()))
        };
        Self {
            locks,
            keys: (0..count).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub(super) fn one(&self, index: u32) -> SlotGuard<'_> {
        match &self.locks {
            PublicationLocks::Single(lock) => SlotGuard::Single {
                _lock: lock_one(lock),
            },
            PublicationLocks::Striped(stripes) => {
                let bulk = stripes
                    .bulk
                    .read()
                    .unwrap_or_else(|error| error.into_inner());
                let stripe = &stripes.slots[index as usize % STRIPES];
                SlotGuard::Striped {
                    _stripe: lock_one(stripe),
                    _bulk: bulk,
                }
            }
        }
    }

    pub(super) fn all(&self) -> SlotGuard<'_> {
        match &self.locks {
            PublicationLocks::Single(lock) => SlotGuard::Single {
                _lock: lock.lock().unwrap_or_else(|error| error.into_inner()),
            },
            PublicationLocks::Striped(stripes) => SlotGuard::Bulk {
                _bulk: stripes
                    .bulk
                    .write()
                    .unwrap_or_else(|error| error.into_inner()),
            },
        }
    }

    pub(super) fn read(&self, index: u32) -> Option<u64> {
        let _slot = self.one(index);
        self.get(index)
    }

    /// Reads the key while holding the corresponding publication guard.
    pub(super) fn get(&self, index: u32) -> Option<u64> {
        self.keys
            .get(index as usize)
            .map(|key| key.load(Ordering::Relaxed))
    }

    /// Sets the key while holding the corresponding publication guard.
    pub(super) fn set(&self, index: u32, value: u64) {
        self.keys[index as usize].store(value, Ordering::Relaxed);
    }
}

#[inline]
fn lock_one(lock: &Mutex<()>) -> MutexGuard<'_, ()> {
    #[cfg(test)]
    if let Some(guard) = super::recording_contention::lock(lock) {
        return guard;
    }
    lock.lock().unwrap_or_else(|error| error.into_inner())
}
