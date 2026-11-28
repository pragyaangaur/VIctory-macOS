use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

const STRIPES: usize = 64;

/// Host publication keys. Recorders can publish independent stripes in
/// parallel. Batch publication and eviction exclude all recorders with one
/// lock so their descriptor copies and retention updates stay batched.
pub(super) struct SlotKeys {
    bulk: RwLock<()>,
    stripes: Option<Box<[Mutex<()>; STRIPES]>>,
    keys: Box<[AtomicU64]>,
}

pub(super) enum SlotGuard<'a> {
    Exclusive {
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
        Self {
            bulk: RwLock::new(()),
            stripes: enabled.then(|| Box::new([const { Mutex::new(()) }; STRIPES])),
            keys: (0..count).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub(super) fn one(&self, index: u32) -> SlotGuard<'_> {
        let Some(stripes) = &self.stripes else {
            return SlotGuard::Exclusive { _bulk: self.all() };
        };
        let bulk = self.bulk.read().unwrap_or_else(|error| error.into_inner());
        let stripe = &stripes[index as usize % STRIPES];
        #[cfg(test)]
        let probed = super::recording_contention::lock(stripe);
        #[cfg(not(test))]
        let probed = None;
        SlotGuard::Striped {
            _stripe: probed
                .unwrap_or_else(|| stripe.lock().unwrap_or_else(|error| error.into_inner())),
            _bulk: bulk,
        }
    }

    pub(super) fn all(&self) -> RwLockWriteGuard<'_, ()> {
        self.bulk.write().unwrap_or_else(|error| error.into_inner())
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
