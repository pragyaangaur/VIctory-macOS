use super::{LOOSE, buffers_read};
use std::sync::OnceLock;

/// A portion of a shadow's title range that currently maps physical memory.
/// The section identifier is local to the mapping snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalCopyExtent {
    pub address: usize,
    pub len: usize,
    pub section: usize,
    pub offset: usize,
}

type PhysicalCopies = fn(&[(usize, usize)], &mut Vec<PhysicalCopyExtent>);
static LOOKUP: OnceLock<PhysicalCopies> = OnceLock::new();

/// Installs the runtime's snapshot of the current physical backing of
/// title address ranges.
pub fn on_physical_copy_extents(lookup: PhysicalCopies) {
    let _ = LOOKUP.set(lookup);
}

/// Counts byte coverage within disjoint domains, merging overlaps within
/// each domain. The input intervals are end-exclusive.
fn covered(mut intervals: Vec<(usize, usize, usize)>) -> usize {
    intervals.sort_unstable();
    let mut bytes = 0;
    let mut previous = None;
    for (domain, start, end) in intervals {
        let from = match previous {
            Some((last_domain, last_end)) if last_domain == domain => start.max(last_end),
            _ => start,
        };
        bytes += end.saturating_sub(from);
        previous = Some((domain, from.max(end)));
    }
    bytes
}

fn measure(extents: &[PhysicalCopyExtent]) -> [usize; 3] {
    let copied = extents.iter().map(|extent| extent.len).sum();
    let title = covered(
        extents
            .iter()
            .map(|extent| (0, extent.address, extent.address + extent.len))
            .collect(),
    );
    let physical = covered(
        extents
            .iter()
            .map(|extent| (extent.section, extent.offset, extent.offset + extent.len))
            .collect(),
    );
    [copied, title, physical]
}

pub(super) fn log() {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_COPY_CENSUS")) {
        return;
    }
    let Some(lookup) = LOOKUP.get() else { return };
    let started = std::time::Instant::now();
    let mut ranges = Vec::new();
    {
        let buffers = buffers_read();
        ranges.extend(
            buffers
                .values()
                .filter_map(|entry| entry.shadow.as_deref())
                .filter(|copy| !copy.aliased)
                .map(|copy| (copy.start as usize, copy.len as usize)),
        );
    }
    {
        let loose = LOOSE.lock().unwrap_or_else(|error| error.into_inner());
        ranges.extend(
            loose
                .values()
                .filter(|copy| !copy.aliased)
                .map(|copy| (copy.start as usize, copy.len as usize)),
        );
    }
    let mut extents = Vec::new();
    lookup(&ranges, &mut extents);
    let [copied, title, physical] = measure(&extents);
    tracing::info!(
        "physical backing of registered and loose copies: {copied} copied bytes, {title} distinct title bytes, \
         {physical} distinct physical bytes, {} bytes at additional title addresses; census {:.2} ms",
        title - physical,
        started.elapsed().as_secs_f64() * 1e3,
    );
}

#[cfg(test)]
#[test]
fn distinguishes_same_address_overlap_from_physical_aliases() {
    let extent = |address, len, section, offset| PhysicalCopyExtent {
        address,
        len,
        section,
        offset,
    };
    let extents = [
        extent(100, 60, 1, 20),
        extent(120, 20, 1, 40), // overlap of two shadows at the same address
        extent(200, 60, 1, 50), // another address shares 30 backing bytes
        extent(300, 60, 2, 20), // equal offset in a different section
    ];
    assert_eq!(measure(&extents), [200, 180, 150]);
}
