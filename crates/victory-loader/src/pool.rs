//! Periodic, read-only pool observations for the identified main executable.

mod residents;
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod windows;

const BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const EMPTY: u32 = 0x1fffe;
const SLOTS: [usize; 3] = [0xa9ec938, 0xabbe2e8, 0x98d79a8];
const TRAPS: [usize; 2] = [0x5149aec, 0x63966f];

fn enabled(value: Option<&str>, hash: &str, is_dll: bool) -> bool {
    value == Some("1") && !is_dll && hash == BUILD
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sample {
    pointer: usize,
    backing: u64,
    metadata: u64,
    stride: u32,
    capacity: u32,
    used: u32,
    head: u64,
}

impl Sample {
    fn parse(pointer: usize, stride: u32, bytes: &[u8; 52]) -> Result<Self, &'static str> {
        let word = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let wide = |i| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        let capacity = word(12);
        let used = word(16);
        let head = wide(32);
        let index = head as u32;
        if word(8) != stride
            || !(1..=EMPTY).contains(&capacity)
            || used > capacity
            || (index != EMPTY && index >= capacity)
            || word(48) != 0x1ffff
        {
            return Err("invalid-header");
        }
        Ok(Self {
            pointer,
            backing: wide(0),
            metadata: wide(40),
            stride,
            capacity,
            used,
            head,
        })
    }

    fn activity(self, previous: Option<Self>) -> Result<(u32, Option<i64>), &'static str> {
        let old = previous.ok_or("no-contiguous-sample")?;
        if self.pointer != old.pointer
            || self.capacity != old.capacity
            || self.stride != old.stride
            || self.backing == 0
            || self.backing != old.backing
            || self.metadata == 0
            || self.metadata != old.metadata
        {
            return Err("storage-changed-or-unavailable");
        }
        let tag = (self.head >> 32) as u32;
        let old_tag = (old.head >> 32) as u32;
        let delta = tag.wrapping_sub(old_tag);
        let pops =
            (tag >= old_tag).then(|| i64::from(self.used) - i64::from(old.used) + i64::from(delta));
        Ok((delta, pops))
    }
}

#[derive(Clone, Copy)]
struct Map {
    slots: [usize; 3],
    traps: [usize; 2],
}

impl Map {
    fn new(base: usize, size: usize) -> Option<Self> {
        base.checked_add(size)?;
        if base == 0
            || SLOTS.iter().any(|rva| rva + 8 > size)
            || TRAPS.iter().any(|rva| *rva >= size)
        {
            return None;
        }
        Some(Self {
            slots: SLOTS.map(|rva| base + rva),
            traps: TRAPS.map(|rva| base + rva),
        })
    }

    fn select(self, at: usize, pointer: usize, globals: [Option<usize>; 3]) -> Option<usize> {
        let candidates = match self.traps.iter().position(|trap| *trap == at)? {
            0 => 0..2,
            _ => 2..3,
        };
        let mut found = None;
        for i in candidates {
            if globals[i]? == pointer && pointer != 0 {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn header() -> [u8; 52] {
        let mut bytes = [0; 52];
        bytes[..8].copy_from_slice(&0x1000_u64.to_le_bytes());
        bytes[8..12].copy_from_slice(&224_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&17_u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&10_u32.to_le_bytes());
        bytes[32..40].copy_from_slice(&(9_u64 << 32 | u64::from(EMPTY)).to_le_bytes());
        bytes[40..48].copy_from_slice(&0x2000_u64.to_le_bytes());
        bytes[48..52].copy_from_slice(&0x1ffff_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn accepts_only_opted_in_exact_original_main_and_bounded_rvas() {
        assert!(enabled(Some("1"), BUILD, false));
        assert!(!enabled(None, BUILD, false));
        assert!(!enabled(Some("0"), BUILD, false));
        assert!(!enabled(Some("1"), &BUILD[..16], false));
        assert!(!enabled(Some("1"), BUILD, true));
        assert_eq!(
            Map::new(0x1000, 0xc000000).unwrap().slots[2],
            0x1000 + SLOTS[2]
        );
        assert!(Map::new(usize::MAX - 12, 0xc000000).is_none());
        assert!(Map::new(0x1000, 0xabbe2ef).is_none());
    }

    #[test]
    fn matches_only_one_readable_global_at_its_own_trap() {
        let map = Map {
            slots: [11, 12, 13],
            traps: [14, 15],
        };
        let globals = [Some(21), Some(22), Some(23)];
        assert_eq!(map.select(14, 22, globals), Some(1));
        assert_eq!(map.select(15, 23, globals), Some(2));
        assert_eq!(map.select(14, 23, globals), None);
        assert_eq!(map.select(91, 21, globals), None);
        assert_eq!(map.select(14, 0, [Some(0); 3]), None);
        assert_eq!(map.select(14, 21, [Some(21), Some(21), Some(23)]), None);
        assert_eq!(map.select(14, 21, [Some(21), None, Some(23)]), None);
    }

    #[test]
    fn parses_complete_tag_and_rejects_malformed_headers() {
        let bytes = header();
        let got = Sample::parse(0x800, 224, &bytes).unwrap();
        assert_eq!(
            (got.used, got.capacity, got.head),
            (10, 17, 9_u64 << 32 | u64::from(EMPTY))
        );
        assert!(Sample::parse(0x800, 272, &bytes).is_err());
        for (offset, value) in [
            (12, 0),
            (12, EMPTY + 1),
            (16, 18),
            (32, 17),
            (32, 0xffff_fffe),
            (48, 0),
        ] {
            let mut bad = bytes;
            bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Sample::parse(0x800, 224, &bad).is_err());
        }
    }

    #[test]
    fn activity_preserves_signed_skew_marks_gaps_replacement_and_wrap() {
        let a = Sample::parse(0x800, 224, &header()).unwrap();
        let b = Sample {
            used: 6,
            head: a.head + (3_u64 << 32),
            ..a
        };
        assert_eq!(b.activity(Some(a)), Ok((3, Some(-1))));
        assert!(b.activity(None).is_err());
        for changed in [
            Sample { pointer: 1, ..a },
            Sample { backing: 0, ..a },
            Sample { metadata: 0, ..a },
            Sample { stride: 272, ..a },
            Sample { capacity: 18, ..a },
        ] {
            assert!(b.activity(Some(changed)).is_err());
        }
        let wrapped = Sample { head: 2, ..a };
        assert_eq!(wrapped.activity(Some(a)), Ok((0_u32.wrapping_sub(9), None)));
    }
}
