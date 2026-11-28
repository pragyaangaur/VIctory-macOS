//! Bounded observations of BUILDING slots and the title's removal queues.
use super::Sample;
pub(super) mod owner;
mod primary;
pub(super) mod request;
use std::io::{self, Seek, SeekFrom, Write};

const LIMIT: usize = 22_500;
const TIMING: usize = 384 + owner::COUNT * owner::BYTES;
const PASS: usize = TIMING + 64;
const PRIMARY: usize = PASS + 32;
const HEADER: usize = PRIMARY + primary::SIZE;
const RECORD: usize = 40;
const FRAME: usize = HEADER + LIMIT * RECORD + 8;
const INPUTS: usize = 0x77d8d90;
const MIXED: usize = 0x77d9070;
const VTABLE: usize = 0x62e3fe0;
#[cfg(all(windows, feature = "diagnostics"))]
pub(super) const IMAGE_RANGES: [(usize, usize); 13] = [
    (INPUTS, 248),
    (MIXED, 16),
    (VTABLE, 8),
    (0xa024230, 232),
    (primary::GLOBAL, 8),
    (0xabcd210, 10),
    (0x764aa00, 4),
    (0x655ce30, 8),
    (0x764b240, 8),
    (0xab44324, 4),
    (0xabbef44, 4),
    (0x7bd24f8, 4),
    (0x7450df8, 19),
];

fn word(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn wide(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

type Reader<'a> = dyn FnMut(usize, &mut [u8]) -> bool + 'a;

fn copy<const N: usize>(read: &mut Reader<'_>, at: usize) -> Option<[u8; N]> {
    let mut bytes = [0; N];
    read(at, &mut bytes).then_some(bytes)
}

fn valid_queue(q: &[u8]) -> bool {
    word(q, 4) <= word(q, 0) && (word(q, 0) == 0 || wide(q, 8) != 0)
}

fn payload<const WIDTH: usize>(
    q: Option<&[u8]>,
    limit: usize,
    entries: &mut Vec<u8>,
    check: &mut Vec<u8>,
    read: &mut Reader<'_>,
) -> bool {
    let Some(q) = q.filter(|q| valid_queue(q) && word(q, 4) as usize <= limit) else {
        return false;
    };
    let n = word(q, 4) as usize * WIDTH;
    entries.resize(n, 0);
    check.resize(n, 0);
    let at = wide(q, 8) as usize;
    n == 0
        || (at.is_multiple_of(WIDTH)
            && at.checked_add(n).is_some()
            && read(at, entries)
            && read(at, check)
            && entries == check)
}

pub(super) struct Census {
    frame: Vec<u8>,
    meta: [Vec<u8>; 2],
    slots: [Vec<u8>; 2],
    input: Vec<u8>,
    mixed: Vec<u8>,
    check: Vec<u8>,
    milestones: u8,
    roster: owner::Roster,
    primary: [primary::Pass; 2],
}

impl Census {
    pub(super) fn new() -> Self {
        Self {
            frame: vec![0; FRAME],
            meta: Default::default(),
            slots: Default::default(),
            input: Vec::new(),
            mixed: Vec::new(),
            check: Vec::new(),
            milestones: 0,
            roster: Default::default(),
            primary: Default::default(),
        }
    }

    fn read_queues(&mut self, base: usize, read: &mut Reader<'_>) -> (bool, bool) {
        let table = copy::<248>(read, base + INPUTS);
        let mix = copy::<16>(read, base + MIXED);
        let mut input_ok = payload::<4>(
            table
                .as_ref()
                .filter(|b| (2..=15).contains(&word(&b[..], 4)))
                .map(|b| &b[24..40]),
            LIMIT,
            &mut self.input,
            &mut self.check,
            read,
        );
        let mut mixed_ok = payload::<8>(
            mix.as_ref().map(|b| &b[..]),
            65_536,
            &mut self.mixed,
            &mut self.check,
            read,
        );
        let end = copy::<248>(read, base + INPUTS);
        let mut mask = 0_u32;
        if let (Some(a), Some(b)) = (table, end) {
            let count = word(&a, 4);
            if count == word(&b, 4) && count <= 15 {
                for i in 0..count as usize {
                    let offset = 8 + i * 16;
                    let q = &a[offset..offset + 16];
                    if q == &b[offset..offset + 16] && valid_queue(q) {
                        self.frame[128 + i * 16..144 + i * 16].copy_from_slice(q);
                        mask |= 1 << i;
                    }
                }
            }
        }
        input_ok &= mask & 2 != 0;
        if let Some(q) =
            mix.filter(|q| valid_queue(q) && copy::<16>(read, base + MIXED) == Some(*q))
        {
            self.frame[368..384].copy_from_slice(&q);
            mask |= 1 << 15;
        } else {
            mixed_ok = false;
        }
        self.frame[28..32].copy_from_slice(&mask.to_le_bytes());
        (input_ok, mixed_ok)
    }

    fn clock(&mut self, base: usize, read: &mut Reader<'_>) {
        for (bit, rva, out) in [(1, 0xabbef44, 92), (2, 0x7bd24f8, 96)] {
            if let Some(n) =
                copy::<4>(read, base + rva).filter(|n| copy::<4>(read, base + rva) == Some(*n))
            {
                self.frame[88] |= bit;
                self.frame[out..out + 4].copy_from_slice(&n);
            }
        }
        if let Some(b) = copy::<19>(read, base + 0x7450df8)
            .filter(|b| copy::<19>(read, base + 0x7450df8) == Some(*b))
        {
            self.frame[88] |= 4;
            self.frame[89] = u8::from(b[8..10] == 9_u16.to_le_bytes() && &b[10..] == b"Prestream");
            self.frame[104..112].copy_from_slice(&b[..8]);
        }
    }

    pub(super) fn capture(
        &mut self,
        base: usize,
        start: Option<Sample>,
        mut read: impl FnMut(usize, &mut [u8]) -> bool,
        end: impl FnOnce() -> Option<Sample>,
    ) {
        self.primary[0].read(base, &mut read);
        self.capture_residents(base, start, &mut read, end);
        self.primary[1].read(base, &mut read);
        primary::encode(
            &self.primary[0],
            &self.primary[1],
            &mut self.frame[PRIMARY..HEADER],
        );
    }

    fn capture_residents(
        &mut self,
        base: usize,
        start: Option<Sample>,
        mut read: impl FnMut(usize, &mut [u8]) -> bool,
        end: impl FnOnce() -> Option<Sample>,
    ) {
        self.frame.fill(0);
        self.frame[..8].copy_from_slice(b"BRES0002");
        self.clock(base, &mut read);
        owner::timing(base, &mut self.frame[TIMING..PASS], &mut read);
        owner::admissions(base, &mut self.frame[PASS..PRIMARY], &mut read);
        let Some(s) = start else { return };
        self.frame[32..40].copy_from_slice(&(s.pointer as u64).to_le_bytes());
        self.frame[40..48].copy_from_slice(&s.backing.to_le_bytes());
        self.frame[48..56].copy_from_slice(&s.metadata.to_le_bytes());
        self.frame[56..60].copy_from_slice(&s.capacity.to_le_bytes());
        self.frame[60..64].copy_from_slice(&s.used.to_le_bytes());
        self.frame[64..72].copy_from_slice(&s.head.to_le_bytes());
        let n = s.capacity as usize;
        self.frame[24] = 2;
        if !(1..=LIMIT).contains(&n)
            || s.stride != 224
            || s.backing == 0
            || s.metadata == 0
            || !s.backing.is_multiple_of(8)
            || !s.metadata.is_multiple_of(4)
            || s.backing.checked_add((n * 224) as u64).is_none()
            || s.metadata.checked_add((n * 4) as u64).is_none()
        {
            return;
        }
        for m in &mut self.meta {
            m.resize(n * 4, 0);
        }
        for b in &mut self.slots {
            b.resize(n * 224, 0);
        }
        self.frame[24] = 3;
        if !read(s.metadata as usize, &mut self.meta[0])
            || !read(s.backing as usize, &mut self.slots[0])
        {
            return;
        }
        let (input_ok, mixed_ok) = self.read_queues(base, &mut read);
        if !read(s.backing as usize, &mut self.slots[1])
            || !read(s.metadata as usize, &mut self.meta[1])
        {
            return;
        }
        let rows = &mut self.frame[HEADER..HEADER + n * RECORD];
        let input_ok = input_ok
            && self
                .input
                .as_chunks::<4>()
                .0
                .iter()
                .all(|b| (word(b, 0) & 0xffff) < s.capacity);
        let mut valid = 0_u32;
        for (i, row) in rows.as_chunks_mut::<RECORD>().0.iter_mut().enumerate() {
            if self.meta[0][i * 4..i * 4 + 4] != self.meta[1][i * 4..i * 4 + 4] {
                continue;
            }
            let meta = word(&self.meta[0], i * 4);
            row[4..8].copy_from_slice(&meta.to_le_bytes());
            if meta & 0x1ffff != 0x1ffff {
                row[0] = 1;
                continue;
            }
            let fields = |snapshot: &[u8]| {
                let b = &snapshot[i * 224..][..224];
                (
                    wide(b, 0),
                    b[0x30],
                    b[0x96] & 15,
                    b[0xb0] & 1,
                    word(b, 0x9c),
                    wide(b, 0x20),
                    wide(b, 0xa8),
                    (b[0xb4] >> 1) & 31,
                    word(b, 0xcc),
                )
            };
            let a = fields(&self.slots[0]);
            if a != fields(&self.slots[1]) {
                row.fill(0);
                continue;
            }
            let (vtable, ty, state, pending, handle, descriptor, reference, class, raw_id) = a;
            let expected = (meta << 5 & 0xffc00000) | (1 << 17) | i as u32;
            if vtable != (base + VTABLE) as u64 || ty != 1 || handle != expected {
                row[0] = 3;
                continue;
            }
            valid += 1;
            row[..4].copy_from_slice(&[2, ty, state, pending]);
            row[8..12].copy_from_slice(&handle.to_le_bytes());
            row[12] = u8::from(input_ok);
            row[14] = u8::from(mixed_ok);
            row[15] = class;
            row[16..24].copy_from_slice(&descriptor.to_le_bytes());
            row[24..32].copy_from_slice(&reference.to_le_bytes());
            row[32..36].copy_from_slice(&raw_id.to_le_bytes());
        }
        if input_ok {
            for entry in self.input.as_chunks::<4>().0 {
                let i = (word(entry, 0) & 0xffff) as usize;
                let row = &mut rows[i * RECORD..][..RECORD];
                if row[0] == 2 {
                    row[12] = if row[12] == 1 { 2 } else { 3 };
                    row[13] = if row[12] == 2 { entry[2] } else { 0 };
                }
            }
        }
        if mixed_ok {
            for entry in self.mixed.as_chunks::<8>().0 {
                if let Some(off) = wide(entry, 0).checked_sub(s.backing)
                    && off.is_multiple_of(224)
                    && off / 224 < n as u64
                {
                    let row = &mut rows[off as usize / 224 * RECORD..][..RECORD];
                    if row[0] == 2 {
                        row[14] = if row[14] == 1 { 2 } else { 3 };
                    }
                }
            }
        }
        let (header, data) = self.frame.split_at_mut(HEADER);
        let eligible = self.roster.select(&data[..n * RECORD]);
        header[114..116].copy_from_slice(&(eligible as u16).to_le_bytes());
        header[112] = owner::observe_owners(
            base,
            &data[..n * RECORD],
            s.backing,
            s.metadata,
            &mut header[384..TIMING],
            &mut read,
            &self.roster,
        ) as u8;
        header[76..80].copy_from_slice(&valid.to_le_bytes());
        self.frame[24] = 4;
        let Some(e) = end().filter(|e| {
            (e.pointer, e.backing, e.metadata, e.capacity, e.stride)
                == (s.pointer, s.backing, s.metadata, s.capacity, s.stride)
        }) else {
            self.frame[76..80].fill(0);
            self.frame[112..116].fill(0);
            self.frame[384..TIMING].fill(0);
            self.frame[HEADER..HEADER + n * RECORD].fill(0);
            return;
        };
        self.frame[72..76].copy_from_slice(&e.used.to_le_bytes());
        self.frame[80..88].copy_from_slice(&e.head.to_le_bytes());
        self.frame[24] = 1;
    }

    pub(super) fn publish(
        &mut self,
        out: &mut (impl Write + Seek),
        seq: u64,
        ms: u64,
    ) -> io::Result<()> {
        self.frame[8..16].copy_from_slice(&seq.to_le_bytes());
        self.frame[16..24].copy_from_slice(&ms.to_le_bytes());
        self.frame[FRAME - 8..].copy_from_slice(&(!seq).to_le_bytes());
        let mut save = |slot: u64| {
            out.seek(SeekFrom::Start(slot * FRAME as u64))?;
            out.write_all(&self.frame)
        };
        save(seq % 64)?;
        let capacity = word(&self.frame, 56);
        let used = word(&self.frame, 60);
        if self.frame[24] == 1 && word(&self.frame, 76) > 0 {
            for (i, threshold) in [0, 50, 80, 95].into_iter().enumerate() {
                if self.milestones & (1 << i) == 0 && used > 0 && used * 100 >= capacity * threshold
                {
                    save(64 + i as u64)?;
                    self.milestones |= 1 << i;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::tests::header;
    use std::io::Cursor;

    const BASE: usize = 0x1000_0000;
    const BACKING: usize = 0x3000_0000;
    const META: usize = 0x4000_0000;
    const INPUT: usize = 0x5000_0000;
    const MIXED: usize = 0x6000_0000;

    struct Memory {
        sample: Sample,
        slots: Vec<u8>,
        meta: [u32; 3],
        queues: [u8; 248],
        mixed: [u8; 16],
        input_entries: [u8; 8],
        mixed_entries: [u8; 16],
    }

    impl Memory {
        fn new() -> Self {
            let mut sample = Sample::parse(8, 224, &header()).unwrap();
            sample.capacity = 3;
            sample.used = 2;
            sample.backing = BACKING as u64;
            sample.metadata = META as u64;
            let mut slots = vec![0; 224 * 3];
            for index in [0, 2] {
                let row = &mut slots[index * 224..][..224];
                row[..8].copy_from_slice(&((BASE + 0x62e3fe0) as u64).to_le_bytes());
                row[0x20..0x28].copy_from_slice(&70_u64.to_le_bytes());
                row[0x30] = 1;
                row[0x96] = 0x85;
                row[0x9c..0xa0].copy_from_slice(&(0x00c20000_u32 + index as u32).to_le_bytes());
                row[0xa8..0xb0].copy_from_slice(&60_u64.to_le_bytes());
                row[0xb0] = 3;
                row[0xb4] = 10;
                row[0xcc..0xd0].copy_from_slice(&51_u32.to_le_bytes());
            }
            let mut queues = [0; 248];
            queues[4..8].copy_from_slice(&15_u32.to_le_bytes());
            queues[24..28].copy_from_slice(&7_u32.to_le_bytes());
            queues[28..32].copy_from_slice(&2_u32.to_le_bytes());
            queues[32..40].copy_from_slice(&(INPUT as u64).to_le_bytes());
            let mut mixed = [0; 16];
            mixed[..4].copy_from_slice(&3_u32.to_le_bytes());
            mixed[4..8].copy_from_slice(&2_u32.to_le_bytes());
            mixed[8..].copy_from_slice(&(MIXED as u64).to_le_bytes());
            let mut mem = Self {
                sample,
                slots,
                meta: [0x7ffff, 0x20002, 0x7ffff],
                queues,
                mixed,
                input_entries: [0; 8],
                mixed_entries: [0; 16],
            };
            mem.input_entries[..4].copy_from_slice(&0x240000_u32.to_le_bytes());
            mem.input_entries[4..].copy_from_slice(&0x130002_u32.to_le_bytes());
            for ptr in mem.mixed_entries.as_chunks_mut::<8>().0 {
                ptr.copy_from_slice(&((BACKING + 448) as u64).to_le_bytes());
            }
            mem
        }

        fn read(&self, at: usize, out: &mut [u8]) -> bool {
            let meta = self.meta.map(u32::to_le_bytes).concat();
            let data: &[u8] = match at {
                BACKING => &self.slots,
                META => &meta,
                INPUT => &self.input_entries,
                MIXED => &self.mixed_entries,
                a if a == BASE + 0x77d8d90 => &self.queues,
                a if a == BASE + 0x77d9070 => &self.mixed,
                _ => return false,
            };
            if data.len() != out.len() {
                return false;
            }
            out.copy_from_slice(data);
            true
        }
    }

    #[test]
    fn published_snapshot_retains_only_matching_occupants_and_observed_queue_membership() {
        let mem = Memory::new();
        let mut scratch = Census::new();
        scratch.capture(
            BASE,
            Some(mem.sample),
            |at, dst| mem.read(at, dst),
            || Some(mem.sample),
        );
        let rows = &scratch.frame[HEADER..HEADER + 3 * RECORD];
        assert_eq!(&rows[..4], &[2, 1, 5, 1]);
        assert_eq!(&rows[4..8], &0x7ffff_u32.to_le_bytes());
        assert_eq!(&rows[8..12], &0xc20000_u32.to_le_bytes());
        assert_eq!(&rows[12..16], &[2, 0x24, 1, 5]);
        assert_eq!(&rows[16..24], &70_u64.to_le_bytes());
        assert_eq!(&rows[24..32], &60_u64.to_le_bytes());
        assert_eq!(&rows[32..36], &51_u32.to_le_bytes());
        assert_eq!(rows[RECORD], 1);
        assert_eq!(&rows[RECORD + 8..RECORD * 2], &vec![0; RECORD - 8]);
        assert_eq!(&rows[2 * RECORD + 12..2 * RECORD + 15], &[2, 0x13, 3]);
        assert_eq!(&scratch.frame[24..28], &1_u32.to_le_bytes());
    }

    #[test]
    fn reuse_changed_fields_and_queue_races_cannot_be_reported_as_correlated() {
        for (addr, byte, expected) in [(META, 0, 0), (BACKING, 0xa8, 0), (INPUT, 2, 2)] {
            let mem = Memory::new();
            let mut scratch = Census::new();
            let mut passes = 0;
            scratch.capture(
                BASE,
                Some(mem.sample),
                |at, dst| {
                    let ok = mem.read(at, dst);
                    if ok && at == addr {
                        passes += 1;
                        if passes == 2 {
                            dst[byte] ^= 1;
                        }
                    }
                    ok
                },
                || Some(mem.sample),
            );
            let row = &scratch.frame[HEADER..][..RECORD];
            assert_eq!(row[0], expected);
            assert_eq!(row[12], 0);
            if expected == 0 {
                assert!(row.iter().all(|v| *v == 0));
            }
        }
        let mut mem = Memory::new();
        mem.meta[0] = 0x9ffff;
        let mut scratch = Census::new();
        scratch.capture(
            BASE,
            Some(mem.sample),
            |at, dst| mem.read(at, dst),
            || Some(mem.sample),
        );
        let row = &scratch.frame[HEADER..][..RECORD];
        assert_eq!(row[0], 3);
        assert_eq!(&row[8..], &vec![0; RECORD - 8]);
    }

    #[test]
    fn lost_backing_replaced_storage_and_oversize_counts_invalidate_observations() {
        let mut mem = Memory::new();
        let mut scratch = Census::new();
        for (missing, replaced) in [(BACKING, false), (0, true)] {
            let mut end = mem.sample;
            if replaced {
                end.metadata += 64;
            }
            scratch.capture(
                BASE,
                Some(mem.sample),
                |at, dst| at != missing && mem.read(at, dst),
                || Some(end),
            );
            assert_ne!(scratch.frame[24], 1);
            assert!(
                scratch.frame[HEADER..HEADER + RECORD * 3]
                    .iter()
                    .all(|b| *b == 0)
            );
        }
        mem.mixed[..8].copy_from_slice(&[1, 0, 1, 0, 1, 0, 1, 0]);
        scratch.capture(
            BASE,
            Some(mem.sample),
            |at, dst| {
                assert_ne!(at, MIXED);
                mem.read(at, dst)
            },
            || Some(mem.sample),
        );
        assert_eq!(scratch.frame[HEADER + 14], 0);
        assert_eq!(scratch.frame[HEADER + 12], 2);
        let mut larger = mem.sample;
        larger.capacity = 22501;
        scratch.capture(
            BASE,
            Some(larger),
            |at, _| {
                assert_ne!(at, BACKING);
                false
            },
            || Some(larger),
        );
        assert_ne!(scratch.frame[24], 1);
    }

    #[test]
    fn frame_footer_rejects_interrupted_wrap_and_milestones_are_retained() {
        fn complete(bytes: &[u8]) -> Option<u64> {
            if bytes.len() != FRAME || &bytes[..8] != b"BRES0002" {
                return None;
            }
            let seq = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            let footer = u64::from_le_bytes(bytes[FRAME - 8..].try_into().unwrap());
            (footer == !seq).then_some(seq)
        }
        let mem = Memory::new();
        let mut scratch = Census::new();
        scratch.capture(
            BASE,
            Some(mem.sample),
            |at, dst| mem.read(at, dst),
            || Some(mem.sample),
        );
        let mut out = Cursor::new(Vec::new());
        scratch.publish(&mut out, 0, 111).unwrap();
        let old = out.get_ref()[..FRAME].to_vec();
        assert_eq!(complete(&old), Some(0));
        scratch.publish(&mut out, 64, 222).unwrap();
        let bytes = out.into_inner();
        assert_eq!(complete(&bytes[..FRAME]), Some(64));
        assert_eq!(complete(&bytes[64 * FRAME..65 * FRAME]), Some(0));
        assert_eq!(complete(&bytes[65 * FRAME..66 * FRAME]), Some(0));
        assert!(bytes.len() <= 68 * FRAME);
        let mut partial = old;
        partial[..FRAME - 8].copy_from_slice(&bytes[..FRAME - 8]);
        assert_eq!(complete(&partial), None);
        assert_eq!(complete(&bytes[..FRAME - 1]), None);
    }
}
