//! Two bounded protected copies of the primary streaming request pool.
use super::{Reader, copy, wide, word};
const LIMIT: usize = 2048;
const STRIDE: usize = 168;
pub(super) const TABLE: usize = 80;
pub(super) const SIZE: usize = TABLE + LIMIT * 32;
pub(super) const GLOBAL: usize = 0x7bc5d90;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct Shape {
    pool: usize,
    backing: usize,
    allocated: usize,
    completed: usize,
}
struct Header {
    shape: Shape,
    count: u32,
    sequence: u64,
    head: u32,
}

fn header(base: usize, read: &mut Reader<'_>) -> Result<Header, u8> {
    let pool = usize::from_le_bytes(copy(read, base.checked_add(GLOBAL).ok_or(2)?).ok_or(3)?);
    if pool == 0 || !pool.is_multiple_of(8) {
        return Err(2);
    }
    let h = copy::<152>(read, pool.checked_add(24).ok_or(2)?).ok_or(3)?;
    let shape = Shape {
        pool,
        backing: wide(&h, 0) as usize,
        allocated: wide(&h, 40) as usize,
        completed: wide(&h, 56) as usize,
    };
    let count = word(&h, 16);
    if word(&h, 8) != STRIDE as u32
        || word(&h, 12) != LIMIT as u32
        || count > LIMIT as u32
        || word(&h, 64) != 64
        || word(&h, 68) != LIMIT as u32
        || [
            (shape.backing, STRIDE * LIMIT, 8),
            (shape.allocated, 256, 4),
            (shape.completed, 256, 4),
        ]
        .iter()
        .any(|&(p, n, alignment)| {
            p == 0 || !p.is_multiple_of(alignment) || p.checked_add(n).is_none()
        })
    {
        return Err(2);
    }
    Ok(Header {
        shape,
        count,
        sequence: wide(&h, 144),
        head: word(&h, 32),
    })
}

fn module_range(base: usize, read: &mut Reader<'_>) -> Option<(u32, u32)> {
    let r = copy::<20>(read, base.checked_add(0xa024230)?)?;
    let first = word(&r, 8);
    let count = word(&r, 16);
    let end = first.checked_add(count)?;
    (wide(&r, 0) == base.checked_add(0x61ced80)? as u64
        && (1..=0x3fffff).contains(&count)
        && end <= 0xfffffff)
        .then_some((first, end))
}

pub(super) struct Pass {
    data: Vec<u8>,
    allocated: [u8; 256],
    completed: [u8; 256],
    shape: Shape,
    count: u32,
    start: u64,
    end: u64,
    head_start: u32,
    head_end: u32,
    status: u8,
    range: Option<(u32, u32)>,
}
impl Default for Pass {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            allocated: [0; 256],
            completed: [0; 256],
            shape: Shape::default(),
            count: 0,
            start: 0,
            end: 0,
            head_start: 0,
            head_end: 0,
            status: 0,
            range: None,
        }
    }
}
impl Pass {
    pub(super) fn read(&mut self, base: usize, read: &mut Reader<'_>) {
        self.range = module_range(base, read);
        self.status = self.load(base, read).err().unwrap_or(1);
    }
    fn load(&mut self, base: usize, read: &mut Reader<'_>) -> Result<(), u8> {
        let h = header(base, read)?;
        self.data.resize(LIMIT * STRIDE, 0);
        if !read(h.shape.allocated, &mut self.allocated)
            || !read(h.shape.completed, &mut self.completed)
            || !read(h.shape.backing, &mut self.data)
        {
            return Err(3);
        }
        let end = header(base, read)?;
        if h.shape != end.shape {
            return Err(4);
        }
        let occupied = self.allocated.iter().map(|b| b.count_ones()).sum::<u32>();
        if h.count != occupied || end.count != occupied {
            return Err(5);
        }
        self.shape = h.shape;
        self.count = h.count;
        self.start = h.sequence;
        self.end = end.sequence;
        self.head_start = h.head;
        self.head_end = end.head;
        Ok(())
    }
}

fn tuple(k: &[u8]) -> [u8; 32] {
    let mut row = [0; 32];
    let sid = word(k, 0) & 0xfffffff;
    row[0] = if k[137] == 0 && sid != 0xfffffff {
        2
    } else {
        4
    };
    row[1] = k[160];
    row[3] = k[137];
    row[4..8].copy_from_slice(&sid.to_le_bytes());
    row[8..16].copy_from_slice(&k[120..128]);
    row[16..20].copy_from_slice(&k[144..148]);
    row[20..22].copy_from_slice(&k[8..10]);
    row[24..32].copy_from_slice(&k[152..160]);
    row
}

pub(super) fn encode(a: &Pass, b: &Pass, out: &mut [u8]) {
    out.fill(0);
    if let Some((low, high)) = a.range.filter(|range| b.range == Some(*range)) {
        out[1] = 1;
        out[12..16].copy_from_slice(&low.to_le_bytes());
        out[16..20].copy_from_slice(&high.to_le_bytes());
    }
    if a.status != 1 || b.status != 1 {
        out[0] = if a.status != 1 { a.status } else { b.status };
        return;
    }
    if a.shape != b.shape {
        out[0] = 4;
        return;
    }
    out[0] = 1;
    out[2] = u8::from(
        a.start == a.end
            && a.end == b.start
            && b.start == b.end
            && a.head_start == a.head_end
            && a.head_end == b.head_start
            && b.head_start == b.head_end,
    );
    out[72..76].copy_from_slice(&a.head_start.to_le_bytes());
    out[76..80].copy_from_slice(&b.head_end.to_le_bytes());
    out[4..8].copy_from_slice(&a.count.to_le_bytes());
    out[8..12].copy_from_slice(&b.count.to_le_bytes());
    for (i, v) in [
        a.shape.pool as u64,
        a.shape.backing as u64,
        a.shape.allocated as u64,
        a.shape.completed as u64,
        a.start,
        b.end,
    ]
    .iter()
    .enumerate()
    {
        out[24 + i * 8..32 + i * 8].copy_from_slice(&v.to_le_bytes());
    }
    for (i, out) in out[TABLE..].as_chunks_mut::<32>().0.iter_mut().enumerate() {
        let mask = 1 << (i % 8);
        let byte = i / 8;
        if (a.allocated[byte] ^ b.allocated[byte]) & mask != 0
            || (a.completed[byte] ^ b.completed[byte]) & mask != 0
        {
            out[0] = 3;
            continue;
        }
        if a.allocated[byte] & mask == 0 {
            out[0] = if a.completed[byte] & mask == 0 { 1 } else { 3 };
            continue;
        }
        let k = i * STRIDE;
        let first = tuple(&a.data[k..k + STRIDE]);
        let second = tuple(&b.data[k..k + STRIDE]);
        if first != second {
            out[0] = 3;
            continue;
        }
        *out = second;
        out[2] = u8::from(b.completed[byte] & mask != 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BASE: usize = 0x10000000;
    const POOL: usize = 0x20000000;
    const BACK: usize = 0x30000000;
    const ALLOC: usize = 0x40000000;
    const DONE: usize = 0x50000000;
    struct Memory {
        hdr: [u8; 152],
        alloc: [u8; 256],
        done: [u8; 256],
        requests: Vec<u8>,
    }
    impl Memory {
        fn new() -> Self {
            let mut mem = Self {
                hdr: [0; 152],
                alloc: [0; 256],
                done: [0; 256],
                requests: vec![0; 344064],
            };
            mem.hdr[..8].copy_from_slice(&BACK.to_le_bytes());
            mem.hdr[8..12].copy_from_slice(&168_u32.to_le_bytes());
            mem.hdr[12..16].copy_from_slice(&2048_u32.to_le_bytes());
            mem.hdr[16..20].copy_from_slice(&2_u32.to_le_bytes());
            mem.hdr[40..48].copy_from_slice(&ALLOC.to_le_bytes());
            mem.hdr[56..64].copy_from_slice(&DONE.to_le_bytes());
            mem.hdr[64..68].copy_from_slice(&64_u32.to_le_bytes());
            mem.hdr[68..72].copy_from_slice(&2048_u32.to_le_bytes());
            mem.alloc[0] = 6;
            mem.done[0] = 2;
            for i in 1..=2 {
                let k = &mut mem.requests[168 * i..][..168];
                k[..4].copy_from_slice(&0xf0000017_u32.to_le_bytes());
                k[120..128].copy_from_slice(&35_u64.to_le_bytes());
                k[144..148].copy_from_slice(&500_u32.to_le_bytes());
                k[160] = if i == 1 { 2 } else { 4 };
            }
            mem
        }
        fn read(&self, at: usize, bytes: &mut [u8]) -> bool {
            let data: &[u8] = match at {
                a if a == BASE + GLOBAL => &POOL.to_le_bytes(),
                a if a == POOL + 24 => &self.hdr,
                BACK => &self.requests,
                ALLOC => &self.alloc,
                DONE => &self.done,
                _ => return false,
            };
            if data.len() != bytes.len() {
                return false;
            }
            bytes.copy_from_slice(data);
            true
        }
    }

    #[test]
    fn primary_preserves_two_matching_requests_and_completed_zero_stamp() {
        let mut mem = Memory::new();
        let mut a = Pass::default();
        let mut b = Pass::default();
        a.read(BASE, &mut |at, bytes| mem.read(at, bytes));
        mem.hdr[144..152].copy_from_slice(&1_u64.to_le_bytes());
        b.read(BASE, &mut |at, bytes| mem.read(at, bytes));
        let mut out = vec![0; SIZE];
        encode(&a, &b, &mut out);
        assert_eq!(out[0], 1);
        assert_eq!((wide(&out, 56), wide(&out, 64)), (0, 1));
        assert_eq!(out[TABLE], 1);
        let first = &out[TABLE + 32..TABLE + 64];
        assert_eq!(&first[..4], &[2, 2, 1, 0]);
        assert_eq!(
            (
                word(first, 4),
                wide(first, 8),
                word(first, 16),
                wide(first, 24)
            ),
            (23, 35, 500, 0)
        );
        assert_eq!(&out[TABLE + 64..TABLE + 68], &[2, 4, 0, 0]);
    }

    #[test]
    fn changed_release_head_prevents_continuous_join_with_stable_endpoint_identity() {
        let mut mem = Memory::new();
        mem.requests[STRIDE + 8..STRIDE + 10].copy_from_slice(&[3, 9]);
        let mut a = Pass::default();
        let mut b = Pass::default();
        a.read(BASE, &mut |at, dst| mem.read(at, dst));
        mem.hdr[32..36].copy_from_slice(&0x10005_u32.to_le_bytes());
        b.read(BASE, &mut |at, dst| mem.read(at, dst));
        let mut out = vec![0; SIZE];
        encode(&a, &b, &mut out);
        assert_eq!((word(&out, 72), word(&out, 76)), (0, 0x10005));
        assert_eq!(out[2], 0);
        assert_eq!(&out[TABLE + 32 + 20..TABLE + 32 + 22], &[3, 9]);
        encode(&b, &b, &mut out);
        assert_eq!(out[2], 1);
    }

    #[test]
    fn primary_reuse_and_torn_completion_invalidate_only_affected_slots() {
        for (offset, in_done) in [(152, false), (120, false), (160, false), (0, true)] {
            let mut mem = Memory::new();
            let mut a = Pass::default();
            let mut b = Pass::default();
            a.read(BASE, &mut |at, bytes| mem.read(at, bytes));
            if in_done {
                mem.done[0] ^= 2;
            } else {
                mem.requests[STRIDE + offset] ^= 1;
            }
            b.read(BASE, &mut |at, bytes| mem.read(at, bytes));
            let mut out = vec![0; SIZE];
            encode(&a, &b, &mut out);
            assert_eq!(out[TABLE + 32], 3);
            assert!(out[TABLE + 33..TABLE + 64].iter().all(|&v| v == 0));
            assert_eq!(out[TABLE + 64], 2);
        }
    }

    #[test]
    fn module_counts_require_the_concrete_registry_while_all_pool_counts_remain_available() {
        let mem = Memory::new();
        let mut out = vec![0; SIZE];
        for (vtable, expected) in [(BASE + 0x61ced80, 1), (BASE + 0x61ced88, 0)] {
            let mut pass = Pass::default();
            pass.read(BASE, &mut |at, b| {
                if at != BASE + 0xa024230 {
                    return mem.read(at, b);
                }
                b[..8].copy_from_slice(&vtable.to_le_bytes());
                b[8..12].copy_from_slice(&20_u32.to_le_bytes());
                b[16..20].copy_from_slice(&100_u32.to_le_bytes());
                true
            });
            encode(&pass, &pass, &mut out);
            assert_eq!(out[0], 1);
            assert_eq!(out[1], expected);
            assert_eq!(out[TABLE + 32], 2);
            assert_eq!(out[TABLE + 64], 2);
        }
    }

    #[test]
    fn an_unoccupied_slot_requires_two_clear_completion_bits() {
        let mut mem = Memory::new();
        let mut clear = Pass::default();
        let mut set = Pass::default();
        clear.read(BASE, &mut |at, b| mem.read(at, b));
        mem.done[0] |= 1;
        set.read(BASE, &mut |at, b| mem.read(at, b));
        let mut out = vec![0; SIZE];
        for (a, b, expected) in [
            (&clear, &clear, 1),
            (&clear, &set, 3),
            (&set, &clear, 3),
            (&set, &set, 3),
        ] {
            encode(a, b, &mut out);
            assert_eq!(out[TABLE], expected);
            assert_eq!(out[TABLE + 32], 2);
            assert_eq!(out[TABLE + 3 * 32], 1);
        }
    }

    #[test]
    fn malformed_or_unreadable_request_storage_cannot_look_empty() {
        for (offset, unreadable) in [
            (8, 0),
            (12, 0),
            (40, 0),
            (64, 0),
            (68, 0),
            (0, DONE),
            (0, BACK),
        ] {
            let mut mem = Memory::new();
            let mut a = Pass::default();
            if unreadable == 0 {
                mem.hdr[offset] ^= 1;
            }
            let expect_rejected_before_copy = unreadable == 0 && offset != 40;
            a.read(BASE, &mut |at, bytes| {
                if expect_rejected_before_copy {
                    assert_ne!(at, BACK);
                }
                at != unreadable && mem.read(at, bytes)
            });
            let mut out = vec![0; SIZE];
            encode(&a, &a, &mut out);
            assert_ne!(out[0], 1);
            assert!(out[TABLE..].iter().all(|&v| v == 0));
        }
    }
}
