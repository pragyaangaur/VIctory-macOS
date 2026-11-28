use super::{Reader, copy, wide, word};

pub(super) const COUNT: usize = 16;
pub(super) const BYTES: usize = 128;
const REG: usize = 0xa024230;
const MANAGER: usize = 0x764b240;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Owner {
    pub(crate) record: usize,
    pub(crate) pointer: usize,
    pub(crate) array: usize,
    pub(crate) count: u32,
    stream_id: u32,
    record_24: u32,
    record_28: u32,
    owner_c4: u32,
}

fn pointer(read: &mut Reader<'_>, at: usize) -> Option<usize> {
    let p = usize::from_le_bytes(copy(read, at)?);
    (p != 0 && p.is_multiple_of(8)).then_some(p)
}

pub(crate) fn owner(base: usize, id: u32, read: &mut Reader<'_>) -> Option<Owner> {
    let index = id & 0x3fffff;
    let reg = copy::<104>(read, base.checked_add(REG)?)?;
    if index == 0x3fffff
        || index >= word(&reg, 16)
        || id >> 28 != 0
        || (id >> 22) != word(&reg, 12) & 63
    {
        return None;
    }
    let bits = wide(&reg, 96) as usize;
    let records = wide(&reg, 56) as usize;
    let stride = word(&reg, 64);
    if bits == 0 || records == 0 || stride < 44 {
        return None;
    }
    if copy::<1>(read, bits.checked_add((index / 8) as usize)?)?[0] & (1 << (index % 8)) == 0 {
        return None;
    }
    let record = records.checked_add(index.checked_mul(stride)? as usize)?;
    let r = copy::<44>(read, record)?;
    let p = wide(&r, 0) as usize;
    if p == 0 {
        return Some(Owner {
            record,
            pointer: 0,
            array: 0,
            count: 0,
            stream_id: word(&reg, 8).checked_add(index)? & 0xfffffff,
            record_24: word(&r, 36),
            record_28: word(&r, 40),
            owner_c4: 0,
        });
    }
    if !p.is_multiple_of(8) {
        return None;
    }
    let o = copy::<200>(read, p)?;
    let array = wide(&o, 8) as usize;
    let count = word(&o, 16);
    if wide(&o, 0) != base.checked_add(0x655ce30)? as u64
        || word(&o, 20) != id
        || !array.is_multiple_of(8)
        || (count != 0 && array == 0)
        || array.checked_add(count as usize * 8).is_none()
    {
        return None;
    }
    Some(Owner {
        record,
        pointer: p,
        array,
        count,
        stream_id: word(&reg, 8).checked_add(index)? & 0xfffffff,
        record_24: word(&r, 36),
        record_28: word(&r, 40),
        owner_c4: word(&o, 196),
    })
}

fn bitmap(read: &mut Reader<'_>, at: usize, index: u32) -> Option<([u8; 16], u32)> {
    let h = copy::<16>(read, at)?;
    let p = wide(&h, 0) as usize;
    let words = word(&h, 8);
    let bits = word(&h, 12);
    if p == 0
        || !p.is_multiple_of(4)
        || index >= bits
        || index / 32 >= words
        || u64::from(bits) > u64::from(words) * 32
    {
        return None;
    }
    let v = u32::from_le_bytes(copy(read, p.checked_add((index / 32) as usize * 4)?)?);
    Some((h, v))
}

fn module_flags(
    base: usize,
    id: u32,
    module: u8,
    read: &mut Reader<'_>,
) -> Option<([usize; 7], u16)> {
    let reg = base.checked_add(REG)?;
    let modules = copy::<10>(read, base.checked_add(0xabcd210)?)?;
    let table = wide(&modules, 0) as usize;
    let n = u16::from_le_bytes(modules[8..10].try_into().unwrap());
    if table == 0
        || !table.is_multiple_of(8)
        || u16::from(module) >= n
        || pointer(read, table.checked_add(module as usize * 8)?)? != reg
    {
        return None;
    }
    let registry = copy::<232>(read, reg)?;
    let vtable = wide(&registry, 0) as usize;
    let index = id & 0x3fffff;
    let bound = word(&registry, 16);
    let occupancy = wide(&registry, 96) as usize;
    let flags = wide(&registry, 224) as usize;
    if id >> 28 != 0
        || index == 0x3fffff
        || index >= bound
        || id >> 22 != word(&registry, 12) & 63
        || vtable == 0
        || usize::from_le_bytes(copy(read, vtable.checked_add(0x98)?)?)
            != base.checked_add(0x4f83000)?
        || occupancy == 0
        || flags == 0
        || !flags.is_multiple_of(2)
    {
        return None;
    }
    let byte = copy::<1>(read, occupancy.checked_add(index as usize / 8)?)?[0];
    if byte & (1 << (index % 8)) == 0 {
        return None;
    }
    let value = u16::from_le_bytes(copy(read, flags.checked_add(index as usize * 2)?)?);
    Some((
        [
            table,
            n as usize,
            vtable,
            occupancy,
            flags,
            bound as usize,
            byte as usize,
        ],
        value,
    ))
}

fn upstream(
    base: usize,
    m: usize,
    sid: u32,
    id: u32,
    module: u8,
    read: &mut Reader<'_>,
) -> [u8; 16] {
    let mut observe = || {
        let bucket = m
            .checked_add(160 * (sid % 503) as usize + 0x58)
            .and_then(|at| bitmap(read, at, sid / 503));
        let global = m.checked_add(0x16850).and_then(|at| bitmap(read, at, sid));
        let flags = module_flags(base, id, module, read);
        let generation = m.checked_add(0x16700).and_then(|at| copy::<2>(read, at));
        let age = base
            .checked_add(0x764aa00)
            .and_then(|at| copy::<4>(read, at));
        let ordinary = m
            .checked_add(160 * (sid % 503) as usize + 0x68)
            .and_then(|at| bitmap(read, at, sid / 503));
        (bucket, global, flags, generation, age, ordinary)
    };
    let (b, g, f, n, a, ordinary) = observe();
    let end = observe();
    let mut out = [0; 16];
    if let Some((_, w)) = b.filter(|v| end.0 == Some(*v)) {
        out[0] |= 1;
        out[1] = 1 + u8::from(w & (1 << ((sid / 503) % 32)) != 0);
    }
    if let Some((_, w)) = g.filter(|v| end.1 == Some(*v)) {
        out[0] |= 2;
        out[2] = 1 + u8::from(w & (1 << (sid % 32)) != 0);
    }
    if let Some((_, v)) = f.filter(|v| end.2 == Some(*v)) {
        out[0] |= 4;
        out[4..6].copy_from_slice(&v.to_le_bytes());
    }
    if let Some(v) = n.filter(|v| end.3 == Some(*v)) {
        out[0] |= 8;
        out[6..8].copy_from_slice(&v);
    }
    if let Some((_, w)) = ordinary.filter(|v| end.5 == Some(*v)) {
        out[0] |= 32;
        out[12] = 1 + u8::from(w & (1 << ((sid / 503) % 32)) != 0);
    }
    if let Some(v) = a.filter(|v| end.4 == Some(*v)) {
        out[0] |= 16;
        out[8..12].copy_from_slice(&v);
    }
    out
}

fn gate(base: usize, sid: u32, id: u32, read: &mut Reader<'_>) -> (u8, [u8; 28], [u8; 16]) {
    let mut lookup = || -> Option<_> {
        if sid == 0xfffffff {
            return None;
        }
        let m = pointer(read, base.checked_add(MANAGER)?)?;
        let selector = m.checked_add(0x14)?;
        let phase = u32::from_le_bytes(copy(read, selector)?);
        if phase > 1 {
            return None;
        }
        let at = m.checked_add((160 * (sid % 503) + 16 * phase + 32) as usize)?;
        let descriptor = copy::<16>(read, at)?;
        let n = word(&descriptor, 4) as usize;
        if !super::valid_queue(&descriptor) || n > 64 {
            return None;
        }
        let data = wide(&descriptor, 8) as usize;
        let mut entries = [0_u8; 64 * 28];
        let mut check = [0_u8; 64 * 28];
        let bytes = n * 28;
        if n > 0
            && (data == 0
                || !data.is_multiple_of(4)
                || data.checked_add(bytes).is_none()
                || !read(data, &mut entries[..bytes]))
        {
            return None;
        }
        let mut matches = entries[..bytes]
            .as_chunks::<28>()
            .0
            .iter()
            .filter(|r| (word(&r[..], 0) & 0xfffffff) == sid);
        let result = match (matches.next(), matches.next()) {
            (Some(r), None) => (2, *r, upstream(base, m, sid, id, r[0x16], read)),
            (None, _) => (1, [0; 28], [0; 16]),
            _ => (3, [0; 28], [0; 16]),
        };
        if (n > 0 && (!read(data, &mut check[..bytes]) || entries[..bytes] != check[..bytes]))
            || copy(read, at)? != descriptor
            || u32::from_le_bytes(copy(read, selector)?) != phase
            || pointer(read, base.checked_add(MANAGER)?)? != m
        {
            return None;
        }
        let (status, entry, mut fields) = result;
        if status == 2 {
            fields[3] = phase as u8;
        }
        Some((status, entry, fields))
    };
    lookup().unwrap_or((0, [0; 28], [0; 16]))
}

#[derive(Default)]
pub(super) struct Roster {
    seen: std::collections::HashSet<u32>,
    indices: Vec<usize>,
    offset: usize,
}
impl Roster {
    pub(super) fn select(&mut self, rows: &[u8]) -> usize {
        self.indices.clear();
        self.seen.clear();
        for (index, row) in rows.as_chunks::<{ super::RECORD }>().0.iter().enumerate() {
            if row[0] == 2 && !matches!(row[15], 15 | 21) && self.seen.insert(word(row, 32)) {
                self.indices.push(index);
            }
        }
        let population = self.indices.len();
        if population > COUNT {
            let rotating = &mut self.indices[COUNT / 2..];
            let start = self.offset % rotating.len();
            rotating.rotate_left(start);
            self.offset = (start + COUNT / 2) % rotating.len();
            self.indices.truncate(COUNT);
        }
        population
    }
}

#[cfg(test)]
fn cohort(
    base: usize,
    rows: &[u8],
    backing: u64,
    meta: u64,
    out: &mut [u8],
    read: &mut Reader<'_>,
) -> usize {
    let mut roster = Roster::default();
    roster.select(rows);
    observe_owners(base, rows, backing, meta, out, read, &roster)
}

pub(super) fn observe_owners(
    base: usize,
    rows: &[u8],
    backing: u64,
    meta: u64,
    out: &mut [u8],
    read: &mut Reader<'_>,
    roster: &Roster,
) -> usize {
    let mut n = 0;
    let mut array = Vec::new();
    let mut check = Vec::new();
    for &index in &roster.indices {
        let row = &rows[index * super::RECORD..][..super::RECORD];
        let id = word(row, 32);
        let entry = &mut out[n * BYTES..][..BYTES];
        n += 1;
        entry[..4].copy_from_slice(&(index as u32).to_le_bytes());
        entry[4..8].copy_from_slice(&id.to_le_bytes());
        if let Some(o) = owner(base, id, read) {
            let addr = backing + (index * 224) as u64;
            let output_read = if o.pointer != 0 && o.count <= 4096 {
                let bytes = o.count as usize * 8;
                array.resize(bytes, 0);
                check.resize(bytes, 0);
                bytes == 0 || read(o.array, &mut array)
            } else {
                false
            };
            let (g, gate, upstream) = gate(base, o.stream_id, id, read);
            // Bracket the gate lookup: withdrawal can clear an output in place.
            let (status, idx) = if output_read
                && (array.is_empty() || (read(o.array, &mut check) && array == check))
            {
                let mut found = array
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| wide(&b[..], 0) == addr);
                match (found.next(), found.next()) {
                    (Some((i, _)), None) => (2, i as u32),
                    (None, _) => (1, 0),
                    _ => (3, 0),
                }
            } else {
                (0, 0)
            };
            // Recheck the selected identity after the owner, output and gate reads.
            let slot = copy::<224>(read, addr as usize);
            let metadata = copy::<4>(read, meta as usize + index * 4);
            if owner(base, id, read) == Some(o)
                && metadata.is_some_and(|m| m[..] == row[4..8])
                && slot.is_some_and(|s| {
                    wide(&s, 0) == (base + 0x62e3fe0) as u64
                        && s[0x30] == 1
                        && word(&s, 0x9c) == word(row, 8)
                        && s[0xb4] >> 1 & 31 == row[15]
                        && word(&s, 0xcc) == id
                })
            {
                entry[8] = if o.pointer == 0 { 2 } else { 1 };
                entry[9] = status;
                entry[10] = g;
                entry[12..16].copy_from_slice(&idx.to_le_bytes());
                entry[16..24].copy_from_slice(&(o.record as u64).to_le_bytes());
                entry[24..32].copy_from_slice(&(o.pointer as u64).to_le_bytes());
                entry[32..40].copy_from_slice(&(o.array as u64).to_le_bytes());
                entry[40..44].copy_from_slice(&o.count.to_le_bytes());
                entry[44..48].copy_from_slice(&o.stream_id.to_le_bytes());
                entry[48..52].copy_from_slice(&o.record_24.to_le_bytes());
                entry[52..56].copy_from_slice(&o.record_28.to_le_bytes());
                entry[56..60].copy_from_slice(&o.owner_c4.to_le_bytes());
                entry[60..62].copy_from_slice(&gate[12..14]);
                entry[62..64].copy_from_slice(&gate[20..22]);
                entry[64] = gate[24];
                entry[66..68].copy_from_slice(&gate[26..28]);
                entry[80..108].copy_from_slice(&gate);
                entry[108..124].copy_from_slice(&upstream);
            }
        }
    }
    n
}

pub(super) fn admissions(base: usize, out: &mut [u8], read: &mut Reader<'_>) {
    let mut sample = || {
        let m = pointer(read, base.checked_add(MANAGER)?)?;
        let phase = u32::from_le_bytes(copy(read, m.checked_add(0x14)?)?);
        if phase > 1 {
            return None;
        }
        let attempts = copy::<4>(read, m.checked_add(0x16918)?);
        let cost = copy::<1>(read, m.checked_add(0x16988)?);
        Some((m, phase, attempts, cost))
    };
    if let (Some((m, phase, attempts, cost)), Some((m2, phase2, count2, cost2))) =
        (sample(), sample())
        && m == m2
        && phase == phase2
    {
        out[0] = 1;
        out[8..16].copy_from_slice(&(m as u64).to_le_bytes());
        out[16..20].copy_from_slice(&phase.to_le_bytes());
        if let Some(v) = attempts.filter(|v| count2 == Some(*v)) {
            out[0] |= 2;
            out[20..24].copy_from_slice(&v);
        }
        if let Some(v) = cost.filter(|v| cost2 == Some(*v)) {
            out[0] |= 4;
            out[24] = v[0];
        }
    }
}

pub(super) fn timing(base: usize, out: &mut [u8], read: &mut Reader<'_>) {
    fn observe(base: usize, read: &mut Reader<'_>) -> Option<[u64; 7]> {
        let m = pointer(read, base.checked_add(MANAGER)?)?;
        let record_28 = copy::<4>(read, base.checked_add(0xab44324)?)?;
        let mode = copy::<4>(read, m)?;
        let data = copy::<56>(read, m.checked_add(0x166c8)?)?;
        let callback = copy::<8>(read, m.checked_add(0x168e0)?)?;
        Some([
            m as u64,
            u32::from_le_bytes(record_28) as u64,
            u32::from_le_bytes(mode) as u64,
            word(&data, 0) as u64,
            wide(&data, 40),
            u64::from_le_bytes(callback),
            wide(&data, 48),
        ])
    }
    if let Some(fields) = observe(base, read).filter(|f| observe(base, read) == Some(*f)) {
        out[0] = 1;
        for (dst, value) in out[8..].as_chunks_mut::<8>().0.iter_mut().zip(fields) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const BASE: usize = 0x1000_0000;
    type Memory = BTreeMap<usize, Vec<u8>>;
    fn load(mem: &Memory, at: usize, out: &mut [u8]) -> bool {
        let Some((base, source)) = mem.range(..=at).next_back() else {
            return false;
        };
        let Some(bytes) = source.get(at - base..at - base + out.len()) else {
            return false;
        };
        out.copy_from_slice(bytes);
        true
    }
    fn registry() -> Memory {
        let mut registry = vec![0; 0x68];
        registry[8..12].copy_from_slice(&600_u32.to_le_bytes());
        registry[12..16].copy_from_slice(&2_u32.to_le_bytes());
        registry[16..20].copy_from_slice(&11_u32.to_le_bytes());
        registry[56..64].copy_from_slice(&0x3000_u64.to_le_bytes());
        registry[64..68].copy_from_slice(&48_u32.to_le_bytes());
        registry[96..104].copy_from_slice(&0x5000_u64.to_le_bytes());
        let mut record = vec![0; 48];
        record[..8].copy_from_slice(&0x8000_u64.to_le_bytes());
        let mut object = vec![0; 200];
        object[..8].copy_from_slice(&((BASE + 0x655ce30) as u64).to_le_bytes());
        object[8..16].copy_from_slice(&0x9000_u64.to_le_bytes());
        object[16..20].copy_from_slice(&9_u32.to_le_bytes());
        object[20..24].copy_from_slice(&0x800005_u32.to_le_bytes());
        BTreeMap::from([
            (BASE + 0xa024230, registry),
            (0x3000 + 48 * 5, record),
            (0x5000, vec![32]),
            (0x8000, object),
        ])
    }

    #[test]
    fn registry_requires_bounded_occupied_record_and_bidirectional_owner_identity() {
        let mem = registry();
        let obs = |m: &Memory, id| owner(BASE, id, &mut |a, b| load(m, a, b));
        let o = obs(&mem, 0x800005).unwrap();
        assert_eq!(
            (o.pointer, o.array, o.count, o.stream_id),
            (0x8000, 0x9000, 9, 605)
        );
        for (addr, offset) in [
            (0x5000, 0),
            (0x8000, 0),
            (0x8000, 20),
            (BASE + 0xa024230, 16),
        ] {
            let mut bad = mem.clone();
            bad.get_mut(&addr).unwrap()[offset] = 0;
            if addr == 0x8000 && offset == 0 {
                bad.get_mut(&addr).unwrap()[0] = 1;
            }
            assert!(obs(&bad, 0x800005).is_none());
        }
        assert!(obs(&mem, 0xc00005).is_none());
        assert!(obs(&mem, 0xbfffff).is_none());
    }

    fn resident_with_output() -> (Memory, [u8; super::super::RECORD], u64, u64) {
        let mut mem = registry();
        let backing = 0xa000_u64;
        let metadata = 0xb000_u64;
        let id = 0x800005_u32;
        let mut row = [0_u8; super::super::RECORD];
        row[0] = 2;
        row[4..8].copy_from_slice(&0x7ffff_u32.to_le_bytes());
        row[8..12].copy_from_slice(&0xc20000_u32.to_le_bytes());
        row[15] = 12;
        row[32..36].copy_from_slice(&id.to_le_bytes());
        let mut storage = vec![0; 224];
        storage[..8].copy_from_slice(&((BASE + 0x62e3fe0) as u64).to_le_bytes());
        storage[0x30] = 1;
        storage[0x9c..0xa0].copy_from_slice(&row[8..12]);
        storage[0xb4] = 24;
        storage[0xcc..0xd0].copy_from_slice(&row[32..36]);
        let mut array = vec![0; 9 * 8];
        array[56..64].copy_from_slice(&backing.to_le_bytes());
        mem.insert(0x9000, array);
        mem.insert(backing as usize, storage);
        mem.insert(metadata as usize, row[4..8].to_vec());
        (mem, row, backing, metadata)
    }

    #[test]
    fn confirmed_output_detached_record_and_reused_slot_are_distinct() {
        let (mut mem, row, backing, metadata) = resident_with_output();
        let observe = |m: &Memory| {
            let mut out = [0_u8; BYTES * COUNT];
            assert_eq!(
                cohort(BASE, &row, backing, metadata, &mut out, &mut |a, b| load(
                    m, a, b
                )),
                1
            );
            (out[8], out[9], word(&out, 12))
        };
        assert_eq!(observe(&mem), (1, 2, 7));
        mem.get_mut(&(0x3000 + 48 * 5)).unwrap()[..8].fill(0);
        assert_eq!(observe(&mem), (2, 0, 0));
        mem.get_mut(&(metadata as usize)).unwrap()[2] ^= 2;
        assert_eq!(observe(&mem), (0, 0, 0));
    }

    #[test]
    fn rotation_preserves_eight_anchors_and_gives_each_eligible_id_one_turn() {
        let mut rows = Vec::new();
        for id in (0..32_u32).chain(8..24) {
            let mut row = [0; super::super::RECORD];
            row[0] = 2;
            row[15] = if id == 30 { 15 } else { 12 };
            if id == 31 {
                row[0] = 3;
            }
            row[32..36].copy_from_slice(&id.to_le_bytes());
            rows.extend_from_slice(&row);
        }
        let mut roster = Roster::default();
        for expected in [
            vec![8, 9, 10, 11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20, 21, 22, 23],
            vec![24, 25, 26, 27, 28, 29, 8, 9],
        ] {
            assert_eq!(roster.select(&rows), 30);
            assert_eq!(&roster.indices[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(&roster.indices[8..16], &expected);
        }
        assert_eq!(roster.select(&rows[..3 * super::super::RECORD]), 3);
        assert_eq!(&roster.indices, &[0, 1, 2]);
    }

    #[test]
    fn upstream_flags_and_independent_bitmap_words_require_stable_bounded_inputs() {
        let id = 0x800005;
        let sid = 605_u32;
        let m = 0x12120000_usize;
        let table = 0x14140000_usize;
        let flags = 0x17170000_usize;
        let bits = 0x16160000_usize;
        let mut mem = registry();
        let reg = mem.get_mut(&(BASE + REG)).unwrap();
        reg.resize(232, 0);
        reg[..8].copy_from_slice(&((BASE + 0x61ced80) as u64).to_le_bytes());
        reg[224..232].copy_from_slice(&flags.to_le_bytes());
        mem.insert(
            BASE + 0x61cee18,
            ((BASE + 0x4f83000) as u64).to_le_bytes().to_vec(),
        );
        mem.insert(
            BASE + 0xabcd210,
            [table.to_le_bytes().as_slice(), &3_u16.to_le_bytes()].concat(),
        );
        mem.insert(table + 8 * 2, ((BASE + REG) as u64).to_le_bytes().to_vec());
        mem.insert(flags + 10, 0x179_u16.to_le_bytes().to_vec());
        mem.insert(BASE + 0x764aa00, 77_u32.to_le_bytes().to_vec());
        mem.insert(m + 0x16700, 13_u16.to_le_bytes().to_vec());
        let mut bitmap = vec![0; 16];
        bitmap[..8].copy_from_slice(&bits.to_le_bytes());
        bitmap[8..12].copy_from_slice(&32_u32.to_le_bytes());
        bitmap[12..16].copy_from_slice(&1024_u32.to_le_bytes());
        let bucket = m + 0xa0 * 102 + 0x58;
        mem.insert(bucket, bitmap.clone());
        let mut ordinary = bitmap.clone();
        ordinary[..8].copy_from_slice(&(bits + 0x1000).to_le_bytes());
        mem.insert(bucket + 0x10, ordinary);
        mem.insert(bits + 0x1000, 0_u32.to_le_bytes().to_vec());
        mem.insert(m + 0x16850, bitmap);
        mem.insert(bits, 2_u32.to_le_bytes().to_vec());
        mem.insert(bits + 4 * (605 / 32), 0_u32.to_le_bytes().to_vec());
        let observe = |memb: &Memory| upstream(BASE, m, sid, id, 2, &mut |at, b| load(memb, at, b));
        let out = observe(&mem);
        assert_eq!(out[0], 63);
        assert_eq!(out[12], 1);
        assert_eq!((out[1], out[2]), (2, 1));
        assert_eq!(&out[4..6], &0x179_u16.to_le_bytes());
        assert_eq!(&out[6..8], &13_u16.to_le_bytes());
        assert_eq!(word(&out, 8), 77);
        mem.get_mut(&bucket).unwrap()[12..16].copy_from_slice(&1_u32.to_le_bytes());
        let mut reads = 0;
        let changed = upstream(BASE, m, sid, id, 2, &mut |at, b| {
            let ok = load(&mem, at, b);
            if at == bits + 4 * (605 / 32) {
                reads += 1;
                if reads == 2 {
                    b[0] ^= 1;
                }
            }
            ok
        });
        assert_eq!((changed[0], changed[1], changed[2]), (60, 0, 0));
        mem.insert(table + 16, 0xdead0000_usize.to_le_bytes().to_vec());
        let out = observe(&mem);
        assert_eq!(out[0] & 4, 0);
    }

    #[test]
    fn gate_requires_stable_phase_header_and_unique_matching_record() {
        let data = 0x7777_0000_usize;
        let map = 0x6666_0000_usize;
        let bucket = map + 0xa0 * 102 + 0x20;
        let mut descriptor = vec![0; 16];
        descriptor[..4].copy_from_slice(&2_u32.to_le_bytes());
        descriptor[4..8].copy_from_slice(&2_u32.to_le_bytes());
        descriptor[8..16].copy_from_slice(&(data as u64).to_le_bytes());
        let mut records = vec![0; 56];
        records[..4].copy_from_slice(&605_u32.to_le_bytes());
        records[12..14].copy_from_slice(&9_u16.to_le_bytes());
        let mut mem = BTreeMap::from([
            (BASE + 0x764b240, (map as u64).to_le_bytes().to_vec()),
            (map + 0x14, vec![0; 4]),
            (bucket, descriptor),
            (data, records),
        ]);
        let result = |m: &Memory| gate(BASE, 605, 0x800005, &mut |a, b| load(m, a, b));
        let (status, r, _) = result(&mem);
        assert_eq!((status, r[12]), (2, 9));
        let mut duplicate = mem[&data].clone();
        duplicate[28..32].copy_from_slice(&605_u32.to_le_bytes());
        mem.insert(data, duplicate);
        assert_eq!(result(&mem).0, 3);
        mem.get_mut(&data).unwrap().fill(0);
        assert_eq!(result(&mem).0, 1);
        mem.get_mut(&bucket).unwrap()[4] = 65;
        assert_eq!(result(&mem).0, 0);
    }

    #[test]
    fn cleared_output_cannot_be_joined_to_newly_published_stream_gate() {
        let (mut mem, row, backing, metadata) = resident_with_output();
        let manager = 0x77770000_usize;
        let data = 0x88880000_usize;
        let mut bucket = vec![0; 16];
        bucket[..4].copy_from_slice(&1_u32.to_le_bytes());
        bucket[4..8].copy_from_slice(&1_u32.to_le_bytes());
        bucket[8..].copy_from_slice(&data.to_le_bytes());
        mem.insert(BASE + MANAGER, manager.to_le_bytes().to_vec());
        mem.insert(manager + 0x14, vec![0; 4]);
        mem.insert(manager + 0x20 + 160 * (605 % 503), bucket);
        mem.insert(data, vec![0; 28]);
        let mut out = [0; BYTES * COUNT];
        let mut changed = false;
        cohort(BASE, &row, backing, metadata, &mut out, &mut |addr, buf| {
            if addr == BASE + MANAGER && !changed {
                assert_eq!(wide(&mem[&0x9000], 56), backing);
                assert_eq!(word(&mem[&data], 0), 0);
                mem.get_mut(&0x9000).unwrap()[56..64].fill(0);
                mem.get_mut(&data).unwrap()[..4].copy_from_slice(&605_u32.to_le_bytes());
                changed = true;
            }
            load(&mem, addr, buf)
        });
        assert!(changed);
        assert_eq!(&out[8..11], &[1, 0, 2]);
    }

    const START: u64 = 0x1234_5678_9876_ffff;
    fn timed_stream(change_start: bool) -> [u8; 64] {
        let m = 0x13130000_usize;
        let mut data = vec![0; 56];
        data[..4].copy_from_slice(&500_u32.to_le_bytes());
        data[40..48].copy_from_slice(&333_u64.to_le_bytes());
        data[48..56].copy_from_slice(&START.to_le_bytes());
        let mut mem = Memory::from([
            (BASE + MANAGER, m.to_le_bytes().to_vec()),
            (BASE + 0xab44324, 2_u32.to_le_bytes().to_vec()),
            (m, vec![0; 4]),
            (m + 0x166c8, data),
            (m + 0x168e0, vec![0; 8]),
        ]);
        let mut out = [0; 64];
        timing(BASE, &mut out, &mut |at, b| {
            let ok = load(&mem, at, b);
            if at == m + 0x166c8 && change_start {
                mem.get_mut(&at).unwrap()[48..56].copy_from_slice(&(START + 1).to_le_bytes());
            }
            ok
        });
        out
    }

    #[test]
    fn timing_retains_interval_start_after_callback() {
        let out = timed_stream(false);
        assert_eq!(out[0], 1);
        assert_eq!(wide(&out, 48), 0);
        assert_eq!(wide(&out, 56), START);
    }

    #[test]
    fn timing_discards_changed_interval_start() {
        assert_eq!(timed_stream(true), [0; 64]);
    }
}
