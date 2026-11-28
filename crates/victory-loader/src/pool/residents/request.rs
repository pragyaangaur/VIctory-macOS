use super::owner::Owner;
use super::{Reader, copy, wide, word};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Request {
    pub(crate) source: usize,
    pub(crate) key: u64,
    pub(crate) override_byte: u8,
    pub(crate) descriptor: usize,
    pub(crate) task: usize,
    pub(crate) list: usize,
    pub(crate) index: u32,
    pub(crate) end: u32,
    pub(crate) count: u16,
    pub(crate) array: usize,
    pub(crate) id: u32,
    pub(crate) matched_owner: Option<Owner>,
}
pub(crate) fn capture(
    base: usize,
    registers: [usize; 6],
    read: &mut Reader<'_>,
) -> Option<Request> {
    let [sp, source, descriptor_slot, bitset, index, end] = registers;
    if descriptor_slot != sp.checked_add(0xb8)?
        || index >= end
        || end > u16::MAX as usize
        || source == 0
    {
        return None;
    }
    let task = bitset.checked_sub(32)?;
    let mut observe = || -> Option<Request> {
        let stack = copy::<0x140>(read, sp)?;
        for (offset, rva) in [
            (0x38, 0xa5db22),
            (0x88, 0x5453ac7),
            (0xd8, 0x23d0f9a),
            (0x138, 0x23d1211),
        ] {
            if wide(&stack, offset) != base.checked_add(rva)? as u64 {
                return None;
            }
        }
        let descriptor = wide(&stack, 0xb8) as usize;
        if descriptor == 0
            || usize::from_le_bytes(copy(read, descriptor)?) != base.checked_add(0x6501de0)?
        {
            return None;
        }
        let args = copy::<32>(read, task)?;
        let list = wide(&args, 0) as usize;
        let array = wide(&args, 16) as usize;
        if list == 0 || array == 0 || !array.is_multiple_of(8) || args[24] != 12 {
            return None;
        }
        let hdr = copy::<10>(read, list)?;
        let data = wide(&hdr, 0) as usize;
        let count = u16::from_le_bytes(hdr[8..10].try_into().unwrap());
        let size = word(&args, 12);
        let start = word(&args, 8).checked_mul(size)?;
        if size == 0
            || data == 0
            || !data.is_multiple_of(8)
            || index < start as usize
            || start.checked_add(size)?.min(u32::from(count)) as usize != end
            || usize::from_le_bytes(copy(read, data.checked_add(index * 8)?)?) != source
        {
            return None;
        }
        let b = copy::<0x7e>(read, source)?;
        Some(Request {
            source,
            key: wide(&b, 0x20),
            override_byte: b[0x7d],
            descriptor,
            task,
            list,
            index: index as u32,
            end: end as u32,
            count,
            array,
            id: word(&args, 28),
            matched_owner: None,
        })
    };
    let mut r = observe()?;
    if observe()? != r {
        return None;
    }
    r.matched_owner = super::owner::owner(base, r.id, read).filter(|o| {
        o.pointer != 0
            && o.array == r.array
            && o.count >= r.end
            && super::owner::owner(base, r.id, read) == Some(*o)
    });
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn failed_batch_requires_observed_stack_chunk_and_list_source_to_agree() {
        let base = 0x10000000_usize;
        let sp = 0x21000000_usize;
        let src = 0x22000000_usize;
        let desc = 0x23000000_usize;
        let task = 0x24000000_usize;
        let list = 0x25000000_usize;
        let data = 0x26000000_usize;
        let mut stack = vec![0; 0x140];
        for (i, rva) in [
            (0x38, 0xa5db22),
            (0x88, 0x5453ac7),
            (0xd8, 0x23d0f9a),
            (0x138, 0x23d1211),
        ] {
            stack[i..i + 8].copy_from_slice(&(base + rva).to_le_bytes());
        }
        stack[0xb8..0xc0].copy_from_slice(&desc.to_le_bytes());
        let mut source = vec![0; 0x7e];
        source[0x20..0x28].copy_from_slice(&123_u64.to_le_bytes());
        let mut args = vec![0; 32];
        args[..8].copy_from_slice(&list.to_le_bytes());
        args[8..12].copy_from_slice(&1_u32.to_le_bytes());
        args[12..16].copy_from_slice(&4_u32.to_le_bytes());
        args[16..24].copy_from_slice(&0x27000000_usize.to_le_bytes());
        args[24] = 12;
        args[28..32].copy_from_slice(&99_u32.to_le_bytes());
        let mut hdr = vec![0; 10];
        hdr[..8].copy_from_slice(&data.to_le_bytes());
        hdr[8..10].copy_from_slice(&7_u16.to_le_bytes());
        let mut mem = BTreeMap::from([
            (sp, stack),
            (src, source),
            (desc, (base + 0x6501de0).to_le_bytes().to_vec()),
            (task, args),
            (list, hdr),
            (data + 40, src.to_le_bytes().to_vec()),
        ]);
        let regs = [sp, src, sp + 0xb8, task + 32, 5, 7];
        let observe = |m: &BTreeMap<usize, Vec<u8>>, args| {
            capture(base, args, &mut |at, out| {
                let Some((&p, b)) = m.range(..=at).next_back() else {
                    return false;
                };
                let Some(b) = b.get(at - p..at - p + out.len()) else {
                    return false;
                };
                out.copy_from_slice(b);
                true
            })
        };
        let r = observe(&mem, regs).unwrap();
        assert_eq!(
            (r.key, r.index, r.end, r.count, r.id, r.matched_owner),
            (123, 5, 7, 7, 99, None)
        );
        let mut bad = regs;
        bad[5] = 8;
        assert!(observe(&mem, bad).is_none());
        mem.get_mut(&(data + 40)).unwrap()[0] ^= 1;
        assert!(observe(&mem, regs).is_none());
    }
}
