use std::ops::Range;

/// Rebase a compressed band onto its first staged row. Reducing the height
/// would change the distance between depth slices.
pub(super) fn compact(
    base: u64,
    offset: u64,
    pitch: u32,
    height: u32,
    depth: u32,
    block: u32,
    rows: &Range<u32>,
) -> Option<(u64, Range<u32>)> {
    let texels = rows.start.saturating_mul(block)..rows.end.saturating_mul(block).min(height);
    let offset = base
        .checked_add(offset)?
        .checked_add(u64::from(rows.start) * u64::from(pitch))?;
    (depth == 1 && !texels.is_empty() && offset.is_multiple_of(512)).then_some((offset, texels))
}

#[cfg(test)]
mod tests {
    use super::compact;

    #[test]
    fn compact_band_reads_the_staged_rows_at_a_nonzero_base() {
        // The allocation starts at 1024 and this subresource starts 512 bytes in.
        let mut staging = [0xFF; 3080];
        staging[2560..2568].fill(0x45);
        staging[2816..2824].fill(0x56);
        let (source, texels) = compact(1024, 512, 256, 25, 1, 4, &(4..6)).unwrap();
        let source = source as usize;
        assert_eq!(texels, 16..24);
        assert_eq!(staging[source..source + 8], [0x45; 8]);
        assert_eq!(staging[source + 256..source + 264], [0x56; 8]);
    }

    #[test]
    fn an_unaligned_offset_or_multiple_depth_slices_preclude_rebasing() {
        assert_eq!(compact(1024, 512, 256, 25, 1, 4, &(3..5)), None);
        assert_eq!(compact(1024, 512, 256, 25, 2, 4, &(4..6)), None);
    }

    #[test]
    fn compact_band_clips_the_final_block_row() {
        assert_eq!(
            compact(1024, 512, 256, 25, 1, 4, &(6..7)),
            Some((3072, 24..25))
        );
        assert_eq!(compact(1024, 512, 256, 25, 1, 4, &(7..7)), None);
        assert_eq!(
            compact(1024, 512, 256, 25, 1, 4, &(0..7)),
            Some((1536, 0..25))
        );
    }
}
