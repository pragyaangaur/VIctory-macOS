//! Packing of the generic elements of a register row into one element.
//!
//! A D3D12 signature can split one register row across several elements:
//! `TEXCOORD0` holds the columns 0-1 of row 5 and `TEXCOORD1` its columns
//! 2-3. The host makes a member of its Metal struct for every element of a
//! signature, so a split row costs more than the components it carries.
//! Packing merges the generic elements of a row into one element, which the
//! mesh and the pixel shader agree on because both pack the same rows.

use std::collections::{BTreeMap, BTreeSet};

use super::container::SigElement;

/// The name a packed element carries: its row, so that the mesh and the
/// pixel shader name it the same way whatever the row's elements were called.
const PACKED: &str = "VREG";

/// The rows whose generic elements can be packed into one element.
///
/// A row qualifies when it holds two or more elements, all of them generic
/// 32-bit floats, and they tile the row from its first column without a gap.
/// A row with a system value or a hole keeps its elements: a packed element
/// would overlap them, and the two stages would not agree on the row.
pub(crate) fn packable_rows(elements: &[SigElement]) -> BTreeSet<u32> {
    grouped(elements)
        .into_iter()
        .filter(|(_, group)| packable(group))
        .map(|(row, _)| row)
        .collect()
}

/// The rows the mesh's outputs and the pixel shader's inputs can pack.
///
/// A row is packed when both stages hold the same semantics in the same
/// layout and both can pack them: the two stages are linked by semantic
/// name, so a packed element must stand for the same semantics on both
/// sides.
pub(crate) fn shared_rows(mesh: &[SigElement], pixel: &[SigElement]) -> BTreeSet<u32> {
    // A signature that already names an element `VREG` keeps its elements,
    // as a packed element would carry the same name twice.
    if mesh
        .iter()
        .chain(pixel)
        .any(|e| e.name.eq_ignore_ascii_case(PACKED))
    {
        return BTreeSet::new();
    }
    let mesh_rows = packable_rows(mesh);
    let pixel_rows = packable_rows(pixel);
    let mut shared = BTreeSet::new();
    for &row in mesh_rows.intersection(&pixel_rows) {
        let (Some(m), Some(p)) = (row_elements(mesh, row), row_elements(pixel, row)) else {
            continue;
        };
        if same_layout(&m, &p) {
            shared.insert(row);
        }
    }
    shared
}

/// Packs the generic elements of `rows` into one element per row.
///
/// The packed element is named [`PACKED`] and indexed by its row, so the
/// mesh and the pixel shader, which pack the same rows, name it the same
/// way. Its columns are the row's, so the components keep their place.
pub(crate) fn coalesce(elements: &mut Vec<SigElement>, rows: &BTreeSet<u32>) {
    if rows.is_empty() {
        return;
    }
    let mut packed: BTreeMap<u32, SigElement> = BTreeMap::new();
    for (row, group) in grouped(elements) {
        if !rows.contains(&row) {
            continue;
        }
        if let Some(element) = packed_element(row, &group) {
            packed.insert(row, element);
        }
    }
    if packed.is_empty() {
        return;
    }
    let mut out = Vec::with_capacity(elements.len());
    let mut done: BTreeSet<u32> = BTreeSet::new();
    for e in elements.drain(..) {
        match e.row {
            Some(row) if packed.contains_key(&row) => {
                if done.insert(row) {
                    out.push(packed[&row].clone());
                }
            }
            _ => out.push(e),
        }
    }
    *elements = out;
}

/// The elements of each row, in the order they were declared.
fn grouped(elements: &[SigElement]) -> BTreeMap<u32, Vec<&SigElement>> {
    let mut rows: BTreeMap<u32, Vec<&SigElement>> = BTreeMap::new();
    for e in elements {
        if let Some(row) = e.row {
            rows.entry(row).or_default().push(e);
        }
    }
    rows
}

/// The elements of one row.
fn row_elements(elements: &[SigElement], row: u32) -> Option<Vec<&SigElement>> {
    let group: Vec<&SigElement> = elements.iter().filter(|e| e.row == Some(row)).collect();
    (!group.is_empty()).then_some(group)
}

/// Whether two rows hold the same semantics in the same layout.
fn same_layout(a: &[&SigElement], b: &[&SigElement]) -> bool {
    let key = |e: &SigElement| {
        (
            e.name.to_ascii_uppercase(),
            e.index,
            e.col,
            e.cols,
            e.kind,
            e.comp,
            e.interp,
        )
    };
    let mut a: Vec<_> = a.iter().map(|e| key(e)).collect();
    let mut b: Vec<_> = b.iter().map(|e| key(e)).collect();
    a.sort_unstable();
    b.sort_unstable();
    a == b
}

/// Whether the elements of one row can be packed into one element.
fn packable(group: &[&SigElement]) -> bool {
    packed_element(group.first().map_or(0, |e| e.row.unwrap_or(0)), group).is_some()
}

/// The packed element of a row, or `None` when the row cannot be packed.
fn packed_element(row: u32, group: &[&SigElement]) -> Option<SigElement> {
    // One element is already one member of the host's struct.
    if group.len() < 2 {
        return None;
    }
    // A system value or an integer keeps its own element, as the two stages
    // would not agree on the row.
    if group.iter().any(|e| e.kind != 0 || e.comp != 9 || e.half) {
        return None;
    }
    let mut sorted = group.to_vec();
    sorted.sort_by_key(|e| e.col);
    let interp = sorted[0].interp;
    if sorted.iter().any(|e| e.interp != interp) {
        return None;
    }
    // The elements must tile the row from its first column.
    let mut col = 0u8;
    for e in &sorted {
        if e.col != col {
            return None;
        }
        col += e.cols;
    }
    if col > 4 {
        return None;
    }
    Some(SigElement {
        name: PACKED.to_owned(),
        index: row,
        kind: 0,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: col,
        usage: 0,
        half: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(
        name: &str,
        index: u32,
        kind: u8,
        comp: u8,
        row: u32,
        col: u8,
        cols: u8,
    ) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 2,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: false,
        }
    }

    fn generic(name: &str, index: u32, row: u32, col: u8, cols: u8) -> SigElement {
        element(name, index, 0, 9, row, col, cols)
    }

    /// A row of two floats is packed into one element named after its row.
    #[test]
    fn a_split_row_is_packed_into_one_element() {
        let mut elements = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        assert_eq!(packable_rows(&elements), BTreeSet::from([1]));
        coalesce(&mut elements, &BTreeSet::from([1]));
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[1].name, "VREG");
        assert_eq!(elements[1].index, 1);
        assert_eq!(elements[1].row, Some(1));
        assert_eq!(elements[1].col, 0);
        assert_eq!(elements[1].cols, 4);
    }

    /// A row of one element is already one member of the host's struct.
    #[test]
    fn a_row_of_one_element_is_left_alone() {
        let elements = vec![generic("TEXCOORD0", 0, 1, 0, 4)];
        assert!(packable_rows(&elements).is_empty());
    }

    /// A system value in the row would overlap the packed element.
    #[test]
    fn a_row_with_a_system_value_is_left_alone() {
        let elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            element("SV_IsFrontFace", 0, 13, 5, 1, 2, 1),
            generic("TEXCOORD1", 1, 1, 3, 1),
        ];
        assert!(packable_rows(&elements).is_empty());
    }

    /// An integer keeps its own element, as the two stages would not agree
    /// on the row's type.
    #[test]
    fn a_row_with_an_integer_is_left_alone() {
        let elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            element("COLOR0", 0, 0, 5, 1, 2, 2),
        ];
        assert!(packable_rows(&elements).is_empty());
    }

    /// A 16-bit float keeps its own element: the packed element is 32-bit.
    #[test]
    fn a_row_with_a_16_bit_float_is_left_alone() {
        let mut elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        elements[1].half = true;
        assert!(packable_rows(&elements).is_empty());
    }

    /// A hole in the row would leave a component the packed element does not
    /// cover.
    #[test]
    fn a_row_with_a_hole_is_left_alone() {
        let elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 1),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        assert!(packable_rows(&elements).is_empty());
    }

    /// The rows the packing is asked for are the only ones packed.
    #[test]
    fn only_the_given_rows_are_packed() {
        let mut elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
            generic("TEXCOORD2", 2, 2, 0, 2),
            generic("TEXCOORD3", 3, 2, 2, 2),
        ];
        coalesce(&mut elements, &BTreeSet::from([1]));
        assert_eq!(elements.len(), 3);
        assert_eq!(elements[0].name, "VREG");
        assert_eq!(elements[0].index, 1);
        assert_eq!(elements[1].name, "TEXCOORD2");
        assert_eq!(elements[1].row, Some(2));
    }

    /// The packed element keeps the row's columns, so the components keep
    /// their place.
    #[test]
    fn a_packed_element_covers_the_row() {
        let mut elements = vec![
            generic("TEXCOORD0", 0, 1, 0, 1),
            generic("TEXCOORD1", 1, 1, 1, 3),
        ];
        assert_eq!(packable_rows(&elements), BTreeSet::from([1]));
        coalesce(&mut elements, &BTreeSet::from([1]));
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].cols, 4);
        assert_eq!(elements[0].col, 0);
    }

    /// A row the two stages hold alike is packed on both sides.
    #[test]
    fn a_row_the_two_stages_share_is_packed() {
        let mesh = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        let pixel = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        assert_eq!(shared_rows(&mesh, &pixel), BTreeSet::from([1]));
    }

    /// A row the two stages do not hold alike is packed on neither.
    #[test]
    fn a_row_the_two_stages_do_not_share_is_left_alone() {
        let mesh = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        let pixel = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD2", 2, 1, 2, 2),
        ];
        assert!(shared_rows(&mesh, &pixel).is_empty());
    }

    /// The pixel shader's row of one element is not packed, so the mesh's
    /// row of two is not either: the two stages would not match.
    #[test]
    fn a_row_only_the_mesh_splits_is_left_alone() {
        let mesh = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        let pixel = vec![generic("TEXCOORD0", 0, 1, 0, 4)];
        assert!(shared_rows(&mesh, &pixel).is_empty());
    }

    /// A signature that already names an element `VREG` keeps its elements.
    #[test]
    fn a_signature_naming_vreg_is_left_alone() {
        let mesh = vec![
            generic("TEXCOORD0", 0, 1, 0, 2),
            generic("TEXCOORD1", 1, 1, 2, 2),
        ];
        let mut pixel = mesh.clone();
        pixel[0].name = "VREG".into();
        assert!(shared_rows(&mesh, &pixel).is_empty());
    }
}
