use super::*;

const CASES: [(&str, &[u32], bool); 3] = [
    ("upper", &MASK_LANES_40_41, true),
    ("inner_empty", &MASK_LANES_8_9, false),
    ("outer_empty", &MASK_EMPTY, false),
];

fn saved_exec_program(mask: &[u32]) -> Vec<u32> {
    let mut w = program(mask, true);
    let inner = w.iter().position(|&word| word == 0xbf88_0005).unwrap();
    let prefix = [
        0x7d84_00a8,         // select lane 40 within the existing outer EXEC
        0xbea6_246a,         // s[38:39] = outer EXEC, EXEC &= VCC
        0xbea0_047e,         // keep the narrower inner mask
        0xbefe_0426,         // restore the outer mask for the first branch
        s_cbranch_execz(18), // skip to the original full EXEC restore
        0xbefe_0420,         // enter with the inner mask
        0xbea2_04c1,         // s[34:35] = -1
        0xbf06_8001,         // group 0 takes the next skip
        0xbf85_0001,
        0xbea2_0480, // group 1 sets s[34:35] to zero
        0xbea0_2422, // save the inner mask, then apply the joined operand
        0xbea4_0420, // s[36:37] = saved inner mask
        0xbf06_8001,
        0xbf85_0001, // copies in the previous block dominate the restore
        0xbf80_0000,
        0xbefe_0424, // restore the old EXEC before the inner visit
    ];
    w.splice(inner..inner, prefix);
    w.insert(inner + prefix.len() + 6, 0xbea6_0380); // distinct inner join
    w
}

#[test]
fn saved_pair_restores_upper_lanes_across_a_join() {
    let w = saved_exec_program(&MASK_LANES_40_41);
    assert!(visit_admitted(&w, &[0, 5]));
}

#[test]
fn restored_exec_publishes_both_scalar_components() {
    for (case, mask, _) in CASES {
        let n = native_for_groups(&saved_exec_program(mask), true, 64, 0, 2);
        let split = lowered_with(&n, true, true).expect("restored EXEC publication");
        assert_eq!(live_sources(&split), [(None, 0), (None, 1)]);
        eprintln!(
            "{case} cap={} DXIL MD5={:02x?}",
            super::super::super::loop_limit(),
            super::super::super::container::md5(&split.dxil)
        );
    }
}

#[cfg(windows)]
#[test]
fn restored_exec_publication_matches_gpu() {
    if !in_bounded_gpu_child(
        "emit::halves_tests::normal_publication::saved_exec::restored_exec_publication_matches_gpu",
    ) {
        return;
    }
    for (case, mask, visit) in CASES {
        check_publication_gpu(&saved_exec_program(mask), [visit; 2], case);
    }
}

#[test]
fn narrowed_exec_and_partial_saved_pairs_cannot_authorize_a_visit() {
    let w = saved_exec_program(&MASK_LANES_40_41);
    let at = w.iter().position(|&word| word == 0xbefe_0424).unwrap();
    for replacement in [
        0xbf80_0000, // keep the narrowed EXEC produced by saveexec
        0xbefe_0422, // use the joined operand
    ] {
        let mut bad = w.clone();
        bad[at] = replacement;
        assert!(!visit_admitted(&bad, &[0, 5]));
    }
    for overwrite in [0xbea4_0322, 0xbea5_0323] {
        let mut bad = w.clone();
        bad[at - 1] = overwrite; // replace one dword on a single predecessor
        assert!(!visit_admitted(&bad, &[0, 5]));
    }
}
