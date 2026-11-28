use super::*;
use victory_shader::emit::{Options, emit_with};
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use victory_shader::{Container, RootParam, io, ir, resource, structure};

const LOADS: [(u32, i32, u32, usize); 5] = [
    (0, -4, 125, 1),
    (1, 12, 2, 2),
    (2, 0, 125, 4),
    (3, 32, 125, 8),
    (4, 64, 125, 16),
];

fn pipeline() -> HostPipeline {
    let mut words = vec![0xBE85_0380, 0xBE84_0303]; // s[4:5] = output table at s3
    words.extend([0xF408_0602, 0xFA00_0000]); // s_load_dwordx4 s[24:27], s[4:5], 0
    let mut out = 0;
    for (op, immediate, soffset, dwords) in LOADS {
        words.extend([
            0xF400_0800 | op << 18, // s_load_dword[xN] s[32..], s[0:1], ...
            soffset << 25 | (immediate as u32 & 0x1F_FFFF),
            0xBF8C_0000, // s_waitcnt
        ]);
        for word in 0..dwords as u32 {
            words.extend([
                0x7E00_0200 | (32 + word), // v_mov_b32 v0, s[32 + word]
                0xE070_0000 | out,
                0x8006_0000, // buffer_store_dword v0, off, s[24:27], 0
            ]);
            out += 4;
        }
    }
    words.push(0xBF81_0000); // s_endpgm
    let param = |root, user_data, slot, sgprs| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            param(RootType::Descriptor, 0, 0, 2),
            param(RootType::Constants, 2, 1, 1),
            param(RootType::Table, 3, 2, 1),
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 3,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("scalar loads IR");
    let st = structure::structure(&p).expect("scalar loads control flow");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("scalar load uses");
    assert_eq!(res.scalar.iter().filter(|(_, a)| a.param == 0).count(), 5);
    let io = io::raise(&p, &native).expect("scalar loads IO");
    let host = emit_with(
        &Container { parts: Vec::new() },
        &native,
        &p,
        &st,
        &io,
        &res,
        Options::default(),
    )
    .expect("lower scalar loads from a root descriptor");
    assert!(host.memory);
    assert!(host.root_srvs.is_empty());
    let shader = crate::pipeline::Shader::for_test(
        "scalar root reads",
        crate::pipeline::Translated {
            host,
            used: native.used,
            used_written: Vec::new(),
            write_unknown: true,
            threads: None,
        },
    );
    HostPipeline::new(&[shader], None, |_, _| {}).expect("scalar loads compute pipeline")
}

#[test]
fn scalar_loads_read_root_memory_across_copies_and_after_rebinding() {
    let Some(host) = crate::host::host() else {
        assert!(
            std::env::var_os("VICTORY_REQUIRE_META_GPU").is_none(),
            "a D3D12 device is required"
        );
        eprintln!("no D3D12 device; skipped");
        return;
    };
    let pipeline = pipeline();
    let page = crate::watch::PAGE;
    let source = crate::watch::tests::pool(2);
    let output = crate::watch::tests::pool(1);
    let make_word = |word| 0xA514_0000u32 ^ word as u32;
    // SAFETY: one committed two-page source and one committed output page.
    unsafe {
        for word in 0..2 * page / 4 {
            (source as *mut u32).add(word).write(make_word(word));
        }
        (output as *mut u32).write(0xDEAD_BEEF);
    }
    let a = buffer_at_address(source as u64, page as u64);
    let b = buffer_at_address((source + page) as u64, page as u64);
    let dst = buffer_at_address(output as u64, page as u64);
    let shadow =
        |address: usize| crate::shadow::find(address as u64, page as u64).expect("page shadow");
    assert!(!std::sync::Arc::ptr_eq(
        &shadow(source),
        &shadow(source + page)
    ));
    let output_shadow = shadow(output);
    let (heap, title_heap, _) = view_heap(1);
    let null = std::ptr::null();
    let raw_uav = [39, 1, 0, 0, 32, 0, 0, 0, 1, 0];
    crate::descriptor::create_uav(null, dst, null, &raw_uav, title_heap.start);
    for (root_offset, dynamic) in [(page - 4, 8), (page + 256, 44)] {
        let mut state = State::default();
        let list = host.new_list().expect("open list");
        state.set_heaps(&[heap]);
        state.set_view(true, 0, (source + root_offset) as u64);
        state.set_constants(true, 1, &[dynamic], 0);
        state.set_table(true, 2, title_heap.start);
        assert!(state.bind(&list, &pipeline, true));
        // SAFETY: the compute pipeline and roots are bound on the open list.
        unsafe { list.list.Dispatch(1, 1, 1) };
        list.close();
        host.execute(&[&list]);
        let expected: Vec<_> = LOADS
            .iter()
            .flat_map(|&(_, immediate, soffset, dwords)| {
                let extra = if soffset == 2 { dynamic } else { 0 };
                let offset = root_offset.wrapping_add_signed(immediate as isize) + extra as usize;
                (offset / 4..offset / 4 + dwords).map(make_word)
            })
            .collect();
        assert_eq!(
            output_shadow.read_host(output as u64, expected.len()),
            expected,
            "root byte offset {root_offset:#x}, scalar offset {dynamic}"
        );
    }
    // SAFETY: plain device health query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    drop(output_shadow);
    for (object, address) in [(a, source), (b, source + page), (dst, output)] {
        crate::shadow::unregister(object as usize, address as u64);
        crate::object::release_ref(object);
    }
    crate::object::release_ref(heap);
    crate::watch::tests::free(source);
    crate::watch::tests::free(output);
    eprintln!("executed 2 scalar root-load GPU dispatches, checked 62 dwords");
}
