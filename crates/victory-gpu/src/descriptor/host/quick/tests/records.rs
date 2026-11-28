use super::*;
use crate::pipeline::{Shader, Translated, host::HostPipeline};
use victory_shader::emit::{Options, emit_with};
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use victory_shader::{
    Container, ResourceType, RootParam, RootType, UsedResource, io, ir, resource, structure,
};

fn metadata_pipeline() -> HostPipeline {
    let mut words = vec![
        0xBE84_0300,
        0xBE85_0380, // s[4:5] = input table at s0
        0xBE86_0302,
        0xBE87_0380, // s[6:7] = heap base at s2
        0xBE88_0301,
        0xBE89_0380, // s[8:9] = result table at s1
        0x8F0A_8503, // s_lshl_b32 s10, s3, 5: heap index to bytes
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0xF408_0403,
        0x1400_0000, // s_load_dwordx4 s[16:19], s[6:7], s10
        0xF400_0503,
        0x1400_0008, // s_load_dword s20, s[6:7], s10 + 8
        0xF408_0604,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[8:9], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    for (i, sgpr) in [12, 13, 14, 16, 17, 18, 20].into_iter().enumerate() {
        words.extend([
            0x7E00_0200 | sgpr, // v_mov_b32 v0, sgpr
            0xE070_0000 | (4 * i as u32),
            0x8006_0000, // buffer_store_dword v0, off, s[24:27], 0
        ]);
    }
    words.push(0xBF81_0000); // s_endpgm
    let param = |root, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: slot,
        slot,
    };
    let view = |kind, user_data, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            param(RootType::Table, 0),
            param(RootType::Table, 1),
            param(RootType::ResourceHeap, 2),
            param(RootType::Constants, 3),
        ],
        used: vec![
            view(ResourceType::Srv, 0, 8, 16),
            view(ResourceType::Uav, 1, 0, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("metadata IR");
    let st = structure::structure(&p).expect("metadata control flow");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("metadata uses");
    let io = io::raise(&p, &native).expect("metadata IO");
    let host = emit_with(
        &Container { parts: Vec::new() },
        &native,
        &p,
        &st,
        &io,
        &res,
        Options::default(),
    )
    .expect("lower descriptor addresses and record counts");
    assert!(host.view_addresses);
    let shader = Shader::for_test(
        "descriptor record counts",
        Translated {
            host,
            used: native.used,
            used_written: Vec::new(),
            write_unknown: true,
            threads: None,
        },
    );
    HostPipeline::new(&[shader], None, |_, _| {}).expect("metadata compute pipeline")
}

#[test]
fn gpu_reads_record_counts_from_tables_and_heaps_after_both_sync_modes() {
    let Some(host) = host() else {
        assert!(
            std::env::var_os("VICTORY_REQUIRE_META_GPU").is_none(),
            "a D3D12 device is required"
        );
        eprintln!("no D3D12 device; skipped");
        return;
    };
    let pipeline = metadata_pipeline();
    let stage = &pipeline.stages[0];
    let (user_data, dwords) = stage.user_data.expect("user data binding");
    let metadata = stage.view_addresses.expect("view addresses binding");
    let samplers = crate::descriptor::fallback_samplers().expect("sampler heap");
    let source = crate::resource::test_buffer(4096);
    let result = crate::resource::test_buffer(256);
    let (src, len) = crate::resource::buffer_bytes(source).expect("input buffer");
    let (dst, dst_len) = crate::resource::buffer_bytes(result).expect("output buffer");
    let address = src as u64;
    crate::shadow::register(source as usize, address, len);
    crate::shadow::register(result as usize, dst as u64, dst_len);
    let out = crate::shadow::find(dst as u64, dst_len).expect("output buffer shadow");

    let null = std::ptr::null();
    const SLOTS: u32 = 8;
    let mut title = vec![0u8; SLOTS as usize * 32];
    let start = title.as_mut_ptr() as u64;
    let at = |i: u32| start + u64::from(i) * 32;
    let raw = |first, count| [39, 1, 0x1688, 0, first, 0, count, 0, 1, 0];
    let structured = |first, count| [0, 1, 0x1688, 0, first, 0, count, 16, 0, 0];
    let typed = [42, 1, 0x1688, 0, 6, 0, 257, 0, 0, 0];
    let output = [39, 1, 0, 0, 16, 0, 0, 0, 1, 0];
    let mut placed = [0u32; 16];
    placed[..10].copy_from_slice(&raw(4, 1));
    let placed_base = address + 32;
    (placed[10], placed[11]) = (placed_base as u32, (placed_base >> 32) as u32);
    let mut placed_buffer = [0u32; 14];
    (placed_buffer[0], placed_buffer[4]) = (1, 256);
    let fields = |address: u64, count| [address as u32, (address >> 32) as u32 & 0xFFFF, count];

    for quick in [false, true] {
        let heap = HostHeap::new(false, SLOTS).expect("view heap");
        let view_addresses = heap.view_addresses().expect("metadata buffer");
        crate::descriptor::create_srv(null, source, &raw(4, 13), at(0));
        crate::descriptor::create_srv(null, source, &structured(4, 9), at(2));
        crate::descriptor::create_srv(null, source, &typed, at(3));
        crate::descriptor::create_placed_srv(null, &placed_buffer, &placed, at(4));
        crate::descriptor::create_uav(null, source, null, &[39, 1, 4, 0, 3, 0, 0, 0, 1, 0], at(5));
        crate::descriptor::create_uav(null, result, null, &output, at(7));
        let dispatch = |index: u32, table: [u32; 3], view: [u32; 3]| {
            let args = [32, 7 * 32, 0, index]; // input table starts at slot 1
            assert_eq!(dwords, args.len() as u32);
            let list = host.new_list().expect("open list");
            // SAFETY: open list with live descriptor heaps, root signature and PSO.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(heap.heap.clone()), Some(samplers.heap.clone())]);
                list.list.SetComputeRootSignature(&pipeline.root);
                list.list.SetPipelineState(&pipeline.pso);
                list.list
                    .SetComputeRoot32BitConstants(user_data, dwords, args.as_ptr().cast(), 0);
                list.list
                    .SetComputeRootShaderResourceView(metadata, view_addresses);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            let [lo, hi, count] = view;
            let expected = [table[0], table[1], table[2], lo, hi, count, count];
            assert_eq!(
                out.read_host(dst as u64, expected.len()),
                expected,
                "quick={quick}, heap slot={index}"
            );
        };

        heap.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title), true, quick);
        let table = fields(address + 64, 9);
        for (slot, view) in [
            (0, fields(address + 16, 52)),  // raw: bytes
            (2, table),                     // structured: elements
            (3, fields(address + 24, 257)), // typed: elements
            (4, fields(address + 48, 4)),   // placed raw
            (5, fields(address + 16, 12)),  // raw UAV
            (6, [0; 3]),                    // null
        ] {
            dispatch(slot, table, view);
        }

        // Replace an SRV with the same base address and only a different count.
        crate::descriptor::create_srv(null, source, &raw(4, 1), at(0));
        title[2 * 32..3 * 32].fill(0);
        title[4 * 32..5 * 32].fill(0);
        heap.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title), true, quick);
        dispatch(0, [0; 3], fields(address + 16, 4));
        dispatch(4, [0; 3], [0; 3]);
    }
    // SAFETY: plain device health query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    crate::object::release_ref(result);
    crate::object::release_ref(source);
    eprintln!("executed 16 descriptor metadata GPU dispatches, checked 112 dwords");
}
