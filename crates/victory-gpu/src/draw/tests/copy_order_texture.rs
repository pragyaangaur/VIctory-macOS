use super::*;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ID3D12DescriptorHeap,
};

#[derive(Clone, Copy, Debug)]
enum Sequence {
    SameSegment,
    AfterAnEarlierRead,
    CrossList,
    SeparateSubmissions,
}

#[test]
fn title_copy_changes_texture_uploaded_by_the_same_segment() {
    copy_then_sample(Sequence::SameSegment, false);
}

#[test]
fn title_copy_changes_texture_sampled_again_in_the_same_title_list() {
    copy_then_sample(Sequence::AfterAnEarlierRead, false);
}

#[test]
fn title_copy_changes_aliased_texture_sampled_by_the_next_list() {
    copy_then_sample(Sequence::CrossList, true);
}

#[test]
fn title_copy_changes_texture_with_separate_submissions() {
    copy_then_sample(Sequence::SeparateSubmissions, false);
}

fn copy_then_sample(sequence: Sequence, aliased: bool) {
    let Some(host) = gpu_host() else { return };
    let _mode = ModeGuard::new(false, false);
    let page = crate::watch::PAGE;
    let (copy_address, texture_address) = if aliased {
        let (copy, texture) = crate::watch::tests::twin_views(1);
        (copy as u64, texture as u64)
    } else {
        let address = title_memory(page);
        (address, address)
    };
    let destination = buffer_at_address(copy_address, page as u64);
    let texture = texture_at_address(texture_address, &texture_words(4, 4, 42, 0x100, 0));
    let native = crate::resource::host_resource(texture).expect("title texture host resource");
    let source = crate::resource::test_buffer(256);
    write_words(destination, OLD);
    write_words(source, NEW);
    let (root, pso) = compile_cs(
        r#"
Texture2D<uint> tex : register(t0);
RWByteAddressBuffer result : register(u0);
cbuffer Params : register(b0) { uint offset; }
[numthreads(1, 1, 1)]
void main() {
    result.Store(offset, tex.Load(int3(0, 0, 0)));
}
"#,
        &[
            Param::Table(&[srv(0, 1)]),
            Param::Uav { register: 0 },
            Param::Constants {
                register: 0,
                count: 1,
            },
        ],
    );
    let (output, mapped) = host.shared_buffer(8).expect("pixels' readback");
    // SAFETY: both dwords are mapped, with no work submitted yet.
    unsafe { mapped.cast::<[u32; 2]>().write([0; 2]) };
    // SAFETY: a shader-visible heap and a live hardware R32_UINT texture.
    let heap: ID3D12DescriptorHeap = unsafe {
        let heap: ID3D12DescriptorHeap = host
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();
        host.device.CreateShaderResourceView(
            &native,
            None,
            heap.GetCPUDescriptorHandleForHeapStart(),
        );
        heap
    };
    let sample = |list, offset| {
        let list = crate::list::host_list(list).unwrap();
        crate::texture::upload(list, texture, &native);
        // SAFETY: an open list and live heap, pipeline, and output buffer.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.list
                .SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
            list.list.SetComputeRoot32BitConstant(2, offset, 0);
            list.list.Dispatch(1, 1, 1);
        }
    };

    let commands = Commands::new();
    let [first, second] = commands.lists;
    if matches!(sequence, Sequence::AfterAnEarlierRead) {
        sample(first, 0);
    }
    slot::<extern "system" fn(This, This, u64, This, u64, u64)>(first, 0x78)(
        first,
        destination,
        0,
        source,
        0,
        BYTES,
    );
    match sequence {
        Sequence::SameSegment | Sequence::AfterAnEarlierRead => {
            sample(first, 4);
            commands.submit(&[0]);
        }
        Sequence::CrossList => {
            sample(second, 4);
            commands.submit(&[0, 1]);
        }
        Sequence::SeparateSubmissions => {
            commands.submit(&[0]);
            sample(second, 4);
            commands.submit(&[1]);
        }
    }
    commands.drain();
    // SAFETY: the submitted work finished and wrote the output and copied bytes.
    let [before, after] = unsafe { mapped.cast::<[u32; 2]>().read() };
    // SAFETY: the copied destination remains live.
    let copied = unsafe { bytes(destination).cast::<[u32; WORDS]>().read() };
    for object in [source, destination, texture] {
        release_ref(object);
    }
    assert_eq!(copied, NEW);
    if matches!(sequence, Sequence::AfterAnEarlierRead) {
        assert_eq!(before, OLD[0], "{sequence:?} before the title copy");
    }
    assert_eq!(after, NEW[0], "{sequence:?} after the title copy");
}
