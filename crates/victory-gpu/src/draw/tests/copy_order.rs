use super::*;
use crate::host::{Host, ModeGuard};
use crate::object::{guid, release_ref, slot};
use std::ptr::null;

const WORDS: usize = 4;
const BYTES: u64 = (WORDS * size_of::<u32>()) as u64;
const OLD: [u32; WORDS] = [0x1122_3344, 0x4536_2718, 0x7e80_0001, 0x4432_8791];
const NEW: [u32; WORDS] = [0x8765_4321, 0x1a2b_3c4d, 0x8100_e0e1, 0x7788_99aa];
const GPU: [u32; WORDS] = [0xabc0_0000, 0xabc0_0011, 0xabc0_0022, 0xabc0_0033];

const READ_BUFFER: &str = r#"
ByteAddressBuffer source : register(t0);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) { uint offset; }
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output.Store(offset + 4 * id.x, source.Load(4 * id.x));
}
"#;

const WRITE_BUFFER: &str = r#"
RWByteAddressBuffer destination : register(u0);
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    destination.Store(4 * id.x, 0xabc00000u + 17u * id.x);
}
"#;

struct Commands {
    allocator: This,
    lists: [This; 2],
    queue: This,
}

impl Commands {
    fn new() -> Self {
        let queue = crate::queue::create_direct_for_test();
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let mut allocator = null();
        assert_eq!(
            crate::list::create_allocator(null(), 0, &allocator_iid, &mut allocator),
            S_OK
        );
        let lists = [(); 2].map(|()| {
            let mut list = null();
            assert_eq!(
                crate::list::create_list(null(), 0, 0, allocator, null(), &list_iid, &mut list),
                S_OK
            );
            list
        });
        Self {
            allocator,
            lists,
            queue,
        }
    }

    fn submit(&self, indices: &[usize]) {
        let lists: Vec<_> = indices.iter().map(|&index| self.lists[index]).collect();
        for &list in &lists {
            assert_eq!(
                slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
                S_OK
            );
        }
        crate::queue::execute_for_test(self.queue, &lists);
    }

    fn drain(&self) {
        crate::queue::drain_submit(crate::queue::Why::Present);
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        self.drain();
        for list in self.lists {
            release_ref(list);
        }
        release_ref(self.allocator);
        release_ref(self.queue);
    }
}

fn gpu_host() -> Option<&'static Host> {
    let host = crate::host::host();
    if host.is_none() {
        assert!(
            std::env::var_os("VICTORY_REQUIRE_GPU_TESTS").is_none(),
            "D3D12 device required for command-order readback"
        );
        eprintln!("skipped: no D3D12 device for command-order readback");
    }
    host
}

fn bytes(resource: This) -> *mut u32 {
    crate::resource::buffer_bytes(resource).unwrap().0.cast()
}

fn write_words(resource: This, words: [u32; WORDS]) {
    let address = bytes(resource);
    let _write = crate::watch::writing(address as usize, BYTES as usize);
    // SAFETY: a live fixture buffer holds at least WORDS dwords.
    unsafe { address.cast::<[u32; WORDS]>().write(words) };
}

fn record_writer(
    list: This,
    address: u64,
    shader: &(ID3D12RootSignature, ID3D12PipelineState),
    read_precedes: bool,
) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    };

    let list = crate::list::host_list(list).unwrap();
    let gpu = root_view(list, address, false, true).expect("destination root UAV");
    let shadow = read_precedes.then(|| crate::shadow::find(address, BYTES).unwrap());
    let transition = |buffer, before, after| {
        let barrier = crate::host::transition(buffer, before, after);
        // SAFETY: an open list and a live view of the buffer.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    };
    if let Some(shadow) = &shadow {
        transition(
            &shadow.buffer,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
    }
    // SAFETY: open list, live pipeline and root buffer.
    unsafe {
        list.list.SetComputeRootSignature(&shader.0);
        list.list.SetPipelineState(&shader.1);
        list.list.SetComputeRootUnorderedAccessView(0, gpu);
        list.list.Dispatch(1, 1, 1);
    }
    if let Some(shadow) = &shadow {
        transition(
            &shadow.buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Writer {
    None,
    Source,
    Destination,
}

#[derive(Clone, Copy, Debug)]
enum CopyCall {
    Region,
    Resource,
    Memory,
}

fn two_buffer_views() -> (This, This) {
    let (a, b) = crate::watch::tests::twin_views(1);
    let page = crate::watch::PAGE as u64;
    (
        buffer_at_address(a as u64, page),
        buffer_at_address(b as u64, page),
    )
}

fn copy_sequence(
    host: &Host,
    same_list: bool,
    aliased: bool,
    writer: Writer,
    call: CopyCall,
) -> [[u32; WORDS]; 2] {
    let commands = Commands::new();
    let (read_alias, destination) = if aliased {
        two_buffer_views()
    } else {
        let buffer = crate::resource::test_buffer(256);
        (buffer, buffer)
    };
    let (write_source, source) = if aliased && matches!(writer, Writer::Source) {
        two_buffer_views()
    } else {
        let buffer = crate::resource::test_buffer(256);
        (buffer, buffer)
    };
    write_words(read_alias, OLD);
    write_words(write_source, NEW);

    let shader = compile_cs(
        READ_BUFFER,
        &[
            Param::Srv {
                register: 0,
                space: 0,
            },
            Param::Uav { register: 0 },
            Param::Constants {
                register: 0,
                count: 1,
            },
        ],
    );
    let writer_shader = compile_cs(WRITE_BUFFER, &[Param::Uav { register: 0 }]);
    let (output, mapped) = host
        .shared_buffer(2 * BYTES)
        .expect("shader readback buffer");
    // SAFETY: the resource stays live through both submissions.
    let output_address = unsafe { output.GetGPUVirtualAddress() };
    let read = |list: This, offset: u32| {
        let list = crate::list::host_list(list).unwrap();
        let gpu = root_view(list, bytes(read_alias) as u64, true, false).expect("source root SRV");
        // SAFETY: open list and live root buffers, with disjoint output ranges.
        unsafe {
            list.list.SetComputeRootSignature(&shader.0);
            list.list.SetPipelineState(&shader.1);
            list.list.SetComputeRootShaderResourceView(0, gpu);
            list.list
                .SetComputeRootUnorderedAccessView(1, output_address);
            list.list.SetComputeRoot32BitConstant(2, offset, 0);
            list.list.Dispatch(1, 1, 1);
        }
    };

    let first = commands.lists[0];
    let second = commands.lists[usize::from(!same_list)];
    read(first, 0);
    match writer {
        Writer::None => (),
        Writer::Source => record_writer(first, bytes(write_source) as u64, &writer_shader, false),
        Writer::Destination => record_writer(first, bytes(read_alias) as u64, &writer_shader, true),
    }
    match call {
        CopyCall::Region => {
            slot::<extern "system" fn(This, This, u64, This, u64, u64)>(second, 0x78)(
                second,
                destination,
                0,
                source,
                0,
                BYTES,
            );
        }
        CopyCall::Resource => {
            slot::<extern "system" fn(This, This, This)>(second, 0x88)(second, destination, source);
        }
        CopyCall::Memory => {
            let offset = if crate::list::has_newer_table(second) {
                0x1f0
            } else {
                0x1e0
            };
            slot::<extern "system" fn(This, u64, u64, u64, u32)>(second, offset)(
                second,
                bytes(destination) as u64,
                bytes(source) as u64,
                BYTES,
                0,
            );
        }
    }
    read(second, BYTES as u32);
    commands.submit(if same_list { &[0] } else { &[0, 1] });
    commands.drain();

    // SAFETY: the queue has finished and the readback holds both dispatch outputs.
    let result = unsafe { mapped.cast::<[[u32; WORDS]; 2]>().read() };
    release_ref(source);
    if write_source != source {
        release_ref(write_source);
    }
    release_ref(read_alias);
    if aliased {
        release_ref(destination);
    }
    result
}

fn check_buffer_order(same_list: bool, aliased: bool, writer: Writer, call: CopyCall) {
    let Some(host) = gpu_host() else { return };
    let expected = match writer {
        Writer::Source => [OLD, GPU],
        Writer::None | Writer::Destination => [OLD, NEW],
    };
    for (flight, worker) in [(false, false), (true, false), (false, true), (true, true)] {
        let _mode = ModeGuard::new(flight, worker);
        let result = copy_sequence(host, same_list, aliased, writer, call);
        assert_eq!(
            result, expected,
            "same list {same_list}, alias {aliased}, writer {writer:?}, call {call:?}, Flight-1 {flight}, worker {worker}"
        );
    }
}

#[test]
fn copy_order_same_list() {
    check_buffer_order(true, false, Writer::None, CopyCall::Region);
}

#[test]
fn copy_order_cross_list() {
    check_buffer_order(false, false, Writer::None, CopyCall::Region);
}

#[test]
fn copy_order_alias_same_list() {
    check_buffer_order(true, true, Writer::None, CopyCall::Region);
}

#[test]
fn copy_order_alias_cross_list() {
    check_buffer_order(false, true, Writer::None, CopyCall::Region);
}

#[test]
fn copy_order_gpu_source_same_list() {
    check_buffer_order(true, false, Writer::Source, CopyCall::Region);
}

#[test]
fn copy_order_gpu_source_cross_list() {
    check_buffer_order(false, false, Writer::Source, CopyCall::Region);
}

#[test]
fn copy_order_gpu_destination_same_list() {
    check_buffer_order(true, false, Writer::Destination, CopyCall::Region);
}

#[test]
fn copy_order_gpu_destination_cross_list() {
    check_buffer_order(false, false, Writer::Destination, CopyCall::Region);
}

#[test]
fn copy_order_gpu_source_alias() {
    for same_list in [true, false] {
        check_buffer_order(same_list, true, Writer::Source, CopyCall::Region);
    }
}

#[test]
fn copy_order_gpu_destination_alias() {
    for same_list in [true, false] {
        check_buffer_order(same_list, true, Writer::Destination, CopyCall::Region);
    }
}

#[test]
fn copy_order_copy_resource() {
    for same_list in [true, false] {
        check_buffer_order(same_list, false, Writer::None, CopyCall::Resource);
    }
}

#[test]
fn copy_order_copy_memory() {
    for same_list in [true, false] {
        check_buffer_order(same_list, true, Writer::None, CopyCall::Memory);
    }
}

#[test]
fn copy_order_batches_consecutive_copy_only_commands() {
    let Some(_host) = gpu_host() else { return };
    let _mode = ModeGuard::new(false, false);
    let commands = Commands::new();
    let list = commands.lists[0];
    let buffers = [(); 3].map(|()| crate::resource::test_buffer(256));
    write_words(buffers[0], NEW);
    for (source, destination) in [(buffers[0], buffers[1]), (buffers[1], buffers[2])] {
        slot::<extern "system" fn(This, This, u64, This, u64, u64)>(list, 0x78)(
            list,
            destination,
            0,
            source,
            0,
            BYTES,
        );
    }
    assert_eq!(crate::list::segments(list).len(), 1);
    commands.submit(&[0]);
    // SAFETY: one live title buffer, after queue completion.
    let result = unsafe { bytes(buffers[2]).cast::<[u32; WORDS]>().read() };
    for buffer in buffers {
        release_ref(buffer);
    }
    assert_eq!(result, NEW);
}

#[test]
fn copy_order_lands_an_earlier_flight_before_copying_an_aliased_gpu_source() {
    let Some(host) = gpu_host() else { return };
    let _mode = ModeGuard::with_prep_ranges();
    for same_list in [true, false] {
        let result = copy_sequence(host, same_list, true, Writer::Source, CopyCall::Region);
        assert_eq!(result, [OLD, GPU], "same list {same_list}");
    }
}
