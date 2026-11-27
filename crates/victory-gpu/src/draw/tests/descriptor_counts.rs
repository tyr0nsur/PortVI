use super::compile_cs;
use crate::{descriptor, object, resource};
use std::ptr::null;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN,
};
use windows::core::{IUnknown, Interface};

/// The API's buffer union starts at a different word in SRVs and UAVs.
/// Read NUM_RECORDS (dword 2 of each 16-byte table entry) as a shader does.
#[test]
fn shader_reads_buffer_record_counts_after_a_slot_rewrite() {
    let Some(host) = crate::host::host() else {
        eprintln!("no D3D12 device; skipped buffer record-count readback");
        return;
    };
    const BYTES: u32 = 1024;
    let input = resource::test_buffer(u64::from(BYTES));
    // SAFETY: test_buffer returned one owned reference.
    let _input = unsafe { IUnknown::from_raw(input.cast_mut().cast()) };
    let base = resource::buffer_bytes(input).unwrap().0 as u64;
    let heap_desc = [0u32, 4, 1, 0, 0, 0, 0, 0];
    let mut heap = null();
    assert_eq!(
        descriptor::create_heap(
            null(),
            heap_desc.as_ptr().cast(),
            std::ptr::from_ref(&ID3D12DescriptorHeap::IID).cast(),
            &mut heap
        ),
        object::S_OK
    );
    // SAFETY: create_heap returned one owned reference.
    let _heap = unsafe { IUnknown::from_raw(heap.cast_mut().cast()) };
    let heap = descriptor::heap(heap);
    let slot = |i: u32| heap.start + u64::from(i * descriptor::increment_size(null(), 0));
    let srv = |format: DXGI_FORMAT, first, count, stride| {
        [
            format.0 as u32,
            D3D12_SRV_DIMENSION_BUFFER.0 as u32,
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            0,
            first,
            0,
            count,
            stride,
            0,
            0,
        ]
    };
    let uav = |first, count, stride| {
        [
            DXGI_FORMAT_UNKNOWN.0 as u32,
            D3D12_UAV_DIMENSION_BUFFER.0 as u32,
            first,
            0,
            count,
            stride,
            0,
            0,
            0,
            0,
        ]
    };
    let placed = |view: [u32; 10]| {
        let mut words = [0; 16];
        words[..10].copy_from_slice(&view);
        words[10] = base as u32;
        words[11] = (base >> 32) as u32;
        words
    };
    let buffer_desc = [1u32, 0, 0, 0, BYTES, 0, 1, 1 | (1 << 16), 0, 1, 0, 1, 0, 0];
    let write_placed_srv = |count| {
        let view = placed(srv(DXGI_FORMAT_UNKNOWN, 5, count, 12));
        descriptor::create_placed_srv(null(), &buffer_desc, &view, slot(0));
    };
    write_placed_srv(37);
    descriptor::create_placed_uav(null(), &buffer_desc, &placed(uav(5, 37, 12)), slot(1));
    descriptor::create_srv(null(), input, &srv(DXGI_FORMAT_R32_UINT, 3, 19, 0), slot(2));
    descriptor::create_uav(null(), input, null(), &uav(7, 11, 12), slot(3));

    let addresses = heap.host().unwrap().view_addresses().unwrap();
    let (output, mapped) = host.shared_buffer(16).unwrap();
    let (root, pso) = compile_cs(
        r#"
        ByteAddressBuffer views : register(t0);
        RWByteAddressBuffer counts : register(u0);
        [RootSignature("SRV(t0), UAV(u0)")]
        [numthreads(4, 1, 1)]
        void main(uint lane : SV_GroupIndex) {
            counts.Store(4 * lane, views.Load(16 * lane + 8));
        }
    "#,
    );
    let read = || {
        let list = host.new_list().unwrap();
        // SAFETY: both root descriptors refer to live buffers. All four shader
        // lanes fit in the table and output. execute waits for completion.
        unsafe {
            mapped.cast::<[u32; 4]>().write_unaligned([u32::MAX; 4]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootShaderResourceView(0, addresses);
            list.list
                .SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the submission finished writing the four mapped words.
        unsafe { mapped.cast::<[u32; 4]>().read_unaligned() }
    };
    let initial = read();
    write_placed_srv(13);
    // D3D12 raw views count four-byte elements. Their zero-stride RDNA2
    // descriptor counts bytes, unlike the typed and structured views above.
    let mut raw_srv = srv(DXGI_FORMAT_R32_TYPELESS, 4, 5, 0);
    raw_srv[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
    descriptor::create_srv(null(), input, &raw_srv, slot(2));
    let mut raw_uav = uav(8, 7, 0);
    raw_uav[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
    raw_uav[8] = D3D12_BUFFER_UAV_FLAG_RAW.0 as u32;
    descriptor::create_uav(null(), input, null(), &raw_uav, slot(3));
    let rewritten = read();
    eprintln!("shader-visible record counts before and after rewrite: {initial:?} {rewritten:?}");
    assert_eq!([initial, rewritten], [[37, 37, 19, 11], [13, 37, 20, 28]]);
}
