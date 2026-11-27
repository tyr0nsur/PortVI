//! GPU half of the source-authored DirectStorage fixture, enabled only for tests.
//!
//! The renderer schedules each changed subresource through [`crate::texture::upload`].
//! The submission's upload list detiles its source and copies it before the draw.

use super::{
    HeapProperties, RESOURCE_IIDS, ResourceDesc, create_committed, host_resource, resource_release,
    texture_layout, texture_memory,
};
use crate::host::{self, Host};
use crate::object::{S_OK, This};
use std::ptr::null;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC4_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{Interface, s};

const TEXELS: usize = 8 * 4;
const OUTPUT_BYTES: u64 = (TEXELS * size_of::<f32>()) as u64;
const PROBE_SHADER: &str = r#"
Texture2D<float> image : register(t0);
SamplerState pointSampler : register(s0);
RWByteAddressBuffer outputTexels : register(u0);
[RootSignature("DescriptorTable(SRV(t0)), UAV(u0), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)")]
[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float2 uv = (float2(tid.xy) + 0.5) / float2(8, 4);
    outputTexels.Store(4 * (tid.x + 8 * tid.y), asuint(image.SampleLevel(pointSampler, uv, 0)));
}
"#;

struct TitleTexture(This);

impl Drop for TitleTexture {
    fn drop(&mut self) {
        resource_release(self.0);
    }
}

/// The probe shader's bytecode, its root signature inside.
fn compile() -> Vec<u8> {
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names outlive the call; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            PROBE_SHADER.as_ptr().cast(),
            PROBE_SHADER.len(),
            s!("bc4_probe"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let text = |b: &ID3DBlob| {
        // SAFETY: the blob's own bytes.
        let bytes = unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) };
        bytes.to_vec()
    };
    if let Err(e) = compiled {
        let log = errors.as_ref().map(text).unwrap_or_default();
        panic!("compile BC4 sampler: {e}: {}", String::from_utf8_lossy(&log));
    }
    text(&code.expect("BC4 sampler bytecode"))
}

/// A committed buffer of `size` bytes in `kind`'s heap, in `state`.
fn buffer(
    host: &Host,
    size: u64,
    kind: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain creation with valid descriptions.
    unsafe {
        host.device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made)
            .expect("BC4 probe buffer");
    }
    made.expect("BC4 probe buffer")
}

struct Sampler {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    output: ID3D12Resource,
    readback: ID3D12Resource,
}

impl Sampler {
    fn new(host: &Host, image: &ID3D12Resource) -> Self {
        let code = compile();
        // SAFETY: the bytecode contains the root signature. All borrowed COM
        // references and the shader bytes outlive pipeline and view creation.
        let (root, pipeline, views) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pipeline = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            host.device.CreateShaderResourceView(
                image,
                None,
                views.GetCPUDescriptorHandleForHeapStart(),
            );
            (root, pipeline, views)
        };
        Self {
            root,
            pipeline,
            views,
            output: buffer(
                host,
                OUTPUT_BYTES,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ),
            readback: buffer(
                host,
                OUTPUT_BYTES,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
        }
    }

    fn record(&self, list: &host::HostList) {
        // SAFETY: the live descriptor, pipeline and buffers match the shader.
        // All 32 lanes write a distinct float before the readback copy.
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[host::transition(
                &self.output,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            l.SetDescriptorHeaps(&[Some(self.views.clone())]);
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&self.pipeline);
            l.SetComputeRootDescriptorTable(0, self.views.GetGPUDescriptorHandleForHeapStart());
            l.SetComputeRootUnorderedAccessView(1, self.output.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[host::transition(
                &self.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&self.readback, 0, &self.output, 0, OUTPUT_BYTES);
            l.ResourceBarrier(&[host::transition(
                &self.output,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    fn submit(
        &self,
        host: &Host,
        texture: &TitleTexture,
        expected_uploads: u64,
        split: bool,
        record: impl FnOnce(&host::HostList, &host::HostList),
    ) -> [f32; TEXELS] {
        let first = host.new_list().expect("BC4 command list");
        let second = split.then(|| host.new_list().expect("second BC4 command list"));
        let before = crate::texture::uploads(texture.0);
        record(&first, second.as_ref().unwrap_or(&first));
        first.close();
        if let Some(second) = second {
            second.close();
            host.execute(&[&first, &second]);
        } else {
            host.execute(&[&first]);
        }
        // SAFETY: execute waits for this list. The readback buffer holds 32
        // aligned floats, and the copy completes before its mapped span is read.
        unsafe {
            host.device
                .GetDeviceRemovedReason()
                .expect("device survived BC4 dispatch");
            let mut ptr = std::ptr::null_mut();
            self.readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: OUTPUT_BYTES as usize,
                    }),
                    Some(&mut ptr),
                )
                .unwrap();
            let texels = ptr.cast::<[f32; TEXELS]>().read();
            self.readback
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            assert_eq!(
                crate::texture::uploads(texture.0) - before, expected_uploads,
                "BC4 upload count, first sampled texel = {:.1}/255", texels[0] * 255.0
            );
            texels
        }
    }
}

/// Exercise the production texture upload on the host device. The storage
/// callback receives the two adjacent blocks in the console texture itself.
/// The device is whichever `host::host` made (the default adapter, or WARP
/// where a diagnostics build asks for it); `require_hardware` rejects a
/// software adapter.
pub fn run(fill: impl FnOnce(&mut [u8; 16]), require_hardware: bool) {
    let host = host::host().expect("the BC4 probe needs a D3D12 device");
    // SAFETY: query the adapter belonging to the actual rendering device.
    let (info, driver) = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let driver = adapter.CheckInterfaceSupport(&IDXGIDevice::IID).ok();
        (adapter.GetDesc1().unwrap(), driver)
    };
    let software = (info.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
        || (info.VendorId, info.DeviceId) == (0x1414, 0x8c);
    assert!(
        !require_hardware || !software,
        "the physical BC4 probe selected a software adapter"
    );
    let end = info
        .Description
        .iter()
        .position(|&v| v == 0)
        .unwrap_or(info.Description.len());
    let driver = driver.map_or_else(
        || "unknown".to_owned(),
        |v| {
            let v = v as u64;
            format!("{}.{}.{}.{}", v >> 48, (v >> 32) & 0xffff, (v >> 16) & 0xffff, v & 0xffff)
        },
    );
    eprintln!(
        "BC4 adapter: {} (PCI\\VEN_{:04X}&DEV_{:04X}&SUBSYS_{:08X}&REV_{:02X}, driver {driver}, software={software})",
        String::from_utf16_lossy(&info.Description[..end]),
        info.VendorId,
        info.DeviceId,
        info.SubSysId,
        info.Revision,
    );

    let desc = ResourceDesc {
        dimension: 3,
        alignment: 0,
        width: 8,
        height: 4,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: DXGI_FORMAT_BC4_UNORM.0 as u32,
        sample_count: 1,
        sample_quality: 0,
        layout: 0x105,
        flags: 0,
    };
    let heap = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let mut raw = null();
    assert_eq!(
        create_committed(
            null(),
            &heap,
            0,
            &desc,
            0,
            null(),
            &RESOURCE_IIDS[0],
            &mut raw
        ),
        S_OK
    );
    let texture = TitleTexture(raw);
    let (memory, _) = texture_memory(raw).unwrap();
    let layout = &texture_layout(&desc).unwrap().planes[0].layout;
    let first = layout.offset(0, 0, 0, 0).unwrap();
    assert_eq!(layout.offset(0, 1, 0, 0).unwrap(), first + 8);
    assert!(first + 16 <= layout.size);
    // SAFETY: the layout places two consecutive BC4 blocks in this owned
    // texture. A mutable borrow spans only the writes, never a GPU submission.
    let address = unsafe { memory.cast_mut().add(first as usize).cast::<[u8; 16]>() };
    let image = host_resource(raw).expect("BC4 host texture");
    let sampler = Sampler::new(host, &image);
    let request = |list: &host::HostList| crate::texture::upload(list, raw, &image);
    let shader_read = |list: &host::HostList| {
        crate::list::ensure_state(list, raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32);
    };
    let sample = |upload: bool| {
        sampler.submit(host, &texture, u64::from(upload), false, |_, list| {
            if upload {
                request(list);
            }
            shader_read(list);
            sampler.record(list);
        })
    };
    let sentinel = [51u8, 51, 0, 0, 0, 0, 0, 0];
    {
        // SAFETY: the two blocks lie in this texture and no work has been
        // submitted; its staging (and any guard) does not exist yet.
        let blocks = unsafe { &mut *address };
        blocks[..8].copy_from_slice(&sentinel);
        blocks[8..].copy_from_slice(&sentinel);
    }
    let original = sample(true);
    assert!(
        original.iter().all(|&v| (v * 255.0 - 51.0).abs() < 0.01),
        "sentinel upload: {original:?}"
    );
    // SAFETY: the previous list has completed and the callback waits for its
    // storage request, which opens guarded pages first (`watch::writing`).
    fill(unsafe { &mut *address });
    let stale = sample(false);
    let updated = sample(true);
    assert_eq!(
        stale, original,
        "the skipped upload must preserve the previous GPU contents"
    );

    // The source encoder selects successive sevenths between (138,125)
    // and (139,126). This oracle reads neither the compressed bytes nor the GPU.
    let matches = |samples: &[f32; TEXELS]| {
        samples
            .iter()
            .enumerate()
            .filter(|(i, v)| {
                let (x, y) = (i % 8, i / 8);
                let block = x / 4;
                let step = (4 * y + x % 4 + block) % 8;
                let expected = 138.0 + block as f32 - 13.0 * step as f32 / 7.0;
                let tolerance = if step == 0 || step == 7 { 0.01 } else { 0.75 };
                (**v * 255.0 - expected).abs() <= tolerance
            })
            .count()
    };
    assert_eq!(matches(&stale), 0, "the reference must reject stale texels");
    assert_eq!(
        matches(&updated),
        TEXELS,
        "unexpected GPU readback: {updated:?}"
    );
    eprintln!("BC4 readback: 32/32 expected texels; skipped upload: 32/32 stale texels detected");

    // A leading memory copy writes the sampled blocks. A later one overwrites
    // them after the draw. Each readback must see exactly its own generation.
    let solid = |v| [v, v, 0, 0, 0, 0, 0, 0, v, v, 0, 0, 0, 0, 0, 0];
    let (first, last, overwrite) = (solid(107), solid(213), solid(173));
    for split in [false, true] {
        let current = sampler.submit(host, &texture, 1, split, |a, b| {
            a.defer_copy(address.cast(), first.as_ptr(), first.len());
            request(b);
            shader_read(b);
            sampler.record(b);
            b.defer_copy(address.cast(), last.as_ptr(), last.len());
        });
        assert!(
            current.iter().all(|&v| (v * 255.0 - 107.0).abs() < 0.01),
            "the first draw must see the preceding copy: {current:?}"
        );
        let next = sample(true);
        assert!(
            next.iter().all(|&v| (v * 255.0 - 213.0).abs() < 0.01),
            "the next submission must see the later copy: {next:?}"
        );

        // A raw dispatch has no deferred texture or heap step. Its visible
        // texture is still the old generation when the overwrite follows it.
        let earlier = sampler.submit(host, &texture, 0, split, |a, b| {
            // The previous submission already left the texture shader-readable.
            sampler.record(a);
            b.defer_copy(address.cast(), overwrite.as_ptr(), overwrite.len());
            request(b);
        });
        assert!(
            earlier.iter().all(|&v| (v * 255.0 - 213.0).abs() < 0.01),
            "a later copy must not change a preceding raw dispatch: {earlier:?}"
        );
        let after = sample(true);
        assert!(
            after.iter().all(|&v| (v * 255.0 - 173.0).abs() < 0.01),
            "the next submission must observe the deferred overwrite: {after:?}"
        );
    }
    eprintln!("BC4 raw dispatch: 64/64 preserved texels, 64/64 later texels");
    eprintln!("BC4 ordered copies: 64/64 current texels and 64/64 next-submission texels");
}
