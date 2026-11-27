//! Reads the production display-plane draw from an offscreen D3D12 texture.

use super::*;

const WIDTH: usize = 3;
const OVERLAY: [u8; 12] = [0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 255, 255];

fn texture(host: &Host, width: usize, format: DXGI_FORMAT, output: bool) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: if output {
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: an ordinary single-sample 2D texture on a live device.
    unsafe {
        host.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                if output {
                    D3D12_RESOURCE_STATE_COMMON
                } else {
                    D3D12_RESOURCE_STATE_COPY_DEST
                },
                None,
                &mut resource,
            )
            .unwrap();
    }
    resource.unwrap()
}

fn upload(
    host: &Host,
    list: &HostList,
    width: usize,
    format: DXGI_FORMAT,
    bytes: &[u8],
) -> ID3D12Resource {
    let resource = texture(host, width, format, false);
    // SAFETY: subresource zero is a single row. The upload stays alive with
    // the list and the fixture retains its destination through submission.
    unsafe {
        let desc = resource.GetDesc();
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut row_bytes, mut total) = (0, 0);
        host.device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            Some(&mut row_bytes),
            Some(&mut total),
        );
        assert_eq!(row_bytes, bytes.len() as u64);
        let (buffer, mapped) = list.upload(host, total).unwrap();
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            mapped.add(footprint.Offset as usize),
            bytes.len(),
        );
        let dest = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(buffer)),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        list.list.CopyTextureRegion(&dest, 0, 0, 0, &from, None);
        drop(std::mem::ManuallyDrop::into_inner(from.pResource));
        drop(std::mem::ManuallyDrop::into_inner(dest.pResource));
    }
    resource
}

/// Each variant uploads the same supplied bytes and finishes its own command
/// list before SRV slots can be reused. The legacy SRV stays legal by using
/// a typeless source resource for the comparison.
fn read_composite(
    host: &Host,
    storage: DXGI_FORMAT,
    title: DXGI_FORMAT,
    bytes: &[u8],
    legacy: bool,
) -> Vec<u8> {
    let list = host.new_list().unwrap();
    let base = upload(host, &list, 1, storage, bytes);
    let hud = upload(host, &list, WIDTH, DXGI_FORMAT_R8G8B8A8_UNORM, &OVERLAY);
    // SAFETY: both texture descriptions belong to live resources.
    let (base_desc, hud_desc) = unsafe { (base.GetDesc(), hud.GetDesc()) };
    let plan = display_format(&base_desc, Some(title), &[]).unwrap();
    let source = if legacy {
        swap_format(base_desc.Format)
    } else {
        plan.source
    };
    let hud_view = display_format(&hud_desc, None, &[]).unwrap().source;
    let byte_width = if plan.swap == DXGI_FORMAT_R16G16B16A16_FLOAT {
        8
    } else {
        4
    };
    // A typeless offscreen texture permits the sRGB RTV explicitly. A real
    // flip-model UNORM backbuffer permits this same sRGB RTV.
    let storage = if plan.target == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB {
        DXGI_FORMAT_R8G8B8A8_TYPELESS
    } else {
        plan.swap
    };
    let dest = texture(host, WIDTH, storage, true);
    let comp = Compositor::new(&host.device, plan.target).unwrap();
    let state = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
    let restore = comp.record(
        &host.device,
        &list.list,
        &dest,
        (WIDTH as u32, 1),
        &[(&base, state, source), (&hud, state, hud_view)],
    );
    // SAFETY: the production draw set the destination to RENDER_TARGET.
    let (buffer, footprint) = unsafe {
        list.list.ResourceBarrier(&restore);
        list.list.ResourceBarrier(&[transition(
            &dest,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        crate::capture::read_back(&host.device, &list.list, &dest).unwrap()
    };
    list.close();
    host.execute(&[&list]);
    assert_eq!(
        (
            footprint.Footprint.Width as usize,
            footprint.Footprint.Height
        ),
        (WIDTH, 1)
    );
    let mut data = std::ptr::null_mut();
    // SAFETY: execute waited for the copy. Only actual texels of row zero
    // are read, excluding the footprint's row-pitch padding.
    unsafe {
        host.device.GetDeviceRemovedReason().unwrap();
        buffer.Map(0, None, Some(&mut data)).unwrap();
        let start = data.cast::<u8>().add(footprint.Offset as usize);
        let result = std::slice::from_raw_parts(start, WIDTH * byte_width).to_vec();
        buffer.Unmap(0, None);
        result
    }
}

fn check_half(label: &str, bytes: &[u8], expected: [u16; 12]) {
    let actual: Vec<_> = bytes
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    assert_eq!(actual.len(), expected.len());
    eprintln!("{label} RGBA16F bits {actual:04x?}");
    for (i, (&word, want)) in actual.iter().zip(expected).enumerate() {
        // Only the middle texel blends fractional UNORM8 coverage with an
        // FP16 destination. Permit one FP16 ULP for rounding that operation.
        let tolerance = u16::from((4..8).contains(&i));
        assert!(
            word.abs_diff(want) <= tolerance,
            "{label} channel {i}: {word:#06x} versus {want:#06x}"
        );
    }
}

#[test]
#[ignore = "requires an explicitly selected hardware GPU"]
fn reads_typed_scene_and_overlay_offscreen() {
    let host = host().expect("D3D12 hardware device required");
    // SAFETY: query the adapter of the exact device executing these lists.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let gpu: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        gpu.GetDesc1().unwrap()
    };
    let len = adapter
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(adapter.Description.len());
    let name = String::from_utf16_lossy(&adapter.Description[..len]);
    eprintln!(
        "adapter={name} vendor={:04x} device={:04x} luid={:08x}:{:08x}",
        adapter.VendorId,
        adapter.DeviceId,
        adapter.AdapterLuid.HighPart,
        adapter.AdapterLuid.LowPart
    );
    assert_eq!(
        adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32,
        0,
        "hardware adapter required"
    );
    if let Ok(expected) = std::env::var("PORTVI_EXPECT_ADAPTER") {
        assert!(
            name.to_lowercase().contains(&expected.to_lowercase()),
            "expected {expected}, selected {name}"
        );
    }
    let texel: Vec<u8> = [0x3c00u16, 0x3800, 0x3400, 0x3c00]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let original = read_composite(
        host,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,
        DXGI_FORMAT_R16G16B16A16_UNORM,
        &texel,
        true,
    );
    let corrected = read_composite(
        host,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,
        DXGI_FORMAT_R16G16B16A16_UNORM,
        &texel,
        false,
    );
    let float_colors = [
        0x3c00, 0x3800, 0x3400, 0x3c00, 0x37f8, 0x3a02, 0x2ff8, 0x3c00, 0, 0, 0x3c00, 0x3c00,
    ];
    check_half("legacy FP16 view", &original, float_colors);
    check_half(
        "correct UNORM16 view",
        &corrected,
        [
            0x3380, 0x3300, 0x3280, 0x3380, 0x2f78, 0x38e3, 0x2e79, 0x38f3, 0, 0, 0x3c00, 0x3c00,
        ],
    );
    assert_ne!(original, corrected);
    // Packed R11=1, G11=0.5, B10=0.25. The implicit alpha is 1.
    let packed = read_composite(
        host,
        DXGI_FORMAT_R11G11B10_FLOAT,
        DXGI_FORMAT_R11G11B10_FLOAT,
        &0x681c03c0u32.to_le_bytes(),
        false,
    );
    check_half("typed R11G11B10", &packed, float_colors);
    let srgb = read_composite(
        host,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        &[128, 64, 192, 255],
        false,
    );
    let expected = [128, 64, 192, 255, 92, 192, 140, 255, 0, 0, 255, 255];
    eprintln!("sRGB bytes {srgb:?}");
    assert_eq!(srgb.len(), expected.len());
    for (i, (&got, want)) in srgb.iter().zip(expected).enumerate() {
        // The sRGB transfer and blend may differ by one 8-bit RGB code.
        let tolerance = u8::from(i < 8 && i % 4 != 3);
        assert!(
            got.abs_diff(want) <= tolerance,
            "sRGB channel {i}: {got} versus {want}"
        );
    }
}
