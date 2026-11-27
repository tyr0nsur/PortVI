//! Four offscreen texels through the regular compute binding and command-list paths.
use super::*;
use crate::pipeline::{Shader, Translated};
use victory_shader::{Container, emit, io, ir, native::fixture, resource, structure};
use windows::Win32::Graphics::Dxgi::*;

type Row = [[u8; 4]; 4];
const RED: Row = [[233, 11, 19, 255], [197, 24, 37, 255], [165, 32, 51, 255], [129, 46, 63, 255]];
const GREEN: Row = [[7, 237, 31, 255], [13, 197, 59, 255], [23, 169, 73, 255], [41, 131, 83, 255]];

struct Owner(This);

impl Drop for Owner {
    fn drop(&mut self) {
        crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

struct Pool(usize);

impl Drop for Pool {
    fn drop(&mut self) {
        crate::watch::tests::free(self.0);
    }
}

fn input_texture(at: usize, pixels: &Row) -> Owner {
    // SAFETY: our 64KiB writable fixture allocation, with the first four
    // texels of the first linear RGBA8 row beginning at the allocation.
    unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), at as *mut [u8; 4], pixels.len()) };
    Owner(texture_at_address(at as u64, &texture_words(4, 1, 28, 0x100, 0)))
}

fn fixture_heap() -> Owner {
    // One real, normally laid-out capped shader-visible heap. Both sizes
    // stay in thousands of descriptors regardless of the adapter's limit.
    let words = [0u32, 4096, 1, 0, 0, 0, 0, 0];
    let iid = ID3D12DescriptorHeap::IID;
    let iid = crate::object::guid(iid.data1, iid.data2, iid.data3, iid.data4);
    let mut object = std::ptr::null();
    assert_eq!(crate::descriptor::create_heap(std::ptr::null(), words.as_ptr().cast(), &iid, &mut object), S_OK);
    let owned = Owner(object);
    crate::descriptor::heap(object).set_test_view_capacity(2048);
    owned
}

fn pipeline(indexed: bool, remap: bool) -> HostPipeline {
    let native = fixture::root_indexed_texture(indexed);
    let program = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let structured = structure::structure(&program).unwrap();
    let resources = resource::track(&program, &native.layout, native.stage, Some(&structured)).unwrap();
    let io = io::raise(&program, &native).unwrap();
    let host = emit::emit_with(
        &Container { digest: [0; 16], version: (1, 0), parts: Vec::new() },
        &native, &program, &structured, &io, &resources,
        emit::Options { view_remap: remap, ..Default::default() },
    ).unwrap();
    assert_eq!(host.root_indexed_views.len(), usize::from(indexed));
    let shader = Shader::for_test("compute", Translated { host, used: native.used });
    HostPipeline::new(&[shader], None).expect("runtime compute PSO")
}

fn identify(host: &crate::host::Host) {
    // SAFETY: query the adapter LUID of the exact device used for the commands.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(host.device.GetAdapterLuid()).unwrap();
        adapter.GetDesc1().unwrap()
    };
    let end = adapter.Description.iter().position(|&c| c == 0).unwrap_or(adapter.Description.len());
    let name = String::from_utf16_lossy(&adapter.Description[..end]);
    println!("adapter={name} vendor={:04x} device={:04x} luid={:08x}:{:08x} remap={}",
        adapter.VendorId, adapter.DeviceId, adapter.AdapterLuid.HighPart, adapter.AdapterLuid.LowPart, host.view_remap);
    assert_eq!(adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0, "hardware adapter required");
    if let Ok(expected) = std::env::var("PORTVI_EXPECT_ADAPTER") {
        assert!(name.to_lowercase().contains(&expected.to_lowercase()), "expected {expected}, got {name}");
    }
    assert!(host.view_remap, "run with VICTORY_VIEW_REMAP=1 on the capped adapter");
}

#[test]
#[ignore = "requires an explicitly selected hardware GPU with descriptor remapping"]
fn reads_root_indexed_textures() {
    let host = crate::host::host().expect("D3D12 hardware device required");
    identify(host);
    let storage = [(); 3].map(|()| Pool(crate::watch::tests::pool(1)));
    let red = input_texture(storage[0].0, &RED);
    let green = input_texture(storage[1].0, &GREEN);
    let output = Owner(texture_at_address(storage[2].0 as u64, &texture_words(4, 1, 28, 0x100, 4)));
    let target = crate::resource::host_resource(output.0).unwrap();
    let owner = fixture_heap();
    let heap = crate::descriptor::heap(owner.0);
    let host_heap = heap.host().unwrap();
    let direct = host_heap.count();
    let base = direct.checked_sub(2).unwrap();
    assert!(direct > 4 && base + 5 < 4096);
    let address = |slot: u32| heap.start + 32 * u64::from(slot);
    let srv = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
    for (offset, object) in [(0, red.0), (1, red.0), (4, green.0), (5, red.0)] {
        crate::descriptor::create_srv(std::ptr::null(), object, &srv, address(base + offset));
    }
    let uav = [28, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    crate::descriptor::create_uav(std::ptr::null(), output.0, std::ptr::null(), &uav, address(0));
    let fixed = pipeline(false, host.view_remap);
    let indexed = pipeline(true, host.view_remap);
    let list = host.new_list().unwrap();
    let mut state = State::default();
    state.set_heaps(&[owner.0]);
    state.set_table(true, 3, address(0));
    state.set_table(true, 7, address(base));
    let mut results = Vec::new();
    let mut snapshots = Vec::new();
    let phases = [
        ("static", &fixed, 0, 0, &RED, 0),
        ("indexed-direct", &indexed, 0, 1, &RED, 0),
        ("indexed-new", &indexed, 3, 4, &GREEN, 1),
        ("constant-changed", &indexed, 4, 5, &RED, 2),
        ("indexed-cached", &indexed, 3, 4, &GREEN, 2),
    ];
    for (label, pso, constant, offset, expected, count) in phases {
        if label == "indexed-cached" {
            // A cached descriptor does not imply the texture is still readable.
            crate::list::ensure_state(&list, green.0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        }
        state.set_constants(true, 2, &[71, constant], 0);
        assert!(state.bind(&list, pso, true), "{label} State::bind");
        assert_eq!(state.fresh_tables.len(), count, "{label} fresh table entries");
        let runs: Vec<_> = state.dynamic.iter().map(|&(_, _, first, len)| (first, len)).collect();
        if label == "indexed-new" { assert_eq!(runs[0].1, 5); }
        if label == "constant-changed" { assert_eq!(runs[1].1, 6); snapshots = runs.clone(); }
        if label == "indexed-cached" {
            assert_eq!(runs, snapshots, "reused the same physical snapshot spans");
        }
        // SAFETY: State::bind recorded the runtime's PSO, descriptor heaps,
        // root arguments and input/output resource barriers on this open list.
        unsafe { list.list.Dispatch(1, 1, 1) };
        crate::list::ensure_state(&list, output.0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let read = crate::capture::read_back(&host.device, &list.list, &target).unwrap();
        results.push((label, offset, expected, runs, read));
    }
    list.close();
    host.execute(&[&list]);
    // The list commits its recorded resource states only at submission.
    let final_state = crate::resource::made_copy(green.0).unwrap().state.load(Ordering::Relaxed);
    let read = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE).0 as u32;
    assert_eq!(final_state & read, read, "cached SRV transition");
    for (label, offset, expected, runs, (read, footprint)) in results {
        assert_eq!((footprint.Footprint.Width, footprint.Footprint.Height), (4, 1));
        // SAFETY: the synchronous submission finished. The footprint holds
        // the entire one-row texture at its reported byte offset.
        let row = unsafe {
            let mut data = std::ptr::null_mut();
            read.Map(0, None, Some(&mut data)).unwrap();
            let row = data.cast::<u8>().add(footprint.Offset as usize).cast::<Row>().read_unaligned();
            read.Unmap(0, None);
            row
        };
        println!("{label} title_slot={} direct_limit={direct} dynamic_spans={runs:?} rgba8={row:?}", base + offset);
        assert_eq!(&row, expected, "{label}: all four texels");
    }
    // SAFETY: plain health query of the same live device.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    assert_eq!(host.log_messages(), None, "D3D12 debug layer error");
}
