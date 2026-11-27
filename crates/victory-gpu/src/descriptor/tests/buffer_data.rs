use super::*;
use std::ptr::null;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV_FLAG_RAW, D3D12_BUFFER_UAV_FLAG_RAW, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_SRV_DIMENSION_BUFFER, D3D12_UAV_DIMENSION_BUFFER,
};
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{IUnknown, Interface};

/// Read what the guest shader sees both on the table's first use and after
/// replacing an existing buffer view. Each slot spans two address words,
/// a record count and a packed format/type word.
#[test]
fn adjacent_buffer_descriptors_report_their_fields_after_a_rewrite() {
    let Some(device) = crate::host::host() else {
        eprintln!("no D3D12 device; skipped buffer descriptor readback");
        return;
    };
    let mut guest = null();
    assert_eq!(
        create(
            HeapDesc {
                kind: VIEWS_KIND,
                count: 3,
                flags: SHADER_VISIBLE,
                nodes: 0
            },
            Placement {
                address: 0,
                size: 0
            },
            &mut guest,
        ),
        S_OK
    );
    // SAFETY: create handed out one reference to the guest descriptor heap.
    let _heap_owner = unsafe { IUnknown::from_raw(guest.cast_mut().cast()) };
    let guest_heap = heap(guest);
    let input = crate::resource::test_buffer(256);
    // SAFETY: test_buffer hands out one reference to the guest buffer.
    let _input_owner = unsafe { IUnknown::from_raw(input.cast_mut().cast()) };
    let base = crate::resource::buffer_bytes(input).unwrap().0 as u64;
    let handle =
        |slot: u32| guest_heap.start + u64::from(slot * increment_size(null(), VIEWS_KIND));
    let write_srv = |slot: u32, format: DXGI_FORMAT, first: u32, count: u32, padding: u32| {
        // The guest CreateShaderResourceView argument is ten DWORDs, including its
        // padding. The buffer union begins at DWORD 4 with the 64-bit FirstElement.
        let desc = [
            format.0 as u32,
            D3D12_SRV_DIMENSION_BUFFER.0 as u32,
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            padding,
            first,
            0,
            count,
            0,
            0,
            0,
        ];
        create_srv(null(), input, &desc, handle(slot));
    };
    write_srv(1, DXGI_FORMAT_R32_UINT, 4, 5, 0);
    write_srv(2, DXGI_FORMAT_R32_FLOAT, 12, 11, 0);

    let shader = descriptor_reader(2);
    assert!(shader.view_addresses && shader.root_srvs.is_empty() && !shader.memory);
    let host = victory_shader::emit::host::Host::from_device(device.device.clone());
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let mirror = guest_heap.host().expect("host view heap");
    let addresses = mirror.view_addresses().expect("view-address table");
    let (output, mapped) = device.shared_buffer(32).unwrap();
    mirror.raw_view_at(0, Some(&output), 0, 32, true);
    let dispatch = |base_slot: u32| {
        let list = device.new_list().unwrap();
        let mut table_base = vec![0u32; shader.user_data_dwords() as usize];
        let table = shader.user_data.iter().find(|u| u.slot == 0 && u.dword == 0).unwrap();
        table_base[table.index as usize] = base_slot * 32;
        // SAFETY: the root table begins at base_slot. The shader reads only the two
        // valid input slots and writes all eight DWORDs of the output. Each
        // submission finishes before rewriting a descriptor or reusing the output.
        unsafe {
            std::ptr::write_bytes(mapped, 0xa5, 32);
            list.list.SetDescriptorHeaps(&[Some(mirror.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRoot32BitConstants(
                0,
                table_base.len() as u32,
                table_base.as_ptr().cast(),
                0,
            );
            list.list.SetComputeRootShaderResourceView(1, addresses);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        device.execute(&[&list]);
        // SAFETY: eight mapped DWORDs written before the submission's fence.
        unsafe { mapped.cast::<[[u32; 4]; 2]>().read_unaligned() }
    };
    let before = dispatch(0);
    write_srv(1, DXGI_FORMAT_R32_SINT, 20, 7, 0);
    let after = dispatch(0);
    // AMD RDNA2 ISA, buffer resource descriptor: NUM_RECORDS is DWORD 2,
    // FORMAT is bits 18:12 of DWORD 3. R32_UINT/SINT/FLOAT encode as 20/21/22.
    // The address takes DWORD 1 bits 15:0 only: STRIDE lies above them.
    let fields = |[lo, hi, count, word3]: [u32; 4]| {
        let address = u64::from(lo) | (u64::from(hi & 0xffff) << 32);
        (address.wrapping_sub(base), count, (word3 >> 12) & 0x7f)
    };
    let actual = [before.map(fields), after.map(fields)];
    let expected = [[(16, 5, 20), (48, 11, 22)], [(80, 7, 21), (48, 11, 22)]];
    eprintln!("descriptor readbacks (byte offset, count, FORMAT) before and after: {actual:?}");
    let mut reads = Vec::new();

    // SRV first element 3, five records. UAV first element 7, nine
    // four-byte records. The expected offsets and GFX10 FORMAT codes
    // below are independent of the runtime's DXGI conversion.
    for (srv, srv_offset, srv_code, uav, uav_code) in [
        (DXGI_FORMAT_R8_UINT, 3, 5, DXGI_FORMAT_R32_UINT, 20),
        (DXGI_FORMAT_R8_SNORM, 3, 2, DXGI_FORMAT_R32_SINT, 21),
        (DXGI_FORMAT_R16_SINT, 6, 12, DXGI_FORMAT_R32_FLOAT, 22),
        (DXGI_FORMAT_R16_FLOAT, 6, 13, DXGI_FORMAT_R32_UINT, 20),
        (DXGI_FORMAT_R8G8_UNORM, 6, 14, DXGI_FORMAT_R32_SINT, 21),
        (DXGI_FORMAT_R16G16_FLOAT, 12, 29, DXGI_FORMAT_R32_FLOAT, 22),
        (DXGI_FORMAT_R32G32_FLOAT, 24, 64, DXGI_FORMAT_R32_UINT, 20),
        (
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            24,
            71,
            DXGI_FORMAT_R32_SINT,
            21,
        ),
        (
            DXGI_FORMAT_R32G32B32A32_UINT,
            48,
            75,
            DXGI_FORMAT_R32_FLOAT,
            22,
        ),
        (
            DXGI_FORMAT_R11G11B10_FLOAT,
            12,
            36,
            DXGI_FORMAT_R32_UINT,
            20,
        ),
        (
            DXGI_FORMAT_R10G10B10A2_UNORM,
            12,
            50,
            DXGI_FORMAT_R32_SINT,
            21,
        ),
        (
            DXGI_FORMAT_R8G8B8A8_UNORM,
            12,
            56,
            DXGI_FORMAT_R32_FLOAT,
            22,
        ),
    ] {
        // If SRV API padding leaks into the guest FORMAT, it reads as 127.
        write_srv(1, srv, 3, 5, 0x0007_f000);
        let desc = [
            uav.0 as u32,
            D3D12_UAV_DIMENSION_BUFFER.0 as u32,
            7,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
        ];
        create_uav(null(), input, null(), &desc, handle(2));
        let actual = dispatch(0).map(fields);
        let expected = [(srv_offset, 5, srv_code), (28, 9, uav_code)];
        reads.push((srv, uav, actual, expected));
    }
    let mut matches = [0u32; 3];
    for (_, _, observed, want) in &reads {
        for (a, e) in observed.iter().zip(want) {
            matches[0] += u32::from(a.0 == e.0);
            matches[1] += u32::from(a.1 == e.1);
            matches[2] += u32::from(a.2 == e.2);
        }
    }
    eprintln!("24 rewritten SRV/UAV fields of each kind: address {}, count {}, FORMAT {}",
              matches[0], matches[1], matches[2]);
    assert_eq!(actual, expected, "initial slots and their first rewrite");
    for (srv, uav, observed, want) in reads {
        assert_eq!(observed, want, "SRV {srv:?}, UAV {uav:?}");
    }

    // The renderer moves tables past the title slots on capped heaps. The
    // root table names the new offset for both resource and metadata reads.
    let relocated = mirror.dynamic(3).expect("three table slots");
    mirror.raw_view_at(relocated, Some(&output), 0, 32, true);
    let srv = super::view(handle(1)).unwrap();
    let uav = super::view(handle(2)).unwrap();
    mirror.make_at(relocated + 1, &srv);
    mirror.make_at(relocated + 2, &uav);
    let old = dispatch(0);
    assert_eq!(dispatch(relocated), old, "rebuilt table sees the same descriptor words");
    mirror.make_null(relocated + 1);
    assert_eq!(dispatch(relocated), [[0; 4], old[1]], "clearing one relocated slot");
    mirror.make_at(relocated + 1, &uav);
    mirror.raw_view_at(relocated + 2, Some(&output), 0, 32, true);
    assert_eq!(dispatch(relocated), [old[1], [0; 4]], "reuse as a host-only raw view");
    assert_eq!(dispatch(0), old, "original slots survive every dynamic rewrite");
    mirror.release(relocated, 3);
    eprintln!("rebuilt descriptor table: original, cleared and reused slots matched on GPU");
}

/// Each buffer view kind's descriptor as the shader's SGPRs hold it, before
/// and after every slot is rewritten as another kind: dword 1 carries the
/// address bits 32-47 and STRIDE (bits 29:16) side by side, the element size
/// of typed views (packed and 12/16-byte formats too), the structure stride
/// of structured ones, 0 for raw ones (records in bytes). A table relocated
/// past the title slots holds the same words.
#[test]
fn shader_reads_high_address_and_stride_of_each_buffer_view_kind() {
    const SLOTS: u32 = 6;
    let Some(device) = crate::host::host() else {
        eprintln!("no D3D12 device; skipped buffer stride readback");
        return;
    };
    let mut guest = null();
    let desc = HeapDesc { kind: VIEWS_KIND, count: SLOTS + 1, flags: SHADER_VISIBLE, nodes: 0 };
    assert_eq!(create(desc, Placement { address: 0, size: 0 }, &mut guest), S_OK);
    // SAFETY: create handed out one reference to the guest descriptor heap.
    let _heap_owner = unsafe { IUnknown::from_raw(guest.cast_mut().cast()) };
    let guest_heap = heap(guest);
    let input = crate::resource::test_buffer(4096);
    // SAFETY: test_buffer hands out one reference to the guest buffer.
    let _input_owner = unsafe { IUnknown::from_raw(input.cast_mut().cast()) };
    let base = crate::resource::buffer_bytes(input).unwrap().0 as u64;
    let handle =
        |slot: u32| guest_heap.start + u64::from(slot * increment_size(null(), VIEWS_KIND));
    // A view: its API format, structure stride and raw flag.
    type Kind = (DXGI_FORMAT, u32, bool);
    let typed = |format| (format, 0, false);
    let structured = |stride| (DXGI_FORMAT_UNKNOWN, stride, false);
    let raw: Kind = (DXGI_FORMAT_R32_TYPELESS, 0, true);
    // Unordered, kind, first element, count; then the expected STRIDE,
    // NUM_RECORDS and FORMAT.
    type Case = (bool, Kind, u32, u32, u32, u32, u32);
    let write = |slot: u32, &(unordered, (format, stride, raw), first, count, ..): &Case| {
        if unordered {
            let flags = if raw { D3D12_BUFFER_UAV_FLAG_RAW.0 as u32 } else { 0 };
            let dimension = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
            let desc = [format.0 as u32, dimension, first, 0, count, stride, 0, 0, flags, 0];
            create_uav(null(), input, null(), &desc, handle(slot));
        } else {
            let flags = if raw { D3D12_BUFFER_SRV_FLAG_RAW.0 as u32 } else { 0 };
            let dimension = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
            let mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            let desc = [format.0 as u32, dimension, mapping, 0, first, 0, count, stride, flags, 0];
            create_srv(null(), input, &desc, handle(slot));
        }
        if unordered && raw {
            let view = super::view(handle(slot)).unwrap();
            let (copy, address, len) = crate::descriptor::host::view_copy(&view).unwrap();
            assert_eq!((address, len), (base + u64::from(first) * 4, u64::from(count) * 4));
            assert_eq!(copy.offset(address) % 16, 0, "raw UAV at slot {slot}");
        }
        // SAFETY: only query the live device's diagnostic state.
        assert!(unsafe { device.device.GetDeviceRemovedReason() }.is_ok(), "invalid descriptor at slot {slot}: {format:?} stride={stride} raw={raw} first={first}");
    };
    // Raw and structured SRVs start on 16 bytes; every range fits 4 KiB.
    let before: [Case; SLOTS as usize] = [
        (false, typed(DXGI_FORMAT_R32G32B32_FLOAT), 1, 4, 12, 4, 74),
        (false, structured(24), 2, 3, 24, 3, 20),
        (false, raw, 4, 8, 0, 32, 20),
        (true, typed(DXGI_FORMAT_R16G16B16A16_FLOAT), 3, 5, 8, 5, 71),
        (true, structured(20), 1, 6, 20, 6, 20),
        (true, raw, 2, 6, 0, 24, 20),
    ];
    // Each slot rewritten as another kind: no stale STRIDE survives.
    let after: [Case; SLOTS as usize] = [
        (false, raw, 4, 4, 0, 16, 20),
        (false, typed(DXGI_FORMAT_R10G10B10A2_UNORM), 5, 7, 4, 7, 50),
        (false, structured(16), 1, 2, 16, 2, 20),
        (true, raw, 6, 3, 0, 12, 20),
        (true, typed(DXGI_FORMAT_R32G32B32A32_UINT), 2, 4, 16, 4, 75),
        (true, structured(2048), 0, 1, 2048, 1, 20),
    ];
    // RDNA2 buffer descriptor: BASE_ADDRESS in dword 0 and dword 1 bits
    // 15:0, STRIDE dword 1 bits 29:16 (30-31 clear here), NUM_RECORDS dword
    // 2, FORMAT dword 3 bits 18:12. Each read apart from the others.
    let fields = |[lo, hi, records, word3]: [u32; 4]| {
        (lo, hi & 0xFFFF, hi >> 16 & 0x3FFF, hi >> 30, records, word3 >> 12 & 0x7F)
    };
    let expected = |cases: [Case; SLOTS as usize]| {
        cases.map(|(_, (_, _, raw), first, _, stride, records, format)| {
            let element = if raw { 4 } else { u64::from(stride) };
            let address = base + u64::from(first) * element;
            (address as u32, (address >> 32) as u32 & 0xFFFF, stride, 0, records, format)
        })
    };

    let shader = descriptor_reader(SLOTS);
    let host = victory_shader::emit::host::Host::from_device(device.device.clone());
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let mirror = guest_heap.host().expect("host view heap");
    let addresses = mirror.view_addresses().expect("view-address table");
    const BYTES: u64 = 16 * SLOTS as u64;
    let (output, mapped) = device.shared_buffer(BYTES).unwrap();
    mirror.raw_view_at(0, Some(&output), 0, BYTES, true);
    let dispatch = |base_slot: u32| {
        let list = device.new_list().unwrap();
        let mut table_base = vec![0u32; shader.user_data_dwords() as usize];
        let table = shader.user_data.iter().find(|u| u.slot == 0 && u.dword == 0).unwrap();
        table_base[table.index as usize] = base_slot * 32;
        // SAFETY: the root table begins at base_slot; the shader reads its
        // SLOTS input descriptors as data only and writes all of the
        // output. Each submission finishes before the next rewrite.
        unsafe {
            std::ptr::write_bytes(mapped, 0xa5, BYTES as usize);
            list.list.SetDescriptorHeaps(&[Some(mirror.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRoot32BitConstants(0, table_base.len() as u32, table_base.as_ptr().cast(), 0);
            list.list.SetComputeRootShaderResourceView(1, addresses);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        device.execute(&[&list]);
        // SAFETY: BYTES mapped bytes written before the submission's fence.
        unsafe { mapped.cast::<[[u32; 4]; SLOTS as usize]>().read_unaligned() }
    };

    for (slot, case) in (1..).zip(&before) {
        write(slot, case);
    }
    let first = dispatch(0);
    for (slot, case) in (1..).zip(&after) {
        write(slot, case);
    }
    let rewritten = dispatch(0);
    eprintln!(
        "(address low, bits 32-47, STRIDE, bits 30-31, records, FORMAT) before and after: {:?} {:?}",
        first.map(fields),
        rewritten.map(fields)
    );
    assert_eq!(first.map(fields), expected(before), "the first views");
    assert_eq!(rewritten.map(fields), expected(after), "every slot rewritten as another kind");

    // Descriptor-as-data: a relocated table holds the same whole words.
    let relocated = mirror.dynamic(SLOTS + 1).expect("relocated table slots");
    mirror.raw_view_at(relocated, Some(&output), 0, BYTES, true);
    for slot in 1..=SLOTS {
        mirror.make_at(relocated + slot, &super::view(handle(slot)).unwrap());
    }
    assert_eq!(dispatch(relocated), rewritten, "the relocated table's words");
    mirror.release(relocated, SLOTS + 1);
}

/// A synthetic one-thread GFX10 shader copies the four descriptor words at
/// each table slot 1 to `slots` into the raw UAV at slot 0, 16 bytes a slot.
/// No input is dereferenced.
fn descriptor_reader(slots: u32) -> victory_shader::emit::HostShader {
    use victory_shader::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };

    let mut words = vec![
        0xbe81_0380u32, // s_mov_b32 s1, 0: the table begins at s0
        0xf408_0100,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0 (output UAV)
    ];
    for slot in 1..=slots {
        words.extend([
            0xf408_0200,
            0xfa00_0000 | (32 * slot), // s_load_dwordx4 s[8:11], s[0:1], 32 * slot
            0xbf8c_0000,             // s_waitcnt 0
        ]);
        for v in 0..4 {
            words.push(0x7e00_0200 | v << 17 | (8 + v)); // v_mov_b32 vN, s[8 + N]
        }
        // buffer_store_dwordx4 v[0:3], off, s[4:7], 0 offset:16 * (slot - 1)
        words.extend([0xe078_0000 | (16 * (slot - 1)), 0x8001_0000]);
    }
    words.push(0xbf81_0000); // s_endpgm
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let signatures = victory_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let ir = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&ir).unwrap();
    let res =
        victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&ir, &native).unwrap();
    victory_shader::emit::emit(&signatures, &native, &ir, &st, &io, &res).unwrap()
}
