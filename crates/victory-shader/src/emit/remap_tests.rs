//! GPU and lowering tests for [`super::VIEW_REMAP`] virtual-to-physical
//! resource-heap remapping above [`super::HEAP_VIEWS`].

use crate::dxbc::Container;
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{io, ir, resource, structure};

use super::host::{Dispatch, Host, Root, Stride, View, view_strides};
use super::{HEAP_VIEWS_PORTABLE, Options, REMAP_VIEWS, VIEW_REMAP, VIEW_STRIDES, emit_with};

fn param(root: RootType, user_data: u8, slot: u8, sgprs: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: u32::from(sgprs),
        base_register: 0,
        user_data,
        slot,
    }
}

/// Builds a compute shader reading a structured buffer SRV at virtual
/// `ResourceDescriptorHeap[s0 >> 5]` (element `tid.x`, stride from `VIEW_STRIDES`)
/// and writing the result into a raw UAV at virtual `ResourceDescriptorHeap[s1 >> 5]`.
fn build_remapped_cs(capped_heap: bool) -> super::HostShader {
    // s0 = root constant 0 (byte offset of SRV in virtual heap)
    // s1 = root constant 1 (byte offset of UAV in virtual heap)
    // s2..s3 = resource heap base (user_data 2, 0)
    // v0 = tid.x
    let words: &[u32] = &[
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0x0000_0000, // s_load_dwordx4 s[4:7], s[2:3], s0
        0xF408_0201,
        0x0200_0000, // s_load_dwordx4 s[8:11], s[2:3], s1
        0xE030_2000,
        0x8001_0100, // buffer_load_dword v1, v0, s[4:7], 0 idxen
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xE070_1000,
        0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [4, 1, 1],
        }),
        vgprs: 4,
        wave64: false,
        layout: vec![
            param(RootType::Constants, 0, 0, 2),
            param(RootType::ResourceHeap, 2, 1, 1),
        ],
        used: Vec::new(),
        code,
    };
    let p = ir::build(&native.code, false, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structured");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let raised = io::raise(&p, &native).expect("io");
    let bytes = crate::dxbc::build(&[]);
    let empty = Container::parse(&bytes).expect("container");
    emit_with(
        &empty,
        &native,
        &p,
        &st,
        &raised,
        &res,
        Options {
            view_marks: false,
            view_remap: capped_heap,
            ..Default::default()
        },
    )
    .expect("emitted")
}

#[test]
fn remapped_high_heap_indices_execute_above_portable_views_limit() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    super::with_view_heap(HEAP_VIEWS_PORTABLE, 721_469, || {
        let without = build_remapped_cs(false);
        assert_eq!(without.root_srvs, [VIEW_STRIDES]);

        let shader = build_remapped_cs(true);
        assert!(shader.root_srvs.contains(&VIEW_REMAP));
        assert!(shader.root_srvs.contains(&VIEW_STRIDES));

        // Virtual console slots well above `HEAP_VIEWS_PORTABLE` (1,000,000), mapped
        // into physical slots 3 (SRV, stride 8 -> elements [101, 202, 303, 404]) and 4 (UAV).
        let virt_srv: u32 = 1_150_000;
        let virt_uav: u32 = 1_500_000;
        let virt_unmapped: u32 = 1_800_000;
        assert!(virt_srv > HEAP_VIEWS_PORTABLE && virt_unmapped < REMAP_VIEWS);

        let srv_words: [u32; 8] = [101, 999, 202, 999, 303, 999, 404, 999];
        let views = [
            View::RawUav(1),
            View::RawUav(1),
            View::RawUav(1),
            View::Buffer {
                data: &srv_words,
                stride: Stride::Raw,
            },
            View::RawUav(4),
            View::NullBuffers,
        ];
        let mut strides = view_strides(&views);
        strides.resize(HEAP_VIEWS_PORTABLE as usize, 0);
        strides[3] = 8;

        let mut remap: Vec<u32> = vec![HEAP_VIEWS_PORTABLE - 1; REMAP_VIEWS as usize];
        for (i, slot) in remap.iter_mut().take(3).enumerate() {
            *slot = i as u32;
        }
        remap[virt_srv as usize] = 3;
        remap[virt_uav as usize] = 4;

        // `root_signature_for` sorts `root_srvs` ascending: param 0 = user constants,
        // param 1 = VIEW_REMAP (254), param 2 = VIEW_STRIDES (255).
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else {
                assert!(!warp, "WARP must be available");
                continue;
            };
            let (root, pso) = host.compute_pipeline(&shader).expect("compute PSO");

            // 1) Mapped virtual slots 1,150,000 -> phys 3 and 1,500,000 -> phys 4.
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(vec![virt_srv * 32, virt_uav * 32, 0, 0])),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: remap.clone(),
                                },
                            ),
                            (
                                2,
                                Root::View {
                                    srv: true,
                                    data: strides.clone(),
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch mapped high slots");
            assert_eq!(out[3], [101, 202, 303, 404], "warp={warp}");

            // 2) Unmapped high slot 1,800,000 resolves to `HEAP_VIEWS - 1` (stride 0 ->
            // `NULL_BUFFER_VIEW`) and reads zeros without faulting.
            let out_null = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (
                                0,
                                Root::Constants(vec![virt_unmapped * 32, virt_uav * 32, 0, 0]),
                            ),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: remap.clone(),
                                },
                            ),
                            (
                                2,
                                Root::View {
                                    srv: true,
                                    data: strides.clone(),
                                },
                            ),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch unmapped high slot");
            assert_eq!(out_null[3], [0, 0, 0, 0], "warp={warp}");
            eprintln!("remapped high SRV/UAV and null slot matched on warp={warp}");
        }
    });
}
