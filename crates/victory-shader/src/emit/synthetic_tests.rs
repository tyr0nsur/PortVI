//! Synthetic GFX10 programs of what the 2020 compiler does not produce,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP. Gathers at a level (`image_gather4_l`, `_l_o`), which
//! DXIL lacks: at level 0 the host's own gather (`image_gather4_lz`) of the
//! same coordinates is the reference on the hardware adapter (RDNA2, as the
//! console): it rounds the exact product of coordinate and size to 1/256 of
//! a texel, half up, which WARP does not. Lane permutes (`ds_bpermute_b32`).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Per lane (32, wave32): the record [offset word, u, v, level] at entry 1
/// of the table at s0 (a 2D texture at entry 0, a raw UAV at entry 2); the
/// level converted from an integer (so it is whole). Gathers red at the
/// level, with the offset, and at level 0 (`lz`) into 12 dwords at lane Ã— 48.
fn program(whole: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        // The culling shaders' sampler: border, transparent black, levels
        // 0-12, linear filters.
        0xB008_01B6, // s_movk_i32 s8, 0x1b6
        0xBE89_03FF,
        0x00C0_0000, // s_mov_b32 s9, 0xc00000
        0xBE8A_03FF,
        0x1850_0000, // s_mov_b32 s10, 0x18500000
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x1628_00B0, // v_mul_u32_u24 v20, 48, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    if whole {
        w.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
    }
    w.extend([
        0xF110_0108,
        0x0043_040D, // image_gather4_l v[4:7], v[13:15], s[12:19], s[8:11]
        0xF150_0108,
        0x0043_080C, // image_gather4_l_o v[8:11], v[12:15], s[12:19], s[8:11]
        0xF11C_0108,
        0x0043_100D, // image_gather4_lz v[16:19], v[13:14], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // buffer_store_dwordx4 v[4:7], v20, s[40:43], 0 offen
        0xE078_1010,
        0x800A_0814, // buffer_store_dwordx4 v[8:11], v20, s[40:43], 0 offen offset:16
        0xE078_1020,
        0x800A_1014, // buffer_store_dwordx4 v[16:19], v20, s[40:43], 0 offen offset:32
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    build_wave(words, false)
}

/// [`build`] for waves of 64 (`wave64`, one wave of 64 threads) or 32.
fn build_wave(words: &[u32], wave64: bool) -> Result<HostShader, crate::ShaderError> {
    let wave = if wave64 { 64 } else { 32 };
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 16),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [wave, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let lowered = lowered(&native)?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    })
}

fn lowered(native: &NativeProgram) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        host_wave32: false,
        view_remap: false,
        view_marks: false,
    })
}

/// The first texel of a footprint (see `Lower::gather_first`).
fn first(u: f32, size: u32) -> i64 {
    let u = if u.is_nan() { 0.0 } else { u.clamp(-2.0, 2.0) };
    ((f64::from(u) * f64::from(size) * 512.0).floor() as i64 - 255) >> 9
}

#[test]
fn gathers_at_a_level_match_the_hardware() {
    // 24Ã—16 then 12Ã—8, 6Ã—4, 3Ã—2; red = level Ã— 1000 + y Ã— 32 + x + 1.
    let mut texels = Vec::new();
    for l in 0..4 {
        let (w, h) = (24 >> l, 16 >> l);
        for y in 0..h {
            for x in 0..w {
                texels.push([(l * 1000 + y * 32 + x + 1) as f32, 0.0, 0.0, 1.0]);
            }
        }
    }
    let texel = |l: u32, x: i64, y: i64| -> f32 {
        let (w, h) = (24i64 >> l, 16i64 >> l);
        let skip: i64 = (0..l).map(|k| (24i64 >> k) * (16 >> k)).sum();
        if (0..w).contains(&x) && (0..h).contains(&y) {
            texels[(skip + y * w + x) as usize][0]
        } else {
            0.0
        }
    };
    // Records: lanes 0-15 at level 0 where rounding decides (coordinates
    // whose f32 product with 24 lands on the other side of a texel's
    // threshold than the exact one, ties of 1/512 on the 16 texels of y,
    // NaN, infinities, zeros); lanes 16-31 at other levels and offsets.
    let mut records: Vec<[u32; 4]> = Vec::new();
    let rounded = |u: f32| ((f64::from(u * 24.0) * 512.0).floor() as i64 - 255) >> 9;
    let mut straddling = (1..24).filter_map(|i| {
        let u0 = ((f64::from(i) + 127.5 / 256.0) / 24.0) as f32;
        (-1024i32..1024)
            .map(|d| f32::from_bits(u0.to_bits().wrapping_add_signed(d)))
            .find(|&u| rounded(u) != first(u, 24))
    });
    for k in 0..16 {
        let tie = (f64::from(k - 3) + 0.5 - 1.0 / 512.0) / 16.0;
        let u = match k {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            _ => straddling.next().unwrap(),
        };
        records.push([0, u.to_bits(), (tie as f32).to_bits(), 0]);
    }
    for k in 0..16u32 {
        let level = [1, 2, 3, 7][k as usize % 4];
        let offset = [0x0000, 0x3F01, 0x0302, 0x3E3D][k as usize / 4];
        let (u, v) = (
            (k as f32 * 0.77 + 0.3) / 13.0 - 0.1,
            (k as f32 * 0.41 + 0.2) / 9.0,
        );
        records.push([offset, u.to_bits(), v.to_bits(), level]);
    }
    let input: Vec<u32> = records.iter().flatten().copied().collect();
    let shader = build(&program(true)).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Mipped(24, 16, 4, &texels),
        // The program reads these dwords by byte offset through a raw handle.
        View::Buffer {
            data: &input,
            stride: Stride::Raw,
        },
        View::RawUav(32 * 12),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        ran += 1;
        for (lane, r) in records.iter().enumerate() {
            let got = |k: usize| -> Vec<f32> {
                out[0][lane * 12 + 4 * k..][..4]
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect()
            };
            let [offset, u, v, level] = *r;
            let level = level.min(3);
            let (w, h) = (24 >> level, 16 >> level);
            let at = |ox: i64, oy: i64| {
                let (x, y) = (
                    first(f32::from_bits(u), w) + ox,
                    first(f32::from_bits(v), h) + oy,
                );
                vec![
                    texel(level, x, y + 1),
                    texel(level, x + 1, y + 1),
                    texel(level, x + 1, y),
                    texel(level, x, y),
                ]
            };
            let sext = |b: u32| i64::from((b as i32) << 26 >> 26);
            assert_eq!(got(0), at(0, 0), "{label}: lane {lane}");
            assert_eq!(
                got(1),
                at(sext(offset), sext(offset >> 8)),
                "{label}: lane {lane}, offset"
            );
            // WARP's own gather truncates where the hardware rounds.
            if lane < 16 && !warp {
                assert_eq!(got(2), got(0), "{label}: lane {lane} against its gather");
            }
        }
    }
    assert!(ran > 0, "no device");
}

#[test]
fn gathers_at_a_level_that_may_be_a_fraction_stop() {
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&program(false)) else {
        panic!("translated a gather at a level read from memory");
    };
    assert_eq!(what, "a gather at a level that may not be whole");
}

#[test]
fn permutes_read_the_lane_named_and_zero_from_lanes_left_out() {
    // Lanes below 24 run the permute; each reads lane (13 Ã— lane + 5) mod 32
    // (its address in dwords, offset 20 bytes) of 1000 + lane.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x1602_00B4, // v_mul_u32_u24 v1, 52, v0
        0x4A04_00FF,
        0x0000_03E8, // v_add_nc_u32 v2, 1000, v0
        0x7E06_02FF,
        0x0000_DEAD, // v_mov_b32 v3, 0xdead
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xDACC_0014,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|l| match (13 * l + 5) % 32 {
            _ if l >= 24 => 0xDEAD,
            src if src < 24 => 1000 + src,
            _ => 0,
        })
        .collect();
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The program declares the console's wave of 32, which WARP refuses.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

/// Builds, DXIL-validates, and runs a single-wave `ds_bpermute_b32` compute
/// shader (`wave` = 32 or 64) where `body` leaves each lane's result in `v3`.
fn run_permute_cs(wave64: bool, body: &[u32], want: &[u32]) {
    let wave = if wave64 { 64 } else { 32 };
    assert_eq!(want.len(), wave as usize);
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let shader = build_wave(&words, wave64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(wave),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // Fixed WaveSize pipelines are refused by WARP, and WaveSize(64) by
        // 32-lane hardware adapters.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp || wave64, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of {wave}; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

#[test]
fn permutes_mask_high_address_bits_and_divergent_branches() {
    // High byte-address bits (0x100) and sub-dword bits (+3) are ignored, so
    // only bits 6:2 select the source lane. Lanes 16..31 skip the permute via
    // `s_cbranch_execz`, so reads from them yield 0 while they keep `v3`.
    let body = [
        0x4A04_00FF,
        0x0000_3000, // v_add_nc_u32 v2, 0x3000, v0
        0x4A06_00FF,
        0xBEEF_0000, // v_add_nc_u32 v3, 0xbeef0000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xBF88_0002, // s_cbranch_execz +2
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
    ];
    let want: Vec<u32> = (0..32)
        .map(|l| {
            if l < 8 {
                0x3000 + l + 8
            } else if l < 16 {
                0
            } else {
                0xBEEF_0000 + l
            }
        })
        .collect();
    run_permute_cs(false, &body, &want);
}

#[test]
fn wave64_permutes_stay_within_each_32_lane_half() {
    // RDNA 2 ISA §10.4.4: wave64 permutes lanes 0..31 and 32..63 independently.
    // v1 = ((lane & 31) << 2) + 0x103 with offset:12 exercises both high/low
    // bit masking and wrapping within each 32-lane half (29..31 -> 0..2,
    // 61..63 -> 32..34; lane 32 pulls from lane 35, not lane 3).
    let body = [
        0x4A04_00FF,
        0x0000_4000, // v_add_nc_u32 v2, 0x4000, v0
        0x3602_009F, // v_and_b32 v1, 31, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
    ];
    let want: Vec<u32> = (0..64)
        .map(|l| 0x4000 + ((l & !31) | ((l + 3) & 31)))
        .collect();
    run_permute_cs(true, &body, &want);
}

#[test]
fn wave64_permutes_check_exec_within_each_half() {
    // Active lanes 0..39 in wave64: lanes 32..39 pull from upper-half lanes
    // 40..47 (inactive -> 0), even though lower-half lanes 8..15 are active;
    // inactive lanes 40..63 preserve 0xcafe0000 + lane.
    let body = [
        0x4A04_00FF,
        0x0000_5000, // v_add_nc_u32 v2, 0x5000, v0
        0x4A06_00FF,
        0xCAFE_0000, // v_add_nc_u32 v3, 0xcafe0000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3602_029F, // v_and_b32 v1, 31, v1
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    ];
    let want: Vec<u32> = (0..64)
        .map(|l| {
            if l < 32 {
                0x5000 + ((l + 8) & 31)
            } else if l < 40 {
                0
            } else {
                0xCAFE_0000 + l
            }
        })
        .collect();
    run_permute_cs(true, &body, &want);
}

/// A lane mask joined with data at a uniform loop's header, as the title's
/// wave-uniform BVH walks push lane masks to memory and pop them
/// (`shadowmask:CS_TraceShadowRays_Checkerboard`): the register holds the
/// mask on entry and a number from the back edge, read both as a number and
/// as a mask.
#[test]
fn a_mask_joined_with_data_at_a_uniform_loop_is_its_number() {
    // s6 = lanes below 24, shifted right by 4 and passed through a VGPR in
    // each of two iterations; v4 takes each lane's bit of it on every pass.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
        0xBE87_0380, // s_mov_b32 s7, 0
        0xD501_0004,
        0x0019_0280, // header: v_cndmask_b32 v4, 0, 1, s6
        0x9006_8406, // s_lshr_b32 s6, s6, 4
        0x7E0A_0206, // v_mov_b32 v5, s6
        0x7E0C_0505, // v_readfirstlane_b32 s6, v5
        0x8007_8107, // s_add_u32 s7, s7, 1
        0xBF0A_8207, // s_cmp_lt_u32 s7, 2
        0xBF85_FFF8, // s_cbranch_scc1 header
        0x7E06_0206, // v_mov_b32 v3, s6
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xE070_1080,
        0x800A_0410, // buffer_store_dword v4, v16, s[40:43], 0 offen offset:128
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(64),
    ];
    let mut want = vec![0x00FF_FFFF >> 8; 32];
    want.extend((0..32).map(|lane| u32::from(lane < 20)));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

/// A lane mask a compare makes inside a region the other lanes skip, joined
/// with data in the same register on the skip edge (a reused register): the
/// skipping lanes read 0, their bit of the compare, as on the console. A
/// mask holding lanes outside its edge (the saved EXEC) stays refused.
#[test]
fn a_compare_mask_joined_with_data_on_the_skip_edge_is_0_there() {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
        0xBF88_0002, // s_cbranch_execz +2
        0xD4C4_0006,
        0x0002_0088, // v_cmp_gt_u32 s6, 8, v0
        0xBEFE_0308, // s_mov_b32 exec_lo, s8
        0xD501_0003,
        0x0019_0280, // v_cndmask_b32 v3, 0, 1, s6
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let shader = build(&words).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let want: Vec<u32> = (0..32).map(|lane| u32::from(lane < 8)).collect();
    {
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let Ok((root, pso)) = host.compute_pipeline(&shader) else {
                assert!(warp, "{label}: no pipeline for a wave of 32");
                eprintln!("{label}: no pipeline for a wave of 32; skipped");
                continue;
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(out[0], want, "{label}");
        }
    }
    // The saved EXEC instead (every lane): the skipping lanes' bit would be
    // 1 on the console.
    words[10..12].copy_from_slice(&[0xBE86_0308, 0xBF80_0000]); // s_mov_b32 s6, s8; s_nop 0
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&words) else {
        panic!("a mask holding lanes off its edge joined with data translated");
    };
    assert!(what.contains("lane-mask join"), "{what}");
}

/// A register array (v3-v14) indexed through M0 in waterfall loops, as
/// `tiled_lighting:CS_PrescalarizeLightLoop` builds its per-tile light
/// masks: each lane's index (clamped to 0-11 by `v_med3_u32`) becomes M0 in
/// the loop pass that handles the lanes holding it; a `v_movrels` reads the
/// lane's array element, a `v_movreld` writes it back with the lane's bit
/// added. Twice, so the second pass reads what the first wrote.
#[test]
fn relative_moves_in_waterfall_loops_reach_each_lane_s_element() {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    ];
    w.extend((3..15u32).map(|k| 0x7E00_0280 | k << 17)); // v_mov_b32 vk, 0
    w.extend([
        0x2C38_0081, // v_lshrrev_b32 v28, 1, v0
        0xD559_001C,
        0x022D_011C, // v_med3_u32 v28, v28, 0, 11
        0x7E3A_0281, // v_mov_b32 v29, 1
        0x343A_3B00, // v_lshlrev_b32 v29, v0, v29
    ]);
    let pass = [
        0xBEA4_037E, // s_mov_b32 s36, exec_lo
        0x7E4C_051C, // read: v_readfirstlane_b32 s38, v28
        0xD4C2_0006,
        0x0002_3826, // v_cmp_eq_u32 s6, s38, v28
        0xBE86_3C06, // s_and_saveexec_b32 s6, s6
        0xBEFC_0326, // s_mov_b32 m0, s38
        0x7E3E_8703, // v_movrels_b32 v31, v3
        0x897E_067E, // s_xor_b32 exec_lo, exec_lo, s6
        0xBF89_FFF8, // s_cbranch_execnz read
        0xBEFE_0324, // s_mov_b32 exec_lo, s36
        0x383A_3B1F, // v_or_b32 v29, v31, v29
        0x7E4C_051C, // write: v_readfirstlane_b32 s38, v28
        0xD4C2_0006,
        0x0002_3826, // v_cmp_eq_u32 s6, s38, v28
        0xBE86_3C06, // s_and_saveexec_b32 s6, s6
        0xBEFC_0326, // s_mov_b32 m0, s38
        0x7E06_851D, // v_movreld_b32 v3, v29
        0x897E_067E, // s_xor_b32 exec_lo, exec_lo, s6
        0xBF89_FFF8, // s_cbranch_execnz write
        0xBEFE_0324, // s_mov_b32 exec_lo, s36
    ];
    w.extend(pass);
    w.extend([0x7E3A_02FF, 0x8000_0000]); // v_mov_b32 v29, 0x80000000
    w.extend(pass);
    for k in 3..15u32 {
        // buffer_store_dword vk, v16, s[40:43], 0 offen offset:128 * (k - 3)
        w.extend([0xE070_1000 | (128 * (k - 3)), 0x800A_0010 | k << 8]);
    }
    w.push(0xBF81_0000); // s_endpgm
    let shader = build(&w).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(12 * 32),
    ];
    let want: Vec<u32> = (0..12u32)
        .flat_map(|k| (0..32u32).map(move |lane| if (lane >> 1).min(11) == k { 1 << lane | 0x8000_0000 } else { 0 }))
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}
#[test]
fn ordered_pixel_sections_use_rasterizer_ordered_views() {
    // A pixel program's ordered section (POPS) as the title's compiler
    // writes it: the collision id after the primitive mask (s10, user data
    // ending at s9), a wait loop on the exiting wave id, a read-modify-write
    // of a UAV, the end of the section.
    let code: Vec<u32> = vec![
        0xBE84_0308, // s_mov_b32 s4, s8
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x9000_9F0A, // s_lshr_b32 s0, s10, 31
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_81EF, // s_add_i32 s2, src_pops_exiting_wave_id, 1
        0xBF08_8502, // s_cmp_gt_u32 s2, 5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0004, // s_cbranch_vccnz +4
        0xBF8E_0001, // s_sleep 1
        0x8100_81EF, // s_add_i32 s0, src_pops_exiting_wave_id, 1
        0xBF0A_8500, // s_cmp_lt_u32 s0, 5
        0xBF84_FFFC, // s_cbranch_scc0 -4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_3F88,
        0x0003_0200, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm dlc glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_1F08,
        0x0003_0200, // image_store v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_1890,
        0x0000_0000, // exp null off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = |db_shader_control| NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 8,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 8,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let l = lowered(&native(1 << 16)).expect("lower");
    assert!(l.rovs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Without the ordered mode the collision id is no value the host has.
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowered(&native(0)) else {
        panic!("translated a collision id outside an ordered section");
    };
    assert_eq!(what, "a hardware SGPR input");
}

#[test]
fn pops_wait_loop_exits_immediately_for_inactive_lanes() {
    use super::container::SigElement;
    use super::host::Draw;
    use crate::native::{Usage, VsInput};

    let pos_sig = |name: &str, kind: u8, interp: u8| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let to_host = |native: &NativeProgram, inputs: Vec<SigElement>, outputs: Vec<SigElement>| {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
        let io = io::raise(&p, native).expect("io");
        let l = lower::lower(&lower::Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs,
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
            host_wave32: false,
            view_remap: false,
            view_marks: false,
        })
        .expect("lower");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = l.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        HostShader {
            stage: native.stage,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            memory: l.memory,
            memory_writes: l.memory_writes,
            view_addresses: l.view_addresses,
            heap: l.heap,
            heap_writes: l.heap_writes,
            view_marks: false,
        }
    };

    // Pass-through full-screen vertex shader (attributes in v4..v7; v0..v3 hold hardware ids).
    let vs_code: [u32; 3] = [
        0xF800_08CF,
        0x0706_0504, // exp pos0, v4, v5, v6, v7 done
        0xBF81_0000, // s_endpgm
    ];
    let vs_native = NativeProgram {
        stage: HwStage::Vs,
        wave64: true,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: vec![VsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0xF,
                vgpr: 4,
            }],
            outputs: Vec::new(),
        },
        code: vs_code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let vs = to_host(
        &vs_native,
        vec![pos_sig("POSITION", 0, 0)],
        vec![pos_sig("SV_Position", 3, 4)],
    );

    // Pixel shader with narrowed EXEC before the title's wrap-around POPS
    // wait loop: lanes with SV_Position.x >= 2.0 have EXEC cleared at
    // s_cbranch_vccnz so they fall into the wait loop, then restore EXEC and
    // export the loop iteration count in s12. If the collision id has
    // wave_id == collision_id (0 == 0), wave_id + ~collision_id is 0xffff_ffff
    // so `s_cmp_lt_u32 s5, s0` never holds and the loop runs to LOOP_LIMIT
    // (1 << 20) instead of exiting after one pass.
    let ps_code: [u32; 28] = [
        0xBE8C_0380, // s_mov_b32 s12, 0
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7C28_00F4, // v_cmpx_gt_f32 exec, 2.0, v0  (active for x < 2.0)
        0x9382_FF09,
        0x000A_0000, // s_bfe_u32 s2, s9, lit(0x000a0000) (collision id)
        0x9385_FF09,
        0x000A_0010, // s_bfe_u32 s5, s9, lit(0x000a0010) (wave id)
        0xBE84_0702, // s_not_b32 s4, s2
        0x9000_9F09, // s_lshr_b32 s0, s9, 31 (overlap bit)
        0x8105_0405, // s_add_i32 s5, s5, s4
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_04EF, // s_add_i32 s2, src_pops_exiting_wave_id, s4
        0xBF08_0502, // s_cmp_gt_u32 s2, s5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0005, // s_cbranch_vccnz +5
        0xBF8E_FFFF, // s_sleep 0xffff
        0x810C_810C, // s_add_i32 s12, s12, 1
        0x8100_04EF, // s_add_i32 s0, src_pops_exiting_wave_id, s4
        0xBF0A_0005, // s_cmp_lt_u32 s5, s0
        0xBF84_FFFB, // s_cbranch_scc0 -5
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0x7E08_0C0C, // v_cvt_f32_u32 v4, s12
        0xF800_180F,
        0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
        0xBF81_0000, // s_endpgm
    ];
    let ps_bytes: Vec<u8> = ps_code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let ps_native = NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 1 << 8, // SV_Position.x in v0
            db_shader_control: 1 << 16,
            patches: Vec::new(),
        },
        code: ps_bytes,
    };
    let ps = to_host(
        &ps_native,
        vec![pos_sig("SV_Position", 3, 4)],
        vec![pos_sig("SV_Target", 16, 0)],
    );

    let host = Host::new(true).expect("WARP host");
    let (root, pso) = host.pipeline(&vs, &ps).expect("pipeline");
    let pixels = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &[
                    -1.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0,
                ],
                stride: 16,
                size: (4, 1),
                mesh_groups: 0,
            },
        )
        .expect("draw");
    // Active pixels (x < 2.0) skip the wait loop (0 iterations); inactive
    // pixels (x >= 2.0) fall through once and exit immediately (1 iteration),
    // never spinning to LOOP_LIMIT (1 << 20).
    assert_eq!(pixels, vec![[0.0; 4], [0.0; 4], [1.0; 4], [1.0; 4]]);
}

// ---- Lane-mask SCC and lane masks joined with data (PR 2289) ----
//
// One wave (32 or 64 lanes). Lane l reads [a, selector, b, 0] at l × 16 of
// the raw SRV at table entry 1 into v[5:7] and stores v3, v4 at l × 8 of the
// raw UAV at entry 2. Each program restores EXEC before the stores, so lanes
// outside a narrowed EXEC record too.

const MASK_PROLOGUE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0502,
    0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x3420_0084, // v_lshlrev_b32 v16, 4, v0
    0x3422_0083, // v_lshlrev_b32 v17, 3, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE03C_1000,
    0x8005_0510, // buffer_load_dwordx3 v[5:7], v16, s[20:23], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
];

const MASK_EPILOGUE: &[u32] = &[
    0xE070_1000,
    0x800A_0311, // buffer_store_dword v3, v17, s[40:43], 0 offen
    0xE070_1004,
    0x800A_0411, // buffer_store_dword v4, v17, s[40:43], 0 offen offset:4
    0xBF81_0000, // s_endpgm
];

fn mask_program(body: &[u32]) -> Vec<u32> {
    [MASK_PROLOGUE, body, MASK_EPILOGUE].concat()
}

/// Lane records: a and b nonzero (and different lane to lane) exactly in
/// the lanes of masks `a` and `b`; every lane's selector `sel`.
fn lane_records(lanes: u32, a: u64, sel: u32, b: u64) -> Vec<u32> {
    (0..lanes)
        .flat_map(|l| {
            let word = |m: u64, k: u32| if m >> l & 1 != 0 { 3 * l + k } else { 0 };
            [word(a, 1), sel, word(b, 2), 0]
        })
        .collect()
}

/// The wave size `PSV0` requires (its runtime info's minimum and maximum
/// wave lane counts, equal), none where any will do. `dxil.dll`, where
/// present, checks them against the module's `WaveSize`.
fn required_wave(shader: &HostShader) -> Option<u32> {
    let c = crate::dxbc::Container::parse(&shader.dxil).expect("container");
    let psv = c.part(crate::dxbc::FourCC::PSV0).expect("PSV0").data;
    let min = crate::dxbc::u32_at(psv, 4 + 16).expect("PSV0 min wave lanes");
    let max = crate::dxbc::u32_at(psv, 4 + 20).expect("PSV0 max wave lanes");
    match (min, max) {
        (0, u32::MAX) => None,
        (n, m) => {
            assert_eq!(n, m, "PSV0 wave lanes {n}..{m}");
            Some(n)
        }
    }
}

/// Runs `shader` (one wave of `lanes`) once per input on each adapter whose
/// wave lane counts include `lanes`: the UAV's dwords, by adapter and
/// input. The program must require that wave size (it ballots the
/// console's lane masks over the physical wave), on any host.
fn run_inputs(
    shader: &HostShader,
    lanes: u32,
    inputs: &[Vec<u32>],
) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    assert_eq!(
        required_wave(shader),
        Some(lanes),
        "the program must declare the console's wave of {lanes}"
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut runs = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // Pipeline creation may accept a wave size the device never runs
        // (a wave of 64 on a 32-lane GPU): its lane counts decide.
        match host.wave_lanes() {
            Some((min, max)) if (min..=max).contains(&lanes) => {}
            bounds => {
                eprintln!("{label}: wave lanes {bounds:?}, no wave of {lanes}; skipped");
                continue;
            }
        }
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: no pipeline for a wave of {lanes}: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let outs = inputs
            .iter()
            .map(|data| {
                assert_eq!(data.len(), 4 * lanes as usize);
                let views = [
                    View::Texture(1, 1, &[[0.0; 4]]),
                    View::Buffer {
                        data: &data[..],
                        stride: Stride::Raw,
                    },
                    View::RawUav(2 * lanes),
                ];
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud.clone()))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                out[0].clone()
            })
            .collect();
        runs.push((label, outs));
    }
    if runs.is_empty() {
        eprintln!("SKIPPED: no adapter runs a wave of {lanes}; numerics unchecked");
    }
    runs
}

/// `s_and_saveexec` of the lanes whose a is set, then `s_cselect_b32 s6, 7,
/// 9` on its SCC: lanes of the new EXEC write s6 to v3 (64 elsewhere);
/// after EXEC is restored, every lane writes s6 to v4. Per mask: v3 is 7 in
/// its lanes and 64 outside; v4 is 7 in every lane if the mask has a lane
/// (SCC 1), else 9 (SCC 0).
fn check_saveexec_scc(wave64: bool, masks: &[u64]) {
    let lanes = if wave64 { 64 } else { 32 };
    let body: [u32; 7] = [
        0x7E06_02C0, // v_mov_b32 v3, 64
        0x7D8A_0A80, // v_cmp_ne_u32_e32 vcc, 0, v5
        if wave64 {
            0xBE98_246A // s_and_saveexec_b64 s[24:25], vcc
        } else {
            0xBE98_3C6A // s_and_saveexec_b32 s24, vcc_lo
        },
        0x8506_8987, // s_cselect_b32 s6, 7, 9
        0x7E06_0206, // v_mov_b32 v3, s6
        if wave64 {
            0xBEFE_0418 // s_mov_b64 exec, s[24:25]
        } else {
            0xBEFE_0318 // s_mov_b32 exec_lo, s24
        },
        0x7E08_0206, // v_mov_b32 v4, s6
    ];
    let shader = build_wave(&mask_program(&body), wave64).expect("lower");
    // Both SCC values are exercised.
    assert!(masks.contains(&0) && masks.iter().any(|&m| m != 0));
    let inputs: Vec<_> = masks
        .iter()
        .map(|&m| lane_records(lanes, m, 0, 0))
        .collect();
    for (label, outs) in run_inputs(&shader, lanes, &inputs) {
        for (&m, got) in masks.iter().zip(&outs) {
            let want: Vec<u32> = (0..lanes)
                .flat_map(|l| {
                    let inside = m >> l & 1 != 0;
                    [if inside { 7 } else { 64 }, if m != 0 { 7 } else { 9 }]
                })
                .collect();
            assert_eq!(got, &want, "{label}: wave{lanes}, mask {m:#018x}");
        }
    }
}

#[test]
fn saveexec_scc_reads_whether_the_new_exec_has_a_lane_wave32() {
    check_saveexec_scc(false, &[0x8421_00F6, 0, 0x8000_0000, 0x1, 0xFFFF_FFFF]);
}

#[test]
fn saveexec_scc_reads_both_dwords_of_the_new_exec_wave64() {
    check_saveexec_scc(
        true,
        &[
            // Low dword only, high dword only (SCC from lanes 32-63 alone),
            // none, lanes 31 and 32 across the boundary, all.
            0x0000_0000_0000_0110,
            0x8000_0400_0000_0000,
            0,
            0x0000_0001_8000_0000,
            u64::MAX,
        ],
    );
}

/// EXEC narrowed to the lanes whose b is set; `v_cmp` of the lanes whose a
/// is set; then a uniform branch on lane 0's selector: nonzero takes
/// `s_and s2, exec, vcc` (a tracked mask: a & b), zero `s_mov` of numbers.
/// The join is one scalar data phi per dword. EXEC is restored before the
/// stores.
#[test]
fn data_phi_of_a_tracked_mask_or_a_number_wave32() {
    let body = [
        0x7E0E_0506, // v_readfirstlane_b32 s7, v6
        0x7D8A_0E80, // v_cmp_ne_u32_e32 vcc_lo, 0, v7
        0xBE98_3C6A, // s_and_saveexec_b32 s24, vcc_lo
        0x7D8A_0A80, // v_cmp_ne_u32_e32 vcc_lo, 0, v5
        0xBF06_8007, // s_cmp_eq_u32 s7, 0
        0xBF84_0003, // s_cbranch_scc0 +3 (the mask)
        0xBE82_03FF,
        0x0BAD_F00D, // s_mov_b32 s2, 0xbadf00d
        0xBF82_0001, // s_branch +1 (the join)
        0x8702_6A7E, // s_and_b32 s2, exec_lo, vcc_lo
        0xBEFE_0318, // s_mov_b32 exec_lo, s24
        0x7E06_0202, // v_mov_b32 v3, s2
        0x2C08_0700, // v_lshrrev_b32 v4, v0, v3
        0x3608_0881, // v_and_b32 v4, 1, v4
    ];
    let shader = build(&mask_program(&body)).expect("lower");
    // (a, selector, b): s2 is a & b for a nonzero selector, else the number.
    let cases: [(u64, u32, u64); 5] = [
        (0xF0F0_A5C3, 1, 0x3C3C_FF0F),
        (0xF0F0_A5C3, 0, 0x3C3C_FF0F),
        (0x0000_000F, 7, 0x0000_00F0),
        (0xFFFF_FFFF, 1, 0x8000_0001),
        (0x0000_0000, 0, 0xFFFF_FFFF),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(a, sel, b)| lane_records(32, a, sel, b))
        .collect();
    for (label, outs) in run_inputs(&shader, 32, &inputs) {
        for (&(a, sel, b), got) in cases.iter().zip(&outs) {
            let s2 = if sel != 0 { (a & b) as u32 } else { 0x0BAD_F00D };
            let want: Vec<u32> = (0..32).flat_map(|l| [s2, s2 >> l & 1]).collect();
            assert_eq!(got, &want, "{label}: a {a:#x}, selector {sel}, b {b:#x}");
        }
    }
}

// ---- Loop-carried lane-mask joins ----

/// Every lane stores its `v_mbcnt_lo` count to v4. Lanes whose b is set
/// then run a uniform loop of three passes (EXEC narrowed by
/// `s_and_saveexec`): pass k compares bit k of a (`v_cmp`, clear outside
/// EXEC) and XORs that into s0 (s[0:1]), the loop's mask phi, entered with
/// what `init` writes. After EXEC is restored, every lane selects 1 or 0 by
/// its bit of the result (`v_cndmask_b32`, reading the join) into v3.
fn check_loop_mask_join(
    wave64: bool,
    init: &[u32],
    init_mask: impl Fn(u32) -> u64,
    cases: &[(u64, u32, u64)],
) {
    let lanes: u32 = if wave64 { 64 } else { 32 };
    let mut body = vec![
        0xD765_0004,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v4, -1, 0
    ];
    // Before EXEC is narrowed (it may read lane 0).
    body.extend_from_slice(init);
    body.extend([
        0x7D8A_0E80, // v_cmp_ne_u32_e32 vcc, 0, v7
        if wave64 {
            0xBE98_246A // s_and_saveexec_b64 s[24:25], vcc
        } else {
            0xBE98_3C6A // s_and_saveexec_b32 s24, vcc_lo
        },
        0xBE88_0380, // s_mov_b32 s8, 0
        // loop:
        0x2C10_0A08, // v_lshrrev_b32 v8, s8, v5
        0x3610_1081, // v_and_b32 v8, 1, v8
        0x7D8A_1080, // v_cmp_ne_u32_e32 vcc, 0, v8
        if wave64 {
            0x8980_6A00 // s_xor_b64 s[0:1], s[0:1], vcc
        } else {
            0x8900_6A00 // s_xor_b32 s0, s0, vcc_lo
        },
        0x8008_8108, // s_add_u32 s8, s8, 1
        0xBF0A_8308, // s_cmp_lt_u32 s8, 3
        0xBF85_FFF9, // s_cbranch_scc1 loop
        if wave64 {
            0xBEFE_0418 // s_mov_b64 exec, s[24:25]
        } else {
            0xBEFE_0318 // s_mov_b32 exec_lo, s24
        },
        0xD501_0003,
        0x0001_0280, // v_cndmask_b32_e64 v3, 0, 1, s0 (s[0:1])
    ]);
    let shader = build_wave(&mask_program(&body), wave64).expect("lower");
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(a, sel, b)| lane_records(lanes, a, sel, b))
        .collect();
    for (label, outs) in run_inputs(&shader, lanes, &inputs) {
        for (&(a, sel, b), got) in cases.iter().zip(&outs) {
            let init = init_mask(sel);
            let want: Vec<u32> = (0..lanes)
                .flat_map(|l| {
                    let aw = if a >> l & 1 != 0 { 3 * l + 1 } else { 0 };
                    // Lanes outside EXEC compare as clear on every pass.
                    let flips = if b >> l & 1 != 0 {
                        (aw ^ (aw >> 1) ^ (aw >> 2)) & 1
                    } else {
                        0
                    };
                    [((init >> l) as u32 & 1) ^ flips, l.min(32)]
                })
                .collect();
            assert_eq!(
                got, &want,
                "{label}: wave{lanes}, a {a:#x}, selector {sel}, b {b:#x}, init {init:#x}"
            );
        }
    }
}

/// (a, selector, b) for a wave of 32: some lanes active, all, none, two.
const LOOP_CASES_32: [(u64, u32, u64); 4] = [
    (0xF0F0_A5C3, 0, 0x3C3C_FF0F),
    (0xFFFF_FFFF, 5, 0xFFFF_FFFF),
    (0x0F0F_0F0F, 0, 0),
    (0x7777_0001, 5, 0x8000_0001),
];

#[test]
fn loop_mask_join_from_zero_wave32() {
    // s_mov_b32 s0, 0
    check_loop_mask_join(false, &[0xBE80_0380], |_| 0, &LOOP_CASES_32);
}

#[test]
fn loop_mask_join_from_all_lanes_wave32() {
    // s_mov_b32 s0, -1: lanes outside EXEC keep their set bit.
    check_loop_mask_join(false, &[0xBE80_03C1], |_| 0xFFFF_FFFF, &LOOP_CASES_32);
}

#[test]
fn loop_mask_join_from_a_constant_lane_set_wave32() {
    // s_mov_b32 s0, 0xa5c35a3c
    check_loop_mask_join(
        false,
        &[0xBE80_03FF, 0xA5C3_5A3C],
        |_| 0xA5C3_5A3C,
        &LOOP_CASES_32,
    );
}

#[test]
fn loop_mask_join_from_a_uniform_select_wave32() {
    // Entered with none or every lane, by lane 0's selector (read before
    // EXEC is narrowed): a uniform number on the entry edge.
    check_loop_mask_join(
        false,
        &[
            0x7E0E_0506, // v_readfirstlane_b32 s7, v6
            0xBF06_8007, // s_cmp_eq_u32 s7, 0
            0x8500_C180, // s_cselect_b32 s0, 0, -1
        ],
        |sel| if sel == 0 { 0 } else { 0xFFFF_FFFF },
        &LOOP_CASES_32,
    );
}

/// Wave64: pairs of inputs whose results share the low dword and differ in
/// the high one (and the reverse), a high dword alone, every lane. Each
/// lane's bit must come from its own dword of the joined pair.
#[test]
fn loop_mask_join_keeps_both_dwords_wave64() {
    let cases = [
        (0xFFFF_0000_F0F0_A5C3, 0, 0x00FF_FF00_3C3C_FF0F),
        (0x0000_FFFF_F0F0_A5C3, 0, 0xFF00_00FF_3C3C_FF0F),
        (0x1234_5678_FFFF_0000, 5, 0x3C3C_FF0F_00FF_FF00),
        (0x1234_5678_0000_FFFF, 5, 0x3C3C_FF0F_FF00_00FF),
        (0x8000_0001_0000_0000, 0, u64::MAX),
        (u64::MAX, 5, u64::MAX),
    ];
    // s_mov_b64 s[0:1], 0, then -1.
    check_loop_mask_join(true, &[0xBE80_0480], |_| 0, &cases);
    check_loop_mask_join(true, &[0xBE80_04C1], |_| u64::MAX, &cases);
}

/// A scalar join reads the accumulated mask as a number on its nonzero arm.
/// Its ballot discovers the boolean loop phi only while filling the data
/// phi. Leaving that final phi empty produces invalid LLVM bitcode.
#[test]
fn data_join_materializes_a_loop_mask_wave32() {
    let body = [
        0x7E0E_0506, // v_readfirstlane_b32 s7, v6
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE88_0380, // s_mov_b32 s8, 0
        0x7D8A_0A80, // loop: v_cmp_ne_u32_e32 vcc_lo, 0, v5
        0x8900_6A00, // s_xor_b32 s0, s0, vcc_lo
        0x8008_8108, // s_add_u32 s8, s8, 1
        0xBF0A_8308, // s_cmp_lt_u32 s8, 3
        0xBF85_FFFB, // s_cbranch_scc1 loop
        0x7D8A_0E80, // v_cmp_ne_u32_e32 vcc_lo, 0, v7
        0xBF06_8007, // s_cmp_eq_u32 s7, 0
        0xBF84_0003, // s_cbranch_scc0 mask
        0xBE82_03FF,
        0x0BAD_F00D, // s_mov_b32 s2, literal
        0xBF82_0001, // s_branch joined
        0x8702_6A00, // mask: s_and_b32 s2, s0, vcc_lo
        0x7E06_0202, // joined: v_mov_b32 v3, s2
        0x2C08_0700, // v_lshrrev_b32 v4, v0, v3
        0x3608_0881, // v_and_b32 v4, 1, v4
    ];
    let shader = build(&mask_program(&body)).expect("lower");
    let cases = [
        (0xF0F0_A5C3, 1, 0x3C3C_FF0F),
        (0xF0F0_A5C3, 0, 0x3C3C_FF0F),
        (0x8000_0001, 2, 0xFFFF_FFFF),
        (0, 1, 0xFFFF_FFFF),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(a, sel, b)| lane_records(32, a, sel, b))
        .collect();
    for (adapter, outputs) in run_inputs(&shader, 32, &inputs) {
        for (&(a, sel, b), actual) in cases.iter().zip(outputs) {
            let word = if sel == 0 {
                0x0BAD_F00D
            } else {
                (a & b) as u32
            };
            let expected: Vec<_> = (0..32)
                .flat_map(|lane| [word, (word >> lane) & 1])
                .collect();
            assert_eq!(
                actual, expected,
                "{adapter}, selector {sel}, inputs {a:#x} {b:#x}"
            );
        }
    }
}

/// The wave64 form: s[2:3] joins a & b (both dwords of a tracked mask) or
/// two numbers; every lane stores s2 and s3.
#[test]
fn data_phi_of_a_tracked_mask_or_a_number_wave64() {
    let body = [
        0x7E0E_0506, // v_readfirstlane_b32 s7, v6
        0x7D8A_0E80, // v_cmp_ne_u32_e32 vcc, 0, v7
        0xBE98_246A, // s_and_saveexec_b64 s[24:25], vcc
        0x7D8A_0A80, // v_cmp_ne_u32_e32 vcc, 0, v5
        0xBF06_8007, // s_cmp_eq_u32 s7, 0
        0xBF84_0005, // s_cbranch_scc0 +5 (the mask)
        0xBE82_03FF,
        0x0BAD_F00D, // s_mov_b32 s2, 0xbadf00d
        0xBE83_03FF,
        0x5EED_1234, // s_mov_b32 s3, 0x5eed1234
        0xBF82_0001, // s_branch +1 (the join)
        0x8782_6A7E, // s_and_b64 s[2:3], exec, vcc
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        0x7E06_0202, // v_mov_b32 v3, s2
        0x7E08_0203, // v_mov_b32 v4, s3
    ];
    let shader = build_wave(&mask_program(&body), true).expect("lower");
    let cases: [(u64, u32, u64); 5] = [
        (0xFF00_00F0_8000_0F0F, 1, 0x0F0F_F0F0_8000_0FF0),
        (0xFF00_00F0_8000_0F0F, 0, 0x0F0F_F0F0_8000_0FF0),
        // High dword only, then low dword only.
        (0x8000_0001_0000_0000, 3, u64::MAX),
        (u64::MAX, 3, 0x0000_0000_4000_0002),
        (0x0000_00FF_0000_0000, 9, 0x0000_FF00_0000_00FF),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(a, sel, b)| lane_records(64, a, sel, b))
        .collect();
    for (label, outs) in run_inputs(&shader, 64, &inputs) {
        for (&(a, sel, b), got) in cases.iter().zip(&outs) {
            let (lo, hi) = if sel != 0 {
                ((a & b) as u32, ((a & b) >> 32) as u32)
            } else {
                (0x0BAD_F00D, 0x5EED_1234)
            };
            let want: Vec<u32> = (0..64).flat_map(|_| [lo, hi]).collect();
            assert_eq!(got, &want, "{label}: a {a:#x}, selector {sel}, b {b:#x}");
        }
    }
}
