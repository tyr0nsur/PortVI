//! Console wave64 compute programs translated for a host of 32-lane waves
//! (`lower::halves`): what reads within a 32-lane half runs exactly in two
//! host waves of 32; what reads across them is traded between the two where
//! every thread of the group meets, and refused with its reason elsewhere.
//!
//! The GPU test runs one program on every hardware adapter: in halves
//! everywhere (`WaveSize(32)`), and also as one wave64 (`WaveSize(64)`)
//! where the adapter has 64-lane waves (AMD), which is the console's own
//! form; the results must be the console's, computed here, in both.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{gfx10, io, ir, resource, structure};

pub(super) const S_ENDPGM: u32 = 0xBF81_0000;
pub(super) const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
pub(super) const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
pub(super) const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// Per thread `l` of a group of 64 (one console wave64), eight dwords at
/// `l * 32`: its thread id, `v_mbcnt` of all lanes (its lane), `l ^ 5`
/// through `ds_swizzle` (xor mode, within 32 lanes), `l ^ 9` through
/// `ds_bpermute_b32` (the address `l ^ 41` keeps the caller’s half), `0x777` written by `v_writelane` to
/// lane 40 only, whether the constant mask -4 (every lane but 0 and 1; lane
/// 32 is bit 0 of its high dword) has its lane, and `l ^ 1` through DPP
/// (`quad_perm:[1,0,3,2]`).
fn in_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0xD8D4_141F,
        0x0200_0000, // ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM xor 5)
        0x3A08_00A9, // v_xor_b32 v4, 41, v0 (request a lane in the opposite half)
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xDACC_0000,
        0x0300_0004, // ds_bpermute_b32 v3, v4, v0
        0xBE88_03FF,
        0x0000_0777, // s_mov_b32 s8, 0x777
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD761_0005,
        0x0001_5008, // v_writelane_b32 v5, s8, 40
        0xBE94_04C4, // s_mov_b64 s[20:21], -4
        0xD501_0006,
        0x0051_0280, // v_cndmask_b32 v6, 0, 1, s[20:21]
        0x7E0E_02FA,
        0xFF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] (DPP)
        S_WAITCNT_LGKM,
    ]);
    for (vdata, slot) in [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`in_halves`]'s lane `l`.
fn console(l: u32) -> [u32; 8] {
    [
        l,
        l,
        l ^ 5,
        l ^ 9,
        if l == 40 { 0x777 } else { 0 },
        u32::from(l >= 2),
        l ^ 1,
        0,
    ]
}

pub(super) fn native(words: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    NativeProgram {
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
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            // One user-data SGPR; LDS in blocks of 512 bytes (bits 15-23).
            rsrc2: 1 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

pub(super) fn lowered(
    native: &NativeProgram,
    host_wave32: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
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
        view_marks: false,
        host_wave32,
        view_remap: false,
    })
}

pub(super) fn shader(l: lower::Lowered) -> super::HostShader {
    super::HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: Vec::new(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    }
}

/// Hardware case from VIctory-NVIDIA #30, including groups with a partial wave.
/// Every invocation stores one at its console flat ID. An unwritten dword
/// exposes a missing thread even when another thread stores twice at the same ID.
#[test]
fn two_dimensional_groups_cover_their_console_thread_ids() {
    let _guard = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    for (shift, height) in [(5, 2), (5, 3), (4, 3), (3, 7)] {
        let width = 1u32 << shift;
        let threads = width * height;
        let mut words = PROLOGUE.to_vec();
        words.extend([
            0xD765_0003, 0x0001_007E,  // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
            0x3408_0280 | shift,       // v_lshlrev_b32 v4, shift, v1
            0x4A08_0104,               // v_add_nc_u32 v4, v4, v0
            0x3420_0882,               // v_lshlrev_b32 v16, 2, v4
            0x7E0A_0281,               // v_mov_b32 v5, 1
        ]);
        words.extend(store(5, 0));
        words.push(S_ENDPGM);
        let mut program = native(&words, false, threads, 0);
        if let StageIo::Cs(cs) = &mut program.io {
            cs.threads = [width, height, 1];
            cs.rsrc2 |= 1 << 11; // thread IDs x and y in v0 and v1
        }
        let program = shader(lowered(&program, true).expect("lower wave32 program"));
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut program.dxil.clone()).expect("validate thread allocation");
        }
        for (adapter, host) in &hosts {
            let (root, pso) = host.compute_pipeline(&program).expect("create wave32 pipeline");
            let ud = vec![0u32; program.user_data_dwords().max(1) as usize];
            let views = [
                View::Texture(1, 1, &[[0.0; 4]]),
                View::Buffer { data: &[0; 4], stride: Stride::Structured(16) },
                View::RawUav(threads),
            ];
            let out = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1; 3],
            }).expect("read console thread IDs");
            let written = out[0][..threads as usize].iter().filter(|&&word| word == 1).count();
            eprintln!("{adapter}: {width}x{height}: {written}/{threads} console threads wrote their output");
            assert_eq!(written, threads as usize, "{width}x{height} on {adapter}");
        }
    }
}

fn refusal(words: &[u32], threads: u32, lds_bytes: u32) -> String {
    match lowered(&native(words, true, threads, lds_bytes), true) {
        Ok(_) => "translated".into(),
        Err(e) => e.to_string(),
    }
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: [(Vec<u32>, &[&str]); 34] = [
        (
            sum16(),
            &[
                "v_cvt_f16_u16 v3, v1",
                "s_or_saveexec_b64 s[2:3], -1",
                "v_cndmask_b32 v4, 0, v3, s[2:3]",
                "v_add_f16",
                "row_mirror",
                "v_permlanex16_b32 v5, v4, 0, 0",
                "v_add_f16 v4, v5, v4",
                "v_readlane_b32 s4, v4, 31",
                "v_readlane_b32 s5, v4, 63",
                "v_add_f16 v6, s4, v7",
            ],
        ),
        (
            whole_group_around(true),
            &[
                "s_cmp_eq_u32 s5, 0",
                "s_cbranch_scc1",
                "v_cmp_gt_u32 vcc, 10, v1",
                "s_andn2_b64 exec, exec, vcc",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s13, v1",
            ],
        ),
        (
            whole_group_into(false),
            &[
                "s_cmp_eq_u32 s5, 1",
                "s_cbranch_scc1",
                "v_mov_b32 v2, lit(0x0000dead)",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_readfirstlane_b32 s13, v1",
            ],
        ),
        (
            sum_kept_by_lanes(),
            &[
                "v_add_nc_u32 v8, 1, v0",
                "s_or_saveexec_b64 s[2:3], -1",
                "v_cndmask_b32 v2, 0, v8, s[2:3]",
                "quad_perm:[1,0,3,2]",
                "quad_perm:[2,3,0,1]",
                "row_half_mirror",
                "row_mirror",
                "v_permlanex16_b32 v3, v2, 0, 0",
                "v_add_nc_u32 v2, v3, v2",
                "v_readlane_b32 s4, v2, 32",
                "v_add_nc_u32 v2, s4, v2",
                "v_readlane_b32 s6, v2, 0",
                "s_mov_b64 exec, s[2:3]",
                "v_cmp_gt_u32 vcc, 20, v1",
                "v_mov_b32 v2, 7",
            ],
        ),
        (
            permlanes(),
            &[
                "v_mul_u32_u24 v2, 3, v0",
                "s_mov_b32 s10, lit(0x89abcdef)",
                "v_permlane16_b32 v3, v2, s10, s11",
                "v_permlanex16_b32 v4, v2, s10, s11",
                "v_and_b32 v9, 1, v1",
                "v_cmp_eq_u32 vcc, 0, v9",
                "v_permlane16_b32 v6, v2, s10, s11 bound_ctrl:1",
                "v_permlane16_b32 v7, v2, s10, s11 fi:1",
            ],
        ),
        (
            lanes_that_end(),
            &[
                "v_cmp_gt_u32 vcc, 10, v1",
                "s_andn2_b64 exec, exec, vcc",
                "s_cbranch_execz",
                "v_cmp_gt_u32 vcc, 40, v1",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_readfirstlane_b32 s13, v1",
            ],
        ),
        (
            first_lane_after_a_loop(),
            &[
                "s_mov_b32 s20, 0",
                "s_add_u32 s20, s20, 1",
                "v_mov_b32 v2, s20",
                "v_cmp_lt_u32 vcc, v11, v12",
                "s_and_b64 exec, exec, vcc",
                "s_cbranch_execnz",
                "v_readfirstlane_b32 s12, v2",
            ],
        ),
        (
            wide_loop(true),
            &[
                "v_lshrrev_b32 v12, 4, v1",
                "v_and_b32 v13, 1, v1",
                "v_and_b32 v12, 3, v12",
                "ds_write_b32 v6, v8",
                "s_mov_b64 s[26:27], exec",
                "v_readfirstlane_b32 s12, v1",
                "v_cmp_le_u32 vcc, 32, v1",
                "s_bcnt1_i32_b64 s13, vcc",
                "ds_write_b32 v6, v11",
                "ds_read_b32 v9, v7",
                "v_cmp_lt_u32 vcc, v11, v12",
                "s_and_b64 exec, exec, vcc",
                "s_cbranch_execnz",
                "s_mov_b64 exec, s[26:27]",
            ],
        ),
        (
            lds_tree_step(true),
            &[
                "v_cmp_gt_u32 vcc, 16, v0",
                "v_add_nc_u32 v4, 64, v2",
                "ds_read_b32 v5, v4",
                "v_add_nc_u32 v5, v5, v1",
                "ds_write_b32 v2, v5",
                "ds_read_b32 v6, v3",
            ],
        ),
        (
            lds_tree_step(false),
            &[
                "v_cmp_le_u32 vcc, 48, v0",
                "v_add_nc_u32 v4, lit(0xffffffc0), v2",
                "ds_write_b32 v2, v5",
            ],
        ),
        (
            mask_kept_by_a_shift(true),
            &[
                "v_xor_b32 v3, 5, v0",
                "s_cmp_eq_u32 s5, 0",
                "s_cbranch_scc0",
                "v_cmp_gt_u32 vcc, v3, v4",
                "v_mov_b32 v4, vcc_hi",
                "v_add_nc_u32 v4, v2, v2 row_shr:15",
            ],
        ),
        (
            masks_and_first_lane(),
            &[
                "v_cmp_gt_u32 vcc, 40, v1",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_mov_b32 v3, vcc_lo",
                "v_mov_b32 v4, vcc_hi",
                "v_readfirstlane_b32 s13, v0",
                "buffer_store_dword v5, v16, s[40:43], 0 offen offset:12",
            ],
        ),
        (
            first_lane_in_exec(),
            &[
                "v_cmp_lt_u32 vcc, 35, v1",
                "s_and_saveexec_b64 s[20:21], vcc",
                "v_readfirstlane_b32 s14, v0",
                "s_mov_b64 exec, s[20:21]",
                "v_mov_b32 v6, s14",
            ],
        ),
        (
            wave_sums(),
            &[
                "s_or_saveexec_b64 s[0:1], -1",
                "s_mov_b64 exec, -1\nv_readlane_b32 s3, v1, 63",
                "v_mov_b32 v9, s3",
            ],
        ),
        (
            first_lane_in_a_branch(20),
            &[
                "v_cmp_gt_u32 vcc, 20, v0",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s12, v0",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            first_lane_of_a_buffer_load_in_a_branch(0, 20),
            &[
                "s_cbranch_execz",
                "buffer_load_dword v4, v0, s[44:47], 0 offen",
                "v_readfirstlane_b32 s12, v4",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            mask_in_a_branch(20),
            &[
                "s_cbranch_execz",
                "v_cmp_gt_u32 vcc, 10, v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            sum_under_a_uniform_branch(),
            &[
                "s_cmp_eq_u32 s5, 0",
                "s_cbranch_scc0",
                "v_readlane_b32 s3, v1, 63",
            ],
        ),
        (
            lds_after_a_spin(),
            &[
                "v_mov_b32 v5, lit(0x00000bad)",
                "v_cmp_gt_u32 vcc, 32, v0",
                "s_mov_b32 s22, lit(0x00004e20)",
                "s_sub_u32 s22, s22, 1",
                "s_cmp_lg_u32 s22, 0",
                "s_cbranch_scc1",
                "ds_write_b32 v3, v5",
                "ds_write_b32 v2, v1",
                "ds_read_b32 v4, v3",
            ],
        ),
        (
            lds_across(),
            &[
                "v_add_nc_u32 v1, 1, v0",
                "ds_write_b32 v2, v1",
                "v_xor_b32 v3, lit(0x00000080), v2",
                "ds_read_b32 v4, v3",
            ],
        ),
        (
            lds_across_a_branch(),
            &[
                "ds_write_b32 v2, v5",
                "v_cmp_gt_u32 vcc, 48, v0",
                "s_and_saveexec_b64 s[20:21], vcc",
                "s_cbranch_execz",
                "s_mov_b64 exec, s[20:21]",
                "ds_read_b32 v4, v3",
            ],
        ),
        (
            lds_written_and_read_in_a_branch(),
            &[
                "ds_write_b32 v2, v5",
                "s_cbranch_execz",
                "ds_write_b32 v2, v1",
                "ds_read_b32 v4, v3",
            ],
        ),
        (
            lds_written_twice_in_a_branch(),
            &[
                "ds_write_b32 v2, v1",
                "ds_write_b32 v3, v1",
                "ds_read_b32 v4, v2",
            ],
        ),
        (
            wide_branch(8, true),
            &[
                "v_mul_u32_u24 v6, 3, v0",
                "v_cmp_le_u32 vcc, 8, v1",
                "v_add_nc_u32 v6, lit(0x000003e8), v6",
                "v_readfirstlane_b32 s12, v0",
                "v_cmp_gt_u32 vcc, 40, v1",
                "s_bcnt1_i32_b64 s13, vcc",
                "v_readlane_b32 s14, v0, 50",
                "v_readlane_b32 s15, v6, 7",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            lane_reads(),
            &[
                "v_readlane_b32 s12, v0, 40",
                "s_mov_b32 s20, 63",
                "v_readlane_b32 s13, v0, s20",
                "v_readlane_b32 s14, v0, 5",
                "buffer_store_dword v4, v16, s[40:43], 0 offen offset:8",
            ],
        ),
        (
            lane_read_in_a_branch(),
            &[
                "v_cmp_gt_u32 vcc, 20, v0",
                "s_and_saveexec_b64 s[20:21], vcc",
                "s_cbranch_execz",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            lane_read_under_a_uniform_branch(),
            &["s_cmp_eq_u32 s5, 0", "s_cbranch_scc0"],
        ),
        (
            small_group(),
            &[
                "v_cmp_gt_u32 vcc, 10, v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_mov_b32 v7, vcc_hi",
                "v_readlane_b32 s13, v0, 5",
                "s_or_saveexec_b64 s[0:1], -1",
                "v_add_nc_u32 v1, v1, v1 row_shr:8",
                "v_readlane_b32 s2, v1, 47",
                "s_mov_b64 exec, s[0:1]",
                "buffer_store_dword v8, v16, s[40:43], 0 offen offset:12",
            ],
        ),
        (
            every_id([6, 6, 1]),
            &[
                "v_mul_u32_u24 v3, lit(0x00000006), v2",
                "v_add_nc_u32 v3, v1, v3",
                "v_mul_u32_u24 v3, lit(0x00000006), v3",
                "v_add_nc_u32 v2, v0, v3",
                "v_lshlrev_b32 v16, 2, v2",
                "v_mov_b32 v3, 1",
            ],
        ),
        (
            in_halves(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_mbcnt_hi_u32_b32 v1, -1, v1",
                "ds_swizzle_b32 v2, v0",
                "v_xor_b32 v4, 41, v0",
                "ds_bpermute_b32 v3, v4, v0",
                "v_writelane_b32 v5, s8, 40",
                "v_cndmask_b32 v6, 0, 1, s[20:21]",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2]",
                "buffer_store_dword v7, v16, s[40:43], 0 offen offset:24",
            ],
        ),
        (
            dpp_row_mask(),
            &[
                "v_mov_b32 v7, -1",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc",
                "buffer_store_dword v7, v16, s[40:43], 0 offen",
            ],
        ),
        (
            first_lane_of_the_thread_id(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_readfirstlane_b32 s12, v0",
                "v_mov_b32 v2, s12",
            ],
        ),
        (
            first_lane_of_a_buffer_load(3),
            &[
                "s_load_dwordx4 s[44:47], s[4:5], 0x000020",
                "v_mov_b32 v3, 4",
                "buffer_load_dword v4, v3, s[44:47], 0 offen",
                "s_waitcnt vmcnt(0)",
                "v_readfirstlane_b32 s12, v4",
                "v_mov_b32 v5, s12",
            ],
        ),
        (
            first_lane_alike(),
            &[
                "v_lshrrev_b32 v2, 6, v0",
                "v_readfirstlane_b32 s12, v2",
                "v_mov_b32 v3, s12",
                "buffer_store_dword v1, v16, s[40:43], 0 offen offset:4",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// In halves the program declares waves of 32; without, the console's 64.
#[test]
fn a_wave64_program_reading_lanes_declares_32_lane_waves_in_halves() {
    let n = native(&in_halves(), true, 64, 0);
    let halves = lowered(&n, true).expect("halves");
    let whole = lowered(&n, false).expect("one wave64");
    assert_eq!(halves.wave_size, Some(32));
    assert_eq!(whole.wave_size, Some(64));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut halves.dxil.clone())
            .expect("DXIL validation failed");
    }
    // A wave32 program is the console's either way.
    let n32 = native(&in_halves(), false, 32, 0);
    assert_eq!(lowered(&n32, true).expect("wave32").wave_size, Some(32));
}

/// Reads across the halves are refused where they are made, with a reason
/// naming the host's waves; the same programs translate for 64-lane hosts.
#[test]
fn reads_across_32_lane_halves_are_refused_with_their_reason() {
    let readlane = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xD760_000C,
            0x0001_5100, // v_readlane_b32 s12, v0, 40
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    // A lane read where every thread of the group meets is traded between
    // the halves; the same program translates for 64-lane hosts.
    assert_eq!(refusal(&readlane, 64, 0), "translated");
    assert!(lowered(&native(&readlane, true, 64, 0), false).is_ok());
    // Inside a branch only lanes below 20 take, the halves cannot meet: the
    // branch runs wave-wide.
    assert_eq!(refusal(&lane_read_in_a_branch(), 64, 0), "translated");
    // Under a branch on a uniform condition both halves of a console wave
    // go alike, but two console waves of one group need not: in a group of
    // two, the branch runs wave-wide.
    let uniform = lane_read_under_a_uniform_branch();
    assert_eq!(refusal(&uniform, 64, 0), "translated");
    assert_eq!(refusal(&uniform, 128, 0), "translated");

    let first = standalone_first_lane();
    let standalone = native(&first, true, 64, 0);
    assert_eq!(
        lowered(&standalone, false).expect("native waves").wave_size,
        Some(64)
    );
    assert_eq!(refusal(&first, 64, 0), "translated");
    assert_eq!(refusal(&first_lane_of_the_thread_id(), 32, 0), "translated");

    // With console lanes the program runs in halves, where lane 32 is the
    // upper host wave's first lane: a value that differs between the halves
    // (the thread id) is traded where every thread meets. In a group of 32
    // there is no upper half.
    assert_eq!(refusal(&first_lane_of_the_thread_id(), 64, 0), "translated");
    assert_eq!(refusal(&first_lane_of_the_thread_id(), 32, 0), "translated");
    // A first-lane read nothing uses costs nothing.
    let unused_first = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x7E18_0500, // v_readfirstlane_b32 s12, v0
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    assert_eq!(refusal(&unused_first, 64, 0), "translated");
    // A buffer dword loaded at an offset the same in every lane, through a
    // scalar descriptor, is one value: its first lane is the console's, even
    // inside a branch lanes of both halves take, with no trade. At an offset
    // that differs (the thread id) the branch runs wave-wide and the halves
    // trade.
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(3), 64, 0),
        "translated"
    );
    for offset in [3, 0] {
        let words = first_lane_of_a_buffer_load_in_a_branch(offset, 40);
        assert_eq!(refusal(&words, 64, 0), "translated", "{offset}");
    }
    assert_eq!(
        barriers(&first_lane_of_a_buffer_load_in_a_branch(3, 40), 64, 0),
        0
    );

    // A value read across halves that nothing uses costs nothing.
    let unused = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xD760_000C,
            0x0001_5100, // v_readlane_b32 s12, v0, 40
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    assert_eq!(refusal(&unused, 64, 0), "translated");
}

/// A group of one console wave64 shares LDS without barriers; in halves its
/// accesses are put in the console's order with group barriers. A region
/// only some lanes run cannot be ordered inside, so one that both writes
/// and reads, or writes twice, runs wave-wide, where it can. A group of 32
/// threads or fewer is one host wave, and one of several console waves
/// keeps the console's barriers.
#[test]
fn lds_of_a_one_wave64_group_is_ordered_in_halves() {
    let lds = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0xD834_0000,
            0x0000_0102, // ds_write_b32 v2, v1
            S_WAITCNT_LGKM,
            0xD8D8_0000,
            0x0300_0002, // ds_read_b32 v3, v2
            S_WAITCNT_LGKM,
        ],
        &store(3, 0),
        &[S_ENDPGM],
    ]
    .concat();
    for threads in [64, 48, 33, 32] {
        assert_eq!(refusal(&lds, threads, 256), "translated", "{threads}");
    }
    for threads in [128, 256, 512] {
        let bytes = 4 * threads;
        assert_eq!(refusal(&lds, threads, bytes), "translated");
        assert!(barriers(&lds, threads, bytes) > 0, "{threads} threads");
    }
    for words in [
        lds_written_and_read_in_a_branch(),
        lds_written_twice_in_a_branch(),
    ] {
        assert_eq!(refusal(&words, 64, 256), "translated");
    }
}

/// Every lane writes `tid + 1` at its own dword, then reads the dword of
/// lane `tid ^ 32`, in the other half.
fn lds_across() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        0x3A06_04FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v2
        0xD8D8_0000,
        0x0400_0003, // ds_read_b32 v4, v3
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// Every lane writes 0 at its own dword; lanes below 48 then write `tid + 1`
/// there inside a branch, and after it every lane reads the dword of lane
/// `tid ^ 32`, in the other half.
fn lds_across_a_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD834_0000,
        0x0000_0502, // ds_write_b32 v2, v5
        0x7D88_00B0, // v_cmp_gt_u32 vcc, 48, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0002, // s_cbranch_execz 2
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0x3A06_04FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v2
        0xD8D8_0000,
        0x0400_0003, // ds_read_b32 v4, v3
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// Lanes below 32 first spin 20,000 times; then every lane writes a poison
/// dword into the dword of lane `tid ^ 32` (in the other half), its own
/// `tid + 1` into its own dword, and reads lane `tid ^ 32`'s. On the console
/// all poison is written before any lane's own dword, so every lane reads
/// its partner's `tid + 1`. Two host waves out of step (the upper one ahead)
/// would read poison: the barriers are what keep them in the console's order.
fn lds_after_a_spin() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0x3A06_04FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v2
        0x7E0A_02FF,
        0x0000_0BAD, // v_mov_b32 v5, 0xbad
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0006, // s_cbranch_execz 6
        0xBE96_03FF,
        0x0000_4E20, // s_mov_b32 s22, 20000
        0x4A0C_0C81, // v_add_nc_u32 v6, 1, v6
        0x8096_8116, // s_sub_u32 s22, s22, 1
        0xBF07_8016, // s_cmp_lg_u32 s22, 0
        0xBF85_FFFC, // s_cbranch_scc1 -4
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0503, // ds_write_b32 v3, v5
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        0xD8D8_0000,
        0x0400_0003, // ds_read_b32 v4, v3
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(4, 0));
    w.extend(store(6, 4));
    w.push(S_ENDPGM);
    w
}

/// Every thread writes 0 at its dword; lanes below 48 then write `tid + 1`
/// there and read lane `tid ^ 32`'s, both inside one branch.
fn lds_written_and_read_in_a_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD834_0000,
        0x0000_0502, // ds_write_b32 v2, v5
        0x7D88_00B0, // v_cmp_gt_u32 vcc, 48, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0007, // s_cbranch_execz 7
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        0x3A06_04FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v2
        0xD8D8_0000,
        0x0400_0003, // ds_read_b32 v4, v3
        S_WAITCNT_LGKM,
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// Lanes below 48 write two dwords in two instructions inside one branch
/// (their own and lane `tid ^ 32`'s); after the join every lane reads its
/// own.
fn lds_written_twice_in_a_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0x3A06_04FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v2
        0x7D88_00B0, // v_cmp_gt_u32 vcc, 48, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0004, // s_cbranch_execz 4
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        0xD834_0000,
        0x0000_0103, // ds_write_b32 v3, v1
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD8D8_0000,
        0x0400_0002, // ds_read_b32 v4, v2
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// [`lds_across`], [`lds_across_a_branch`] and [`lds_after_a_spin`] in a
/// group of 64 on every
/// hardware adapter, in halves (two host waves, put in order by barriers)
/// and as one wave64 where the adapter runs those: every lane reads what the
/// other half's lane wrote, as on the console.
#[test]
fn one_wave_lds_crosses_the_halves_in_order_on_every_gpu() {
    let across: Vec<u32> = (0..64).map(|l| (l ^ 32) + 1).collect();
    let mut failed = differences_with_lds(&lds_across(), 64, 256, 1, &across);
    let branch: Vec<u32> = (0..64)
        .map(|l| if l ^ 32 < 48 { (l ^ 32) + 1 } else { 0 })
        .collect();
    failed.extend(differences_with_lds(
        &lds_across_a_branch(),
        64,
        256,
        1,
        &branch,
    ));
    let spin: Vec<u32> = (0..64)
        .flat_map(|l| [(l ^ 32) + 1, if l < 32 { 20_000 } else { 0 }])
        .collect();
    failed.extend(differences_with_lds(&lds_after_a_spin(), 64, 256, 2, &spin));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane reads the dword written by the opposite half of its console
/// wave. A passing dispatch requires ordering between the two host waves.
#[test]
fn straight_line_lds_exchanges_data_across_two_host_waves() {
    let mut words = PROLOGUE.to_vec();
    words.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0: output byte index
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0: LDS byte index
        0xD765_0001, 0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000, 0x0000_0002, // ds_write_b32 v2, v0
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000, 0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
    ]);
    words.extend(store(3, 0));
    words.push(S_ENDPGM);
    for threads in [64, 128, 256, 512] {
        let want: Vec<_> = (0..threads).map(|lane| lane ^ 32).collect();
        let bytes = (4 * threads).max(512);
        assert!(barriers(&words, threads, bytes) > 0, "{threads} threads");
        let wrong = differences_with_lds(&words, threads, bytes, 1, &want);
        assert!(wrong.is_empty(), "{wrong:?}");
    }
}

/// Runs `words`, one group of `threads` of a wave64 program whose lanes store
/// `per_lane` dwords at `l * 32`, on every hardware adapter: in halves, and
/// as one wave64 where the adapter runs 64-lane waves. A line per lane whose
/// dwords differ from `want` (the console's, `per_lane` per lane).
fn differences(words: &[u32], threads: u32, per_lane: usize, want: &[u32]) -> Vec<String> {
    differences_with_lds(words, threads, 0, per_lane, want)
}

/// [`differences`] for a program with `lds_bytes` of LDS.
fn differences_with_lds(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    per_lane: usize,
    want: &[u32],
) -> Vec<String> {
    compared(words, threads, lds_bytes, per_lane, want, true)
}

/// [`differences`] in halves only: for what the console's own form gets
/// wrong on the host for another reason.
fn differences_in_halves(
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
) -> Vec<String> {
    compared(words, threads, 0, per_lane, want, false)
}

/// [`differences_with_lds`], with the console's own form (one wave64) only
/// where `wave64`.
fn compared(
    words: &[u32],
    threads: u32,
    lds_bytes: u32,
    per_lane: usize,
    want: &[u32],
    wave64: bool,
) -> Vec<String> {
    let n = native(words, true, threads, lds_bytes);
    let mut forms = vec![("halves", shader(lowered(&n, true).expect("halves")))];
    if wave64 {
        forms.push(("wave64", shader(lowered(&n, false).expect("one wave64"))));
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(threads * 8),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
    }
    let mut failed = Vec::new();
    for (adapter, host) in &hosts {
        let wave32 = host.wave32().unwrap_or(false);
        for (form, shader) in &forms {
            let label = format!("{adapter} ({form})");
            let Ok((root, pso)) = host.compute_pipeline(shader) else {
                // Only the console's own form may be refused, and only
                // where the host's waves are 32 lanes.
                assert!(*form == "wave64" && wave32, "{label}: no pipeline");
                eprintln!("{label}: no pipeline for 64-lane waves; skipped");
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
            let wrong: Vec<String> = out[0]
                .chunks(8)
                .zip(want.chunks(per_lane))
                .enumerate()
                .filter(|(_, (got, want))| got[..per_lane] != **want)
                .map(|(l, (got, want))| {
                    format!(
                        "{label}: lane {l}: {:?}, the console {want:?}",
                        &got[..per_lane]
                    )
                })
                .collect();
            eprintln!(
                "{label}: {} lanes of {threads} differ from the console",
                wrong.len()
            );
            failed.extend(wrong);
        }
    }
    failed
}

/// A DPP row mask names the console's rows 0-3: with `row_mask:0xc` only
/// rows 2 and 3 (lanes 32-63) take `l ^ 1`, rows 0 and 1 keep -1.
fn dpp_row_mask() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0E_02C1, // v_mov_b32 v7, -1
        0x7E0E_02FA,
        0xCF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc (DPP)
    ]);
    w.extend(store(7, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn dpp_row_masks_name_the_console_rows_on_every_gpu() {
    let want: Vec<u32> = (0..64)
        .map(|l| if l >= 32 { l ^ 1 } else { u32::MAX })
        .collect();
    let failed = differences(&dpp_row_mask(), 64, 1, &want);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_readfirstlane_b32` of the thread id, with console lanes (`v_mbcnt`):
/// the console's first lane is lane 0 of the wave.
fn first_lane_of_the_thread_id() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readfirstlane_b32` of a dword loaded from the buffer at table dword 8,
/// at the byte offset in `v<offset>`: `v3` holds 4 in every lane, `v0` the
/// thread id.
fn first_lane_of_a_buffer_load(offset: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E06_0284, // v_mov_b32 v3, 4
        0xE030_1000,
        0x800B_0400 | offset, // buffer_load_dword v4, v<offset>, s[44:47], 0 offen
        0xBF8C_3F70,          // s_waitcnt vmcnt(0)
        0x7E18_0504,          // v_readfirstlane_b32 s12, v4
        0x7E0A_020C,          // v_mov_b32 v5, s12
    ]);
    w.extend(store(5, 0));
    w.push(S_ENDPGM);
    w
}

/// Per thread `l`, three dwords at `l * 32`: the thread ids of lanes 40, 63
/// (named by an SGPR) and 5 of its console wave, through `v_readlane_b32`.
fn lane_reads() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x7E04_020C, // v_mov_b32 v2, s12
        0xBE94_03BF, // s_mov_b32 s20, 63
        0xD760_000D,
        0x0000_2900, // v_readlane_b32 s13, v0, s20
        0x7E06_020D, // v_mov_b32 v3, s13
        0xD760_000E,
        0x0001_0B00, // v_readlane_b32 s14, v0, 5
        0x7E08_020E, // v_mov_b32 v4, s14
    ]);
    for (vdata, slot) in [(2, 0), (3, 1), (4, 2)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// Lane 40's thread id, read inside a branch only threads below 20 take.
fn lane_read_in_a_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0003, // s_cbranch_execz 3
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x7E04_020C, // v_mov_b32 v2, s12
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// Lane 40's thread id, read under a branch on a scalar compare.
fn lane_read_under_a_uniform_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBF06_8005, // s_cmp_eq_u32 s5, 0
        0xBF84_0003, // s_cbranch_scc0 3
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// [`lane_reads`] on every hardware adapter, in halves (traded between the
/// two host waves) and as one wave64 where the adapter runs those: in groups
/// of one and of two console waves, every lane gets its console wave's lanes.
#[test]
fn lane_reads_across_halves_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let base = l & !63;
                [base + 40, base + 63, base + 5]
            })
            .collect();
        failed.extend(differences(&lane_reads(), threads, 3, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Per thread `l` of a group of 128 (two console wave64s), two dwords at
/// `l * 32`: the first lane of `l >> 6` (one value per console wave, the
/// same in both its halves) and the lane `v_mbcnt` counts.
fn first_lane_alike() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E18_0502, // v_readfirstlane_b32 s12, v2
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(1, 4));
    w.push(S_ENDPGM);
    w
}

/// A single first-lane read of the entry thread id, with no other wave opcode.
fn standalone_first_lane() -> Vec<u32> {
    [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x7E18_0500, // v_readfirstlane_b32 s12, v0
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat()
}

/// First-lane reads kept in halves give the console's value on every
/// hardware adapter: the thread id's in a group of 32 (one host wave holds
/// the whole console wave), and one value per console wave in a group of
/// 128, where each half reads its console wave's.
#[test]
fn first_lanes_kept_in_halves_match_the_console_on_every_gpu() {
    let mut failed = differences(&first_lane_of_the_thread_id(), 32, 1, &[0; 32]);
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).map(|id| id & !63).collect();
        failed.extend(differences(&standalone_first_lane(), threads, 1, &want));
    }
    let want: Vec<u32> = (0..128).flat_map(|l| [l >> 6, l & 63]).collect();
    failed.extend(differences(&first_lane_alike(), 128, 2, &want));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A group of `[x, y, z]` threads (thread ids in v0-v2) running `words`.
fn native_shaped(words: &[u32], [x, y, z]: [u32; 3]) -> NativeProgram {
    let mut n = native(words, true, x * y * z, 0);
    if let StageIo::Cs(cs) = &mut n.io {
        cs.threads = [x, y, z];
        cs.rsrc2 |= 2 << 11; // TIDIG_COMP_CNT 2: v1 holds y, v2 holds z
    }
    n
}

/// Each thread of a group of `[x, y, z]` threads stores 1 at its flat id
/// `(z * y_n + y) * x_n + x` (`x_n`, `y_n` the group's width and height),
/// with console lanes (`v_mbcnt`): a thread id two threads share leaves
/// another one unstored.
fn every_id([x, y, _]: [u32; 3]) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xD765_0005,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
        0x1606_04FF,
        y,           // v_mul_u32_u24 v3, y, v2
        0x4A06_0701, // v_add_nc_u32 v3, v1, v3
        0x1606_06FF,
        x,           // v_mul_u32_u24 v3, x, v3
        0x4A04_0700, // v_add_nc_u32 v2, v0, v3
        0x3420_0482, // v_lshlrev_b32 v16, 2, v2
        0x7E06_0281, // v_mov_b32 v3, 1
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

/// A private LDS dword for thread tickets lets tiled host waves cover
/// the console row. A full 32 KiB allocation leaves room only for the
/// shapes whose first host lanes already start distinct 32-lane runs.
#[test]
fn tiled_groups_need_room_for_thread_tickets_in_halves() {
    let translate = |shape, lds_bytes| {
        let mut n = native_shaped(&every_id(shape), shape);
        if let StageIo::Cs(cs) = &mut n.io {
            cs.rsrc2 = (cs.rsrc2 & !(0x1ff << 15)) | ((lds_bytes / 512) << 15);
        }
        lowered(&n, true)
    };
    for shape in [[32, 2, 1], [64, 4, 1], [32, 2, 2], [6, 6, 1], [24, 2, 1]] {
        assert!(translate(shape, 0u32).is_ok(), "{shape:?}");
        let error = translate(shape, 32u32 * 1024).err().expect("tile without a ticket");
        assert!(error.to_string().contains("may tile"), "{shape:?}: {error}");
    }
    for shape in [[8, 8, 1], [16, 2, 1], [4, 4, 4], [32, 1, 2], [48, 1, 2]] {
        assert!(translate(shape, 32u32 * 1024).is_ok(), "{shape:?}");
    }
}

/// The groups halves keep give every thread its own id on every hardware
/// adapter, in halves and as one wave64 where the adapter runs those.
#[test]
fn every_thread_keeps_its_own_id_in_halves_on_every_gpu() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
    }
    let mut failed = Vec::new();
    for shape in [
        [8, 8, 1],
        [16, 2, 1],
        [4, 4, 4],
        [64, 1, 1],
        [32, 1, 2],
        [48, 1, 2],
        [6, 1, 6],
        [256, 1, 4],
    ] {
        let n = native_shaped(&every_id(shape), shape);
        let threads = shape.iter().product::<u32>();
        for (form, halves) in [("halves", true), ("wave64", false)] {
            let s = shader(lowered(&n, halves).expect("lowers"));
            for (adapter, host) in &hosts {
                let label = format!("{adapter} {shape:?} ({form})");
                let Ok((root, pso)) = host.compute_pipeline(&s) else {
                    assert!(
                        !halves && host.wave32().unwrap_or(false),
                        "{label}: no pipeline"
                    );
                    continue;
                };
                let ud = vec![0u32; s.user_data_dwords().max(1) as usize];
                let views = [
                    View::Texture(1, 1, &[[0.0; 4]]),
                    View::Buffer {
                        data: &[0; 4],
                        stride: Stride::Structured(16),
                    },
                    View::RawUav(threads),
                ];
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
                let stored = out[0]
                    .iter()
                    .take(threads as usize)
                    .filter(|&&v| v == 1)
                    .count();
                eprintln!("{label}: {stored} of {threads} ids stored");
                if stored != threads as usize {
                    failed.push(format!("{label}: {stored} of {threads} ids stored"));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
}

pub(super) const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
}

pub(super) const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
    (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

pub(super) const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

pub(super) const fn dpp16(vsrc0: u8, ctrl: u16) -> u32 {
    0xFF << 24 | (ctrl as u32) << 8 | vsrc0 as u32
}

pub(super) const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xD400_0000 | (op as u32) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
    ]
}

/// The exclusive prefix sum of `v0` over the active lanes, in `v2`, as the
/// console compiler expands `WavePrefixSum` in a wave64 (the section of
/// `wave::tests::exec_minus_two_masks_lane_zero_and_enforces_dpp_guard`):
/// row sums by DPP, the rows below added through lane reads, then shifted
/// by one lane. It writes `s[0:2]` and `v1`-`v5`.
pub(super) fn prefix_sum_section() -> Vec<u32> {
    use crate::ir::EXEC;
    let mut w = vec![sop1(0, 0x25, 193)]; // s_or_saveexec_b64 s[0:1], -1
    w.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s[0:1]
    for shift in [1u16, 2, 4, 8] {
        w.push(vop2(1, 0x25, 0xFA, 1)); // v_add_nc_u32 v1, v1, v1 row_shr:shift
        w.push(dpp16(1, 0x110 + shift));
    }
    // Each row adds the sums of the rows below it (lanes 15, 31, 47).
    for (lane, lo, hi) in [
        (15u16, 0xFFFF_0000u32, 0u32),
        (31, 0, 0x0000_FFFF),
        (47, 0, 0xFFFF_0000),
    ] {
        w.extend(vop3(2, 0x360, 257, 128 + lane, 0)); // v_readlane_b32 s2, v1, lane
        w.extend([sop1(EXEC, 0x03, 255), lo, sop1(EXEC + 1, 0x03, 255), hi]);
        w.push(vop2(1, 0x25, 2, 1)); // v_add_nc_u32 v1, s2, v1
    }
    w.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1
    w.push(vop1(3, 0x01, 257)); // v_mov_b32 v3, v1
    w.push(vop1(3, 0x01, 0xFA)); // v_mov_b32 v3, v1 row_shr:1
    w.push(dpp16(1, 0x111));
    // The first lane of each row takes the last lane of the row below.
    for (lane, lo, hi) in [
        (15u16, 0x0001_0000u32, 0u32),
        (31, 0, 1),
        (47, 0, 0x0001_0000),
    ] {
        w.extend(vop3(2, 0x360, 257, 128 + lane, 0)); // v_readlane_b32 s2, v1, lane
        w.extend([sop1(EXEC, 0x03, 255), lo, sop1(EXEC + 1, 0x03, 255), hi]);
        w.push(vop1(3, 0x01, 2)); // v_mov_b32 v3, s2
    }
    w.push(sop1(EXEC, 0x04, 193)); // s_mov_b64 exec, -1
    w.push(vop1(2, 0x01, 128)); // v_mov_b32 v2, 0
    w.push(vop1(5, 0x01, 135)); // v_mov_b32 v5, 7
    w.push(sop1(EXEC, 0x04, 194)); // s_mov_b64 exec, -2
    w.push(vop1(2, 0x01, 259)); // v_mov_b32 v2, v3
    w.push(sop1(EXEC, 0x04, 0)); // s_mov_b64 exec, s[0:1]
    w
}

/// Per thread `l`, four dwords at `l * 32`: the exclusive prefix sum of the
/// thread ids below it, the number of threads below 10 (`s_bcnt1_i32_b64`
/// of a compare's mask), that mask's high dword and lane 5's thread id.
fn small_group() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,         // v_lshlrev_b32 v16, 5, v0
        0x7D88_008A,         // v_cmp_gt_u32 vcc, 10, v0
        sop1(12, 0x10, 106), // s_bcnt1_i32_b64 s12, vcc
        vop1(6, 0x01, 12),   // v_mov_b32 v6, s12
        vop1(7, 0x01, 107),  // v_mov_b32 v7, vcc_hi
    ]);
    w.extend(vop3(13, 0x360, 256, 128 + 5, 0)); // v_readlane_b32 s13, v0, 5
    w.push(vop1(8, 0x01, 13)); // v_mov_b32 v8, s13
    w.extend(prefix_sum_section());
    for (vdata, slot) in [(2, 0), (6, 1), (7, 2), (8, 3)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// In a group of 32 threads or fewer one host wave holds the whole console
/// wave: masks read as data, wave prefix sums and a lane read of a constant
/// lane below 32 translate in halves with no trade; a lane read past the
/// group is refused.
#[test]
fn a_group_one_host_wave_holds_reads_nothing_across_halves() {
    assert_eq!(refusal(&small_group(), 32, 0), "translated");
    assert_eq!(refusal(&small_group(), 16, 0), "translated");
    let readlane = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xD760_000C,
            0x0001_5100, // v_readlane_b32 s12, v0, 40
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    let why = refusal(&readlane, 32, 0);
    assert!(
        why.contains("a lane read that may cross 32-lane halves"),
        "{why}"
    );
}

/// [`small_group`] in a group of 32 on every hardware adapter, in halves and
/// as one wave64 where the adapter runs those: the console's results.
#[test]
fn a_group_one_host_wave_holds_matches_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..32)
        .flat_map(|l| [l * (l.max(1) - 1) / 2, 10, 0, 5])
        .collect();
    let failed = differences(&small_group(), 32, 4, &want);
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
}

/// [`prefix_sum_section`] that also reads the sum over every lane, lane 63's
/// inclusive sum, into `s3`, where the section sets EXEC to every lane.
fn sum_section() -> Vec<u32> {
    use crate::ir::EXEC;
    let mut w = prefix_sum_section();
    let all = w
        .iter()
        .position(|&x| x == sop1(EXEC, 0x04, 193))
        .expect("s_mov_b64 exec, -1");
    // v_readlane_b32 s3, v1, 63
    w.splice(all + 1..all + 1, vop3(3, 0x360, 257, 128 + 63, 0));
    w
}

/// Per thread `l`, two dwords at `l * 32`: the exclusive prefix sum of the
/// thread ids below it in its console wave, and the sum of all of them,
/// from one whole-wave section ([`sum_section`]).
fn wave_sums() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    w.extend(sum_section());
    w.push(vop1(9, 0x01, 3)); // v_mov_b32 v9, s3
    for (vdata, slot) in [(2, 0), (9, 1)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// Per thread `l`, four dwords at `l * 32`: of the mask of its console
/// wave's lanes below 40, the count (`s_bcnt1_i32_b64`), the low and the
/// high dword (lanes 32-39); then the thread id of the console wave's first
/// lane (`v_readfirstlane_b32`).
fn masks_and_first_lane() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,         // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D88_02A8,         // v_cmp_gt_u32 vcc, 40, v1
        sop1(12, 0x10, 106), // s_bcnt1_i32_b64 s12, vcc
        vop1(2, 0x01, 12),   // v_mov_b32 v2, s12
        vop1(3, 0x01, 106),  // v_mov_b32 v3, vcc_lo
        vop1(4, 0x01, 107),  // v_mov_b32 v4, vcc_hi
        vop1(13, 0x02, 256), // v_readfirstlane_b32 s13, v0
        vop1(5, 0x01, 13),   // v_mov_b32 v5, s13
    ]);
    for (vdata, slot) in [(2, 0), (3, 1), (4, 2), (5, 3)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// Per thread `l`, the thread id of its console wave's first lane above 35,
/// read by `v_readfirstlane_b32` under an EXEC of those lanes only (lane
/// 0's where it holds none): the first lane is in the upper half.
fn first_lane_in_exec() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,         // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D82_02A3,         // v_cmp_lt_u32 vcc, 35, v1
        0xBE94_246A,         // s_and_saveexec_b64 s[20:21], vcc
        vop1(14, 0x02, 256), // v_readfirstlane_b32 s14, v0
        0xBEFE_0414,         // s_mov_b64 exec, s[20:21]
        vop1(6, 0x01, 14),   // v_mov_b32 v6, s14
    ]);
    w.extend(store(6, 0));
    w.push(S_ENDPGM);
    w
}

/// The first lane's thread id, read inside a branch only threads below
/// `below` take, with console lanes (`v_mbcnt`).
fn first_lane_in_a_branch(below: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1,                 // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7D88_0000 | (128 + below), // v_cmp_gt_u32 vcc, below, v0
        0xBE94_246A,                 // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0002,                 // s_cbranch_execz 2
        vop1(12, 0x02, 256),         // v_readfirstlane_b32 s12, v0
        vop1(2, 0x01, 12),           // v_mov_b32 v2, s12
        0xBEFE_0414,                 // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// [`first_lane_of_a_buffer_load`] with the load and the first-lane read
/// inside a branch only threads below `below` take.
fn first_lane_of_a_buffer_load_in_a_branch(offset: u32, below: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1,                 // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E06_0284,                 // v_mov_b32 v3, 4
        0x7E0A_0280,                 // v_mov_b32 v5, 0
        0x7D88_0000 | (128 + below), // v_cmp_gt_u32 vcc, below, v0
        0xBE94_246A,                 // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0005,                 // s_cbranch_execz 5
        0xE030_1000,
        0x800B_0400 | offset, // buffer_load_dword v4, v<offset>, s[44:47], 0 offen
        0xBF8C_3F70,          // s_waitcnt vmcnt(0)
        0x7E18_0504,          // v_readfirstlane_b32 s12, v4
        0x7E0A_020C,          // v_mov_b32 v5, s12
        0xBEFE_0414,          // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(store(5, 0));
    w.push(S_ENDPGM);
    w
}

/// The number of threads below 10, counted inside a branch only threads
/// below `below` take.
fn mask_in_a_branch(below: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,                 // v_lshlrev_b32 v16, 5, v0
        0x7D88_0000 | (128 + below), // v_cmp_gt_u32 vcc, below, v0
        0xBE94_246A,                 // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0003,                 // s_cbranch_execz 3
        0x7D88_008A,                 // v_cmp_gt_u32 vcc, 10, v0
        sop1(12, 0x10, 106),         // s_bcnt1_i32_b64 s12, vcc
        vop1(2, 0x01, 12),           // v_mov_b32 v2, s12
        0xBEFE_0414,                 // s_mov_b64 exec, s[20:21]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// [`wave_sums`]'s prefix sum under a branch on a scalar compare.
fn sum_under_a_uniform_branch() -> Vec<u32> {
    let section = sum_section();
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,                        // v_lshlrev_b32 v16, 5, v0
        0xBF06_8005,                        // s_cmp_eq_u32 s5, 0
        0xBF84_0000 | section.len() as u32, // s_cbranch_scc0 past the section
    ]);
    w.extend(section);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// Where every thread of the group meets, the halves trade first-lane
/// reads, masks read as data and whole-wave sums: groups of 48, 64, 96 and
/// 128 translate. Inside a branch only some lanes take, or one on a uniform
/// condition in a group of several console waves, the region runs
/// wave-wide, where they meet.
#[test]
fn collectives_are_traded_where_every_thread_meets() {
    for threads in [48, 64, 96, 128] {
        for words in [
            masks_and_first_lane(),
            first_lane_in_exec(),
            wave_sums(),
            small_group(),
        ] {
            assert_eq!(refusal(&words, threads, 0), "translated", "{threads}");
        }
    }
    // Inside a branch lanes of both halves take, or one on a uniform
    // condition in a group of several console waves, the region runs
    // wave-wide.
    assert_eq!(refusal(&first_lane_in_a_branch(40), 64, 0), "translated");
    assert_eq!(refusal(&mask_in_a_branch(40), 64, 0), "translated");
    assert_eq!(refusal(&sum_under_a_uniform_branch(), 64, 0), "translated");
    assert_eq!(refusal(&sum_under_a_uniform_branch(), 128, 0), "translated");
}

/// [`masks_and_first_lane`], [`first_lane_in_exec`], [`wave_sums`] and
/// [`small_group`] on every hardware adapter, in halves (traded between the
/// two host waves) and as one wave64 where the adapter runs those: in groups
/// of 48 (an upper half of 16 threads), 64, 96 (a second console wave with
/// no upper half) and 128, every lane gets its console wave's masks, first
/// lanes and sums. [`first_lane_in_exec`] runs in halves only: the console's
/// own form reads the host wave's lane 0 under a narrowed EXEC on `main`
/// (an instruction with a scalar result drops its guard; upstream PR #2830,
/// fuzeofficlal's, fixes that there).
#[test]
fn collectives_cross_the_halves_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [48, 64, 96, 128] {
        // A console wave's first thread and its number of threads.
        let wave = |l: u32| (l & !63, (threads - (l & !63)).min(64));
        let masks: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let (base, n) = wave(l);
                let low = if n >= 32 { u32::MAX } else { (1 << n) - 1 };
                let high = if n > 32 {
                    (1 << (n - 32).min(8)) - 1
                } else {
                    0
                };
                [n.min(40), low, high, base]
            })
            .collect();
        failed.extend(differences(&masks_and_first_lane(), threads, 4, &masks));
        let above: Vec<u32> = (0..threads)
            .map(|l| {
                let (base, n) = wave(l);
                if n > 36 { base + 36 } else { base }
            })
            .collect();
        failed.extend(differences_in_halves(
            &first_lane_in_exec(),
            threads,
            1,
            &above,
        ));
        let sums: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let (base, n) = wave(l);
                [(base..l).sum::<u32>(), (base..base + n).sum::<u32>()]
            })
            .collect();
        failed.extend(differences(&wave_sums(), threads, 2, &sums));
        let small: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let (base, n) = wave(l);
                let below10 = if base == 0 { n.min(10) } else { 0 };
                [(base..l).sum::<u32>(), below10, 0, base + 5]
            })
            .collect();
        failed.extend(differences(&small_group(), threads, 4, &small));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A mask's high dword read as data under a branch on a scalar compare,
/// then kept by a DPP shift after the join in the lanes without a source
/// lane (0-14 of each row). With `ends`, the branch ends the program where
/// a second scalar compare holds (it never does).
fn mask_kept_by_a_shift(ends: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1,               // v_mbcnt_lo_u32_b32 v1, -1, 0
        vop1(2, 0x01, 256),        // v_mov_b32 v2, v0
        vop2(3, 0x1D, 128 + 5, 0), // v_xor_b32 v3, 5, v0
        vop1(4, 0x01, 256),        // v_mov_b32 v4, v0
        0xBF06_8005,               // s_cmp_eq_u32 s5, 0
    ]);
    let mut body = vec![
        0x7D88_0903,        // v_cmp_gt_u32 vcc, v3, v4
        vop1(4, 0x01, 107), // v_mov_b32 v4, vcc_hi
    ];
    if ends {
        body.extend([
            0xBF06_8105, // s_cmp_eq_u32 s5, 1
            0xBF84_0001, // s_cbranch_scc0 1
            S_ENDPGM,
        ]);
    }
    w.push(0xBF84_0000 | body.len() as u32); // s_cbranch_scc0 past it
    w.extend(body);
    w.extend([
        vop2(4, 0x25, 0xFA, 2),
        dpp16(2, 0x11F), // v_add_nc_u32 v4, v2, v2 row_shr:15
    ]);
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// In halves a write under a lane guard (here DPP's lanes without a source)
/// is refused when what the other lanes keep was refused: there, a mask
/// read in a branch on a scalar compare, in a group of two console waves
/// (run wave-wide, the branch able to end the program), where the halves
/// would trade through group-shared memory the program's 32 KiB of LDS
/// leaves no room for. Kept as zero it would store 0 where the console
/// stores the mask. With room, the halves trade.
#[test]
fn a_kept_refused_value_refuses_the_write_in_halves() {
    let why = refusal(&mask_kept_by_a_shift(true), 128, 32768);
    assert!(
        why.contains("past the host's 32 KiB of group-shared memory"),
        "{why}"
    );
    for ends in [true, false] {
        for threads in [64, 128] {
            let words = mask_kept_by_a_shift(ends);
            assert_eq!(
                refusal(&words, threads, 0),
                "translated",
                "{ends} {threads}"
            );
        }
    }
}

/// Every thread writes `tid + 1` at its LDS dword. In a branch only lanes of
/// one half take (below 16 with `low`, from 48 without), each reads the
/// dword 16 threads up (or down), adds its own and writes the sum back: a
/// region that both reads and writes LDS. After the join every thread
/// reads the dword of thread `tid ^ 32`, in the other half.
fn lds_tree_step(low: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,               // v_lshlrev_b32 v16, 5, v0
        vop2(2, 0x1A, 128 + 2, 0), // v_lshlrev_b32 v2, 2, v0
        vop2(1, 0x25, 128 + 1, 0), // v_add_nc_u32 v1, 1, v0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
    ]);
    if low {
        w.push(0x7D88_0090); // v_cmp_gt_u32 vcc, 16, v0
    } else {
        w.push(0x7D86_00B0); // v_cmp_le_u32 vcc, 48, v0
    }
    let step: [u32; 2] = if low {
        [vop2(4, 0x25, 128 + 64, 2), 0] // v_add_nc_u32 v4, 64, v2
    } else {
        [vop2(4, 0x25, 255, 2), 0xFFFF_FFC0] // v_add_nc_u32 v4, -64, v2
    };
    let mut body: Vec<u32> = if low {
        step[..1].to_vec()
    } else {
        step.to_vec()
    };
    body.extend([
        0xD8D8_0000,
        0x0500_0004, // ds_read_b32 v5, v4
        S_WAITCNT_LGKM,
        vop2(5, 0x25, 256 + 5, 1), // v_add_nc_u32 v5, v5, v1
        0xD834_0000,
        0x0000_0502, // ds_write_b32 v2, v5
    ]);
    w.extend([0xBE94_246A, 0xBF88_0000 | body.len() as u32]); // s_and_saveexec_b64 s[20:21], vcc; s_cbranch_execz
    w.extend(body);
    w.extend([
        0xBEFE_0414,           // s_mov_b64 exec, s[20:21]
        vop2(3, 0x1D, 255, 2), // v_xor_b32 v3, 0x80, v2
        0x0000_0080,
        0xD8D8_0000,
        0x0600_0003, // ds_read_b32 v6, v3
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(6, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lds_tree_step`]'s thread `l` in a group of 64.
fn lds_tree_step_want(low: bool, l: u32) -> u32 {
    let i = l ^ 32;
    match low {
        true if i < 16 => (i + 1) + (i + 17),
        false if i >= 48 => (i + 1) + (i - 15),
        _ => i + 1,
    }
}

/// A branch only one half's lanes take (a compare of the thread id with a
/// constant, in a group of one console wave or several) lets that half's
/// host wave run alone: a first-lane read, a mask read as data and LDS both
/// read and written there translate in halves, the first lane and the mask
/// with no barrier.
#[test]
fn one_half_regions_translate_in_halves() {
    for threads in [64, 128] {
        for words in [
            first_lane_in_a_branch(20),
            mask_in_a_branch(20),
            first_lane_of_a_buffer_load_in_a_branch(0, 20),
        ] {
            assert_eq!(refusal(&words, threads, 0), "translated", "{threads}");
        }
    }
    for low in [true, false] {
        assert_eq!(refusal(&lds_tree_step(low), 64, 256), "translated", "{low}");
    }
    for words in [first_lane_in_a_branch(20), mask_in_a_branch(20)] {
        assert_eq!(barriers(&words, 64, 0), 0);
    }
}

/// Group barriers in `words` lowered in halves.
fn barriers(words: &[u32], threads: u32, lds_bytes: u32) -> u32 {
    let n = native(words, true, threads, lds_bytes);
    lowered(&n, true).expect("halves").barriers
}

/// In a branch the console lanes from `from` take (1 to 31: lanes of both
/// halves): the first lane's thread id (`v2`), how many of the branch's
/// lanes are below 40 (`v3`), lane 50's thread id (`v4`), and lane 50's
/// `v6` (three times its thread id, which the branch adds 1000 to), or with
/// `outside` that of lane `from - 1`, which does not take the branch and
/// keeps the old value (`v5`). Lanes outside the branch store 0xdead in
/// each.
fn wide_branch(from: u32, outside: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,               // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop2(6, 0x0B, 128 + 3, 0), // v_mul_u32_u24 v6, 3, v0
    ]);
    for d in 2..6 {
        w.extend([vop1(d, 0x01, 255), 0xDEAD]); // v_mov_b32 vd, 0xdead
    }
    let read = if outside { from - 1 } else { 50 };
    let mut body = vec![
        vop2(6, 0x25, 255, 6), // v_add_nc_u32 v6, 1000, v6
        1000,
        vop1(12, 0x02, 256),                     // v_readfirstlane_b32 s12, v0
        vop1(2, 0x01, 12),                       // v_mov_b32 v2, s12
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 168, // v_cmp_gt_u32 vcc, 40, v1
        sop1(13, 0x10, 106),                     // s_bcnt1_i32_b64 s13, vcc
        vop1(3, 0x01, 13),                       // v_mov_b32 v3, s13
    ];
    body.extend(vop3(14, 0x360, 256, 128 + 50, 0)); // v_readlane_b32 s14, v0, 50
    body.push(vop1(4, 0x01, 14)); // v_mov_b32 v4, s14
    body.extend(vop3(15, 0x360, 256 + 6, 128 + read as u16, 0)); // v_readlane_b32 s15, v6, read
    body.push(vop1(5, 0x01, 15)); // v_mov_b32 v5, s15
    w.extend([
        0x7C00_0000 | 0xC3 << 17 | 1 << 9 | (128 + from), // v_cmp_le_u32 vcc, from, v1
        0xBE94_246A,                                      // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0000 | body.len() as u32,                  // s_cbranch_execz past it
    ]);
    w.extend(body);
    w.push(0xBEFE_0414); // s_mov_b64 exec, s[20:21]
    for (slot, d) in (2..6).enumerate() {
        w.extend(store(d, 4 * slot as u32));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console stores for [`wide_branch`]'s thread `l`.
fn wide_branch_want(from: u32, outside: bool, l: u32) -> [u32; 4] {
    let (base, lane) = (l & !63, l & 63);
    if lane < from {
        return [0xDEAD; 4];
    }
    let read = if outside {
        3 * (base + from - 1)
    } else {
        3 * (base + 50) + 1000
    };
    [base + from, 40 - from, base + 50, read]
}

/// A branch lanes of both halves take, with first lanes, masks, lane reads,
/// a whole-wave sum, or LDS both written and read, runs wave-wide in
/// halves: it translates and pays for its trades only, two group barriers
/// each.
#[test]
fn wave_wide_regions_translate_in_halves() {
    assert_eq!(barriers(&first_lane_in_a_branch(40), 64, 0), 2);
    assert_eq!(barriers(&mask_in_a_branch(40), 64, 0), 2);
    assert_eq!(barriers(&lane_read_in_a_branch(), 64, 0), 2);
    // A first lane, a mask and two lane reads.
    for threads in [64, 128] {
        for outside in [false, true] {
            let words = wide_branch(8, outside);
            assert_eq!(barriers(&words, threads, 0), 8, "{threads} {outside}");
        }
    }
    assert_eq!(refusal(&sum_under_a_uniform_branch(), 128, 0), "translated");
    for words in [
        lds_written_and_read_in_a_branch(),
        lds_written_twice_in_a_branch(),
    ] {
        assert_eq!(refusal(&words, 64, 256), "translated");
    }
}

/// A loop each console lane runs [`wide_loop_rounds`] rounds of (the exit a
/// compare of the round with that count, EXEC narrowed by it): each round
/// adds the first lane's console lane (`v2`) and the number of lanes from
/// 32 still in the loop (`v3`); with `lds`, each lane also writes the round
/// at its LDS dword and adds what lane `lane ^ 33` wrote there (`v5`), which
/// that lane left at its last round once it is out of the loop. After the
/// loop: the rounds (`v11`).
fn wide_loop(lds: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,                  // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop2(12, 0x16, 128 + 4, 1),   // v_lshrrev_b32 v12, 4, v1
        vop2(13, 0x1B, 128 + 1, 1),   // v_and_b32 v13, 1, v1
        vop2(12, 0x25, 256 + 13, 12), // v_add_nc_u32 v12, v13, v12
        vop2(12, 0x1B, 128 + 3, 12),  // v_and_b32 v12, 3, v12
        vop2(12, 0x25, 128 + 1, 12),  // v_add_nc_u32 v12, 1, v12
    ]);
    for d in [2, 3, 5, 11] {
        w.push(vop1(d, 0x01, 128)); // v_mov_b32 vd, 0
    }
    if lds {
        w.extend([
            vop2(6, 0x1A, 128 + 2, 1),  // v_lshlrev_b32 v6, 2, v1
            vop2(7, 0x1D, 128 + 33, 1), // v_xor_b32 v7, 33, v1
            vop2(7, 0x1A, 128 + 2, 7),  // v_lshlrev_b32 v7, 2, v7
            vop1(8, 0x01, 255),         // v_mov_b32 v8, 100
            100,
            0xD834_0000,
            0x0000_0806, // ds_write_b32 v6, v8
            S_WAITCNT_LGKM,
        ]);
    }
    w.push(0xBE9A_047E); // s_mov_b64 s[26:27], exec
    let mut body = vec![
        vop1(12, 0x02, 256 + 1),                 // v_readfirstlane_b32 s12, v1
        vop2(2, 0x25, 12, 2),                    // v_add_nc_u32 v2, s12, v2
        0x7C00_0000 | 0xC3 << 17 | 1 << 9 | 160, // v_cmp_le_u32 vcc, 32, v1
        sop1(13, 0x10, 106),                     // s_bcnt1_i32_b64 s13, vcc
        vop2(3, 0x25, 13, 3),                    // v_add_nc_u32 v3, s13, v3
    ];
    if lds {
        body.extend([
            0xD834_0000,
            0x0000_0B06, // ds_write_b32 v6, v11
            0xD8D8_0000,
            0x0900_0007, // ds_read_b32 v9, v7
            S_WAITCNT_LGKM,
            vop2(5, 0x25, 256 + 9, 5), // v_add_nc_u32 v5, v9, v5
        ]);
    }
    body.extend([
        vop2(11, 0x25, 128 + 1, 11), // v_add_nc_u32 v11, 1, v11
        0x7C00_0000 | 0xC1 << 17 | 12 << 9 | (256 + 11), // v_cmp_lt_u32 vcc, v11, v12
        0x87FE_6A7E,                 // s_and_b64 exec, exec, vcc
    ]);
    // s_cbranch_execnz back to the loop's first instruction.
    let back = -(body.len() as i32 + 1);
    body.push(0xBF89_0000 | (back as u32 & 0xFFFF));
    w.extend(body);
    w.push(0xBEFE_041A); // s_mov_b64 exec, s[26:27]
    for (slot, d) in [2, 3, 5, 11].into_iter().enumerate() {
        w.extend(store(d, 4 * slot as u32));
    }
    w.push(S_ENDPGM);
    w
}

/// The rounds console lane `l` runs of [`wide_loop`]: `((l >> 4) + (l & 1))
/// & 3` plus one, so the low half goes round at most 3 times and the high
/// half 4.
fn wide_loop_rounds(l: u32) -> u32 {
    (((l >> 4) + (l & 1)) & 3) + 1
}

/// What the console stores for [`wide_loop`]'s thread `l` in a group of
/// `threads` (one console wave), round by round as its wave runs the loop.
fn wide_loop_want(lds: bool, threads: u32, l: u32) -> [u32; 4] {
    let k = wide_loop_rounds(l);
    let running = |r: u32| (0..threads).filter(move |&m| wide_loop_rounds(m) > r);
    let first: u32 = (0..k).map(|r| running(r).min().unwrap_or(0)).sum();
    let high: u32 = (0..k)
        .map(|r| running(r).filter(|&m| m >= 32).count() as u32)
        .sum();
    // What lane `l ^ 33` last wrote at its dword: this round, or its last.
    let other = wide_loop_rounds(l ^ 33);
    let read: u32 = if lds {
        (0..k).map(|r| r.min(other - 1)).sum()
    } else {
        0
    };
    [first, high, read, k]
}

/// A loop each console lane goes round [`wide_loop_rounds`] times, writing
/// its count of rounds so far (a scalar) to `v2` each round; after the loop,
/// in every lane, the first lane's `v2` (`v3`): what console lane 0 wrote in
/// its last round, which lanes that went round more times do not hold.
fn first_lane_after_a_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,                  // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop2(12, 0x16, 128 + 4, 1),   // v_lshrrev_b32 v12, 4, v1
        vop2(13, 0x1B, 128 + 1, 1),   // v_and_b32 v13, 1, v1
        vop2(12, 0x25, 256 + 13, 12), // v_add_nc_u32 v12, v13, v12
        vop2(12, 0x1B, 128 + 3, 12),  // v_and_b32 v12, 3, v12
        vop2(12, 0x25, 128 + 1, 12),  // v_add_nc_u32 v12, 1, v12
        vop1(11, 0x01, 128),          // v_mov_b32 v11, 0
        sop1(20, 0x03, 128),          // s_mov_b32 s20, 0
        0xBE9A_047E,                  // s_mov_b64 s[26:27], exec
    ]);
    let body = [
        0x8014_8114,                                     // s_add_u32 s20, s20, 1
        vop1(2, 0x01, 20),                               // v_mov_b32 v2, s20
        vop2(11, 0x25, 128 + 1, 11),                     // v_add_nc_u32 v11, 1, v11
        0x7C00_0000 | 0xC1 << 17 | 12 << 9 | (256 + 11), // v_cmp_lt_u32 vcc, v11, v12
        0x87FE_6A7E,                                     // s_and_b64 exec, exec, vcc
    ];
    w.extend(body);
    let back = -(body.len() as i32 + 1);
    w.push(0xBF89_0000 | (back as u32 & 0xFFFF)); // s_cbranch_execnz back
    w.extend([
        0xBEFE_041A,             // s_mov_b64 exec, s[26:27]
        vop1(12, 0x02, 256 + 2), // v_readfirstlane_b32 s12, v2
        vop1(3, 0x01, 12),       // v_mov_b32 v3, s12
    ]);
    w.extend(store(2, 0));
    w.extend(store(3, 4));
    w.push(S_ENDPGM);
    w
}

/// [`first_lane_after_a_loop`] on every hardware adapter, in halves and as
/// one wave64 where the adapter runs those, in groups of 48, 64 and 128:
/// every lane reads console lane 0's `v2`, 1. Before, the value counted as
/// one in every lane of the console wave (each round writes one scalar), and
/// each half read its own first lane: 3, lane 32's, in the upper half.
#[test]
fn first_lane_after_a_loop_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [48, 64, 128] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| [wide_loop_rounds(l & 63), wide_loop_rounds(0)])
            .collect();
        failed.extend(differences(&first_lane_after_a_loop(), threads, 2, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A loop whose exit is lane-dependent, with a first-lane read, a mask read
/// as data and one-wave LDS written and read inside, runs wave-wide in
/// halves: every thread goes round while any lane of the group does, in a
/// group of one console wave or of several (whose waves the console runs
/// round their own numbers of times).
#[test]
fn wave_wide_loops_translate_in_halves() {
    for threads in [48, 64] {
        assert_eq!(
            refusal(&wide_loop(false), threads, 0),
            "translated",
            "{threads}"
        );
    }
    assert_eq!(refusal(&wide_loop(true), 64, 256), "translated");
    // A first lane and a mask traded each round, and whether to go round.
    assert_eq!(barriers(&wide_loop(false), 64, 0), 6);
    for threads in [96, 128] {
        assert_eq!(
            refusal(&wide_loop(false), threads, 0),
            "translated",
            "{threads}"
        );
    }
}

/// [`wide_loop`] on every hardware adapter, in halves and as one wave64
/// where the adapter runs those: in groups of 48, 64, 96 and 128 (each
/// console wave going round its own rounds on the console), and with LDS in
/// a group of 64, every lane stores what the console does.
#[test]
fn wave_wide_loops_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [48, 64, 96, 128] {
        // Each console wave's lanes: its first lane's id, how many.
        let wave = |l: u32| (l & !63, (threads - (l & !63)).min(64));
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| wide_loop_want(false, wave(l).1, l - wave(l).0))
            .collect();
        failed.extend(differences(&wide_loop(false), threads, 4, &want));
    }
    let want: Vec<u32> = (0..64).flat_map(|l| wide_loop_want(true, 64, l)).collect();
    failed.extend(differences_with_lds(&wide_loop(true), 64, 256, 4, &want));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Console lanes below 10 leave the program (EXEC loses them, and a wave
/// with none left skips to its end); the others store how many of their
/// wave's lanes below 40 are left (`v2`) and their wave's first lane left
/// (`v3`). A region that holds the program's end.
fn lanes_that_end() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,                             // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 138, // v_cmp_gt_u32 vcc, 10, v1
        0x8AFE_6A7E,                             // s_andn2_b64 exec, exec, vcc
    ]);
    let mut rest = vec![
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 168, // v_cmp_gt_u32 vcc, 40, v1
        sop1(12, 0x10, 106),                     // s_bcnt1_i32_b64 s12, vcc
        vop1(2, 0x01, 12),                       // v_mov_b32 v2, s12
        vop1(13, 0x02, 256 + 1),                 // v_readfirstlane_b32 s13, v1
        vop1(3, 0x01, 13),                       // v_mov_b32 v3, s13
    ];
    rest.extend(store(2, 0));
    rest.extend(store(3, 4));
    w.push(0xBF88_0000 | rest.len() as u32); // s_cbranch_execz to the end
    w.extend(rest);
    w.push(S_ENDPGM);
    w
}

/// [`lanes_that_end`] on every hardware adapter, in halves and as one
/// wave64 where the adapter runs those, in groups of 48, 64 and 128: the
/// lanes left read a mask and a first lane over their wave's lanes left
/// (30 and 10), and the lanes that left store nothing.
#[test]
fn regions_that_hold_the_end_match_the_console_on_every_gpu() {
    for threads in [48, 64, 128] {
        assert_eq!(
            refusal(&lanes_that_end(), threads, 0),
            "translated",
            "{threads}"
        );
    }
    let mut failed = Vec::new();
    for threads in [48, 64, 128] {
        let wave = |l: u32| (l & !63, (threads - (l & !63)).min(64));
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let (base, n) = wave(l);
                if l - base < 10 {
                    [0, 0]
                } else {
                    [n.min(40) - 10, 10]
                }
            })
            .collect();
        failed.extend(differences(&lanes_that_end(), threads, 2, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
/// Wave-wide regions on every hardware adapter, in halves and as one wave64
/// where the adapter runs those, in groups of 64 and 128: [`wide_branch`]
/// (with `outside` in halves only, as [`lane_read_in_a_branch`]: the
/// one-wave64 form reads a lane that does not run as the host does, which
/// D3D leaves undefined), [`mask_in_a_branch`] of the lanes below 40, and in
/// a group of 64, LDS written and read, or written twice, in a branch.
#[test]
fn wave_wide_regions_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [64, 128] {
        for from in [8, 20] {
            for outside in [false, true] {
                let words = wide_branch(from, outside);
                let want: Vec<u32> = (0..threads)
                    .flat_map(|l| wide_branch_want(from, outside, l))
                    .collect();
                failed.extend(if outside {
                    differences_in_halves(&words, threads, 4, &want)
                } else {
                    differences(&words, threads, 4, &want)
                });
            }
        }
        let mask: Vec<u32> = (0..threads).map(|l| if l < 40 { 10 } else { 0 }).collect();
        failed.extend(differences(&mask_in_a_branch(40), threads, 1, &mask));
        let read: Vec<u32> = (0..threads).map(|l| if l < 20 { 40 } else { 0 }).collect();
        failed.extend(differences_in_halves(
            &lane_read_in_a_branch(),
            threads,
            1,
            &read,
        ));
    }
    let written_and_read: Vec<u32> = (0..64)
        .map(|l| match l ^ 32 {
            o if l < 48 && o < 48 => o + 1,
            _ => 0,
        })
        .collect();
    failed.extend(differences_with_lds(
        &lds_written_and_read_in_a_branch(),
        64,
        256,
        1,
        &written_and_read,
    ));
    let twice: Vec<u32> = (0..64)
        .map(|l| match l ^ 32 {
            o if o < 48 => o + 1,
            _ if l < 48 => l + 1,
            _ => 0,
        })
        .collect();
    failed.extend(differences_with_lds(
        &lds_written_twice_in_a_branch(),
        64,
        256,
        1,
        &twice,
    ));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`lds_tree_step`] in a group of 64 on every hardware adapter, in halves
/// and as one wave64 where the adapter runs those: the half that runs the
/// branch reads and writes LDS in the console's order, and the other half
/// reads its results after the join.
#[test]
fn one_half_lds_regions_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for low in [true, false] {
        let want: Vec<u32> = (0..64).map(|l| lds_tree_step_want(low, l)).collect();
        failed.extend(differences_with_lds(&lds_tree_step(low), 64, 256, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`in_halves`] on every hardware adapter: in halves, and as one wave64
/// where the adapter runs 64-lane waves; both give the console's results.
#[test]
fn wave64_programs_in_halves_match_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64).flat_map(console).collect();
    let failed = differences(&in_halves(), 64, 8, &want);
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
}

/// Lanes below 10 leave the program and the others read a mask and a first
/// lane over the lanes left, as [`lanes_that_end`], after every lane stores
/// its console lane at dword 2; but where `skip`, a branch on a scalar
/// compare before (`s5` is 0) takes the whole group around all of it, to
/// the program's end, which the lanes that leave also go to.
fn whole_group_around(skip: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
    ]);
    w.extend(store(1, 8));
    let mut rest = vec![
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 168, // v_cmp_gt_u32 vcc, 40, v1
        sop1(12, 0x10, 106),                     // s_bcnt1_i32_b64 s12, vcc
        vop1(2, 0x01, 12),                       // v_mov_b32 v2, s12
        vop1(13, 0x02, 256 + 1),                 // v_readfirstlane_b32 s13, v1
        vop1(3, 0x01, 13),                       // v_mov_b32 v3, s13
    ];
    rest.extend(store(2, 0));
    rest.extend(store(3, 4));
    let leave = [
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 138, // v_cmp_gt_u32 vcc, 10, v1
        0x8AFE_6A7E,                             // s_andn2_b64 exec, exec, vcc
        0xBF88_0000 | rest.len() as u32,         // s_cbranch_execz to the end
    ];
    w.extend([
        0xBF06_8005 | u32::from(!skip) << 8, // s_cmp_eq_u32 s5, 0 (1 without `skip`)
        0xBF85_0000 | (leave.len() + rest.len()) as u32, // s_cbranch_scc1 to the end
    ]);
    w.extend(leave);
    w.extend(rest);
    w.push(S_ENDPGM);
    w
}

/// Lanes below 10 leave the program; the others take how many of them are
/// below 40 (`v2`, 0xdead before), then read the first lane left (`v3`) and
/// store both. Where `skip`, a branch on a scalar compare before (`s5` is 0)
/// takes the whole group straight to the first-lane read, with every lane:
/// into the middle of the region the lanes that stay run.
fn whole_group_into(skip: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop1(2, 0x01, 255),
        0xDEAD, // v_mov_b32 v2, 0xdead
    ]);
    let mut read = vec![
        vop1(13, 0x02, 256 + 1), // v_readfirstlane_b32 s13, v1
        vop1(3, 0x01, 13),       // v_mov_b32 v3, s13
    ];
    read.extend(store(2, 0));
    read.extend(store(3, 4));
    let count = [
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 168, // v_cmp_gt_u32 vcc, 40, v1
        sop1(12, 0x10, 106),                     // s_bcnt1_i32_b64 s12, vcc
        vop1(2, 0x01, 12),                       // v_mov_b32 v2, s12
    ];
    let leave = [
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 138, // v_cmp_gt_u32 vcc, 10, v1
        0x8AFE_6A7E,                             // s_andn2_b64 exec, exec, vcc
        0xBF88_0000 | (count.len() + read.len()) as u32, // s_cbranch_execz to the end
    ];
    w.extend([
        0xBF06_8005 | u32::from(!skip) << 8, // s_cmp_eq_u32 s5, 0 (1 without `skip`)
        0xBF85_0000 | (leave.len() + count.len()) as u32, // s_cbranch_scc1 to the read
    ]);
    w.extend(leave);
    w.extend(count);
    w.extend(read);
    w.push(S_ENDPGM);
    w
}

/// Regions run wave-wide that the whole group may also go around, to their
/// join ([`whole_group_around`]), or into, at a block past their head
/// ([`whole_group_into`]), on a branch it takes together, translate in
/// halves in groups of one and of several console waves, to valid DXIL.
#[test]
fn regions_the_whole_group_goes_around_or_into_translate_in_halves() {
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    for threads in [48, 64, 128] {
        for skip in [false, true] {
            for (name, words) in [
                ("around", whole_group_around(skip)),
                ("into", whole_group_into(skip)),
            ] {
                let l = lowered(&native(&words, true, threads, 0), true)
                    .unwrap_or_else(|e| panic!("{name} {threads} {skip}: {e}"));
                if let Some(v) = validator.as_mut() {
                    v.validate(&mut l.dxil.clone())
                        .unwrap_or_else(|e| panic!("{name} {threads} {skip}: {e}"));
                }
            }
        }
    }
}

/// [`whole_group_around`] and [`whole_group_into`] on every hardware
/// adapter, in halves and as one wave64 where the adapter runs those, in
/// groups of 48, 64 and 128, with the whole group going its own way or not.
#[test]
fn regions_the_whole_group_goes_around_or_into_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [48, 64, 128] {
        let wave = |l: u32| (l & !63, (threads - (l & !63)).min(64));
        for skip in [false, true] {
            let around: Vec<u32> = (0..threads)
                .flat_map(|l| {
                    let (base, n) = wave(l);
                    let lane = l - base;
                    if skip || lane < 10 {
                        [0, 0, lane]
                    } else {
                        [n.min(40) - 10, 10, lane]
                    }
                })
                .collect();
            failed.extend(differences(&whole_group_around(skip), threads, 3, &around));
            let into: Vec<u32> = (0..threads)
                .flat_map(|l| {
                    let (base, n) = wave(l);
                    if skip {
                        [0xDEAD, 0]
                    } else if l - base < 10 {
                        [0, 0]
                    } else {
                        [n.min(40) - 10, 10]
                    }
                })
                .collect();
            failed.extend(differences(&whole_group_into(skip), threads, 2, &into));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A wave64 sum written the way the console compiler expands one (the
/// idiom `wave::sections` reads), of `v8` (the thread id plus one) over the
/// lanes EXEC holds: row sums by DPP, the other row of each 32 by
/// `v_permlanex16_b32`, each half's sum; then lane 32's (the high half's
/// sum) added in every lane, so that lanes 0-31 hold
/// the wave's sum and lanes 32-63 twice the high half's, and the wave's sum
/// read from lane 0 (`s6`). After it, lanes below 20 write 7 to that
/// register and the others keep what the section left: each lane stores
/// it, and the sum.
fn sum_kept_by_lanes() -> Vec<u32> {
    use crate::ir::EXEC;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,           // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop2(8, 0x25, 129, 0), // v_add_nc_u32 v8, 1, v0
        sop1(2, 0x25, 193),    // s_or_saveexec_b64 s[2:3], -1
    ]);
    w.extend(vop3(2, 0x101, 128, 256 + 8, 2)); // v_cndmask_b32 v2, 0, v8, s[2:3]
    for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
        // v_add_nc_u32 v2, v2, v2 quad_perm:[1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror
        w.extend([vop2(2, 0x25, 0xFA, 2), dpp16(2, ctrl)]);
    }
    w.extend(vop3(3, 0x378, 256 + 2, 128, 128)); // v_permlanex16_b32 v3, v2, 0, 0
    w.push(vop2(2, 0x25, 256 + 3, 2)); // v_add_nc_u32 v2, v3, v2
    w.extend(vop3(4, 0x360, 256 + 2, 128 + 32, 0)); // v_readlane_b32 s4, v2, 32
    w.push(vop2(2, 0x25, 4, 2)); // v_add_nc_u32 v2, s4, v2
    w.extend(vop3(6, 0x360, 256 + 2, 128, 0)); // v_readlane_b32 s6, v2, 0
    w.push(sop1(EXEC, 0x04, 2)); // s_mov_b64 exec, s[2:3]
    w.extend([
        0x7C00_0000 | 0xC4 << 17 | 1 << 9 | 148, // v_cmp_gt_u32 vcc, 20, v1
        0xBE94_246A,                             // s_and_saveexec_b64 s[20:21], vcc
        vop1(2, 0x01, 128 + 7),                  // v_mov_b32 v2, 7
        0xBEFE_0414,                             // s_mov_b64 exec, s[20:21]
        vop1(4, 0x01, 6),                        // v_mov_b32 v4, s6
    ]);
    w.extend(store(2, 0));
    w.extend(store(4, 4));
    w.push(S_ENDPGM);
    w
}

/// What the console stores for [`sum_kept_by_lanes`]'s thread `l`.
fn sum_kept_by_lanes_want(l: u32) -> [u32; 2] {
    let base = l & !63;
    let total: u32 = (base..base + 64).map(|k| k + 1).sum();
    let high: u32 = (base + 32..base + 64).map(|k| k + 1).sum();
    let kept = match l - base {
        0..20 => 7,
        20..32 => total,
        _ => 2 * high,
    };
    [kept, total]
}

/// In halves, [`sum_kept_by_lanes`] runs its section step by step: on every
/// hardware adapter, in groups of 64 and 128, the lanes that keep the
/// section's register get the console's values in both halves (main keeps
/// them as zero, which is what one wave64 on a 64-lane host still stores).
/// In a group of a part of a console wave it is refused as before.
#[test]
fn a_whole_wave_sum_kept_by_lanes_matches_the_console_in_halves_on_every_gpu() {
    let partial = refusal(&sum_kept_by_lanes(), 48, 0);
    assert!(
        partial.contains("a whole-wave section step the simulation does not follow"),
        "{partial}"
    );
    let mut failed = Vec::new();
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).flat_map(sum_kept_by_lanes_want).collect();
        failed.extend(differences_in_halves(
            &sum_kept_by_lanes(),
            threads,
            2,
            &want,
        ));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_permlane16_b32` and `v_permlanex16_b32` of `v2` (three times the
/// thread id plus one), each lane reading lane 15 - i of its row of 16 (of
/// the other row of its 32 with x16): with every lane (`v3`, `v4`), then
/// with the even lanes only, the odd ones they read left out: as it is
/// (`v5`), with bound_ctrl (`v6`) and with fi (`v7`). Each register holds
/// 0x3333.. 0x7777 before.
fn permlanes() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,               // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop2(2, 0x0B, 128 + 3, 0), // v_mul_u32_u24 v2, 3, v0
        vop2(2, 0x25, 129, 2),     // v_add_nc_u32 v2, 1, v2
        sop1(10, 0x03, 255),
        0x89AB_CDEF, // s_mov_b32 s10, 0x89abcdef
        sop1(11, 0x03, 255),
        0x0123_4567, // s_mov_b32 s11, 0x01234567
    ]);
    for d in 3..8u8 {
        w.extend([vop1(d, 0x01, 255), 0x1111 * u32::from(d)]); // v_mov_b32 vd, 0xdddd
    }
    w.extend(vop3(3, 0x377, 256 + 2, 10, 11)); // v_permlane16_b32 v3, v2, s10, s11
    w.extend(vop3(4, 0x378, 256 + 2, 10, 11)); // v_permlanex16_b32 v4, v2, s10, s11
    w.extend([
        vop2(9, 0x1B, 129, 1),                   // v_and_b32 v9, 1, v1
        0x7C00_0000 | 0xC2 << 17 | 9 << 9 | 128, // v_cmp_eq_u32 vcc, 0, v9
        0xBE94_246A,                             // s_and_saveexec_b64 s[20:21], vcc
    ]);
    for (d, op_sel) in [(5u8, 0u32), (6, 2), (7, 1)] {
        let [a, b] = vop3(d, 0x377, 256 + 2, 10, 11); // v_permlane16_b32 vd, v2, s10, s11
        w.extend([a | op_sel << 11, b]);
    }
    w.push(0xBEFE_0414); // s_mov_b64 exec, s[20:21]
    for (slot, d) in (3..8).enumerate() {
        w.extend(store(d, 4 * slot as u32));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console stores for [`permlanes`]'s thread `l` in a group of
/// `threads`.
fn permlanes_want(threads: u32, l: u32) -> [u32; 5] {
    let base = l & !63;
    let lane = l - base;
    let (row, pos) = (lane & !15, lane & 15);
    // A lane past the group's is not in EXEC: its reader keeps its own.
    let read = |src: u32, kept: u32| {
        if base + src < threads {
            3 * (base + src) + 1
        } else {
            kept
        }
    };
    let (own, other) = (row + 15 - pos, (row ^ 16) + 15 - pos);
    let even = pos % 2 == 0;
    [
        read(own, 0x3333),
        read(other, 0x4444),
        0x5555,
        if even { 0 } else { 0x6666 },
        if even { read(own, 0x7777) } else { 0x7777 },
    ]
}

/// [`permlanes`] on every hardware adapter, in halves and as one wave64
/// where the adapter runs those, in groups of 48, 64 and 128.
#[test]
fn permlanes_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [48, 64, 128] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| permlanes_want(threads, l))
            .collect();
        failed.extend(differences(&permlanes(), threads, 5, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A 16-bit sum of the console lanes (`v_cvt_f16_u16` of `v1`) as the
/// console compiler writes a whole-wave one: `v_add_f16` with DPP over each
/// row, `v_permlanex16_b32` for the other row of each 32, then lanes 31 and
/// 63 (each half's sum) read and added. Every lane stores its half's sum
/// and the wave's, as f16 in the low half of the dword. The sums are
/// integers below 2,048, so every step is exact in f16, in any order.
fn sum16() -> Vec<u32> {
    use crate::ir::EXEC;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1,            // v_mbcnt_hi_u32_b32 v1, -1, v1
        vop1(3, 0x01, 128),     // v_mov_b32 v3, 0
        vop1(6, 0x01, 128),     // v_mov_b32 v6, 0
        vop1(3, 0x50, 256 + 1), // v_cvt_f16_u16 v3, v1
        sop1(2, 0x25, 193),     // s_or_saveexec_b64 s[2:3], -1
    ]);
    w.extend(vop3(4, 0x101, 128, 256 + 3, 2)); // v_cndmask_b32 v4, 0, v3, s[2:3]
    for ctrl in [0xB1, 0x4E, 0x141, 0x140] {
        // v_add_f16 v4, v4, v4 quad_perm:[1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror
        w.extend([vop2(4, 0x32, 0xFA, 4), dpp16(4, ctrl)]);
    }
    w.extend(vop3(5, 0x378, 256 + 4, 128, 128)); // v_permlanex16_b32 v5, v4, 0, 0
    w.push(vop2(4, 0x32, 256 + 5, 4)); // v_add_f16 v4, v5, v4
    w.extend(vop3(4, 0x360, 256 + 4, 128 + 31, 0)); // v_readlane_b32 s4, v4, 31
    w.extend(vop3(5, 0x360, 256 + 4, 128 + 63, 0)); // v_readlane_b32 s5, v4, 63
    w.push(vop1(7, 0x01, 5)); // v_mov_b32 v7, s5
    w.push(vop2(6, 0x32, 4, 7)); // v_add_f16 v6, s4, v7
    w.push(sop1(EXEC, 0x04, 2)); // s_mov_b64 exec, s[2:3]
    w.extend(store(4, 0));
    w.extend(store(6, 4));
    w.push(S_ENDPGM);
    w
}

/// [`sum16`] in halves on every hardware adapter, in groups of 64 and 128:
/// each half's 16-bit sum (496 and 1,520) and the wave's (2,016), as f16.
#[test]
fn a_16_bit_wave_sum_matches_the_console_in_halves_on_every_gpu() {
    const LOW: u32 = 0x5FC0; // 496.0
    const HIGH: u32 = 0x65F0; // 1520.0
    const WAVE: u32 = 0x67E0; // 2016.0
    let mut failed = Vec::new();
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| [if l & 63 < 32 { LOW } else { HIGH }, WAVE])
            .collect();
        failed.extend(differences_in_halves(&sum16(), threads, 2, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
