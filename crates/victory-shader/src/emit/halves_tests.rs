//! Console wave64 compute programs translated for a host of 32-lane waves
//! (`lower::halves`): what reads within a 32-lane half runs exactly in two
//! host waves of 32, what reads across them is refused with its reason.
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

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
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

fn native(words: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
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

fn lowered(
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

fn shader(l: lower::Lowered) -> super::HostShader {
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
    let programs: [(Vec<u32>, &[&str]); 5] = [
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
    assert!(
        refusal(&readlane, 64, 0).contains("a lane read that may cross 32-lane halves"),
        "{}",
        refusal(&readlane, 64, 0)
    );
    assert!(lowered(&native(&readlane, true, 64, 0), false).is_ok());

    let first = standalone_first_lane();
    let standalone = native(&first, true, 64, 0);
    assert_eq!(
        lowered(&standalone, false).expect("native waves").wave_size,
        Some(64)
    );
    assert_eq!(refusal(&first, 64, 0), "translated");
    assert_eq!(refusal(&first_lane_of_the_thread_id(), 32, 0), "translated");

    // With only high EXEC enabled, its first active lane is 32. The
    // 1D entry formula is invalid after even a straight-line EXEC write.
    let mut masked = first;
    masked.splice(PROLOGUE.len()..PROLOGUE.len(), [0xBEFE_0380]); // s_mov_b32 exec_lo, 0
    assert_ne!(refusal(&masked, 64, 0), "translated");
    let why = refusal(&standalone_first_lane(), 96, 0);
    assert!(
        why.contains("a first-lane read of a value that may differ between 32-lane halves"),
        "{why}"
    );
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
    // scalar descriptor, is one value: its first lane is the console's.
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(3), 64, 0),
        "translated"
    );
    let why = refusal(&first_lane_of_a_buffer_load(0), 64, 0);
    assert!(
        why.contains("a first-lane read of a value that may differ between 32-lane halves"),
        "{why}"
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

/// A second console wave does not supply a group barrier for the first.
#[test]
fn lds_spanning_host_waves_needs_proven_ordering() {
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
    assert_eq!(refusal(&lds, 32, 256), "translated");
    for threads in [33, 64] {
        assert_eq!(refusal(&lds, threads, 256), "translated");
    }
    assert!(refusal(&lds, 128, 512).contains("LDS ordering across 32-lane halves"));
    assert!(lowered(&native(&lds, true, 128, 512), false).is_ok());

    let write = lds.iter().position(|&word| word == 0xD834_0000).unwrap();
    let mut masked = lds.to_vec();
    masked.insert(write + 2, 0xBEFE_0414); // s_mov_b64 exec, s[20:21]
    masked.splice(write..write, [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    ]);
    assert!(refusal(&masked, 64, 256).contains("LDS ordering across 32-lane halves"));
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
    let want: Vec<_> = (0..64).map(|lane| lane ^ 32).collect();
    let wrong = differences_with_lds(&words, 64, 1, &want, 512);
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// Runs `words`, one group of `threads` of a wave64 program whose lanes store
/// `per_lane` dwords at `l * 32`, on every hardware adapter: in halves, and
/// as one wave64 where the adapter runs 64-lane waves. A line per lane whose
/// dwords differ from `want` (the console's, `per_lane` per lane).
fn differences(words: &[u32], threads: u32, per_lane: usize, want: &[u32]) -> Vec<String> {
    differences_with_lds(words, threads, per_lane, want, 0)
}

fn differences_with_lds(words: &[u32], threads: u32, per_lane: usize, want: &[u32], lds: u32) -> Vec<String> {
    let n = native(words, true, threads, lds);
    let forms = [
        ("halves", shader(lowered(&n, true).expect("halves"))),
        ("wave64", shader(lowered(&n, false).expect("one wave64"))),
    ];
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

/// [`in_halves`] on every hardware adapter: in halves, and as one wave64
/// where the adapter runs 64-lane waves; both give the console's results.
#[test]
fn wave64_programs_in_halves_match_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64).flat_map(console).collect();
    let failed = differences(&in_halves(), 64, 8, &want);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
