//! Explicit-gradient samples with 16-bit gradients (`image_sample_d_g16`,
//! `image_sample_d_cl_g16`) end to end, beside their f32 siblings
//! (`image_sample_d`, `image_sample_d_cl`), each with and without a16: a
//! synthetic wave32 compute program puts the address in v1.. and samples a
//! mipped 2D texture whose texels name their level and place; it is lowered
//! to DXIL, validated by `dxil.dll` when present, and run on the host GPU
//! and WARP.
//!
//! Gradient width follows the `_g16` opcode only; a16 packs only the
//! coordinates and the clamp (GFX10.3 has G16). The gradients pick the
//! level and the clamp raises it; the coordinates pick the texel. So a
//! gradient taken as an f32 word, halves swapped, one direction read twice,
//! coordinates or clamp read as halves (or from the wrong VGPR) each show as
//! another level or texel. The other dimensions are lowered and validated
//! only (the host tests have 2D views); the cube, comparison and offset
//! forms are refused.

use super::host::View;
use super::waterfall_tests::run;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Level 0 of the texture: wider than high, so dP/dx and dP/dy taken for
/// each other's (or their halves swapped) select another level.
const W: u32 = 64;
const H: u32 = 16;
/// 64x16 down to 1x1.
const LEVELS: u32 = 7;

/// One of the four explicit-gradient forms.
#[derive(Clone, Copy, Debug)]
struct Form {
    opcode: u16,
    name: &'static str,
    clamp: bool,
    g16: bool,
}

const SAMPLE_D: Form = Form {
    opcode: 0x022,
    name: "image_sample_d",
    clamp: false,
    g16: false,
};
const SAMPLE_D_CL: Form = Form {
    opcode: 0x023,
    name: "image_sample_d_cl",
    clamp: true,
    g16: false,
};
const SAMPLE_D_G16: Form = Form {
    opcode: 0x0a2,
    name: "image_sample_d_g16",
    clamp: false,
    g16: true,
};
const SAMPLE_D_CL_G16: Form = Form {
    opcode: 0x0a3,
    name: "image_sample_d_cl_g16",
    clamp: true,
    g16: true,
};

/// The bits of `x` as an f16; `x` must be zero or a normal f16 exactly.
fn f16(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = b & 0x7F_FFFF;
    assert!(
        (1..31).contains(&exp) && man & 0x1FFF == 0,
        "{x} is not an f16"
    );
    (b >> 31) << 15 | (exp as u32) << 10 | man >> 13
}

/// Two f16s in a dword, the first in the low half.
fn pair(lo: f32, hi: f32) -> u32 {
    f16(lo) | f16(hi) << 16
}

/// The address dwords of a 2D sample: dP/dx (s, t), dP/dy (s, t), then the
/// coordinates and the clamp. The `g16` opcode packs gradients into a
/// dword per direction; a16 packs the coordinates, and the clamp into the
/// low half of the next dword (its high half holds a decoy, 5.0).
fn address_2d(form: Form, a16: bool, case: &Case) -> Vec<u32> {
    let Case {
        dx, dy, uv, clamp, ..
    } = *case;
    let mut w = Vec::new();
    if form.g16 {
        w.extend([pair(dx[0], dx[1]), pair(dy[0], dy[1])]);
    } else {
        w.extend([dx[0], dx[1], dy[0], dy[1]].map(f32::to_bits));
    }
    if a16 {
        w.push(pair(uv[0], uv[1]));
        if form.clamp {
            w.push(pair(clamp, 5.0));
        }
    } else {
        w.extend(uv.map(f32::to_bits));
        if form.clamp {
            w.push(clamp.to_bits());
        }
    }
    w
}

/// The program: the table of views at s0 (the texture at entry 0, the
/// result buffer at dword 16), the samplers' at s1; `address` moved to v1..,
/// the form sampled into v[20:23] (dmask 0xf, `dim`), stored at 16 bytes a
/// lane.
fn program(form: Form, dim: u8, a16: bool, address: &[u32]) -> Vec<u32> {
    assert!(address.len() <= 12, "v1..v12 hold the address");
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    ];
    for (k, &a) in address.iter().enumerate() {
        // v_mov_b32 v(1 + k), a
        w.extend([0x7E00_02FF | (1 + k as u32) << 17, a]);
    }
    let op = u32::from(form.opcode);
    w.extend([
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // image_sample_d* v[20:23], v[1:n], s[24:31], s[8:11] dmask:0xf
        0xF000_0000 | (op & 0x7F) << 18 | 0xF << 8 | u32::from(dim) << 3 | op >> 7,
        u32::from(a16) << 30 | 2 << 21 | 6 << 16 | 20 << 8 | 1,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_1410, // buffer_store_dwordx4 v[20:23], v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// Lowers a program of [`program`]; the DXIL is validated when the
/// validator is there. The views are as [`run`] binds them: textures at
/// heap entries 0 and 1, the UAV at 2.
fn build(words: &[u32]) -> Result<HostShader, ShaderError> {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Uav, 16, 4, 0),
            range(ResourceType::Sampler, 0, 4, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        host_wave32: false,
        view_remap: false,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// The texture: texel (x, y) of level L is (L, x, y, 1).
fn texels() -> Vec<[f32; 4]> {
    let mut t = Vec::new();
    for level in 0..LEVELS {
        let (w, h) = ((W >> level).max(1), (H >> level).max(1));
        for y in 0..h {
            for x in 0..w {
                t.push([level as f32, x as f32, y as f32, 1.0]);
            }
        }
    }
    t
}

/// One sample: gradients, coordinates, clamp (clamp forms only) and the
/// level they select.
struct Case {
    dx: [f32; 2],
    dy: [f32; 2],
    uv: [f32; 2],
    clamp: f32,
    level: u32,
}

/// A small cross term: every f16 half is nonzero, too small to move the
/// level (a few hundredths of a texel at level 0).
const E: f32 = 1.0 / 1024.0;
/// Coordinates at level-0 texel centres (13.5, 5.5) and (50.5, 12.5): never
/// on a texel edge at any level, and exact in f16.
const UV_A: [f32; 2] = [27.0 / 128.0, 11.0 / 32.0];
const UV_B: [f32; 2] = [101.0 / 128.0, 25.0 / 32.0];

/// The cases for a form. Gradients are along the axes (apart from `E`), so
/// any length the sampler takes gives the same level; the levels are whole,
/// half a level from where point mip filtering would round otherwise.
fn cases(clamp: bool) -> Vec<Case> {
    let mut c = vec![
        // dP/dx is 4 texels: level 2. dP/dy is half a texel. With the halves
        // of each dword swapped: level 1.
        Case {
            dx: [4.0 / 64.0, E],
            dy: [E, 0.5 / 16.0],
            uv: UV_A,
            clamp: 0.0,
            level: 2,
        },
        // dP/dy is 8 texels: level 3. dP/dx read for both: level 0.
        Case {
            dx: [1.0 / 64.0, E],
            dy: [E, 8.0 / 16.0],
            uv: UV_B,
            clamp: 0.0,
            level: 3,
        },
        // Level 1.
        Case {
            dx: [2.0 / 64.0, E],
            dy: [E, 1.0 / 16.0],
            uv: UV_A,
            clamp: 0.0,
            level: 1,
        },
    ];
    if clamp {
        c.extend([
            // The clamp raises level 1 to 3. Its f32 word's low half is 0
            // (level 1), its high half 2.125 (level 2).
            Case {
                dx: [2.0 / 64.0, E],
                dy: [E, 1.0 / 16.0],
                uv: UV_B,
                clamp: 3.0,
                level: 3,
            },
            // A clamp below the level leaves it.
            Case {
                dx: [4.0 / 64.0, E],
                dy: [E, 0.5 / 16.0],
                uv: UV_B,
                clamp: 1.0,
                level: 2,
            },
            // To 2.
            Case {
                dx: [E, E],
                dy: [E, 1.0 / 16.0],
                uv: UV_A,
                clamp: 2.0,
                level: 2,
            },
        ]);
    }
    c
}

/// What a case reads: (level, x, y, 1) of the texel under the coordinates
/// at its level.
fn expected(case: &Case) -> [u32; 4] {
    let l = case.level;
    let (w, h) = ((W >> l).max(1), (H >> l).max(1));
    let x = (case.uv[0] * w as f32).floor();
    let y = (case.uv[1] * h as f32).floor();
    [l as f32, x, y, 1.0].map(f32::to_bits)
}

/// Runs every case of `form` (with and without a16) on the hardware GPU
/// and WARP; every lane must read the case's texel.
fn sample_2d(form: Form) {
    let tex = texels();
    let dummy = [[0.0f32; 4]];
    let views = [
        View::Mipped(W, H, LEVELS, &tex),
        View::Texture(1, 1, &dummy),
        View::RawUav(4 * 32),
    ];
    for a16 in [false, true] {
        for (n, case) in cases(form.clamp).iter().enumerate() {
            let address = address_2d(form, a16, case);
            let at = format!("{} a16 {a16} case {n}", form.name);
            let shader = build(&program(form, 1, a16, &address))
                .unwrap_or_else(|e| panic!("{at}: not lowered: {e:?}"));
            let results = run(&shader, &views);
            assert!(!results.is_empty(), "{at}: no device ran");
            let want = expected(case);
            for (label, got) in results {
                assert!(got.len() >= 4 * 32, "{at}, {label}: {} dwords read back", got.len());
                for lane in 0..32 {
                    let texel = &got[4 * lane..4 * lane + 4];
                    assert_eq!(
                        texel,
                        want,
                        "{at}, {label} lane {lane}: got {:?}, want {:?} (level, x, y, 1)",
                        texel.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>(),
                        want.map(f32::from_bits),
                    );
                }
            }
        }
    }
}

#[test]
fn sample_d_selects_the_level_of_its_f32_gradients_on_the_card() {
    // Ordinary opcodes keep full-width gradients with both coordinate widths on RDNA2.
    sample_2d(SAMPLE_D);
    sample_2d(SAMPLE_D_CL);
}

#[test]
fn sample_d_g16_selects_the_level_of_its_f16_gradients_on_the_card() {
    sample_2d(SAMPLE_D_G16);
}

#[test]
fn sample_d_cl_g16_clamps_at_its_f32_or_a16_clamp_on_the_card() {
    sample_2d(SAMPLE_D_CL_G16);
}

/// Address dwords for `dim` (0 1D, 2 3D, 4 1D array, 5 2D array; 3D and 1D
/// gradients pad each 16-bit direction to a whole dword).
fn address_dwords(form: Form, dim: u8, a16: bool) -> usize {
    let (coords, grads): (usize, usize) = match dim {
        0 => (1, 1),
        1 => (2, 2),
        2 => (3, 3),
        4 => (2, 1),
        5 => (3, 2),
        _ => unreachable!(),
    };
    let grads = if form.g16 {
        2 * grads.div_ceil(2)
    } else {
        2 * grads
    };
    let packed = coords + usize::from(form.clamp);
    grads + if a16 { packed.div_ceil(2) } else { packed }
}

#[test]
fn sample_d_g16_lowers_and_validates_where_its_sibling_does() {
    // 1D, 2D, 3D, 1D array, 2D array: the dimensions the f32 forms lower to
    // a gradient sample. The address values are only placeholders here.
    for dim in [0u8, 1, 2, 4, 5] {
        for a16 in [false, true] {
            for (sibling, form) in [(SAMPLE_D, SAMPLE_D_G16), (SAMPLE_D_CL, SAMPLE_D_CL_G16)] {
                for f in [sibling, form] {
                    let n = address_dwords(f, dim, a16);
                    let address: Vec<u32> = (0..n as u32).map(|k| 0x3C00_3800 + k).collect();
                    if let Err(e) = build(&program(f, dim, a16, &address)) {
                        panic!("{} dim {dim} a16 {a16}: {e:?}", f.name);
                    }
                }
            }
        }
    }
}

fn refused(words: &[u32]) -> &'static str {
    match build(words) {
        Err(ShaderError::Unsupported { what, .. }) => what,
        Err(e) => panic!("{e:?}"),
        Ok(_) => panic!("lowered"),
    }
}

#[test]
fn other_g16_forms_and_cubes_are_refused_by_name() {
    // A cube: the f32 form's level estimate is not taken for 16-bit
    // gradients (dim 3: two gradients each way, face coordinates).
    for form in [SAMPLE_D_G16, SAMPLE_D_CL_G16] {
        for a16 in [false, true] {
            let packed = 3 + usize::from(form.clamp);
            let n = 2 + if a16 { packed.div_ceil(2) } else { packed };
            let words = program(form, 3, a16, &vec![0x3C00_3C00; n]);
            assert_eq!(refused(&words), "a cube sample with 16-bit gradients");
        }
    }
    // Comparison and offset forms: not lowered at all, as before.
    for (opcode, name, n) in [
        (0x0aa, "image_sample_c_d_g16", 5),
        (0x0ab, "image_sample_c_d_cl_g16", 6),
        (0x0b2, "image_sample_d_o_g16", 5),
        (0x0b3, "image_sample_d_cl_o_g16", 6),
        (0x0ba, "image_sample_c_d_o_g16", 6),
        (0x0bb, "image_sample_c_d_cl_o_g16", 7),
    ] {
        let form = Form {
            opcode,
            name,
            clamp: false,
            g16: true,
        };
        let words = program(form, 1, false, &vec![0x3C00_3C00; n]);
        assert_eq!(refused(&words), name);
    }
}
