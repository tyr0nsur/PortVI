//! Random wave64 compute programs on a host of 32-lane waves
//! (`lower::halves`), against the same programs as one wave64: every program
//! the translator keeps in halves must store, in every lane, what its
//! one-wave64 form stores on an adapter that runs 64-lane waves (AMD). A
//! program refused in halves is fine, since it is refused with its reason;
//! a different result is not.
//!
//! The programs read across lanes at the top level (DPP, `ds_swizzle_b32`,
//! `ds_bpermute_b32`, `v_readlane_b32`, `v_writelane_b32`,
//! `v_readfirstlane_b32`, lane masks read as data, a whole-wave prefix sum,
//! LDS), compute lane by lane in branches only some lanes take (where they
//! may also read across lanes and write LDS, one inside another or not),
//! go round loops each lane leaves after its own number of rounds (reading
//! across lanes inside), take branches on a scalar compare, and may end
//! some lanes partway. They are integer only, set every
//! register before reading it, read only LDS that is written, read only
//! lanes that run (the one-wave64 form reads another as the host does,
//! which D3D leaves undefined), and run in whole console waves of 1-D
//! groups, so that both forms store one result each. `VICTORY_HALVES_PROGRAMS` sets how many
//! programs a run takes (default 64) and `VICTORY_HALVES_SEED` the first
//! seed (default 1); a difference names its seed and lists its program.
//! `VICTORY_HALVES_TRACE=1` lists each program and names each step as it
//! starts, for a run that ends in a crash.

use std::collections::BTreeMap;

use super::halves_tests::{
    PROLOGUE, S_ENDPGM, S_WAITCNT_LGKM, dpp16, lowered, native, prefix_sum_section, shader, sop1,
    store, vop1, vop2, vop3,
};
use super::host::{Dispatch, Host, Root, Stride, View};

/// The registers a program computes in, stored per thread at the end.
const STATE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// A literal follows the instruction.
const LIT: u16 = 255;

/// VGPR `r` as a source operand.
const fn v(r: u8) -> u16 {
    256 + r as u16
}

/// The inline integer `n` (0 to 64).
const fn k(n: u32) -> u16 {
    128 + n as u16
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Never 0, which xorshift keeps.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// In `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }
}

/// A random program and the group it runs in.
struct Program {
    words: Vec<u32>,
    threads: u32,
    lds_bytes: u32,
}

/// The program of `seed`: registers set from the thread id, 4 to 12 random
/// steps, then `v1`-`v8` stored at the thread's 32 bytes.
fn program(seed: u64) -> Program {
    let mut r = Rng::new(seed);
    let threads = r.pick(&[64, 64, 128, 256]);
    let lds = r.below(3) == 0;
    let mut g = Gen {
        r,
        w: PROLOGUE.to_vec(),
        threads,
        lds,
        lds_written: false,
        same: [false; 9],
        lanes: [0, 64],
        in_loop: false,
        ended: None,
    };
    g.w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    g.init(&STATE);
    for _ in 0..4 + g.r.below(9) {
        g.step();
    }
    for (slot, &reg) in STATE.iter().enumerate() {
        g.w.extend(store(reg, 4 * slot as u32));
    }
    if let Some(at) = g.ended {
        g.w[at] |= (g.w.len() - at - 1) as u32; // to s_endpgm
    }
    g.w.push(S_ENDPGM);
    Program {
        words: g.w,
        threads,
        lds_bytes: if lds { threads * 4 } else { 0 },
    }
}

/// A program being written.
struct Gen {
    r: Rng,
    w: Vec<u32>,
    threads: u32,
    /// The program has LDS, a dword per thread.
    lds: bool,
    /// Every thread has written its LDS dword.
    lds_written: bool,
    /// By register: it holds one value in every lane of the console wave (a
    /// lane read, a mask's count), or in every lane of the branch that wrote
    /// it, whose compiler may see the value as one. No DPP reads such a
    /// register: on the Radeon iGPU of a Ryzen 9 7900X (driver
    /// 32.0.21045.29003) a DPP read of a DPP read of such a value crashes
    /// the driver's shader compiler, in the process, when it makes the
    /// one-wave64 form's pipeline (any of `quad_perm`, `row_shr`,
    /// `row_mirror`, `row_share` and `row_xmask` twice after
    /// `v_readlane_b32` or `v_readfirstlane_b32`; the module validates, and
    /// the halves form makes its pipeline).
    same: [bool; 9],
    /// The console lanes that take the branches around the steps being
    /// written (`low..high`).
    lanes: [u32; 2],
    /// Steps are being written inside a loop, where a lane may have left
    /// (so no lane reads: the one-wave64 form reads a lane that left as the
    /// host does, which D3D leaves undefined).
    in_loop: bool,
    /// Where some lanes left the program: the `s_cbranch_execz` that skips a
    /// wave with none left to the end, its offset set at the end. After it,
    /// no step reads a lane that may have left (permutes, swizzles,
    /// whole-wave sections that enable every lane, LDS other threads wrote).
    ended: Option<usize>,
}

impl Gen {
    fn reg(&mut self) -> u8 {
        self.r.pick(&STATE)
    }

    /// A register whose lanes may hold different values, if one does.
    fn varying(&mut self) -> Option<u8> {
        let regs: Vec<u8> = STATE
            .iter()
            .copied()
            .filter(|&r| !self.same[r as usize])
            .collect();
        (!regs.is_empty()).then(|| self.r.pick(&regs))
    }

    /// Notes what `d` now holds: one value in every lane (of its branch) or
    /// not.
    fn wrote(&mut self, d: u8, same: bool) {
        self.same[d as usize] = same;
    }

    /// `regs` set from the thread id, each its own way; `v1` is the console
    /// lane (`v_mbcnt`), which also makes the program read its console
    /// lanes.
    fn init(&mut self, regs: &[u8]) {
        for &d in regs {
            if d == 1 {
                self.w.extend([
                    0xD765_0001,
                    0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
                    0xD766_0001,
                    0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
                ]);
                self.wrote(1, false);
                continue;
            }
            let (m, a) = (1 + self.r.below(1000), self.r.next());
            self.w.extend([vop2(d, 0x0B, LIT, 0), m]); // v_mul_u32_u24 d, m, v0
            self.w.extend([vop2(d, 0x25, LIT, d), a]); // v_add_nc_u32 d, a, d
            self.wrote(d, false);
        }
    }

    fn step(&mut self) {
        let ended = self.ended.is_some();
        match self.r.below(21) {
            0 | 1 => self.alu(),
            2 | 3 => self.dpp(),
            4 | 5 if ended => self.alu(),
            4 => self.swizzle(),
            5 => self.bpermute(),
            6 => self.read_lane(),
            7 => self.write_lane(),
            8 => self.first_lane(),
            9 | 10 => self.ballot(),
            11 | 12 => self.branch(0),
            13 => self.uniform(),
            14 | 15 if self.lds && !ended => self.lds_step(),
            14 | 15 => self.alu(),
            16 => self.lane_loop(),
            17 => self.early_end(),
            18 => self.around(),
            19 => self.into(),
            _ if ended => self.alu(),
            _ => self.section(),
        }
    }

    /// A branch on a scalar compare (`s5` is 0: with `s_cmp_eq_u32 s5, 0`
    /// the whole group takes it, with 1 none) to the join of a lane branch
    /// ([`Gen::branch`]) right after it, around the region its lanes run.
    /// Where every lane takes the branches around it.
    fn around(&mut self) {
        if self.lanes != [0, 64] || self.in_loop {
            return self.alu();
        }
        let none = self.r.below(2);
        self.w.extend([
            0xBE94_047E,             // s_mov_b64 s[20:21], exec (the join restores it)
            0xBF06_8005 | none << 8, // s_cmp_eq_u32 s5, none
        ]);
        let at = self.w.len();
        self.w.push(0);
        self.branch(0);
        // To the branch's join, its `s_mov_b64 exec, s[20:21]`.
        self.w[at] = 0xBF85_0000 | (self.w.len() - at - 2) as u32; // s_cbranch_scc1
    }

    /// A lane branch (console lanes below a random one, or from one) of a
    /// step, then steps that read across lanes, which a branch on a scalar
    /// compare before it (taken by the whole group or by none, as
    /// [`Gen::around`]) may take the whole group straight into, with every
    /// lane. Where every lane takes the branches around it.
    fn into(&mut self) {
        if self.lanes != [0, 64] || self.in_loop {
            return self.alu();
        }
        let none = self.r.below(2);
        let (from, lanes) = (self.r.below(2) == 0, 1 + self.r.below(63));
        // v_cmp_le_u32 (from) or v_cmp_gt_u32 (below) vcc, lanes, v10
        let op: u32 = if from { 0xC3 } else { 0xC4 };
        self.w.extend([
            0xBE94_047E,             // s_mov_b64 s[20:21], exec (the join restores it)
            0xBF06_8005 | none << 8, // s_cmp_eq_u32 s5, none
        ]);
        let jump = self.w.len();
        self.w.push(0);
        self.w.extend([
            vop2(10, 0x1B, k(63), 0), // v_and_b32 v10, 63, v0
            0x7C00_0000 | op << 17 | 10 << 9 | k(lanes) as u32,
            0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        ]);
        let past = self.w.len();
        self.w.push(0);
        // Lane reads name the branch's lanes, which run either way.
        self.lanes = if from { [lanes, 64] } else { [0, lanes] };
        self.alu();
        let mid = self.w.len();
        for _ in 0..1 + self.r.below(2) {
            match self.r.below(4) {
                0 => self.first_lane(),
                1 => self.ballot(),
                2 => self.dpp(),
                _ => self.read_lane(),
            }
        }
        self.lanes = [0, 64];
        self.w[jump] = 0xBF85_0000 | (mid - jump - 1) as u32; // s_cbranch_scc1 into it
        self.w[past] = 0xBF88_0000 | (self.w.len() - past - 1) as u32; // s_cbranch_execz past it
        self.w.push(0xBEFE_0414); // s_mov_b64 exec, s[20:21]
    }

    /// A lane's own arithmetic.
    fn alu(&mut self) {
        let (d, a, b) = (self.reg(), self.reg(), self.reg());
        let (n, x) = (self.r.below(65), self.r.next());
        let op = self.r.below(6);
        match op {
            0 => self.w.push(vop2(d, 0x25, v(b), a)), // v_add_nc_u32 d, b, a
            1 => self.w.push(vop2(d, 0x1D, v(b), a)), // v_xor_b32 d, b, a
            2 => self.w.extend([vop2(d, 0x1B, LIT, a), x]), // v_and_b32 d, x, a
            3 => self.w.push(vop2(d, 0x1C, k(n), a)), // v_or_b32 d, n, a
            4 => self.w.push(vop2(d, 0x0B, k(n.max(1)), a)), // v_mul_u32_u24 d, n, a
            _ => self.w.push(vop2(d, 0x1A, k(1 + n % 8), a)), // v_lshlrev_b32 d, n, a
        }
        let (same_a, same_b) = (self.same[a as usize], self.same[b as usize]);
        self.wrote(d, same_a && (op > 1 || same_b));
    }

    /// `v_mov_b32` or `v_add_nc_u32` with a DPP16 source: a quad
    /// permutation, a row shift or rotation, a mirror, a row share or xor.
    fn dpp(&mut self) {
        let Some(a) = self.varying() else {
            return self.alu();
        };
        let (d, b) = (self.reg(), self.reg());
        let ctrl = match self.r.below(6) {
            0 => self.r.below(256),
            1 => 0x101 + self.r.below(15),
            2 => 0x111 + self.r.below(15),
            3 => 0x121 + self.r.below(15),
            4 => 0x140 + self.r.below(2),
            _ => 0x150 + self.r.below(32),
        } as u16;
        if self.r.below(2) == 0 {
            self.w.extend([vop1(d, 0x01, 0xFA), dpp16(a, ctrl)]); // v_mov_b32 d, a (DPP)
        } else {
            self.w.extend([vop2(d, 0x25, 0xFA, b), dpp16(a, ctrl)]); // v_add_nc_u32 d, a, b (DPP)
        }
        self.wrote(d, false);
    }

    /// `ds_swizzle_b32` in its bit-mask mode (and, or and xor masks of five
    /// bits) or its quad mode.
    fn swizzle(&mut self) {
        let (d, a) = (self.reg(), self.reg());
        let offset = if self.r.below(2) == 0 {
            self.r.below(32) | self.r.below(32) << 5 | self.r.below(32) << 10
        } else {
            0x8000 | self.r.below(256)
        };
        self.w.extend([
            0xD8D4_0000 | offset,
            (d as u32) << 24 | a as u32, // ds_swizzle_b32 d, a offset:...
            S_WAITCNT_LGKM,
        ]);
        let same = self.same[a as usize];
        self.wrote(d, same);
    }

    /// `ds_bpermute_b32` of `x` from the lane in the low six bits of `a`.
    fn bpermute(&mut self) {
        let (d, a, x) = (self.reg(), self.reg(), self.reg());
        self.w.extend([
            vop2(9, 0x1B, k(63), a), // v_and_b32 v9, 63, a
            vop2(9, 0x1A, k(2), 9),  // v_lshlrev_b32 v9, 2, v9
            0xDACC_0000,
            (d as u32) << 24 | (x as u32) << 8 | 9, // ds_bpermute_b32 d, v9, x
            S_WAITCNT_LGKM,
        ]);
        let same = self.same[x as usize];
        self.wrote(d, same);
    }

    /// `v_readlane_b32` of a lane named by a constant or an SGPR; in a
    /// branch, of a lane that takes it (the one-wave64 form reads a lane
    /// that does not as the host does, which D3D leaves undefined).
    fn read_lane(&mut self) {
        let (d, a) = (self.reg(), self.reg());
        let [low, high] = self.lanes;
        let lane = if low < high {
            low + self.r.below(high - low)
        } else {
            self.r.below(64)
        };
        if self.r.below(2) == 0 {
            self.w.extend(vop3(12, 0x360, v(a), k(lane), 0)); // v_readlane_b32 s12, a, lane
        } else {
            self.w.push(sop1(24, 0x03, 128 + lane as u8)); // s_mov_b32 s24, lane
            self.w.extend(vop3(12, 0x360, v(a), 24, 0)); // v_readlane_b32 s12, a, s24
        }
        self.w.push(vop1(d, 0x01, 12)); // v_mov_b32 d, s12
        self.wrote(d, true);
    }

    /// `v_writelane_b32` of a scalar to a constant lane.
    fn write_lane(&mut self) {
        let (d, lane, x) = (self.reg(), self.r.below(64), self.r.next());
        self.w.extend([sop1(13, 0x03, 255), x]); // s_mov_b32 s13, x
        self.w.extend(vop3(d, 0x361, 13, k(lane), 0)); // v_writelane_b32 d, s13, lane
        self.wrote(d, false);
    }

    /// `v_readfirstlane_b32`, every lane on.
    fn first_lane(&mut self) {
        let (d, a) = (self.reg(), self.reg());
        self.w.push(vop1(14, 0x02, v(a))); // v_readfirstlane_b32 s14, a
        self.w.push(vop1(d, 0x01, 14)); // v_mov_b32 d, s14
        self.wrote(d, true);
    }

    /// A compare's lane mask, read as data (a count, a first lane, either
    /// dword, `v_mbcnt` over it) or as lane booleans (`v_cndmask_b32`).
    fn ballot(&mut self) {
        let (d, a, b) = (self.reg(), self.reg(), self.reg());
        let op = self.r.pick(&[0xC1u32, 0xC2, 0xC4, 0xC5]);
        let src0 = if self.r.below(2) == 0 {
            v(a)
        } else {
            k(self.r.below(65))
        };
        // v_cmp_{lt,eq,gt,ne}_u32 vcc, src0, b
        self.w
            .push(0x7C00_0000 | op << 17 | (b as u32) << 9 | src0 as u32);
        let read = self.r.below(6);
        match read {
            0 => self.w.extend([sop1(15, 0x10, 106), vop1(d, 0x01, 15)]), // s_bcnt1_i32_b64 s15, vcc
            1 => self.w.extend([sop1(15, 0x14, 106), vop1(d, 0x01, 15)]), // s_ff1_i32_b64 s15, vcc
            2 => self.w.push(vop1(d, 0x01, 106)),                         // v_mov_b32 d, vcc_lo
            3 => self.w.push(vop1(d, 0x01, 107)),                         // v_mov_b32 d, vcc_hi
            4 => {
                self.w.extend(vop3(d, 0x765, 106, k(0), 0)); // v_mbcnt_lo_u32_b32 d, vcc_lo, 0
                self.w.extend(vop3(d, 0x766, 107, v(d), 0)); // v_mbcnt_hi_u32_b32 d, vcc_hi, d
            }
            _ => self.w.push(vop2(d, 0x01, v(a), b)), // v_cndmask_b32 d, a, b, vcc
        }
        // A count, a first lane and either dword are one value per wave.
        self.wrote(d, read < 4);
    }

    /// A branch the lanes below a random console lane take, or the lanes
    /// from one (`depth` the branches around it): inside, a lane's own
    /// arithmetic, DPP, lane reads, lane masks, first lanes, a write of its
    /// LDS dword once every thread has one, LDS reads in a group of one
    /// wave, and at the top a branch of its own. Where only one half's lanes
    /// take it, that half runs alone; where both halves' do, the region runs
    /// wave-wide.
    fn branch(&mut self, depth: u32) {
        let (from, lanes) = (self.r.below(2) == 0, 1 + self.r.below(63));
        // The EXEC each branch saves: s[20:21], s[22:23] inside.
        let saved = 20 + 2 * depth;
        // v_cmp_le_u32 (from) or v_cmp_gt_u32 (below) vcc, lanes, v10
        let op: u32 = if from { 0xC3 } else { 0xC4 };
        self.w.extend([
            vop2(10, 0x1B, k(63), 0), // v_and_b32 v10, 63, v0
            0x7C00_0000 | op << 17 | 10 << 9 | k(lanes) as u32,
            0xBE80_246A | saved << 16, // s_and_saveexec_b64 s[saved:], vcc
        ]);
        let at = self.w.len();
        self.w.push(0);
        let around = self.lanes;
        self.lanes = if from {
            [around[0].max(lanes), around[1]]
        } else {
            [around[0], around[1].min(lanes)]
        };
        for _ in 0..1 + self.r.below(3) {
            match self.r.below(10) {
                0 if self.lds_written && self.ended.is_none() => self.lds_write(),
                1 => self.ballot(),
                2 => self.first_lane(),
                3 if self.lds_written && self.threads == 64 && self.ended.is_none() => {
                    self.lds_read()
                }
                4 if !self.in_loop => self.read_lane(),
                5 => self.dpp(),
                6 if depth == 0 => self.branch(1),
                7 if depth == 0 => self.lane_loop(),
                _ => self.alu(),
            }
        }
        self.lanes = around;
        self.w[at] = 0xBF88_0000 | (self.w.len() - at - 1) as u32; // s_cbranch_execz past it
        self.w.push(0xBEFE_0400 | saved); // s_mov_b64 exec, s[saved:]
    }

    /// A loop each console lane goes round a number of times of its own (1
    /// to 4, two bits of the lane, in `v13`; the round in `v14`, which no
    /// other step writes), each round's exit narrowing EXEC: inside,
    /// a lane's own arithmetic, DPP, lane masks, first lanes, LDS writes and
    /// (in a group of one wave) reads, and a branch where the loop is not in
    /// one.
    fn lane_loop(&mut self) {
        let shift = self.r.below(5);
        self.w.extend([
            vop2(10, 0x1B, k(63), 0),     // v_and_b32 v10, 63, v0
            vop2(13, 0x16, k(shift), 10), // v_lshrrev_b32 v13, shift, v10
            vop2(13, 0x1B, k(3), 13),     // v_and_b32 v13, 3, v13
            vop2(13, 0x25, k(1), 13),     // v_add_nc_u32 v13, 1, v13
            vop1(14, 0x01, k(0)),         // v_mov_b32 v14, 0
            0xBE9A_047E,                  // s_mov_b64 s[26:27], exec
        ]);
        let top = self.w.len();
        let outside = self.lanes == [0, 64];
        self.in_loop = true;
        for _ in 0..1 + self.r.below(3) {
            match self.r.below(8) {
                0 => self.ballot(),
                1 => self.first_lane(),
                2 => self.dpp(),
                3 if self.lds_written && self.ended.is_none() => self.lds_write(),
                4 if self.lds_written && self.threads == 64 && self.ended.is_none() => {
                    self.lds_read()
                }
                5 if outside => self.branch(1),
                _ => self.alu(),
            }
        }
        self.in_loop = false;
        self.w.extend([
            vop2(14, 0x25, k(1), 14), // v_add_nc_u32 v14, 1, v14
            0x7C00_0000 | 0xC1 << 17 | 13 << 9 | v(14) as u32, // v_cmp_lt_u32 vcc, v14, v13
            0x87FE_6A7E,              // s_and_b64 exec, exec, vcc
        ]);
        let back = top as i32 - (self.w.len() as i32 + 1);
        self.w.push(0xBF89_0000 | (back as u32 & 0xFFFF)); // s_cbranch_execnz to the top
        self.w.push(0xBEFE_041A); // s_mov_b64 exec, s[26:27]
    }

    /// The console lanes below a random one, or from one, leave the program:
    /// EXEC loses them, and a wave with none left skips to the end. Once a
    /// program; the lanes left are those lane reads may name.
    fn early_end(&mut self) {
        if self.ended.is_some() {
            return self.alu();
        }
        let (from, lanes) = (self.r.below(2) == 0, 1 + self.r.below(63));
        // v_cmp_le_u32 (from) or v_cmp_gt_u32 (below) vcc, lanes, v10
        let op: u32 = if from { 0xC3 } else { 0xC4 };
        self.w.extend([
            vop2(10, 0x1B, k(63), 0), // v_and_b32 v10, 63, v0
            0x7C00_0000 | op << 17 | 10 << 9 | k(lanes) as u32,
            0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
            0xBF88_0000, // s_cbranch_execz to the end (set there)
        ]);
        self.ended = Some(self.w.len() - 1);
        self.lanes = if from { [0, lanes] } else { [lanes, 64] };
    }

    /// A read of the LDS dword of another thread of the console wave.
    fn lds_read(&mut self) {
        let (d, other) = (self.reg(), self.r.below(64));
        self.w.extend([
            vop2(12, 0x1D, LIT, 0), // v_xor_b32 v12, other, v0
            other,
            vop2(12, 0x1A, k(2), 12), // v_lshlrev_b32 v12, 2, v12
            0xD8D8_0000,
            (d as u32) << 24 | 12, // ds_read_b32 d, v12
            S_WAITCNT_LGKM,
        ]);
        self.wrote(d, false);
    }

    /// A branch on a scalar compare (always taken here), around steps that
    /// may read across lanes.
    fn uniform(&mut self) {
        self.w.push(0xBF06_8005); // s_cmp_eq_u32 s5, 0
        let at = self.w.len();
        self.w.push(0);
        for _ in 0..1 + self.r.below(2) {
            match self.r.below(4) {
                0 => self.alu(),
                1 => self.dpp(),
                2 => self.read_lane(),
                _ => self.ballot(),
            }
        }
        self.w[at] = 0xBF84_0000 | (self.w.len() - at - 1) as u32; // s_cbranch_scc0 past it
    }

    /// Every arriving lane writes a register at its thread's LDS dword.
    fn lds_write(&mut self) {
        let x = self.reg();
        self.w.extend([
            vop2(11, 0x1A, k(2), 0), // v_lshlrev_b32 v11, 2, v0
            0xD834_0000,
            (x as u32) << 8 | 11, // ds_write_b32 v11, x
        ]);
    }

    /// LDS: every thread writes its dword (unless one was written before,
    /// sometimes), then reads the dword of another thread of the group, with
    /// the console's barriers around the read in a group of several waves.
    fn lds_step(&mut self) {
        if !self.lds_written || self.r.below(2) == 0 {
            self.lds_write();
            self.lds_written = true;
        }
        let several = self.threads > 64;
        if several {
            self.w.extend([S_WAITCNT_LGKM, 0xBF8A_0000]); // s_waitcnt lgkmcnt(0); s_barrier
        }
        let (d, other) = (self.reg(), self.r.below(self.threads));
        self.w.extend([
            vop2(12, 0x1D, LIT, 0), // v_xor_b32 v12, other, v0
            other,
            vop2(12, 0x1A, k(2), 12), // v_lshlrev_b32 v12, 2, v12
            0xD8D8_0000,
            (d as u32) << 24 | 12, // ds_read_b32 d, v12
            S_WAITCNT_LGKM,
        ]);
        self.wrote(d, false);
        if several {
            self.w.push(0xBF8A_0000); // s_barrier
        }
    }

    /// The console compiler's exclusive prefix sum of the thread ids, into
    /// `v2`; what else it writes is set again.
    fn section(&mut self) {
        self.w.extend(prefix_sum_section());
        self.wrote(2, false);
        self.init(&[1, 3, 4, 5]);
    }
}

/// Names `what` before it starts, with `VICTORY_HALVES_TRACE=1`.
fn trace(what: std::fmt::Arguments) {
    if std::env::var_os("VICTORY_HALVES_TRACE").is_some_and(|v| v == "1") {
        eprintln!("{what}");
    }
}

/// How many programs a run takes, and its first seed.
fn plan() -> (u64, u64) {
    let var = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    (
        var("VICTORY_HALVES_SEED", 1),
        var("VICTORY_HALVES_PROGRAMS", 64),
    )
}

/// One group of `threads` running `s`: eight dwords per thread.
fn run(host: &Host, s: &super::HostShader, threads: u32) -> Result<Vec<u32>, String> {
    let (root, pso) = host
        .compute_pipeline(s)
        .map_err(|e| format!("no pipeline: {e}"))?;
    let ud = vec![0u32; s.user_data_dwords().max(1) as usize];
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(threads * 8),
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
        .map_err(|e| format!("dispatch: {e}"))?;
    Ok(out[0].iter().take(threads as usize * 8).copied().collect())
}

/// `words` as the disassembler reads them.
fn listing(words: &[u32]) -> Result<String, String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::gfx10::decode(&code)
        .map(|insts| {
            insts
                .iter()
                .map(|i| i.text(true))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .map_err(|e| format!("{e:?}"))
}

/// Random wave64 programs translated in halves store what they store as
/// one wave64, lane by lane, on every hardware adapter (see the module).
#[test]
fn random_wave64_programs_in_halves_match_one_wave64() {
    let (first, count) = plan();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    let Some((reference, wave64)) = hosts.iter().find(|(_, h)| h.wave32() == Ok(false)) else {
        eprintln!("no hardware adapter runs 64-lane waves; skipped");
        return;
    };
    let mut refused: BTreeMap<String, u32> = BTreeMap::new();
    let (mut compared, mut failed) = (0, Vec::new());
    for seed in first..first + count {
        let p = program(seed);
        let text = listing(&p.words)
            .unwrap_or_else(|e| panic!("seed {seed}: the program does not decode: {e}"));
        trace(format_args!(
            "seed {seed}: translating
{text}"
        ));
        let n = native(&p.words, true, p.threads, p.lds_bytes);
        let halves = match lowered(&n, true) {
            Ok(l) => shader(l),
            Err(e) => {
                let why = e.to_string();
                let why = why.split_once(": ").map_or(why.as_str(), |w| w.1);
                *refused.entry(why.to_string()).or_default() += 1;
                continue;
            }
        };
        let whole = match lowered(&n, false) {
            Ok(l) => shader(l),
            Err(e) => {
                *refused.entry(format!("as one wave64: {e}")).or_default() += 1;
                continue;
            }
        };
        trace(format_args!("seed {seed}: one wave64 on {reference}"));
        let want = run(wave64, &whole, p.threads)
            .unwrap_or_else(|e| panic!("seed {seed}: one wave64 on {reference}: {e}"));
        for (adapter, host) in &hosts {
            trace(format_args!("seed {seed}: halves on {adapter}"));
            let got = run(host, &halves, p.threads)
                .unwrap_or_else(|e| panic!("seed {seed}: halves on {adapter}: {e}"));
            let lanes: Vec<usize> = (0..p.threads as usize)
                .filter(|&l| got[l * 8..l * 8 + 8] != want[l * 8..l * 8 + 8])
                .collect();
            if let Some(&l) = lanes.first() {
                failed.push(format!(
                    "seed {seed}, {} threads, {} LDS bytes, on {adapter}: {} lanes differ; lane {l}: {:x?}, as one wave64 {:x?}\n{text}",
                    p.threads,
                    p.lds_bytes,
                    lanes.len(),
                    &got[l * 8..l * 8 + 8],
                    &want[l * 8..l * 8 + 8],
                ));
            }
        }
        compared += 1;
    }
    eprintln!(
        "{count} programs from seed {first}: {compared} compared on {} adapters against one wave64 on {reference}; refused in halves: {refused:#?}",
        hosts.len()
    );
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    // Most programs should run in halves, or the test tests little.
    assert!(
        compared * 2 >= count,
        "{compared} of {count} programs compared"
    );
}
