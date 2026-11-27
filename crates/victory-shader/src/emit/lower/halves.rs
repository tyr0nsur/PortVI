//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact; reads into the other half have no
//! host form and are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), `v_permlane16_b32` and `v_permlanex16_b32`
//!   (a row of 16 or its partner row in the same half, by host lane;
//!   fetch-inactive only in whole halves, see `permlane`), and
//!   `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code),
//!   `v_writelane_b32` (the lane named takes the scalar) and the row a DPP
//!   row mask names (rows 2 and 3 are the upper half's) are decided by the
//!   console lane ([`Lower::console_lane`]);
//! * across halves, refused: ballots (a mask read as data holds the other
//!   half's lanes), `v_readlane_b32` of a lane that may be in the other half
//!   (SGPR spills read back are not lane reads), and whole-wave reductions
//!   and prefix sums;
//! * `v_readfirstlane_b32` reads the host wave's first lane, which in the
//!   upper half is console lane 32 or later: kept for a value the same in
//!   every lane of the console wave ([`lane_uniform`], what the compiler
//!   reads first lanes for) and in a group of 32 threads or fewer (one host
//!   wave holds the whole console wave), refused otherwise ([`FIRST_ACROSS`]).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! When one console wave runs on two host waves, its straight-line LDS
//! reads and writes can synchronize at each access. Other exchanges need
//! a proof that all participating lanes reach the same synchronization.
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name. Which values are the same in every lane of the console
//! wave ([`lane_uniform`]): justrealyme's analysis (chippy01/VIctory#1),
//! ported from fxceless/victory-fork (#497 there).

use super::*;
use crate::emit::bitcode::Rmw;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

impl Lower<'_> {
    /// Each lane reads `data` of the lane its address plus the offset
    /// names in dwords within its own 32-lane half (RDNA 2 ISA,
    /// section 10.4.4: wave64 permutes lanes 0..31 and 32..63
    /// independently); 0 from a lane EXEC leaves out. Native wave64
    /// keeps the current hardware lane's half; in halves (wave64 on
    /// a 32-lane host) the host wave is that half, so the operand
    /// is the local lane 0..31; native wave32 is local too.
    pub(super) fn bpermute_lane(&mut self, a: V, x: V, offset: u32, guard: Option<V>) -> V {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (o, two, mask31) = (k(self, offset), k(self, 2), k(self, 31));
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let lane = self.s.m.bin(c, Bin::LShr, at, two);
        let within_half = self.s.m.bin(c, Bin::And, lane, mask31);
        let lane = if self.x.native.wave64 && !self.halves {
            let me = self.lane_index();
            let not31 = k(self, !31);
            let base = self.s.m.bin(c, Bin::And, me, not31);
            self.s.m.bin(c, Bin::Or, base, within_half)
        } else {
            within_half
        };
        let r = self.read_lane(x, lane);
        let on = guard.unwrap_or_else(|| self.s.m.bool(true));
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        let (lo, hi) = (
            self.s.m.extract(c, ballot, 0),
            self.s.m.extract(c, ballot, 1),
        );
        let running = self.lane_bit_of(c, lane, lo, hi);
        let z = k(self, 0);
        self.s.m.select(c, running, r, z)
    }

    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Adapted from VIctory-NVIDIA's group synchronization for split waves.
    /// Only one console wave is involved, every lane follows the same block,
    /// and none of its memory accesses is predicated. Unused reserved LDS
    /// requires no synchronization, even in a group with several waves.
    pub(super) fn halves_lds(x: &Parts, group: u32, lds_dwords: u32) -> Result<bool, Why> {
        if lds_dwords == 0 || group <= HALF {
            return Ok(false);
        }
        let mut accesses = x.p.blocks.iter().enumerate().flat_map(|(b, block)| {
            block.insts.iter().enumerate().filter_map(move |(i, inst)| {
                let name = inst.name();
                (matches!(inst.fields, Fields::Ds { .. }) && !name.starts_with("ds_swizzle")
                    && !name.contains("permute")).then_some((b, i, name))
            })
        }).peekable();
        if accesses.peek().is_none() {
            return Ok(false);
        }
        if group <= 64 && matches!(x.st.exits.as_slice(), [LaneExit::End])
            && accesses.all(|(b, i, name)| x.st.guards[b][i].is_none()
                && (name.starts_with("ds_read") || name.starts_with("ds_write")))
        {
            return Ok(true);
        }
        Err("LDS ordering across 32-lane halves has not been proved")
    }

    /// An unmodified 1D thread id at program entry has a known first value
    /// in each full console wave. Once EXEC changes, its first set bit can
    /// be any lane, including one in the opposite half.
    pub(super) fn first_is_wave_base(
        &self,
        (b, i): (usize, usize),
        operand: Option<&Operand>,
    ) -> bool {
        let Some(Operand::Vector { reg, .. }) = operand else {
            return false;
        };
        let StageIo::Cs(cs) = &self.x.native.io else {
            return false;
        };
        let [width, y, z] = cs.threads;
        let entry = &self.x.p.blocks[0];
        b == 0
            && entry.preds.is_empty()
            && y == 1
            && z == 1
            && width.is_multiple_of(64)
            && entry.insts[i]
                .read(Reg::V(*reg))
                .is_some_and(|v| Some(v) == wave_tid(self.x))
            && !entry.insts[..i].iter().any(|op| {
                op.writes.iter().any(|(r, _)|
                matches!(*r, Reg::S(n) if n == crate::ir::EXEC || n == crate::ir::EXEC + 1))
            })
    }

    /// A flat console thread ID made once in the entry block for the whole group.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let wave = self.s.wave_size.expect("set with console lanes");
        let [x, y, z] = self.s.threads.expect("compute thread group");
        let group = x * y * z;
        // A multidimensional group can be tiled across waves. The first
        // flat IDs of two host waves can therefore round to the same base.
        // The counter occupies one private dword of D3D's 32 KiB of TGSM.
        if group > wave && (y > 1 || z > 1) && self.s.lds_dwords < 8192 {
            let f = self.ticketed_flat(wave, group);
            self.relabelled = Some(f);
            self.s.flags |= flags::WAVE_OPS;
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// Adapted from VIctory-NVIDIA #30 and Vice-Tay c125c0c. Full waves
    /// reserve aligned runs from the start and partial waves reserve from
    /// the end. A group has at most 1024 threads, so the two 16-bit counters
    /// cannot carry into each other. The console's LDS is separately bounded.
    fn ticketed_flat(&mut self, wave: u32, group: u32) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let tickets = self.s.m.global("console_thread_tickets", i32t, 3);
        self.s.m.atomic(e, Rmw::Xchg, tickets, zero);
        let sync = self.s.m.i32(1 | 8);
        self.s.op(e, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[sync]);

        let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(1));
        let count = self.s.op(e, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[one, sum, unsigned]);
        let rank = self.s.op(e, "wavePrefixOp.i32", Attr::NoUnwind, i32t, 121, &[one, sum, unsigned]);
        let width = self.s.m.i32(wave);
        let full = self.s.m.cmp(e, pred::EQ, count, width);
        let leader = self.s.m.cmp(e, pred::EQ, rank, zero);
        let shift = self.s.m.i32(16);
        let partial = self.s.m.bin(e, Bin::Shl, count, shift);
        let reservation = self.s.m.select(e, full, width, partial);
        let add = self.s.m.select(e, leader, reservation, zero);
        let old = self.s.m.atomic(e, Rmw::Add, tickets, add);
        let found = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        let low_mask = self.s.m.i32(0xffff);
        let front = self.s.m.bin(e, Bin::And, found, low_mask);
        let reserved = self.s.m.bin(e, Bin::LShr, found, shift);
        let size = self.s.m.i32(group);
        let end = self.s.m.bin(e, Bin::Sub, size, reserved);
        let back = self.s.m.bin(e, Bin::Sub, end, count);
        let base = self.s.m.select(e, full, front, back);
        self.s.m.bin(e, Bin::Add, base, rank)
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Whether the group fits one host wave: a console wave64 of 32 threads
    /// or fewer has no lane in the upper half.
    pub(super) fn one_host_wave(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() <= HALF)
    }

    /// Refuses a read into the other half of the console wave in halves.
    pub(super) fn same_half(&self, what: Why) -> Result<(), Why> {
        if self.halves { Err(what) } else { Ok(()) }
    }
}

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, and the value is not known to be the same there.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)";

/// Every per-lane branch (by its block) with its region: the blocks
/// reachable from it that do not post-dominate it, which only some lanes
/// run. Not branches on uniform conditions. None with an indirect exit.
fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let end = st.exits.len();
    let n = end + 1;
    let mut succ = vec![Vec::new(); n];
    let mut pred = vec![Vec::new(); n];
    for (b, exit) in st.exits.iter().enumerate() {
        match *exit {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => succ[b].extend([taken, not_taken]),
            LaneExit::End => succ[b].push(end),
            LaneExit::Indirect => return None,
        }
        for &t in &succ[b] {
            pred[t].push(b);
        }
    }
    // Each divergent region ends at its immediate post-dominator. Computing
    // dominators of the reversed CFG avoids a dense block-by-block bit matrix.
    let order = crate::ir::graph_rpo(n, end, |b, i| pred[b].get(i).copied());
    if order.len() != n {
        return None;
    }
    let mut index = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    let post = crate::ir::graph_idoms(&order, &index, |b| &succ[b]);
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else {
            continue;
        };
        if st
            .bools
            .support(when)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(_)))
        {
            continue;
        }
        let mut seen = vec![false; end];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if b == end || b == post[d] || seen[b] {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The thread id's x (`v0` on entry) of a wave64 compute program whose
/// console waves are runs of 64 ids along x (a group 64-aligned along x,
/// or one row): `id >> 6` and up is one value per console wave.
fn wave_tid(x: &Parts) -> Option<Value> {
    let StageIo::Cs(cs) = &x.native.io else {
        return None;
    };
    let [tx, ty, tz] = cs.threads;
    if !x.native.wave64 || !(tx % 64 == 0 || ty * tz == 1) {
        return None;
    }
    x.p.inputs
        .iter()
        .find(|(r, _)| *r == Reg::V(0))
        .map(|(_, v)| *v)
}

/// VGPR values that are the same in every lane of a console wave holding
/// them, so that a first-lane read in either half gives the console's
/// value: written lane by lane from scalars, constants and such values, by
/// an instruction every arriving lane runs or that runs under a condition
/// itself the same in every lane (then only over an old value that is such
/// a value too); loaded from LDS, or from a buffer through a scalar
/// descriptor, at such an address; or the thread id's x shifted right past
/// the console wave ([`wave_tid`]). The console compiler keeps wave-uniform
/// data in VGPRs and fetches it with `v_readfirstlane` (a typed buffer's
/// element, say, which only a vector load reads). A scalar is one value,
/// except the dwords a waterfall loop reads, which are each lane's own.
///
/// A phi counts where every predecessor is a block every lane of the wave
/// reaches together: no branch on a lane-dependent condition can have split
/// them (a branch on alike values does not). The analysis runs to a fixed
/// point from nothing, so a loop-carried value that needs itself to be
/// alike never is.
pub(super) fn lane_uniform(x: &Parts) -> HashSet<Value> {
    let (p, st) = (x.p, x.st);
    let wave_tid = wave_tid(x);
    let regions = lane_regions(st);
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let per_lane = |v: Value| x.res.waterfall.reads.contains(&v);
    let mut same: HashSet<Value> = HashSet::new();
    loop {
        let alike_atom = |same: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => same.contains(&v),
            _ => false,
        };
        let alike_bool = |same: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| alike_atom(same, a))
        };
        // Blocks every lane of a console wave reaches together.
        let wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for (d, region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[*d] else {
                        continue;
                    };
                    if alike_bool(&same, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => vec![false; st.exits.len()],
        };
        let mut next: HashSet<Value> = HashSet::new();
        for b in crate::ir::reverse_postorder(&p.blocks) {
            let block = &p.blocks[b];
            let whole = block
                .preds
                .iter()
                .all(|&q| wave_all.get(q).copied().unwrap_or(false));
            for phi in &block.phis {
                if matches!(phi.reg, Reg::V(_))
                    && whole
                    && phi
                        .args
                        .iter()
                        .all(|a| next.contains(a) || same.contains(a))
                {
                    next.insert(phi.value);
                }
            }
            let known = |next: &HashSet<Value>, v: &Value| next.contains(v) || same.contains(v);
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Under a condition: only one every lane holds alike, and
                // over an old value that is alike too.
                if let Some(g) = st.guards[b][i] {
                    if !alike_bool(&same, g) {
                        continue;
                    }
                    let old_alike = inst
                        .writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .all(|w| inst.read(w.0).is_none_or(|old| known(&next, &old)));
                    if !old_alike {
                        continue;
                    }
                }
                // A buffer load at an alike index and offset, through a scalar
                // descriptor, reads one value in every lane.
                if let Fields::Buffer {
                    addr,
                    rsrc,
                    soffset,
                    offen,
                    idxen,
                    ..
                } = &inst.fields
                    && (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                {
                    let vector = |op: &Operand| match *op {
                        Operand::Vector { reg, dwords } => (0..dwords)
                            .all(|k| inst.read(Reg::V(reg + k)).is_some_and(|v| known(&next, &v))),
                        _ => true,
                    };
                    let scalar = |op: &Operand| match *op {
                        Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                            .all(|k| inst.read(Reg::S(code + k)).is_none_or(|v| !per_lane(v))),
                        _ => true,
                    };
                    if (!(*offen || *idxen) || vector(addr)) && scalar(rsrc) && scalar(soffset) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // An LDS read at an alike address reads one value in every lane.
                if let Fields::Ds {
                    addr: Some(Operand::Vector { reg, .. }),
                    dst: Some(_),
                    data,
                    ..
                } = &inst.fields
                    && name.starts_with("ds_read")
                    && data.is_empty()
                {
                    if inst.read(Reg::V(*reg)).is_some_and(|a| known(&next, &a)) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                let ops: Vec<Operand> = match &inst.fields {
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane") =>
                    {
                        v.src.clone()
                    }
                    _ => continue,
                };
                // The thread id's x shifted past the console wave: one value
                // per console wave (the b32 shift uses five bits of its count).
                if let (
                    Some(tid),
                    "v_lshrrev_b32",
                    [
                        Operand::Const(k @ 128..=192),
                        Operand::Vector { reg, .. },
                        ..,
                    ],
                ) = (wave_tid, name, ops.as_slice())
                    && ((k - 128) & 31) >= 6
                    && inst.read(Reg::V(*reg)) == Some(tid)
                {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                    continue;
                }
                let alike = ops.iter().all(|op| match *op {
                    Operand::Vector { reg, .. } => {
                        // every dword the instruction reads from that register on
                        let read: Vec<Value> = inst
                            .reads
                            .iter()
                            .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                            .map(|r| r.1)
                            .collect();
                        !read.is_empty() && read.iter().all(|x| known(&next, x))
                    }
                    // A scalar is one value; a lane mask is not, unless a
                    // compare of alike values made it.
                    Operand::Scalar { code, .. } if code != NULL => inst
                        .read(Reg::S(code))
                        .is_none_or(|x| !is_mask(x) && !per_lane(x) || known(&next, &x)),
                    _ => true,
                });
                if alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) {
            return same;
        }
        same.extend(next);
    }
}
