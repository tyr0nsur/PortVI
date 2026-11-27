//! `v_permlane16_b32` and `v_permlanex16_b32`: lane shuffles within rows of
//! 16, and the fetch-inactive (FI) flow check they need.
//!
//! Each nibble of {src2, src1} selects a lane in this row of 16
//! (`v_permlane16`) or in the other row of the same 32-lane half
//! (`v_permlanex16`); no form reads across 32-lane halves, so a console
//! wave64 program in halves ([`Lower::halves`], host waves of 32) keeps
//! them exact. Which lane each step uses:
//!
//! * the selected source position and the lane read: the host lane. In
//!   halves host lane `n` of the host wave holding half `h` is console lane
//!   `32h + n`; the row (`n & !15`), its partner row (`^ 16`) and the source
//!   all lie in that half, so they are the console's positions less `32h`,
//!   in the same host wave that `waveReadLaneAt` reads;
//! * whether the source lane is active (without FI): the host ballot's bit
//!   for that host lane, of the console EXEC guard, whose own bits are
//!   decided by the console lane ([`Lower::console_lane`]);
//! * the destination's EXEC: the guard, as every VALU result.
//!
//! FI reads lanes outside EXEC, which have no host form where their host
//! invocation is absent: a partial last row, a block not every host lane
//! reaches ([`all_lane_blocks`]), and in halves a host wave of fewer than
//! 32 invocations ([`PARTIAL_HALF`]).

use super::*;

/// Why a fetch-inactive shuffle stops in halves: the group leaves a host
/// wave of fewer than 32 invocations. Only full 32-lane halves have been
/// measured with FI on 32-lane hosts (`permlane_tests`); in a partial one
/// FI reads, without the ballot the other form checks, a host lane whose
/// invocation exists only if the host packs it into the wave's low lanes.
pub(super) const PARTIAL_HALF: Why =
    "a permlane fetch from a partial 32-lane half (wave64 on a 32-lane host)";

impl Lower<'_> {
    /// `v_permlane16_b32` / `v_permlanex16_b32` (RDNA2 ISA 12.12) under
    /// `guard`. OP_SEL[0:1] are fetch-inactive and bound-control, not half
    /// selects.
    pub(super) fn permlane16(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        if v.abs != 0 || v.neg != 0 || v.opsel & !3 != 0 {
            return Err("permlane input modifiers");
        }
        self.cross_lane(false)?;
        // Console lanes made the program declare its wave: the console's, or
        // 32 in halves, where every row and its partner are one host wave's.
        debug_assert!(!self.halves || self.s.wave_size == Some(32));
        let fetch_inactive = v.opsel & 1 != 0;
        let cross = inst.name() == "v_permlanex16_b32";
        // FI can read a lane outside EXEC, even when its VGPR contains
        // an earlier value. A partial last row has no such host invocation.
        let row_group = if cross { 32 } else { 16 };
        let group = self
            .s
            .threads
            .map(|group| group.iter().map(|&n| u64::from(n)).product::<u64>());
        if fetch_inactive && group.is_some_and(|n| !n.is_multiple_of(row_group)) {
            return Err("a permlane fetch from an incomplete thread-group row");
        }
        if fetch_inactive && self.halves && group.is_some_and(|n| !n.is_multiple_of(32)) {
            return Err(PARTIAL_HALF);
        }
        if fetch_inactive
            && !self
                .full_wave_blocks
                .get_or_init(|| all_lane_blocks(self.x.p, self.x.st))
                .as_ref()
                .is_some_and(|all| all[self.cur_ir])
        {
            // DXIL cannot read the last VGPR value of a host invocation
            // that has left through a lane branch. A guard keeps it here.
            return Err("a permlane fetch from lanes outside the host branch");
        }
        let (x, lo, hi) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
        // The host lane: in halves, the console lane within its half.
        let lane = self.lane_index();
        let (k7, k8, k15, k16, k31, k32, k4, one, zero) = (
            self.s.m.i32(7),
            self.s.m.i32(8),
            self.s.m.i32(15),
            self.s.m.i32(16),
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(4),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let pos = self.s.m.bin(c, Bin::And, lane, k15);
        let lower = self.s.m.cmp(c, pred::IULT, pos, k8);
        let packed = self.s.m.select(c, lower, lo, hi);
        let shift = self.s.m.bin(c, Bin::And, pos, k7);
        let shift = self.s.m.bin(c, Bin::Mul, shift, k4);
        let pick = self.s.m.bin(c, Bin::LShr, packed, shift);
        let pick = self.s.m.bin(c, Bin::And, pick, k15);
        let not15 = self.s.m.i32(!15);
        let mut row = self.s.m.bin(c, Bin::And, lane, not15);
        if cross {
            row = self.s.m.bin(c, Bin::Xor, row, k16);
        }
        // A host lane of this host wave (below 32 in halves).
        let src = self.s.m.bin(c, Bin::Or, row, pick);
        let r = if fetch_inactive {
            self.read_lane(x, src)
        } else {
            // Ballot the console EXEC guard as well as the host's active
            // invocations. Never shuffle from an inactive host lane.
            let enabled = guard.unwrap_or_else(|| self.s.m.bool(true));
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let mask = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[enabled]);
            let lo = self.s.m.extract(c, mask, 0);
            let hi = self.s.m.extract(c, mask, 1);
            let lower = self.s.m.cmp(c, pred::IULT, src, k32);
            let word = self.s.m.select(c, lower, lo, hi);
            let bit = self.s.m.bin(c, Bin::And, src, k31);
            let bit = self.s.m.bin(c, Bin::LShr, word, bit);
            let bit = self.s.m.bin(c, Bin::And, bit, one);
            let active = self.s.m.cmp(c, pred::NE, bit, zero);
            let safe = self.s.m.select(c, active, src, lane);
            let read = self.read_lane(x, safe);
            let fallback = if v.opsel & 2 != 0 {
                zero
            } else {
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a permlane result outside a VGPR");
                };
                self.i32_of(inst.read(Reg::V(reg)).ok_or("a permlane old value")?)?
            };
            self.s.m.select(c, active, read, fallback)
        };
        self.write_i(inst, r, guard)
    }
}

/// Blocks reached by every host lane: a fetch-inactive shuffle cannot
/// recover registers of invocations absent from a divergent host branch.
pub(super) fn all_lane_blocks(p: &Program, st: &Structured) -> Option<Vec<bool>> {
    let exit_block = st.exits.len();
    let n = exit_block + 1;
    let mut successors = vec![Vec::new(); n];
    let mut rpo = vec![usize::MAX; exit_block];
    for (index, block) in crate::ir::reverse_postorder(&p.blocks)
        .into_iter()
        .enumerate()
    {
        rpo[block] = index;
    }
    let mut aborts = vec![false; exit_block];
    for (b, exit) in st.exits.iter().enumerate() {
        match *exit {
            LaneExit::Jump(t) => {
                successors[b].push(t);
                if rpo[b] != usize::MAX && rpo[t] <= rpo[b] {
                    // `Lower::exit` adds a per-lane return at the iteration
                    // limit of an unconditional back edge. It bypasses the
                    // loop's normal exit and every later reconvergence point.
                    successors[b].push(exit_block);
                    aborts[b] = true;
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                successors[b].extend([taken, not_taken]);
            }
            LaneExit::End => successors[b].push(exit_block),
            LaneExit::Indirect => return None,
        }
    }
    // Post-dominators are the ordinary dominators of the reversed graph.
    // One synthetic exit includes early returns from a bounded loop.
    let mut predecessors = vec![Vec::new(); n];
    for (from, next) in successors.iter().enumerate() {
        for &to in next {
            predecessors[to].push(from);
        }
    }
    let order = crate::ir::graph_rpo(n, exit_block, |b, i| predecessors[b].get(i).copied());
    if order.len() != n {
        return None;
    }
    let mut index = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    let post = crate::ir::graph_idoms(&order, &index, |b| &successors[b]);
    let mut all = vec![true; n];
    let mut visited = vec![usize::MAX; n];
    for (branch, exit) in st.exits.iter().enumerate() {
        let divergent = match *exit {
            LaneExit::Branch { when, .. } => !st
                .bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_))),
            _ => false,
        };
        if !divergent && !aborts[branch] {
            continue;
        }
        let mut pending = successors[branch].clone();
        while let Some(b) = pending.pop() {
            if visited[b] == branch || b == post[branch] {
                continue;
            }
            visited[b] = branch;
            all[b] = false;
            pending.extend(successors[b].iter().copied());
        }
    }
    all.truncate(exit_block);
    Some(all)
}
