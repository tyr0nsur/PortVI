//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`; host lane `n` is console lane `32h + n`.
//! Instructions that read another lane use its console lane index:
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
//! * across halves, traded: `v_readlane_b32` ([`Lower::lane_read`]),
//!   `v_readfirstlane_b32` ([`Lower::first_lane`]: the console's first lane
//!   in EXEC, which may be in either half), lane masks read as data
//!   ([`Lower::mask_halves`]: each half's ballot is a dword of the console's
//!   mask) and whole-wave reductions and prefix sums
//!   ([`Lower::reduce_halves`]: each half's part, combined). The halves
//!   store their parts in group-shared memory and meet at a group barrier
//!   ([`Lower::trade`]), which is legal only where every thread of the
//!   group gets there together ([`Exchange`]); elsewhere they are refused;
//! * a value the same in every lane of the console wave ([`lane_uniform`],
//!   what the compiler reads first lanes for) is its own first lane, so its
//!   `v_readfirstlane_b32` is the host wave's anywhere;
//! * a group of 32 threads or fewer is one host wave, which holds its whole
//!   console wave: nothing is in the other half, so its ballots' high dwords
//!   are 0, its reductions and first lanes are the host wave's, and a lane
//!   read of a constant lane below 32 is the host's.
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! For multidimensional groups that the host may tile, a private LDS
//! counter gives each host wave distinct contiguous console IDs. When the
//! group occupies all 32 KiB, `halves_tiles` checks whether the IDs are
//! contiguous without a counter.
//!
//! The console's lanes can share LDS within a wave without an explicit
//! barrier. The two host halves need barriers around dependent accesses,
//! including when several console waves share one group. `lds_fences` and
//! `wide` select points that all physical waves reach together.
//!
//! The cross-half exchanges, region analysis and wave-wide loops are
//! adapted from [chippy01's VIctory implementation](https://github.com/chippy01/VIctory/commit/b32cef72554981f89fad00fecc8358699518da8a).
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name. Which values are the same in every lane of the console
//! wave ([`lane_uniform`]): justrealyme's analysis (chippy01/VIctory#1),
//! ported from fxceless/victory-fork (#497 there).

use super::wide::Plan;
use super::*;
use crate::emit::bitcode::Rmw;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

/// Dwords each half trades at most at once ([`Lower::trade`]).
const WORDS: u32 = 2;

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

    /// Why a program in halves cannot run: a group of several rows (its
    /// height above one) whose rows are 32 threads wide or more, or not a
    /// power of two wide. A host may then fill a wave with a tile of the
    /// group (16 x 2 threads) rather than a run of flat ids, and the
    /// relabelling ([`Lower::relabelled_flat`]) would give two threads one
    /// id. Measured in halves on an RTX 3070 Ti and the Radeon iGPU of a
    /// Ryzen 9 7900X, with the same results on both:
    ///
    /// - 32 x 2, 32 x 8, 64 x 2, 64 x 4, 32 x 32, 48 x 2, 24 x 2, 12 x 4,
    ///   6 x 6 and 32 x 2 x 2 groups store only some of their ids;
    /// - 8 x 4, 8 x 8, 16 x 2, 16 x 4, 16 x 16, 8 x 16, 4 x 4 x 4, 2 x 2 x 16,
    ///   8 x 8 x 4, 16 x 4 x 2 and 8 x 4 x 4 store all of them;
    /// - and so does a group of one row in any depth, whatever its width:
    ///   32 x 1 x 2, 32 x 1 x 4, 32 x 1 x 32, 64 x 1 x 2, 64 x 1 x 4,
    ///   128 x 1 x 2, 256 x 1 x 4, 40 x 1 x 2, 48 x 1 x 2, 96 x 1 x 2,
    ///   24 x 1 x 4, 12 x 1 x 3 and 6 x 1 x 6.
    ///
    /// Upstream PR #2126 measured the tiles on NVIDIA, widths that are not
    /// powers of two too, and hands each wave its ids by ticket instead.
    pub(super) fn halves_tiles([x, y, _]: [u32; 3]) -> Option<Why> {
        (y > 1 && (x >= HALF || !x.is_power_of_two()))
            .then_some("a group whose rows a 32-lane host wave may tile (wave64 on a 32-lane host)")
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

    /// Whether the halves of a console wave trade what one reads of the
    /// other's lanes: in halves, unless the group fits one host wave (no
    /// lane in the other half).
    fn trades(&self) -> bool {
        self.halves && !self.one_host_wave()
    }

    /// The one half of each console wave whose lanes alone run IR block `b`,
    /// if its entry condition shows one ([`super::confine`]).
    fn alone(&self, b: usize) -> Option<u8> {
        self.exchange.half.get(b).copied().flatten()
    }

    /// `why`, a refusal at IR block `b` for want of a meeting, noted for the
    /// next attempt, which may run its region wave-wide ([`super::wide`]).
    fn apart(&self, b: usize, why: Why) -> Why {
        self.plan.ask(b);
        why
    }

    /// Whether every thread of the group meets at the IR block being
    /// lowered ([`Exchange`]), where the halves may trade.
    fn meets(&self) -> bool {
        self.exchange
            .meet
            .get(self.cur_ir)
            .copied()
            .unwrap_or(false)
    }

    /// `v_readlane_b32` (`inst`) of `x` at `lane` in halves: `None` where the
    /// host wave's own read is the console's (not halves; a constant lane
    /// below 32 of a group one host wave holds), the lane's own `x` where `x`
    /// is the same in every lane of the console wave ([`lane_uniform`]) and
    /// every lane holds it ([`Lower::held_everywhere`]), else
    /// what the halves trade ([`Lower::across`]). In a group one host wave
    /// holds, any other lane read is refused: a lane past the group holds
    /// whatever its registers did on the console.
    pub(super) fn lane_read(&mut self, inst: &Inst, x: V, lane: V) -> Result<Option<V>, Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Ok(None);
        };
        if !self.halves {
            return Ok(None);
        }
        if self.one_host_wave() {
            let low = v
                .src
                .get(1)
                .and_then(|&l| crate::resource::lane_of(l))
                .is_some_and(|l| l < HALF);
            return if low {
                Ok(None)
            } else {
                Err("a lane read that may cross 32-lane halves (wave64 on a 32-lane host)")
            };
        }
        if let Some(&Operand::Vector { reg, .. }) = v.src.first()
            && inst
                .read(Reg::V(reg))
                .is_some_and(|r| self.alike.get_or_init(|| lane_uniform(self.x)).contains(&r) && self.held_everywhere(r))
        {
            return Ok(Some(x));
        }
        self.across(x, lane).map(Some)
    }

    /// Whether every lane of the console wave holds value `v`: it was
    /// written where every thread meets, and not wave-wide. A lane outside
    /// the branch a value is written in keeps the register's old value,
    /// which a lane read of it gives.
    fn held_everywhere(&self, v: Value) -> bool {
        let at = self
            .def_block
            .get(&v)
            .copied()
            .or_else(|| self.phi_at.get(&v).map(|&(b, _)| b));
        at.is_none_or(|b| self.exchange.meet.get(b) == Some(&true) && !self.wide.inside[b])
    }

    /// Lane `lane`'s `x` in every lane of the console wave: each half reads
    /// its lane `lane % 32` and the halves trade ([`Lower::trade`]); the half
    /// holding `lane` gives the read.
    fn across(&mut self, x: V, lane: V) -> Result<V, Why> {
        if !self.meets() {
            return Err(self.apart(self.cur_ir, APART));
        }
        let c = self.cur;
        let (k0, k1, k5, k31) = (
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(5),
            self.s.m.i32(31),
        );
        let low = self.s.m.bin(c, Bin::And, lane, k31);
        let mine = self.read_lane(x, low);
        let (lo, hi) = self.trade(c, &[(mine, 0)])?[0];
        let half = self.s.m.bin(c, Bin::LShr, lane, k5);
        let half = self.s.m.bin(c, Bin::And, half, k1);
        let upper = self.s.m.cmp(c, pred::NE, half, k0);
        Ok(self.s.m.select(c, upper, hi, lo))
    }

    /// Each half's `words` (word and identity; each word the same in every
    /// lane of its host wave), traded through group-shared memory in DXIL
    /// block `at` of an IR block where every thread of the group meets
    /// ([`Exchange`]): every lane stores its half's words (one value per
    /// word, so the stores agree), a group barrier, every lane loads both
    /// halves' words, and a second barrier keeps the next trade from
    /// overwriting them first. Per word, the low half's and the high half's;
    /// a half without threads (in a group that ends inside a console wave's
    /// low half) gives the identity. A barrier one half skips hangs the GPU
    /// (D3D12 leaves it undefined; NVIDIA waits for every thread).
    fn trade(&mut self, at: usize, words: &[(V, u32)]) -> Result<Vec<(V, V)>, Why> {
        debug_assert!(words.len() <= WORDS as usize);
        let slots = self.slots()?;
        let f = self.relabelled_flat();
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k5, k6, kw, k2w) = (k(self, 5), k(self, 6), k(self, WORDS), k(self, 2 * WORDS));
        let own = self.s.m.bin(at, Bin::LShr, f, k5);
        let own = self.s.m.bin(at, Bin::Mul, own, kw);
        for (i, &(x, _)) in words.iter().enumerate() {
            let ki = k(self, i as u32);
            let slot = self.s.m.bin(at, Bin::Add, own, ki);
            let p = self.s.m.element(at, slots, slot);
            self.s.m.store(at, p, x);
        }
        self.group_barrier(at);
        let wave = self.s.m.bin(at, Bin::LShr, f, k6);
        let low = self.s.m.bin(at, Bin::Mul, wave, k2w);
        let high = self.s.m.bin(at, Bin::Add, low, kw);
        // Only the last console wave's high half may have no threads.
        let group = self.s.threads.map_or(64, |t| t.iter().product::<u32>());
        let present = (!group.is_multiple_of(64) && group % 64 <= HALF).then(|| {
            let (k64, n) = (k(self, 64), k(self, group - HALF));
            let first = self.s.m.bin(at, Bin::Mul, wave, k64);
            self.s.m.cmp(at, pred::IULT, first, n)
        });
        let mut traded = Vec::with_capacity(words.len());
        for (i, &(_, ident)) in words.iter().enumerate() {
            let ki = k(self, i as u32);
            let load = |l: &mut Self, base: V| {
                let slot = l.s.m.bin(at, Bin::Add, base, ki);
                let p = l.s.m.element(at, slots, slot);
                l.s.m.load(at, p)
            };
            let lo = load(self, low);
            let mut hi = load(self, high);
            if let Some(present) = present {
                let id = k(self, ident);
                hi = self.s.m.select(at, present, hi, id);
            }
            traded.push((lo, hi));
        }
        self.group_barrier(at);
        Ok(traded)
    }

    /// `v_readfirstlane_b32` (`inst`) of `x` in halves: `None` where the host
    /// wave's own first lane is the console's (not in halves, a group one
    /// host wave holds, a scalar or a value the same in every lane of the
    /// console wave), else the console's: the first lane in EXEC (the
    /// instruction's guard), in either half, or lane 0 where EXEC holds none.
    /// Each half reads its own first lane in EXEC (lane 0 with none) and its
    /// ballot of EXEC, and the halves trade; refused where not every thread
    /// meets.
    pub(super) fn first_lane(&mut self, inst: &Inst, x: V) -> Result<Option<V>, Why> {
        let Fields::Valu(v) = &inst.fields else {
            return Ok(None);
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Ok(None);
        };
        // Wave-wide, the lanes outside the block keep older values.
        if !self.trades()
            || !self.wide.inside[self.cur_ir]
                && inst
                    .read(Reg::V(reg))
                    .is_some_and(|r| self.alike.get_or_init(|| lane_uniform(self.x)).contains(&r))
        {
            return Ok(None);
        }
        // Where one half's lanes alone run, its host wave's first lane in
        // EXEC is the console's, unless a guard may leave EXEC empty (the
        // console then reads lane 0, which may not be running here): then
        // the region runs wave-wide instead, where the halves meet.
        let alone = match (self.meets(), self.alone(self.cur_ir)) {
            (true, _) => None,
            (false, Some(h)) => Some(h),
            (false, None) => return Err(self.apart(self.cur_ir, FIRST_ACROSS)),
        };
        let b = self.cur_ir;
        let i = self.x.p.blocks[b]
            .insts
            .iter()
            .position(|j| std::ptr::eq(j, inst))
            .ok_or(FIRST_ACROSS)?;
        let on = match self.guard(b, i)? {
            Some(_) if alone.is_some() => return Err(self.apart(b, FIRST_ACROSS)),
            Some(g) => g,
            None => self.s.m.bool(true),
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        self.s.flags |= flags::WAVE_OPS;
        let ballot = self.s.m.extract(c, ballot, 0);
        let k0 = self.s.m.i32(0);
        let any = self.s.m.cmp(c, pred::NE, ballot, k0);
        let low = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[ballot]);
        let lane = self.s.m.select(c, any, low, k0);
        let mine = self.read_lane(x, lane);
        if alone.is_some() {
            return Ok(Some(mine));
        }
        let traded = self.trade(c, &[(mine, 0), (ballot, 0)])?;
        let ((x0, x1), (b0, b1)) = (traded[0], traded[1]);
        let any0 = self.s.m.cmp(c, pred::NE, b0, k0);
        let any1 = self.s.m.cmp(c, pred::NE, b1, k0);
        let r = self.s.m.select(c, any1, x1, x0);
        Ok(Some(self.s.m.select(c, any0, x0, r)))
    }

    /// Refuses lane mask `mask` read as data in halves unless every thread
    /// of the group meets at the IR block writing it, where the halves trade
    /// their ballots, or one half's lanes alone run it ([`Lower::mask_halves`]);
    /// program inputs are the entry's.
    pub(super) fn mask_meets(&self, mask: Value) -> Result<(), Why> {
        let b = self.def_block.get(&mask).copied().unwrap_or(0);
        let meets = self.exchange.meet.get(b).copied().unwrap_or(false);
        if self.trades() && !meets && self.alone(b).is_none() {
            return Err(self.apart(b, MASK_APART));
        }
        Ok(())
    }

    /// The console mask's dwords from this host wave's ballot `xy` of lane
    /// mask `mask`, taken in DXIL block `at`: in halves each half's ballot is
    /// one of them, traded there, or where one half's lanes alone write it,
    /// that half's (the other's lanes are not in it).
    pub(super) fn mask_halves(
        &mut self,
        mask: Value,
        at: usize,
        xy: (V, V),
    ) -> Result<(V, V), Why> {
        if !self.trades() {
            return Ok(xy);
        }
        let b = self.def_block.get(&mask).copied().unwrap_or(0);
        if !self.exchange.meet.get(b).copied().unwrap_or(false)
            && let Some(h) = self.alone(b)
        {
            let none = self.s.m.i32(0);
            return Ok(if h == 0 { (xy.0, none) } else { (none, xy.0) });
        }
        Ok(self.trade(at, &[(xy.0, 0)])?[0])
    }

    /// In halves, the whole-wave sections the plan runs instruction by
    /// instruction rather than by the simulation's shapes ([`Literal`]):
    /// they are left out of [`Lower::sections`], so each of their steps is
    /// lowered on its own. Where every thread of the group does not meet,
    /// or in a group of a part of a console wave, they keep the simulation
    /// (and the program its refusal).
    pub(super) fn literal_sections(&mut self) {
        let whole = |t: [u32; 3]| t.iter().product::<u32>().is_multiple_of(2 * HALF);
        if !self.trades() || !self.s.threads.is_some_and(whole) {
            return;
        }
        let (literal, meet) = (&self.literal.sections, &self.exchange.meet);
        for (&b, found) in &mut self.sections {
            if meet.get(b).copied().unwrap_or(false) {
                found.retain(|&start, _| !literal.contains(&(b, start)));
            }
        }
        self.sections.retain(|_, found| !found.is_empty());
    }

    /// A value of whole-wave section `section` (block, start) the
    /// simulation could not lower, read after the section: the plan may run
    /// the section instruction by instruction next ([`Literal`]).
    pub(super) fn ask_literal(&self, section: (usize, usize)) {
        self.literal.asked.borrow_mut().insert(section);
    }

    /// Refuses a whole-wave reduction or prefix sum in halves where not
    /// every thread of the group meets ([`Lower::reduce_halves`]).
    pub(super) fn reduction_meets(&self) -> Result<(), Why> {
        if self.trades() && !self.meets() && self.alone(self.cur_ir).is_none() {
            return Err(self.apart(self.cur_ir, REDUCTION_APART));
        }
        Ok(())
    }

    /// The console wave's `op` over its lanes (with `prefix`, the sum over
    /// the lanes below each, and with `Some(true)` it too) from this half's
    /// `r` of `arg` (f32 values where `float`), `ident` the identity: in
    /// halves the halves trade their parts. A reduction combines both, low
    /// half first; a prefix sum in the high half adds the low half's total.
    pub(super) fn reduce_halves(
        &mut self,
        op: super::super::wave::Op,
        prefix: Option<bool>,
        arg: V,
        r: V,
        ident: u32,
        float: bool,
    ) -> Result<V, Why> {
        use super::super::wave::Op;
        // Where one half's lanes alone run ([`Lower::reduction_meets`]), the
        // host wave's reduction is over every lane the console's is.
        if !self.trades() || !self.meets() {
            return Ok(r);
        }
        let c = self.cur;
        let bits = |l: &mut Self, x: V| if float { l.as_int(x) } else { x };
        let value = |l: &mut Self, x: V| if float { l.as_float(x) } else { x };
        if prefix.is_some() {
            let (ty, sfx) = if float {
                (self.s.t.f32, "f32")
            } else {
                (self.s.t.i32, "i32")
            };
            let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
            let total = self.s.op(
                c,
                &format!("waveActiveOp.{sfx}"),
                Attr::NoUnwind,
                ty,
                119,
                &[arg, sum, unsigned],
            );
            let total = bits(self, total);
            let (low, _) = self.trade(c, &[(total, 0)])?[0];
            let low = value(self, low);
            let below = self.s.m.bin(c, Bin::Add, r, low);
            let f = self.relabelled_flat();
            let (k0, k32) = (self.s.m.i32(0), self.s.m.i32(HALF));
            let half = self.s.m.bin(c, Bin::And, f, k32);
            let upper = self.s.m.cmp(c, pred::NE, half, k0);
            return Ok(self.s.m.select(c, upper, below, r));
        }
        let mine = bits(self, r);
        let (lo, hi) = self.trade(c, &[(mine, ident)])?[0];
        let (lo, hi) = (value(self, lo), value(self, hi));
        let pick = |l: &mut Self, p: u8| {
            let t = l.s.m.cmp(c, p, lo, hi);
            l.s.m.select(c, t, lo, hi)
        };
        Ok(match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinU => pick(self, pred::IULT),
            Op::MaxU => pick(self, pred::IUGT),
            Op::MinI => pick(self, pred::SLT),
            Op::MaxI => pick(self, pred::SGT),
            Op::MinF => self.max_min(false, lo, hi),
            Op::MaxF => self.max_min(true, lo, hi),
            // 16-bit sums are refused in halves before they get here.
            Op::AddH => return Err(REDUCTION_APART),
        })
    }

    /// At the start of IR block `b`, after its phis: the group barrier where a
    /// region accessing LDS comes back ([`Lds`]). What LDS accesses are
    /// pending at a block's start is not known.
    pub(super) fn lds_block_start(&mut self, b: usize) {
        let Some(lds) = &self.exchange.lds else {
            return;
        };
        if lds.at_entry.contains(&b) {
            self.group_barrier(self.cur);
            self.exchange.pending = None;
        } else {
            self.exchange.pending = Some(true);
        }
    }

    /// At the end of IR block `b`, before it branches: the group barrier
    /// before a region accessing LDS ([`Lds`]).
    pub(super) fn lds_block_end(&mut self, b: usize) {
        if self
            .exchange
            .lds
            .as_ref()
            .is_some_and(|l| l.before_exit.contains(&b))
        {
            self.group_barrier(self.cur);
        }
    }

    /// Before the LDS access `name` where every thread meets: a group
    /// barrier when an access since the last one may meet it (a write, or a
    /// read before this write), as a console wave orders them.
    pub(super) fn lds_order(&mut self, name: &str) {
        if self.exchange.lds.is_none() || !self.meets() {
            return;
        }
        let writes = !name.starts_with("ds_read");
        let pending = self.exchange.pending;
        if pending == Some(true) || writes && pending == Some(false) {
            self.group_barrier(self.cur);
            self.exchange.pending = None;
        }
        self.exchange.pending = Some(writes || self.exchange.pending == Some(true));
    }

    /// Whether `word` (each half's, the same in every lane of its host wave)
    /// is non-zero in any half of the group, in DXIL block `at` of an IR
    /// block every thread of the group meets at: in a group of one console
    /// wave the halves trade ([`Lower::trade`]); in one of several, every
    /// half stores its word, a group barrier, every lane reads the word of
    /// every half that has threads, and a second barrier.
    pub(super) fn group_any(&mut self, at: usize, word: V) -> Result<V, Why> {
        let group = self.s.threads.map_or(64, |t| t.iter().product::<u32>());
        let zero = self.s.m.i32(0);
        if group <= 64 {
            let (low, high) = self.trade(at, &[(word, 0)])?[0];
            let both = self.s.m.bin(at, Bin::Or, low, high);
            return Ok(self.s.m.cmp(at, pred::NE, both, zero));
        }
        let slots = self.slots()?;
        let f = self.relabelled_flat();
        let (k5, kw) = (self.s.m.i32(5), self.s.m.i32(WORDS));
        let own = self.s.m.bin(at, Bin::LShr, f, k5);
        let own = self.s.m.bin(at, Bin::Mul, own, kw);
        let p = self.s.m.element(at, slots, own);
        self.s.m.store(at, p, word);
        self.group_barrier(at);
        let mut any = zero;
        for h in 0..group.div_ceil(HALF) {
            let slot = self.s.m.i32(h * WORDS);
            let p = self.s.m.element(at, slots, slot);
            let w = self.s.m.load(at, p);
            any = self.s.m.bin(at, Bin::Or, any, w);
        }
        self.group_barrier(at);
        Ok(self.s.m.cmp(at, pred::NE, any, zero))
    }

    /// A group barrier ordering group-shared memory
    /// (`GroupMemoryBarrierWithGroupSync`), in DXIL block `at`.
    fn group_barrier(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// The trade's group-shared slots, made once: [`WORDS`] dwords per half
    /// of each console wave of the group, beside the program's own LDS and
    /// within the host's 32 KiB.
    fn slots(&mut self) -> Result<V, Why> {
        if let Some(g) = self.exchange.slots {
            return Ok(g);
        }
        let group = self.s.threads.map_or(64, |t| t.iter().product::<u32>());
        let dwords = 2 * WORDS * group.div_ceil(64);
        let ticket_dwords = u32::from(self.s.threads.is_some_and(|[_, y, z]| {
            group > HALF && (y > 1 || z > 1) && self.s.lds_dwords < 8192
        }));
        if self.s.lds_dwords + dwords + ticket_dwords > 8192 {
            return Err(
                "a trade between 32-lane halves past the host's 32 KiB of group-shared memory",
            );
        }
        let g = self.s.shared("halves", dwords);
        self.exchange.slots = Some(g);
        Ok(g)
    }
}

/// Why a lane read stops in halves: the halves cannot meet there.
pub(super) const APART: Why = "a lane read across 32-lane halves where not every thread of the group meets (wave64 on a 32-lane host)";

/// Why a lane mask read as data stops in halves: the halves cannot meet
/// where it is written to trade their ballots.
const MASK_APART: Why = "a lane mask read as data, written where not every thread of the group meets (wave64 on a 32-lane host)";

/// Why a whole-wave reduction or prefix sum stops in halves.
const REDUCTION_APART: Why = "a wave reduction over both 32-lane halves where not every thread of the group meets (wave64 on a 32-lane host)";

/// Why a 16-bit sum's shape stops in halves: the console wave's, over both
/// halves ([`Literal`] runs its steps instead).
pub(super) const SUM16_ACROSS: Why =
    "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)";

/// The whole-wave sections run instruction by instruction in halves, over
/// the lowering's attempts ([`super::lower`]), as [`Plan`] does for
/// regions: an attempt refused for a value a section leaves that the
/// simulation cannot lower asks for that section, and the next runs its
/// steps one by one ([`Lower::literal_sections`]).
///
/// Such a value has lanes that hold different things, which no shape
/// describes: the per-half sums and totals of the console compiler's wave64
/// sums, read after the section in lanes a later write leaves as they
/// were. Main lowers no step of it and keeps those lanes as zero on every
/// host; here they get the console's values. Or it is a 16-bit sum over
/// both halves, whose shape takes the sum in f32 and rounds once where the
/// console rounds each step, in its own order: here each step is the
/// console's.
///
/// The steps are exact in halves: DPP and `v_permlane*16` read within a
/// 32-lane half, and the lane reads that cross the halves are traded
/// where every thread of the group meets. The section sets EXEC to every
/// lane of the console wave, so only in a group of whole console waves,
/// where every lane has a host thread to run them.
#[derive(Default)]
pub(super) struct Literal {
    /// By block and start.
    sections: BTreeSet<(usize, usize)>,
    /// What the last attempt asked for.
    asked: std::cell::RefCell<BTreeSet<(usize, usize)>>,
}

impl Literal {
    /// After an attempt refused for `what`: the sections the attempts since
    /// the last call asked for join the plan (a join that read such a value
    /// is missing in the attempts after it, and its read asks no more);
    /// whether another attempt is worth making (some joined).
    pub(super) fn follow(&mut self, what: Why) -> bool {
        let asked = std::mem::take(self.asked.get_mut());
        if what != SECTION_UNFOLLOWED && what != SUM16_ACROSS {
            return false;
        }
        let before = self.sections.len();
        self.sections.extend(asked);
        self.sections.len() > before
    }
}

/// Trades between the two halves of each console wave: where they may
/// meet, and the group-shared slots they trade through.
#[derive(Default)]
pub(super) struct Exchange {
    /// Per IR block: every thread of the group gets there together, the
    /// same number of times ([`meeting`]).
    meet: Vec<bool>,
    /// [`Lower::slots`], once made.
    slots: Option<V>,
    /// Group barriers that keep accesses within each split console wave in order.
    lds: Option<Lds>,
    /// Why the program cannot run in halves ([`lds_fences`]).
    pub(super) refused: Option<Why>,
    /// Per IR block not every thread meets at: the one half of each console
    /// wave whose lanes alone run it, where its entry condition shows one
    /// ([`super::confine`]).
    half: Vec<Option<u8>>,
    /// LDS accesses since the last barrier in this block where every thread
    /// meets: `Some(true)` with a write among them (or not known, at a
    /// block's start), `Some(false)` reads only.
    pending: Option<bool>,
}

impl Exchange {
    /// With `wide`, the IR blocks run wave-wide ([`super::wide`]), which
    /// every thread meets at; `plan` hears of LDS refused elsewhere.
    pub(super) fn new(x: &Parts, halves: bool, wide: &[bool], plan: &Plan) -> Self {
        if !halves {
            return Exchange::default();
        }
        let mut meet = meeting(x);
        for (m, &w) in meet.iter_mut().zip(wide) {
            *m |= w;
        }
        let half = super::confine::confined(x, &meet);
        // Even in a larger group, a console wave may read another lane's
        // earlier write without a group barrier. Add host barriers only at
        // points every wave of this group reaches.
        let split_lds = match &x.native.io {
            StageIo::Cs(cs) => {
                let group = cs.threads.iter().product::<u32>();
                cs.lds_bytes() > 0 && group > HALF
            }
            _ => false,
        };
        let (lds, refused) = match split_lds.then(|| lds_fences(x, &meet, &half, plan)) {
            Some(Ok(l)) => (Some(l), None),
            Some(Err(why)) => (None, Some(why)),
            None => (None, None),
        };
        Exchange {
            meet,
            slots: None,
            lds,
            refused,
            half,
            pending: None,
        }
    }
}

/// Where a group orders its split console waves' LDS: the blocks
/// ending with a group barrier (before they branch into a region only some
/// lanes run that accesses LDS) and those starting with one (where such a
/// region comes back).
#[derive(Default)]
struct Lds {
    before_exit: HashSet<usize>,
    at_entry: HashSet<usize>,
}

/// [`Lds`] of a program, or why it cannot run in halves: a region only some
/// lanes run (a set of blocks not every thread meets in, joined by edges)
/// that both writes and reads LDS, or writes it in two instructions, cannot
/// be put in the console's order without a barrier inside it, unless every
/// block of it accessing LDS is one that only one half's lanes run (`half`),
/// the same half: its host wave runs them in the console's order. `plan`
/// hears of the blocks of a refused region that access LDS.
fn lds_fences(x: &Parts, meet: &[bool], half: &[Option<u8>], plan: &Plan) -> Result<Lds, Why> {
    let (p, st) = (x.p, x.st);
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return Err(MIXED),
        }
    }
    let meets = |b: usize| meet.get(b).copied().unwrap_or(false);
    // The regions: blocks not every thread meets in, by their edges.
    let mut root: Vec<usize> = (0..n).collect();
    fn find(root: &mut [usize], mut b: usize) -> usize {
        while root[b] != b {
            root[b] = root[root[b]];
            b = root[b];
        }
        b
    }
    for (b, next) in succ.iter().enumerate() {
        for &t in next {
            if !meets(b) && !meets(t) {
                let (rb, rt) = (find(&mut root, b), find(&mut root, t));
                root[rb] = rt;
            }
        }
    }
    // Each region's LDS reads, writes and other accesses (atomics), and the
    // one half whose lanes alone make them, if there is one.
    let mut access: HashMap<usize, ([u32; 3], Option<u8>)> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        if meets(b) {
            continue;
        }
        for inst in &block.insts {
            let name = inst.name();
            if !matches!(inst.fields, Fields::Ds { .. })
                || name.starts_with("ds_swizzle")
                || name.contains("permute")
                || name == "ds_nop"
            {
                continue;
            }
            let k = if name.starts_with("ds_read") {
                0
            } else if name.starts_with("ds_write") {
                1
            } else {
                2
            };
            let alone = half.get(b).copied().flatten();
            let e = access.entry(find(&mut root, b)).or_insert(([0; 3], alone));
            e.0[k] += 1;
            if e.1 != alone {
                e.1 = None;
            }
        }
    }
    let mixed: HashSet<usize> = access
        .iter()
        .filter(|(_, e)| {
            let ([r, w, a], alone) = **e;
            alone.is_none() && ((w > 0 && r > 0) || (a > 0 && w + r > 0) || a > 1 || w > 1)
        })
        .map(|(&root, _)| root)
        .collect();
    if !mixed.is_empty() {
        for b in (0..n).filter(|&b| !meets(b)) {
            if mixed.contains(&find(&mut root, b)) {
                plan.ask(b);
            }
        }
        return Err(MIXED);
    }
    let mut lds = Lds::default();
    for (u, next) in succ.iter().enumerate() {
        for &v in next {
            if meets(u) && !meets(v) && access.contains_key(&find(&mut root, v)) {
                lds.before_exit.insert(u);
            }
            if !meets(u) && meets(v) && access.contains_key(&find(&mut root, u)) {
                lds.at_entry.insert(v);
            }
        }
    }
    Ok(lds)
}

/// Per IR block, whether every thread of the group gets there together and
/// the same number of times, so that a group barrier there is legal: no
/// branch on a lane-dependent condition has it in its region (the per-lane
/// form branches on the console's wave-wide tests too). In a group of
/// several console waves, which may each take a uniform branch their own
/// way, no branch at all. Nowhere with an indirect exit.
pub(super) fn meeting(x: &Parts) -> Vec<bool> {
    let Some(regions) = lane_regions(x) else {
        return vec![false; x.st.exits.len()];
    };
    let mut meet = vec![true; x.st.exits.len()];
    for (_, region) in regions {
        for (b, r) in region.iter().enumerate() {
            if *r {
                meet[b] = false;
            }
        }
    }
    meet
}

/// [`branch_regions`] of the branches that may part a group's threads: in
/// a group of several console waves, which may each take a uniform branch
/// their own way, every branch.
pub(super) fn lane_regions(x: &Parts) -> Option<Vec<(usize, Vec<bool>)>> {
    let several = match &x.native.io {
        StageIo::Cs(cs) => cs.threads.iter().product::<u32>() > 64,
        _ => true,
    };
    branch_regions(x.st, several)
}

/// Whether a refusal in halves is for want of a meeting, which a region run
/// wave-wide ([`super::wide`]) may give.
pub(super) fn for_want_of_a_meeting(what: Why) -> bool {
    [APART, FIRST_ACROSS, MASK_APART, REDUCTION_APART, MIXED].contains(&what)
}

/// Why LDS stops in halves where it cannot be put in order.
const MIXED: Why = "LDS a branch only some lanes take both writes and reads (or writes twice), on a host of 32-lane waves";

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, the value is not known to be the same there, and the
/// halves cannot meet to trade it.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves, where not every thread of the group meets (wave64 on a 32-lane host)";

/// Every per-lane branch (by its block) with its region: the blocks
/// reachable from it that do not post-dominate it, which only some lanes
/// run, and the branch's block itself where they lead back to it. Branches on uniform conditions too with `uniform`. None with an
/// indirect exit.
fn branch_regions(st: &Structured, uniform: bool) -> Option<Vec<(usize, Vec<bool>)>> {
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
        if !uniform
            && st
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
            // The branch's own block post-dominates itself, but where its
            // edges lead back to it (a loop's latch, a loop of one block)
            // each lane gets there once a round, as many times as it goes
            // round: it is in the region.
            if b == end || (b != d && b == post[d]) || seen[b] {
                continue;
            }
            seen[b] = true;
            if b != d {
                stack.extend(succ[b].iter().copied());
            }
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
/// them (a branch on alike values does not). A value written inside a loop
/// whose exit is lane-dependent and read after it is not alike, nor what is
/// computed from it there: lanes leave in rounds of their own, each holding
/// what its last round wrote (scalars too, in the per-lane form). The
/// analysis runs to a fixed point
/// from nothing, so a loop-carried value that needs itself to be alike
/// never is.
pub(super) fn lane_uniform(x: &Parts) -> HashSet<Value> {
    let (p, st) = (x.p, x.st);
    let wave_tid = wave_tid(x);
    let regions = branch_regions(st, false);
    let mut def: HashMap<Value, usize> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        def.extend(block.phis.iter().map(|phi| (phi.value, b)));
        for inst in &block.insts {
            def.extend(inst.writes.iter().map(|w| (w.1, b)));
        }
    }
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
        // Loops whose exit is lane-dependent: the blocks of a branch's region
        // that lead back to the branch, where the region holds it.
        let loops: Vec<Vec<bool>> = regions
            .iter()
            .flatten()
            .filter(|(d, region)| {
                region[*d]
                    && matches!(st.exits[*d], LaneExit::Branch { when, .. } if !alike_bool(&same, when))
            })
            .map(|(d, region)| {
                let mut body = vec![false; region.len()];
                let mut stack = vec![*d];
                while let Some(b) = stack.pop() {
                    if body[b] {
                        continue;
                    }
                    body[b] = true;
                    stack.extend(p.blocks[b].preds.iter().copied().filter(|&q| region[q]));
                }
                body
            })
            .collect();
        // Whether `v`, read in block `b`, was written in such a loop `b` is
        // not in.
        let escapes = |v: &Value, b: usize| {
            def.get(v)
                .is_some_and(|&d| loops.iter().any(|body| body[d] && !body[b]))
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
            let known = |next: &HashSet<Value>, v: &Value| {
                (next.contains(v) || same.contains(v)) && !escapes(v, b)
            };
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
                    Operand::Scalar { code, .. } if code != NULL => {
                        inst.read(Reg::S(code)).is_none_or(|x| {
                            !is_mask(x) && !per_lane(x) && !escapes(&x, b) || known(&next, &x)
                        })
                    }
                    _ => true,
                });
                if alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) {
            // A value read outside its loop is not alike where it is read
            // inside either: the alike are values, not reads.
            for (b, block) in p.blocks.iter().enumerate() {
                for inst in &block.insts {
                    for r in &inst.reads {
                        if escapes(&r.1, b) {
                            same.remove(&r.1);
                        }
                    }
                }
            }
            return same;
        }
        same.extend(next);
    }
}
