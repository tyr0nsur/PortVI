//! Which half of each console wave64 can run a block, for programs in halves
//! ([`super::halves`]). A lane branch on a compare of the lane's position
//! with a constant (`tid < 16`, `lane >= 32`) lets only lanes of one half
//! into its region. There one host wave runs alone, in the console's order,
//! so what the halves otherwise trade or put in order through group
//! barriers needs nothing: LDS the region both writes and reads, and lane
//! masks, first lanes and reductions over the region's lanes.
//!
//! A block's entry condition ([`Structured::reach`]) is a decision diagram
//! over atoms. The compares of a lane position (the console lane from
//! `v_mbcnt`, the thread id of a group one row high, and simple arithmetic
//! on them) with constants are known in every lane; the other atoms are
//! not. A half can run a block where some lane of it meets the condition
//! with the unknown atoms either way.

use super::*;
use crate::structure::Bools;

/// Per IR block: the half (0: lanes 0-31, 1: lanes 32-63) of every console
/// wave that holds all the lanes that can run it, where its entry condition
/// shows one; `None` where it does not, and for blocks every thread of the
/// group meets at (`meet`), which need no answer.
pub(super) fn confined(x: &Parts, meet: &[bool]) -> Vec<Option<u8>> {
    let st = x.st;
    let n = st.exits.len();
    let StageIo::Cs(cs) = &x.native.io else {
        return vec![None; n];
    };
    let group = cs.threads.iter().product::<u32>();
    let lanes = Lanes::new(x, cs.threads[1] * cs.threads[2] == 1);
    let mut tables: HashMap<Atom, Option<Vec<u64>>> = HashMap::new();
    (0..n)
        .map(|b| {
            if meet.get(b).copied().unwrap_or(true) {
                return None;
            }
            let reach = st.reach[b];
            let known: HashMap<Atom, Vec<u64>> = st
                .bools
                .support(reach)
                .into_iter()
                .filter_map(|a| {
                    let t = tables.entry(a).or_insert_with(|| lanes.table(a, group));
                    t.clone().map(|t| (a, t))
                })
                .collect();
            if known.is_empty() {
                return None;
            }
            [0u32, 1].into_iter().find_map(|h| {
                let other = 32 * (1 - h);
                let shut = (0..group.div_ceil(64)).all(|w| {
                    (other..other + 32).filter(|l| 64 * w + l < group).all(|l| {
                        let value = |a: Atom| known.get(&a).map(|t| t[w as usize] >> l & 1 == 1);
                        !possible(&st.bools, reach, &value, &mut HashMap::new())
                    })
                });
                shut.then_some(h as u8)
            })
        })
        .collect()
}

/// Whether some lane meets `b` with each atom `value` knows fixed to it and
/// the others either way.
fn possible(
    bools: &Bools,
    b: Bool,
    value: &dyn Fn(Atom) -> Option<bool>,
    memo: &mut HashMap<Bool, bool>,
) -> bool {
    if b == Bool::TRUE || b == Bool::FALSE {
        return b == Bool::TRUE;
    }
    if let Some(&m) = memo.get(&b) {
        return m;
    }
    let Some((atom, lo, hi)) = bools.node(b) else {
        return true;
    };
    let r = match value(atom) {
        Some(true) => possible(bools, hi, value, memo),
        Some(false) => possible(bools, lo, value, memo),
        None => possible(bools, lo, value, memo) || possible(bools, hi, value, memo),
    };
    memo.insert(b, r);
    r
}

/// The values that are a function of where a lane is: by IR value, its
/// value in lane `l` of console wave `w`.
struct Lanes<'a> {
    x: &'a Parts<'a>,
    /// The instruction writing each value, by block and index.
    defs: HashMap<Value, (usize, usize)>,
    /// The thread id's x on entry, in a group one row high (64 w + l).
    tid: Option<Value>,
}

impl<'a> Lanes<'a> {
    fn new(x: &'a Parts<'a>, one_row: bool) -> Self {
        let mut defs = HashMap::new();
        for (b, block) in x.p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs.insert(v, (b, i));
                }
            }
        }
        let tid = one_row
            .then(|| {
                x.p.inputs
                    .iter()
                    .find(|(r, _)| *r == Reg::V(0))
                    .map(|(_, v)| *v)
            })
            .flatten();
        Lanes { x, defs, tid }
    }

    /// Per console wave of a group of `group` threads, the lanes where atom
    /// `a` holds, if it is a compare of values [`Lanes::at`] knows.
    fn table(&self, a: Atom, group: u32) -> Option<Vec<u64>> {
        let Atom::Lane(v) = a else {
            return None;
        };
        let (b, i) = *self.defs.get(&v)?;
        let inst = &self.x.p.blocks[b].insts[i];
        let name = inst.name();
        let rest = name
            .strip_prefix("v_cmpx_")
            .or_else(|| name.strip_prefix("v_cmp_"))?;
        let (op, ty) = rest.split_once('_')?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.dpp.is_some() || f.sdwa.is_some() {
            return None;
        }
        let (s0, s1) = (*f.src.first()?, *f.src.get(1)?);
        (0..group.div_ceil(64))
            .map(|w| {
                let mut bits = 0u64;
                for l in 0..64 {
                    let (a, b) = (
                        self.operand(inst, s0, w, l, 0)?,
                        self.operand(inst, s1, w, l, 0)?,
                    );
                    let (sa, sb) = (a as i32, b as i32);
                    let holds = match (op, ty) {
                        ("lt", "u32") => a < b,
                        ("le", "u32") => a <= b,
                        ("gt", "u32") => a > b,
                        ("ge", "u32") => a >= b,
                        ("lt", "i32") => sa < sb,
                        ("le", "i32") => sa <= sb,
                        ("gt", "i32") => sa > sb,
                        ("ge", "i32") => sa >= sb,
                        ("eq", "u32" | "i32") => a == b,
                        ("ne" | "lg", "u32" | "i32") => a != b,
                        _ => return None,
                    };
                    bits |= u64::from(holds) << l;
                }
                Some(bits)
            })
            .collect()
    }

    /// Operand `op` of `inst` in lane `l` of console wave `w`.
    fn operand(&self, inst: &Inst, op: Operand, w: u32, l: u32, depth: u32) -> Option<u32> {
        match op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Vector { reg, .. } => self.at(inst.read(Reg::V(reg))?, w, l, depth + 1),
            _ => None,
        }
    }

    /// Value `v` in lane `l` of console wave `w`: the thread id, or a short
    /// chain of unguarded lane arithmetic over it and the console lane.
    fn at(&self, v: Value, w: u32, l: u32, depth: u32) -> Option<u32> {
        if Some(v) == self.tid {
            return Some(64 * w + l);
        }
        if depth > 8 {
            return None;
        }
        let (b, i) = *self.defs.get(&v)?;
        if self.x.st.guards[b][i].is_some() {
            return None;
        }
        let inst = &self.x.p.blocks[b].insts[i];
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.dpp.is_some() || f.sdwa.is_some() {
            return None;
        }
        let src = |k: usize| self.operand(inst, *f.src.get(k)?, w, l, depth);
        Some(match inst.name() {
            "v_mov_b32" => src(0)?,
            "v_and_b32" => src(0)? & src(1)?,
            "v_or_b32" => src(0)? | src(1)?,
            "v_add_nc_u32" | "v_add_u32" => src(0)?.wrapping_add(src(1)?),
            "v_lshlrev_b32" => src(1)? << (src(0)? & 31),
            "v_lshrrev_b32" => src(1)? >> (src(0)? & 31),
            // The set bits of the mask in the lanes below this one, of its
            // low or high dword, added to the second source.
            "v_mbcnt_lo_u32_b32" => {
                let below = if l >= 32 { u32::MAX } else { (1u32 << l) - 1 };
                src(1)?.wrapping_add((src(0)? & below).count_ones())
            }
            "v_mbcnt_hi_u32_b32" => {
                let below = if l < 32 { 0 } else { (1u32 << (l - 32)) - 1 };
                src(1)?.wrapping_add((src(0)? & below).count_ones())
            }
            _ => return None,
        })
    }
}
