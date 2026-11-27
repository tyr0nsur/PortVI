//! Wave-wide regions, for programs in halves ([`super::halves`]). The two
//! host waves of a console wave trade at group barriers, which every thread
//! of the group must reach, so inside a branch or a loop only some lanes
//! run a trade is refused. Such a region can instead run wave-wide: every
//! thread runs every block of it in order, and each block acts only in the
//! lanes the console runs there.
//!
//! A region is a set of blocks not every thread meets at, joined by their
//! edges, that every lane enters from one block (its head) and leaves to
//! one (its join).
//!
//! - A block's lanes are those along its edges in; an edge's are those of
//!   the block it leaves that its branch sends that way.
//! - Each vector instruction's guard ([`Lower::guard`]) takes in its block's
//!   lanes: a VGPR write keeps the old value in the others, memory is
//!   accessed by those lanes only, and the wave operations that read EXEC
//!   (ballots, first lanes, DPP source lanes) see only them.
//! - A phi of a block of the region, or of its join, picks lane by lane the
//!   value along the edge the lane came by.
//! - Every block of the region is one where every thread meets
//!   ([`super::halves::Exchange`]): the halves trade there, and LDS is put
//!   in order there, as anywhere else.
//!
//! A loop of the region runs as the console's wave runs it: every thread
//! goes round while any lane of the group does, which every half of the
//! group tells the others at the end of each round ([`Lower::wide_round`]).
//! In a group of several console waves, a wave whose lanes have all left
//! goes round with none, which changes nothing it holds.
//!
//! - The header's lanes are those that enter the loop, then those that take
//!   an edge back to it.
//! - A lane that leaves along an edge out of the loop is out of it from
//!   then on; after the loop the edge's lanes are all that left along it,
//!   in any round.
//! - What a lane holds stays as it left it: VGPR writes are guarded, and
//!   every other value a block of the loop defines (scalars, compare bits,
//!   what whole-wave sections leave, ballots) is kept per lane from the last
//!   round the lane ran the block ([`Lower::wide_keep`]), as each lane holds
//!   it in the per-lane form.
//!
//! A region may hold the program's ends (it then has no join): a lane that
//! gets to an end is in no block after it, and every thread ends together
//! at the region's last block.
//!
//! Scalar code runs in every lane, as it does in the per-lane form, which
//! keeps each lane's values; a region whose scalar code has effects beyond
//! its values (stores, atomics, messages) does not run wave-wide, nor one
//! with a loop entered other than at its header.
//!
//! The lowering takes these regions only where an attempt was refused for
//! want of a meeting ([`Plan`]): a program that needs none is lowered as
//! before, and one that does pays for its regions only.

use std::cell::RefCell;

use super::*;

/// The regions run wave-wide, over the lowering's attempts ([`super::lower`]):
/// an attempt refused where the halves cannot meet asks for the region
/// around each block that wanted a meeting, and the next runs them
/// wave-wide.
#[derive(Default)]
pub(super) struct Plan {
    /// The regions run wave-wide, by their heads.
    heads: BTreeSet<usize>,
    /// The blocks the attempts since the plan last grew wanted a meeting at.
    asked: RefCell<BTreeSet<usize>>,
}

impl Plan {
    /// Notes that IR block `b` wanted every thread of the group to meet.
    pub(super) fn ask(&self, b: usize) {
        self.asked.borrow_mut().insert(b);
    }

    /// After an attempt refused for `what`: the regions around the blocks
    /// the attempts since the last call asked for join the plan (a join
    /// that read a value refused for want of a meeting is missing in the
    /// attempts after, and reading it there asks for nothing); whether
    /// another attempt is worth making (some region joined).
    pub(super) fn widen(&mut self, x: &Parts, what: Why) -> bool {
        let asked = std::mem::take(self.asked.get_mut());
        if asked.is_empty() || !super::halves::for_want_of_a_meeting(what) {
            return false;
        }
        let regions = regions(x);
        let mut grew = false;
        for b in asked {
            if let Some(r) = regions.iter().find(|r| r.blocks[b]) {
                grew |= self.heads.insert(r.head);
            }
        }
        grew
    }
}

/// A region that can run wave-wide.
struct Region {
    /// The block whose lane branch enters it.
    head: usize,
    /// The other edges into it: from blocks whose way on the whole group
    /// takes together ([`regions`]), to a block of the region.
    entries: Vec<(usize, usize)>,
    /// Per IR block: in the region.
    blocks: Vec<bool>,
    /// The block every way out of the region leads to; `None` where each
    /// way out is an end.
    join: Option<usize>,
    /// Its blocks in the order they are lowered: each after the blocks its
    /// forward edges come from, a loop's blocks together from its header.
    order: Vec<usize>,
    loops: Vec<Loop>,
}

/// A loop of a region: the blocks from which its header is reached again
/// without leaving them.
struct Loop {
    header: usize,
    /// Per IR block: in the loop.
    body: Vec<bool>,
    /// The block of the loop lowered last.
    last: usize,
    /// The blocks with an edge back to the header.
    latches: Vec<usize>,
    /// The edges out of the loop.
    exits: Vec<(usize, usize)>,
}

/// The regions of a program in halves that can run wave-wide: the blocks
/// not every thread meets at, joined by their edges, where such a set is
/// entered from one block and left to one (both where every thread meets,
/// the set between them in reverse postorder) or holds the program's ends,
/// has no scalar instruction with effects beyond its values, and its loops
/// are entered at their headers only; else the regions of its branches
/// where every thread meets, where those are.
///
/// Blocks every thread meets at whose way on the whole group takes together
/// (a branch on scalar values in a group of one console wave, the early end
/// of a program that skips its work) may also lead into the region, or
/// around it to its join. Around it, no lane of the group is in the region,
/// and the join takes that edge's values as they are. Into it, the group
/// runs the region from its first block, with every lane in the block that
/// edge leads to and none in the blocks before ([`Lower::wide_entries`]).
fn regions(x: &Parts) -> Vec<Region> {
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
                if not_taken != taken {
                    succ[b].push(not_taken);
                }
            }
            LaneExit::End => {}
            LaneExit::Indirect => return Vec::new(),
        }
    }
    let meet = super::halves::meeting(x);
    let mut rpo = vec![usize::MAX; n];
    for (k, b) in crate::ir::reverse_postorder(&p.blocks)
        .into_iter()
        .enumerate()
    {
        rpo[b] = k;
    }
    // The sets: blocks not every thread meets at, by their edges.
    let apart = |b: usize| rpo[b] != usize::MAX && !meet[b];
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
            if apart(b) && apart(t) {
                let (rb, rt) = (find(&mut root, b), find(&mut root, t));
                root[rb] = rt;
            }
        }
    }
    let mut sets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for b in (0..n).filter(|&b| apart(b)) {
        sets.entry(find(&mut root, b)).or_default().push(b);
    }
    let branches = super::halves::lane_regions(x).unwrap_or_default();
    // Blocks whose way on the whole group takes together: every thread
    // meets there, and no lane branch leaves them (in a group of several
    // console waves every branch is one: each wave may take its own way).
    let together = |q: usize| meet[q] && !branches.iter().any(|(h, _)| *h == q);
    let joins = descriptor_joins(x);
    let region = |members: &[usize]| -> Option<Region> {
        let mut blocks = vec![false; n];
        for &b in members {
            blocks[b] = true;
        }
        let (mut head, mut join, mut ends) = (None, None, false);
        let mut entries = Vec::new();
        for &b in members {
            ends |= st.exits[b] == LaneExit::End;
            if p.blocks[b].insts.iter().any(|i| scalar_effect(i.name())) {
                return None;
            }
            for &q in &p.blocks[b].preds {
                if rpo[q] == usize::MAX || blocks[q] {
                    continue;
                }
                if together(q) {
                    entries.push((q, b));
                } else if *head.get_or_insert(q) != q {
                    return None;
                }
            }
            for &t in &succ[b] {
                if !blocks[t] && *join.get_or_insert(t) != t {
                    return None;
                }
            }
        }
        // Entered from one block only, that is the head, whatever its way on.
        if head.is_none() && entries.iter().all(|&(q, _)| q == entries[0].0) {
            head = entries.first().map(|&(q, _)| q);
            entries.clear();
        }
        // A lane that ends never gets to a join: a region with ends has none.
        let head = head?;
        if ends == join.is_some() {
            return None;
        }
        // shortcut: a descriptor chosen by control flow in a region entered
        // other than from its head is made from each way in's own block
        // (`Lower::uniform_heap_index`), which does not lead to the region's
        // blocks there; such a region is not run wave-wide.
        if !entries.is_empty() && members.iter().any(|b| joins.contains(b)) {
            return None;
        }
        // After the first block it is entered from (an edge in past the head
        // may come first in reverse postorder), before its join.
        let first = entries
            .iter()
            .map(|&(q, _)| rpo[q])
            .fold(rpo[head], usize::min);
        let between = |b: usize| first < rpo[b] && join.is_none_or(|j| rpo[b] < rpo[j]);
        if succ[head].iter().any(|&t| !blocks[t] && Some(t) != join)
            || join.is_some_and(|j| {
                p.blocks[j]
                    .preds
                    .iter()
                    .any(|&q| rpo[q] != usize::MAX && !blocks[q] && q != head && !together(q))
            })
            || !members.iter().all(|&b| between(b))
        {
            return None;
        }
        let mut loops = loops(p, members, &blocks, &succ, &rpo)?;
        // An entry other than the head is not into a loop of the region,
        // nor is the region's first block a loop's header: the entries go
        // there ([`Lower::wide_entries`]).
        if entries
            .iter()
            .any(|&(_, t)| loops.iter().any(|l| l.body[t]))
        {
            return None;
        }
        let order = chain(&blocks, None, &mut loops, &succ, &rpo)?;
        if !entries.is_empty() && loops.iter().any(|l| l.header == order[0]) {
            return None;
        }
        Some(Region {
            head,
            entries,
            blocks,
            join,
            order,
            loops,
        })
    };
    // Where a set cannot run wave-wide as one (two branches' regions that
    // share a join, say), the region of each branch in it where every thread
    // meets may.
    // A branch's region may span two sets (its sides in each), or reach
    // round a loop into another branch's: the first taken keeps its blocks.
    let mut regions = Vec::new();
    let mut taken = vec![false; n];
    for members in sets.into_values() {
        if let Some(r) = region(&members) {
            regions.push(r);
            continue;
        }
        for (head, inside) in &branches {
            if meet[*head] && members.iter().any(|&b| inside[b]) {
                let part: Vec<usize> = (0..n).filter(|&b| inside[b]).collect();
                if part.iter().any(|&b| taken[b]) {
                    continue;
                }
                if let Some(r) = region(&part) {
                    for &b in &part {
                        taken[b] = true;
                    }
                    regions.push(r);
                }
            }
        }
    }
    regions
}

/// The loops of a region (`members`, `blocks`), by header: `None` where one
/// is entered other than at its header, or two hold each other's headers.
/// Each loop's last block is set by [`chain`].
fn loops(
    p: &Program,
    members: &[usize],
    blocks: &[bool],
    succ: &[Vec<usize>],
    rpo: &[usize],
) -> Option<Vec<Loop>> {
    let n = blocks.len();
    let mut latches: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &u in members {
        for &t in &succ[u] {
            if blocks[t] && rpo[t] <= rpo[u] {
                latches.entry(t).or_default().push(u);
            }
        }
    }
    let mut loops = Vec::new();
    for (header, latches) in latches {
        let mut body = vec![false; n];
        body[header] = true;
        let mut stack = latches.clone();
        while let Some(b) = stack.pop() {
            if body[b] {
                continue;
            }
            body[b] = true;
            for &q in &p.blocks[b].preds {
                if rpo[q] == usize::MAX {
                    continue;
                }
                // An edge into the loop past its header.
                if !blocks[q] {
                    return None;
                }
                stack.push(q);
            }
        }
        let exits = (0..n)
            .filter(|&b| body[b])
            .flat_map(|b| succ[b].iter().map(move |&t| (b, t)))
            .filter(|&(_, t)| !body[t])
            .collect();
        loops.push(Loop {
            header,
            body,
            last: header,
            latches,
            exits,
        });
    }
    for a in &loops {
        for b in &loops {
            if a.header != b.header && a.body[b.header] && b.body[a.header] {
                return None;
            }
        }
    }
    Some(loops)
}

/// The blocks of `set` (a region, or with `header` the loop it heads) in
/// the order they are lowered: each after the blocks its edges come from,
/// edges back to `header` aside, the first in reverse postorder first, and
/// each loop inside taken whole, in its own order. `None` where no such
/// order exists.
fn chain(
    set: &[bool],
    header: Option<usize>,
    loops: &mut [Loop],
    succ: &[Vec<usize>],
    rpo: &[usize],
) -> Option<Vec<usize>> {
    let n = set.len();
    // The loops directly inside: those no other loop inside holds.
    let inside: Vec<usize> = (0..loops.len())
        .filter(|&l| set[loops[l].header] && Some(loops[l].header) != header)
        .collect();
    let outer: Vec<usize> = inside
        .iter()
        .copied()
        .filter(|&l| {
            !inside
                .iter()
                .any(|&m| m != l && loops[m].body[loops[l].header])
        })
        .collect();
    // Each block's unit: the loop directly inside that holds it (by its
    // header), or itself.
    let units: Vec<usize> = (0..n)
        .map(|b| {
            outer
                .iter()
                .find(|&&l| loops[l].body[b])
                .map_or(b, |&l| loops[l].header)
        })
        .collect();
    let mut waits = vec![0usize; n];
    for u in (0..n).filter(|&u| set[u]) {
        for &t in &succ[u] {
            if set[t] && Some(t) != header && units[u] != units[t] {
                waits[units[t]] += 1;
            }
        }
    }
    let mut ready: BTreeSet<(usize, usize)> = (0..n)
        .filter(|&b| set[b] && units[b] == b && waits[b] == 0)
        .map(|b| (rpo[b], b))
        .collect();
    let mut order = Vec::new();
    while let Some((_, b)) = ready.pop_first() {
        let part = match outer.iter().find(|&&l| loops[l].header == b) {
            Some(&l) => {
                let body = loops[l].body.clone();
                let part = chain(&body, Some(b), loops, succ, rpo)?;
                loops[l].last = *part.last()?;
                part
            }
            None => vec![b],
        };
        for &u in &part {
            for &t in &succ[u] {
                if set[t] && Some(t) != header && units[t] != b {
                    waits[units[t]] -= 1;
                    if waits[units[t]] == 0 {
                        ready.insert((rpo[units[t]], units[t]));
                    }
                }
            }
        }
        order.extend(part);
    }
    (order.len() == set.iter().filter(|&&s| s).count()).then_some(order)
}

/// Whether scalar instruction `name` has an effect beyond the values it
/// writes, which running it in lanes the console does not would repeat.
fn scalar_effect(name: &str) -> bool {
    const EFFECTS: [&str; 9] = [
        "s_store",
        "s_buffer_store",
        "s_scratch_store",
        "s_atomic",
        "s_buffer_atomic",
        "s_sendmsg",
        "s_trap",
        "s_sethalt",
        "s_dcache_wb",
    ];
    EFFECTS.iter().any(|e| name.starts_with(e))
}

/// A loop run wave-wide, as lowered.
struct Round {
    /// The DXIL block the loop is entered from.
    before: usize,
    /// The header's lanes: a phi of those entering and those going round.
    lanes: V,
    /// The lanes entering.
    entering: V,
    /// Per edge out of the loop: a phi of the lanes that left along it in
    /// earlier rounds.
    left: Vec<((usize, usize), V)>,
    /// Values kept per lane across rounds ([`Lower::wide_keep`]) while the
    /// loop is lowered: the phi at the header, what it starts as, and the
    /// value at the end of a round.
    kept: Vec<(V, V, V)>,
    /// Once the loop's last block is lowered: the DXIL block its rounds end
    /// in, and the one that goes back to the header.
    end: Option<(usize, usize)>,
}

/// The [`Plan`]'s regions laid out for one attempt, and their lanes as
/// lowered.
#[derive(Default)]
pub(super) struct Wide {
    /// Per IR block heading a region or in one: the block lowered after it
    /// (the join after the region's last).
    next: HashMap<usize, usize>,
    /// The last blocks of regions without a join: every thread ends there.
    last: HashSet<usize>,
    /// Per IR block: in a region run wave-wide.
    pub(super) inside: Vec<bool>,
    /// The regions' blocks in the order they are lowered, and the block
    /// outside they are lowered after where that is not the one before
    /// their first: the last of the blocks they are entered from.
    orders: Vec<(Vec<usize>, Option<usize>)>,
    /// Per region entered other than from its head, by its first block:
    /// how ([`Lower::wide_entries`]).
    entered: HashMap<usize, Entered>,
    /// Per edge into a region other than from its head: the region's first
    /// block; once its source is lowered, the DXIL block the edge goes to.
    entry_into: HashMap<(usize, usize), usize>,
    entry_block: HashMap<(usize, usize), usize>,
    /// Per IR block of such a region, its join and its last block: the
    /// region's first block.
    entered_by: HashMap<usize, usize>,
    /// Per such region, by its first block, while it is lowered: the values
    /// [`Lower::wide_entries`] replaced, as they were (and so as the blocks
    /// outside the region read them), and the DXIL values that replaced
    /// them, by the ones replaced (for its phis, which are set last).
    swaps: HashMap<usize, (Vec<Swapped>, HashMap<V, V>)>,
    /// The regions' loops, by header.
    loops: HashMap<usize, Loop>,
    /// Per IR block that is the last of loops: their headers, the inmost
    /// first.
    ends: HashMap<usize, Vec<usize>>,
    /// Per IR block in loops: their headers, the inmost first.
    around: HashMap<usize, Vec<usize>>,
    /// Per IR block in a region or joining one, once the block lowered
    /// before it is: that block's last DXIL block, its predecessor (a loop's
    /// header has a second, where its rounds go back).
    from: HashMap<usize, usize>,
    /// Per IR block in a region: its lanes (DXIL i1), from its start.
    lanes: HashMap<usize, V>,
    /// Per IR edge out of a region's head or block: its lanes. After a loop,
    /// an edge out of it has all the lanes that left along it.
    edges: HashMap<(usize, usize), V>,
    /// The loops being lowered or lowered, by header.
    rounds: HashMap<usize, Round>,
}

impl Wide {
    pub(super) fn new(x: &Parts, plan: &Plan) -> Self {
        let n = x.st.exits.len();
        let mut wide = Wide {
            inside: vec![false; n],
            ..Wide::default()
        };
        if plan.heads.is_empty() {
            return wide;
        }
        let mut doms: Option<Doms> = None;
        for r in regions(x)
            .into_iter()
            .filter(|r| plan.heads.contains(&r.head))
        {
            let first = r.order[0];
            let mut after = None;
            if !r.entries.is_empty() {
                let doms = doms.get_or_insert_with(|| Doms::new(x.p));
                let e = Entered::new(x, &r, doms);
                // Lowered once every block it is entered from is.
                after = e
                    .sources()
                    .max_by_key(|&q| doms.rpo[q])
                    .filter(|&q| doms.rpo[q] > doms.rpo[first]);
                for &edge in &r.entries {
                    wide.entry_into.insert(edge, first);
                }
                for &b in r.order.iter().chain(&r.join) {
                    wide.entered_by.insert(b, first);
                }
                wide.entered.insert(first, e);
            }
            let mut last = r.head;
            for &b in &r.order {
                wide.inside[b] = true;
                wide.next.insert(last, b);
                last = b;
            }
            match r.join {
                Some(j) => {
                    wide.next.insert(last, j);
                }
                None => {
                    wide.last.insert(last);
                }
            }
            // The inmost loop is the one with the fewest blocks.
            let mut loops = r.loops;
            loops.sort_by_key(|l| l.body.iter().filter(|&&b| b).count());
            for l in &loops {
                wide.ends.entry(l.last).or_default().push(l.header);
                for b in (0..n).filter(|&b| l.body[b]) {
                    wide.around.entry(b).or_default().push(l.header);
                }
            }
            wide.loops.extend(loops.into_iter().map(|l| (l.header, l)));
            wide.orders.push((r.order, after));
        }
        wide
    }

    /// `order` (the program's blocks in reverse postorder) with each
    /// region's blocks together in the region's own order, where its first
    /// was, or after the last block it is entered from.
    pub(super) fn lowered_in(&self, order: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(order.len());
        for &b in order {
            if !self.inside[b] {
                out.push(b);
            }
            for (o, after) in &self.orders {
                if *after == Some(b) || after.is_none() && o[0] == b {
                    out.extend(o);
                }
            }
        }
        out
    }
}

/// The blocks where a descriptor of the program is chosen by control flow
/// ([`Source::Phi`], [`Source::LanePhi`]).
fn descriptor_joins(x: &Parts) -> HashSet<usize> {
    fn walk(s: &Source, out: &mut HashSet<usize>) {
        match s {
            Source::Phi { block, args } => {
                out.insert(*block);
                args.iter().for_each(|a| walk(a, out));
            }
            Source::LanePhi { block, args } => {
                out.insert(*block);
                args.iter().flatten().for_each(|a| walk(a, out));
            }
            Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().for_each(|a| walk(a, out));
            }
            _ => {}
        }
    }
    let mut out = HashSet::new();
    for d in &x.res.descriptors {
        walk(&d.source, &mut out);
    }
    out
}

/// A value [`Lower::wide_entries`] replaced, as the lowering held it.
struct Swapped {
    value: Value,
    val: Option<Val>,
    lane_bit: Option<V>,
    // One low word can have several distinct high-word partners.
    mask_phis: Vec<(Value, V)>,
}

/// A program's blocks' immediate dominators, and their places in reverse
/// postorder.
struct Doms {
    idom: Vec<usize>,
    rpo: Vec<usize>,
}

impl Doms {
    fn new(p: &Program) -> Self {
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (k, &b) in order.iter().enumerate() {
            rpo[b] = k;
        }
        Doms {
            idom: crate::ir::idoms(&p.blocks, &order, &rpo),
            rpo,
        }
    }

    /// Whether IR block `d` dominates IR block `b`.
    fn dominates(&self, d: usize, mut b: usize) -> bool {
        loop {
            if b == d {
                return true;
            }
            let up = self.idom[b];
            if up == usize::MAX || up == b {
                return false;
            }
            b = up;
        }
    }
}

/// How a region is entered other than from its head ([`Region::entries`]).
#[derive(Clone)]
struct Entered {
    head: usize,
    /// The other edges in.
    edges: Vec<(usize, usize)>,
    /// The values defined on some ways in and not on others, which the
    /// region may read (by its instructions, or through the lane conditions
    /// and masks the lowering keeps of them): each with, per way in (the
    /// head, then each edge), whether it comes by its definition.
    values: Vec<(Value, Vec<bool>)>,
}

impl Entered {
    fn new(x: &Parts, r: &Region, doms: &Doms) -> Self {
        let p = x.p;
        let mut e = Entered {
            head: r.head,
            edges: r.entries.clone(),
            values: Vec::new(),
        };
        let sources: Vec<usize> = e.sources().collect();
        for (d, block) in p.blocks.iter().enumerate() {
            if r.blocks[d] || doms.rpo[d] == usize::MAX {
                continue;
            }
            let by: Vec<bool> = sources.iter().map(|&s| doms.dominates(d, s)).collect();
            if by.iter().all(|&b| b) || !by.iter().any(|&b| b) {
                continue;
            }
            let defined = block
                .phis
                .iter()
                .map(|phi| phi.value)
                .chain(block.insts.iter().flat_map(|i| &i.writes).map(|w| w.1));
            e.values.extend(defined.map(|v| (v, by.clone())));
        }
        e
    }

    /// The blocks it is entered from: the head, then each edge's source.
    fn sources(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(self.head).chain(self.edges.iter().map(|&(q, _)| q))
    }
}

impl Lower<'_> {
    /// The DXIL block an edge from IR block `from` into a region at `to`,
    /// other than from its head, goes to: a block of its own on to the
    /// region's first block ([`Lower::wide_entries`]).
    pub(super) fn wide_entry(&mut self, from: usize, to: usize) -> Option<usize> {
        let first = *self.wide.entry_into.get(&(from, to))?;
        let e = self.s.m.new_block();
        self.s
            .m
            .terminate(e, Term::Br(self.block[first].expect("reachable")));
        self.wide.entry_block.insert((from, to), e);
        Some(e)
    }

    /// At the start of IR block `b`, the first of a region entered other
    /// than from its head ([`regions`]), which every way in leads to: which
    /// way the group came. Along the head's edges, their lanes, and none
    /// where it came by another; along each other edge, every lane where
    /// it came by that one, and none elsewhere. The blocks before the one
    /// an edge leads to run with no lanes. A value defined outside that
    /// some way in does not come by is zero along it (no lane there reads
    /// it).
    fn wide_entries(&mut self, b: usize, e: &Entered) {
        let c = self.cur;
        let head_tail = self.wide.from[&b];
        let blocks: Vec<usize> = e
            .edges
            .iter()
            .map(|edge| self.wide.entry_block[edge])
            .collect();
        let ways: Vec<usize> = std::iter::once(head_tail)
            .chain(blocks.iter().copied())
            .collect();
        let i1 = self.s.t.i1;
        let (f, t) = (self.s.m.bool(false), self.s.m.bool(true));
        let mut head_edges: Vec<((usize, usize), V)> = self
            .wide
            .edges
            .iter()
            .filter(|((q, _), _)| *q == e.head)
            .map(|(&k, &v)| (k, v))
            .collect();
        head_edges.sort_by_key(|&(k, _)| k);
        for (edge, lanes) in head_edges {
            let phi = self.s.m.phi(c, i1);
            let incoming = std::iter::once(lanes)
                .chain(blocks.iter().map(|_| f))
                .zip(ways.iter().copied())
                .collect();
            self.s.m.set_phi(phi, incoming);
            self.wide.edges.insert(edge, phi);
        }
        for (i, &edge) in e.edges.iter().enumerate() {
            let phi = self.s.m.phi(c, i1);
            let incoming = std::iter::once(f)
                .chain((0..blocks.len()).map(|k| if k == i { t } else { f }))
                .zip(ways.iter().copied())
                .collect();
            self.s.m.set_phi(phi, incoming);
            self.wide.edges.insert(edge, phi);
        }
        let (mut saved, mut swaps) = (Vec::new(), HashMap::new());
        let mut masks_by_low: HashMap<_, Vec<_>> = HashMap::new();
        for (&(low, high), &mask) in &self.mask_phis {
            masks_by_low.entry(low).or_default().push((high, mask));
        }
        for (v, by) in &e.values {
            let mut mask_phis = masks_by_low.remove(v).unwrap_or_default();
            mask_phis.sort_unstable_by_key(|&(high, _)| high);
            let was = Swapped {
                value: *v,
                val: self.vals.get(v).copied(),
                lane_bit: self.lane_bits.get(v).copied(),
                mask_phis,
            };
            // A constant is the same along every way in.
            let made = match was.val {
                Some(Val::I32(x)) if !matches!(x, V::Const(_)) => Some((x, false)),
                Some(Val::Bit(x)) if !matches!(x, V::Const(_)) => Some((x, true)),
                _ => None,
            };
            if let Some((x, bit)) = made {
                let y = self.zero_along(c, x, bit, by, &ways);
                swaps.insert(x, y);
                self.vals
                    .insert(*v, if bit { Val::Bit(y) } else { Val::I32(y) });
            }
            if let Some(x) = was.lane_bit.filter(|x| !matches!(x, V::Const(_))) {
                let y = self.zero_along(c, x, true, by, &ways);
                swaps.insert(x, y);
                self.lane_bits.insert(*v, y);
            }
            for &(high, x) in &was.mask_phis {
                if matches!(x, V::Const(_)) {
                    continue;
                }
                let y = self.zero_along(c, x, true, by, &ways);
                swaps.insert(x, y);
                self.mask_phis.insert((*v, high), y);
            }
            // Made again from the value, here.
            self.floats.remove(v);
            self.low16.remove(v);
            self.ballots.remove(v);
            saved.push(was);
        }
        self.wide.swaps.insert(b, (saved, swaps));
        // Lane booleans worked out before now read the values as they were.
        self.bools.clear();
    }

    /// After the last block of a region [`Lower::wide_entries`] lowered:
    /// the values it replaced are again as they were, for the blocks after
    /// the region (its join reads them through its phis, which map them,
    /// [`Lower::wide_phi`]).
    fn wide_restore(&mut self, last: usize) {
        let Some(&first) = self.wide.entered_by.get(&last) else {
            return;
        };
        if self
            .wide
            .next
            .get(&last)
            .is_some_and(|n| self.wide.entered_by.get(n) == Some(&first) && self.wide.inside[*n])
        {
            return;
        }
        let Some((saved, _)) = self.wide.swaps.get_mut(&first) else {
            return;
        };
        let saved = std::mem::take(saved);
        let restored: HashSet<_> = saved.iter().map(|was| was.value).collect();
        self.mask_phis.retain(|(low, _), _| !restored.contains(low));
        for was in saved {
            let v = was.value;
            match was.val {
                Some(val) => self.vals.insert(v, val),
                None => self.vals.remove(&v),
            };
            match was.lane_bit {
                Some(x) => self.lane_bits.insert(v, x),
                None => self.lane_bits.remove(&v),
            };
            for (high, mask) in was.mask_phis {
                self.mask_phis.insert((v, high), mask);
            }
            self.floats.remove(&v);
            self.low16.remove(&v);
            self.ballots.remove(&v);
        }
        self.bools.clear();
    }

    /// A phi in DXIL block `at` of `x` along the `ways` in that `by` marks,
    /// zero (false where `bit`) along the others.
    fn zero_along(&mut self, at: usize, x: V, bit: bool, by: &[bool], ways: &[usize]) -> V {
        let (ty, zero) = if bit {
            (self.s.t.i1, self.s.m.bool(false))
        } else {
            (self.s.t.i32, self.s.m.i32(0))
        };
        let phi = self.s.m.phi(at, ty);
        let incoming = by
            .iter()
            .zip(ways)
            .map(|(&b, &w)| (if b { x } else { zero }, w))
            .collect();
        self.s.m.set_phi(phi, incoming);
        phi
    }

    /// At the start of IR block `b`, after its phis: where `b` runs
    /// wave-wide, its lanes, those of its edges in. A loop's header takes
    /// the lanes entering the loop, then those going round.
    pub(super) fn wide_enter(&mut self, b: usize) {
        if !self.wide.inside.get(b).copied().unwrap_or(false) {
            return;
        }
        if let Some(e) = self.wide.entered.get(&b).cloned() {
            self.wide_entries(b, &e);
        }
        let c = self.cur;
        let body = self.wide.loops.get(&b).map(|l| l.body.clone());
        let at = match (&body, self.wide.from.get(&b)) {
            (Some(_), Some(&before)) => before,
            _ => c,
        };
        let mut on = self.s.m.bool(false);
        for &q in &self.x.p.blocks[b].preds {
            if body.as_ref().is_some_and(|body| body[q]) {
                continue;
            }
            if let Some(&e) = self.wide.edges.get(&(q, b)) {
                on = self.s.m.bin(at, Bin::Or, on, e);
            }
        }
        if body.is_some() {
            let i1 = self.s.t.i1;
            let lanes = self.s.m.phi(c, i1);
            let exits = self.wide.loops[&b].exits.clone();
            let left = exits
                .into_iter()
                .map(|e| (e, self.s.m.phi(c, i1)))
                .collect();
            self.wide.rounds.insert(
                b,
                Round {
                    before: at,
                    lanes,
                    entering: on,
                    left,
                    kept: Vec::new(),
                    end: None,
                },
            );
            on = lanes;
        }
        self.wide.lanes.insert(b, on);
    }

    /// `x`, an i1 per lane in DXIL block `at`, in IR block `b`'s lanes only
    /// where it runs wave-wide.
    pub(super) fn wide_and(&mut self, b: usize, x: V, at: usize) -> V {
        match self.wide.lanes.get(&b) {
            Some(&on) => self.s.m.bin(at, Bin::And, on, x),
            None => x,
        }
    }

    /// The guard of an instruction in IR block `b` (`g`, if any), with the
    /// block's lanes where it runs wave-wide.
    pub(super) fn wide_guard(&mut self, b: usize, g: Option<V>) -> Option<V> {
        match (g, self.wide.lanes.get(&b)) {
            (Some(g), Some(_)) => Some(self.wide_and(b, g, self.cur)),
            (None, Some(&on)) => Some(on),
            (g, None) => g,
        }
    }

    /// `x`, a value IR block `b` defines in DXIL block `at`, as each lane
    /// holds it: where `b` is in loops run wave-wide, the lanes outside the
    /// block keep what they held, from the last round they ran it (a phi at
    /// each loop's header carries it round; a lane that never ran it holds
    /// zero, which nothing of its own reads).
    pub(super) fn wide_keep(&mut self, b: usize, x: V, at: usize) -> V {
        let (Some(&on), Some(around)) = (self.wide.lanes.get(&b), self.wide.around.get(&b)) else {
            return x;
        };
        let around = around.clone();
        let ty = self.s.m.type_of(x);
        let mut held = if ty == self.s.t.i1 {
            self.s.m.bool(false)
        } else {
            self.s.m.i32(0)
        };
        // From the outmost loop in: each starts with what the one around it
        // holds.
        let mut phis = Vec::new();
        for &h in around.iter().rev() {
            let blk = self.block[h].expect("reachable");
            let phi = self.s.m.phi(blk, ty);
            phis.push((h, phi, held));
            held = phi;
        }
        let kept = self.s.m.select(at, on, x, held);
        for (h, phi, starts) in phis {
            let round = self
                .wide
                .rounds
                .get_mut(&h)
                .expect("a header is lowered first");
            match round.end {
                Some((_, back)) => {
                    let before = round.before;
                    self.s.m.set_phi(phi, vec![(starts, before), (kept, back)]);
                }
                None => round.kept.push((phi, starts, kept)),
            }
        }
        kept
    }

    /// The dwords of a ballot of lane mask `mask`, taken in DXIL block `at`
    /// of the block writing it, as each lane holds them
    /// ([`Lower::wide_keep`]).
    pub(super) fn wide_keep_ballot(&mut self, mask: Value, (x, y): (V, V), at: usize) -> (V, V) {
        match self.def_block.get(&mask).copied() {
            Some(b) => (self.wide_keep(b, x, at), self.wide_keep(b, y, at)),
            None => (x, y),
        }
    }

    /// At the end of IR block `b` in a loop run wave-wide: the values it
    /// defines other than guarded VGPR writes (scalars, compare bits, what
    /// whole-wave sections leave) are held per lane from here on
    /// ([`Lower::wide_keep`]).
    fn wide_keep_block(&mut self, b: usize) {
        if !self.wide.around.contains_key(&b) || !self.wide.lanes.contains_key(&b) {
            return;
        }
        let block = &self.x.p.blocks[b];
        let at = self.cur;
        let sections: HashSet<usize> = self
            .sections
            .get(&b)
            .into_iter()
            .flatten()
            .flat_map(|(&start, s)| start + 1..s.end)
            .collect();
        let mut values: Vec<Value> = block
            .phis
            .iter()
            .filter(|phi| !matches!(phi.reg, Reg::V(_)))
            .map(|phi| phi.value)
            .collect();
        for (i, inst) in block.insts.iter().enumerate() {
            for &(reg, v) in &inst.writes {
                if !matches!(reg, Reg::V(_)) || sections.contains(&i) {
                    values.push(v);
                }
            }
        }
        for v in values {
            match self.vals.get(&v).copied() {
                Some(Val::I32(x)) => {
                    let kept = self.wide_keep(b, x, at);
                    self.vals.insert(v, Val::I32(kept));
                }
                Some(Val::Bit(x)) => {
                    let kept = self.wide_keep(b, x, at);
                    self.vals.insert(v, Val::Bit(kept));
                }
                _ => {}
            }
            if let Some(&bit) = self.lane_bits.get(&v) {
                let kept = self.wide_keep(b, bit, at);
                self.lane_bits.insert(v, kept);
            }
        }
        // Lane booleans worked out before now read the values as they were.
        self.bools.clear();
    }

    /// The exit of IR block `b` where it heads a region run wave-wide or is
    /// in one: on to the block lowered next (or, the last of a region
    /// without a join, the program's end, every thread together), the lanes
    /// of each edge noted, after the end of each loop `b` is the last block
    /// of; `None` elsewhere.
    pub(super) fn wide_exit(&mut self, b: usize) -> Result<Option<Term>, Why> {
        let next = self.wide.next.get(&b).copied();
        if next.is_none() && !self.wide.last.contains(&b) {
            return Ok(None);
        }
        self.wide_keep_block(b);
        let c = self.cur;
        let on = self.wide.lanes.get(&b).copied();
        let edge = |l: &mut Self, to: usize, e: Option<V>| {
            let e = e.unwrap_or_else(|| l.s.m.bool(true));
            l.wide.edges.insert((b, to), e);
        };
        match self.x.st.exits[b] {
            LaneExit::Jump(t) => edge(self, t, on),
            LaneExit::Branch {
                taken, not_taken, ..
            } if taken == not_taken => edge(self, taken, on),
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition goes the other way round (as `exit`).
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        (self.atom(atom, c)?, not_taken, taken)
                    }
                    _ => (self.boolean(when, c)?, taken, not_taken),
                };
                let t = self.s.m.bool(true);
                let not = self.s.m.bin(c, Bin::Xor, cond, t);
                let (yes, no) = match on {
                    Some(on) => (
                        self.s.m.bin(c, Bin::And, on, cond),
                        self.s.m.bin(c, Bin::And, on, not),
                    ),
                    None => (cond, not),
                };
                edge(self, taken, Some(yes));
                edge(self, not_taken, Some(no));
            }
            // Its lanes are done: they are in no block after it.
            LaneExit::End => {}
            LaneExit::Indirect => return Ok(None),
        }
        for h in self.wide.ends.get(&b).cloned().unwrap_or_default() {
            self.wide_round(h)?;
        }
        self.wide_restore(b);
        let Some(next) = next else {
            return Ok(Some(Term::Ret));
        };
        // Into a loop: its count of rounds starts.
        self.enter_loop(b, next, self.cur);
        self.wide.from.insert(next, self.cur);
        Ok(Some(Term::Br(self.block[next].expect("reachable"))))
    }

    /// The end of a round of the loop `header` heads, in the DXIL block
    /// being written, which every thread of the group reaches: every half
    /// tells the others whether any of its lanes takes an edge back to the
    /// header ([`Lower::group_any`]), and every thread goes round again
    /// while one does. After the loop, each edge out of it has the lanes
    /// that left along it in any round.
    ///
    /// Past [`LOOP_LIMIT`] rounds the loop ends for every thread, by its one
    /// way out: a lane still going round takes no edge out, so it is in no
    /// block of the region after the loop. (A second way out, to a return,
    /// made the Radeon iGPU of a Ryzen 9 7900X store, after such a loop, in
    /// lanes whose guard was false.)
    fn wide_round(&mut self, header: usize) -> Result<(), Why> {
        let c = self.cur;
        let (latches, before) = (
            self.wide.loops[&header].latches.clone(),
            self.wide.rounds[&header].before,
        );
        let mut round_again = self.s.m.bool(false);
        for u in latches {
            if let Some(&e) = self.wide.edges.get(&(u, header)) {
                round_again = self.s.m.bin(c, Bin::Or, round_again, e);
            }
        }
        let left: Vec<((usize, usize), V, V)> = self.wide.rounds[&header]
            .left
            .clone()
            .into_iter()
            .map(|(e, phi)| {
                let now = match self.wide.edges.get(&e) {
                    Some(&now) => self.s.m.bin(c, Bin::Or, phi, now),
                    None => phi,
                };
                (e, phi, now)
            })
            .collect();
        // Each half's lanes going round, as a ballot; any lane of the group.
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self.s.op(
            c,
            "waveActiveBallot",
            Attr::NoUnwind,
            four,
            116,
            &[round_again],
        );
        self.s.flags |= flags::WAVE_OPS;
        let mine = self.s.m.extract(c, ballot, 0);
        let any = self.group_any(c, mine)?;
        let (back, after) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(c, Term::CondBr(any, back, after));
        self.cur = back;
        let ok = self.count_iteration(header, pred::IULT);
        let top = self.block[header].expect("reachable");
        self.s.m.terminate(back, Term::CondBr(ok, top, after));
        let f = self.s.m.bool(false);
        let round = self.wide.rounds.get_mut(&header).expect("lowered");
        round.end = Some((c, back));
        let (lanes, entering) = (round.lanes, round.entering);
        let kept = std::mem::take(&mut round.kept);
        self.s
            .m
            .set_phi(lanes, vec![(entering, before), (round_again, back)]);
        for (e, phi, now) in left {
            self.s.m.set_phi(phi, vec![(f, before), (now, back)]);
            self.wide.edges.insert(e, now);
        }
        for (phi, starts, value) in kept {
            self.s.m.set_phi(phi, vec![(starts, before), (value, back)]);
        }
        self.cur = after;
        Ok(())
    }

    /// Sets DXIL phi `v` of IR block `b` from `incoming`, the values along
    /// the edges from IR blocks `from`. Where `b` is in a region run
    /// wave-wide or joins one, its DXIL predecessor brings the value along
    /// the edge each lane came by; a loop's header has two, the block the
    /// loop is entered from and the one its rounds go back from. An edge
    /// into a join around the region (the whole group took it, see
    /// [`regions`]) keeps its own DXIL predecessor. In a region entered
    /// other than from its head, a value replaced along the ways in
    /// ([`Lower::wide_entries`]) is what replaced it.
    pub(super) fn wide_phi(&mut self, v: V, b: usize, incoming: Vec<(V, usize)>, from: &[usize]) {
        let Some(&at) = self.wide.from.get(&b) else {
            self.s.m.set_phi(v, incoming);
            return;
        };
        let around = |q: usize| {
            !self.wide.inside[q]
                && !self.wide.next.contains_key(&q)
                && !self.wide.entry_into.contains_key(&(q, b))
        };
        let (around, edges): (Vec<_>, Vec<_>) = incoming
            .into_iter()
            .zip(from)
            .partition(|&(_, &q)| around(q));
        let swaps = self
            .wide
            .entered_by
            .get(&b)
            .and_then(|first| self.wide.swaps.get(first))
            .map(|(_, s)| s);
        let edges: Vec<(V, usize)> = edges
            .into_iter()
            .map(|((x, _), &q)| (swaps.and_then(|s| s.get(&x)).copied().unwrap_or(x), q))
            .collect();
        let Some((end, back)) = self.wide.rounds.get(&b).and_then(|r| r.end) else {
            let picked = self.wide_pick(b, &edges, at);
            let mut incoming: Vec<(V, usize)> = around.into_iter().map(|(x, _)| x).collect();
            incoming.extend(picked.map(|x| (x, at)));
            self.s.m.set_phi(v, incoming);
            return;
        };
        let body = &self.wide.loops[&b].body;
        let (round_again, entering): (Vec<_>, Vec<_>) = edges.iter().partition(|&&(_, q)| body[q]);
        let mut incoming = Vec::new();
        if let Some(x) = self.wide_pick(b, &entering, at) {
            incoming.push((x, at));
        }
        if let Some(x) = self.wide_pick(b, &round_again, end) {
            incoming.push((x, back));
        }
        self.s.m.set_phi(v, incoming);
    }

    /// One value from `edges` (each a value and the IR block its edge to `b`
    /// leaves), picked lane by lane in DXIL block `at` by the edge the lane
    /// came by; the last where it came by none.
    fn wide_pick(&mut self, b: usize, edges: &[(V, usize)], at: usize) -> Option<V> {
        let mut picked: Option<V> = None;
        for &(x, q) in edges.iter().rev() {
            picked = Some(match picked {
                None => x,
                Some(rest) => {
                    let e = match self.wide.edges.get(&(q, b)) {
                        Some(&e) => e,
                        None => self.s.m.bool(false),
                    };
                    self.s.m.select(at, e, x, rest)
                }
            });
        }
        picked
    }
}
