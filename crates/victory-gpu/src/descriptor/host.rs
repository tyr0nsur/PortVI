//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at, views_current};
use crate::host::host;

/// The title slots a view heap was made from, and where title slots past
/// its direct ones live (`crate::host::Host::view_remap`): a host slot of
/// the overflow range (`count..count + overflow`) while one holds a view,
/// none otherwise. After Marvisak/VIctory#3518.
struct Slots {
    /// The record (view index and generation) each title slot (all of the
    /// title heap's) was last made from; 0 for none.
    made: Vec<u64>,
    /// Title slot (at or past `count`) to its overflow host slot.
    to_phys: std::collections::HashMap<u32, u32>,
    /// Overflow host slot (less `count`) to its title slot; `u32::MAX` free.
    to_title: Vec<u32>,
    /// Free overflow host slots, the last freed taken first.
    free: Vec<u32>,
}

impl Slots {
    fn new(title_count: u32, direct: u32, overflow: u32) -> Self {
        // Only indices that the shader can reach need generation records.
        let tracked = if overflow == 0 { direct } else { title_count.min(victory_shader::emit::REMAP_VIEWS) };
        Slots {
            made: vec![0; tracked as usize],
            to_phys: std::collections::HashMap::new(),
            to_title: vec![u32::MAX; overflow as usize],
            free: (direct..direct + overflow).rev().collect(),
        }
    }

    /// The host slot of title slot `title`: itself below `direct`, its
    /// overflow slot past it; none for one that has none.
    fn phys_of(&self, direct: u32, title: u32) -> Option<u32> {
        if title < direct { Some(title) } else { self.to_phys.get(&title).copied() }
    }

    /// The overflow slot of title slot `title` (at or past `direct`), taken
    /// for it if it had none: the slot and whether it was taken now; none
    /// when every overflow slot is taken.
    fn take_overflow(&mut self, direct: u32, title: u32) -> Option<(u32, bool)> {
        if let Some(&phys) = self.to_phys.get(&title) {
            return Some((phys, false));
        }
        let phys = self.free.pop()?;
        self.to_phys.insert(title, phys);
        self.to_title[(phys - direct) as usize] = title;
        Some((phys, true))
    }

    /// Gives back title slot `title`'s overflow slot, if it has one.
    fn free_overflow(&mut self, direct: u32, title: u32) -> Option<u32> {
        let phys = self.to_phys.remove(&title)?;
        self.to_title[(phys - direct) as usize] = u32::MAX;
        self.free.push(phys);
        Some(phys)
    }

    /// The title slot host slot `phys` holds: itself below `direct`, the
    /// one an overflow slot holds, none for a free one or past them.
    fn title_of(&self, direct: u32, phys: u32) -> Option<u32> {
        if phys < direct {
            return Some(phys);
        }
        let t = *self.to_title.get((phys - direct) as usize)?;
        (t != u32::MAX).then_some(t)
    }
}

/// Overflow host slots a capped view heap keeps for title slots past its
/// direct ones when the title's slots do not all fit
/// (`crate::host::Host::view_remap`), taken from the title range of
/// [`capped_layout`] (the dynamic slots and view pairs keep theirs): 2^18,
/// as upstream, at most half the title range; `VICTORY_VIEW_OVERFLOW=n`
/// another number. Tables reaching past the direct slots are made afresh
/// in dynamic slots.
fn overflow_slots(title: u32) -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let n = *N.get_or_init(|| {
        std::env::var("VICTORY_VIEW_OVERFLOW")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1 << 18)
    });
    n.min(title / 2)
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's, title slot `i` in host slot `i`; view
    /// heaps have `dynamic` more.
    count: u32,
    /// The title heap's slots; past `count`, title slots live in the
    /// overflow range while they hold a view ([`Slots`]), else nowhere.
    title_count: u32,
    /// Host slots `count..count + overflow` for title slots past `count`
    /// (`crate::host::Host::view_remap`; none otherwise).
    overflow: u32,
    /// Dynamic slots from `base` (none in sampler heaps): [`DYNAMIC`],
    /// fewer only in a capped heap too small for it ([`capped_layout`]).
    dynamic: u32,
    /// The first dynamic slot: `count` where the heap mirrors every title
    /// slot below the shaders' clamp; on a host that caps its heaps, the
    /// same for every heap ([`title_views`]), so that an index shaders
    /// take from the heap itself below that bound never reaches a dynamic
    /// slot or a pair (slots from `count` to it stay null).
    base: u32,
    sampler: bool,
    /// The view heap size shaders are translated for
    /// (`victory_shader::emit::heap_views`): indices below it exist, the
    /// stride and mark tables cover them, the null views lie just below it.
    views: u32,
    /// The record each title slot was last made from, and the overflow
    /// slots title slots past `count` hold.
    slots: Mutex<Slots>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one.
    kept: Mutex<Vec<Kept>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per shader-visible slot, title and dynamic ones alike (tables made
    /// afresh lie in dynamic slots, and shaders index the view addresses at
    /// the slots their tables are in), the console descriptor's dwords 0-3
    /// as far as the host knows them (shaders reading a descriptor as data, `crate::memory`;
    /// [`write_address`]): the buffer, its mapping and host address, made on
    /// first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// View heaps whose shaders read title slots through the view-remap
    /// table (`victory_shader::emit::VIEW_REMAP`): a dword per title slot
    /// below `REMAP_VIEWS`, its host slot (the last, null, for none); made
    /// on first use, then kept as slots are taken and given back.
    remap: std::sync::OnceLock<Option<(ID3D12Resource, usize)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The slots a capped heap keeps for host copies' view pairs
/// (`crate::memory`) when the title's slots do not all fit: pairs live as
/// long as their copies, and taken from the dynamic slots instead they
/// would leave no long runs free there (root views take 256 at a time).
const CAPPED_PAIRS: u32 = 1 << 14;

/// The slots at the end of every view heap that stay null: the null raw
/// UAV and SRV (`victory_shader::emit::null_buffer_view`) and the last
/// one, which shaders' clamped indices land on.
const NULL_SLOTS: u32 = 3;

/// How the `views` slots of a view heap are shared on a host that makes no
/// larger one (1,000,000 on NVIDIA), when the title's slots, [`DYNAMIC`]
/// and the null slots do not all fit: the title slots it mirrors at most,
/// its dynamic slots, and the slots kept for view pairs past them, then
/// the [`NULL_SLOTS`].
///
/// The dynamic slots keep their full range where there is room for it: a
/// title heap with more slots than fit is cut short either way, tables
/// reaching past the cut are made afresh in dynamic slots for every draw,
/// and 2^16 of those ran out in seconds.
fn capped_layout(views: u32) -> (u32, u32, u32) {
    let room = views.saturating_sub(NULL_SLOTS);
    let dynamic = DYNAMIC.min(room / 2);
    let pairs = CAPPED_PAIRS.min(room / 8) & !1;
    (room - dynamic - pairs, dynamic, pairs)
}

/// The title slots every view heap of `views` slots mirrors at most, what
/// shaders translated for such heaps bound the indices they take from the
/// heap itself by (`victory_shader::emit::title_views`, told by
/// `crate::host` with this): `views` (no bound past the clamp) on a host
/// that makes heaps of `HEAP_VIEWS_MAX`, else the capped layout's title
/// slots, which every heap there keeps alike (dynamic slots from it on)
/// whatever its title heap's size.
pub(crate) fn title_views(views: u32) -> u32 {
    if views >= victory_shader::emit::HEAP_VIEWS_MAX {
        views
    } else {
        capped_layout(views).0
    }
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        // The host device first: making it says how large its view heaps
        // are (`victory_shader::emit::heap_views`).
        host()?;
        Self::sized(sampler, count, victory_shader::emit::heap_views())
    }

    /// [`HostHeap::new`] for view heaps of `views` slots (what translated
    /// shaders index: `victory_shader::emit::heap_views`), the most the
    /// host makes unless it is `HEAP_VIEWS_MAX`.
    fn sized(sampler: bool, count: u32, views: u32) -> Option<Self> {
        let host = host()?;
        let title_count = count;
        // Title slots past the direct ones live in an overflow range,
        // shaders reading them through the view-remap table.
        let remap = !sampler && host.view_remap;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `views`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(views)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(views)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // Every title slot and the full dynamic range where the heap takes
        // them; else the capped layout. Translated shaders clamp their
        // indices below `views` or `HEAP_SAMPLERS`: slots up to there exist
        // and stay null.
        let samplers = victory_shader::emit::HEAP_SAMPLERS;
        let full = count.saturating_add(DYNAMIC);
        // A host that makes larger heaps may take one past `views` (its null
        // slots then lie among the title's, as before), dynamic slots right
        // after the title's. One that makes none (NVIDIA) gets heaps of
        // exactly `views` slots laid out alike whatever the title heap's
        // size: title slots below `title_views(views)`, the dynamic ones
        // from there, then pairs and the null ones last. Shaders translated
        // for it send an index they take from the heap itself at or past
        // that bound to the last slot, so it must be the same in every heap.
        let uncapped = views >= victory_shader::emit::HEAP_VIEWS_MAX;
        let (heap, count, extra, base) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0, count),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Some(heap) = uncapped.then(|| make(full.max(views)).ok()).flatten() {
            (heap, count, DYNAMIC, count)
        } else {
            // Every slot a shader can come to exists, the null ones last;
            // the title's slots that fit beside the dynamic ones and pairs.
            let (title, dynamic, _) = capped_layout(views);
            if count > title && remap && !uncapped {
                let overflow = overflow_slots(title);
                tracing::info!(
                    count,
                    direct = title - overflow,
                    overflow,
                    dynamic,
                    "host view heap capped at {views}: the title's slots past the direct ones are remapped into the overflow slots"
                );
            } else if count > title {
                // shortcut: the title's slots past the host's are left out
                // (shaders taking them from the heap itself read the null
                // last slot; tables reaching there are made afresh,
                // `crate::draw`); a title with views there needs them
                // remapped (`HostHeap::mirror` counts them).
                tracing::warn!(count, title, dynamic, "host view heap capped at {views}: the title's slots past {title} are left out");
            }
            // On a host that makes `HEAP_VIEWS_MAX` (one that failed to make
            // this heap) shaders have no title bound: the dynamic slots
            // follow the title's, as there.
            let base = if uncapped { count.min(title) } else { title };
            // With the remap, the overflow range ends at `title` (`base`).
            let direct = if remap && !uncapped && count > title {
                title - overflow_slots(title)
            } else {
                count.min(title)
            };
            match make(views) {
                Ok(heap) => (heap, direct, dynamic, base),
                Err(e) => {
                    tracing::warn!(error = %e, count, views, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // Where shaders translated for `views` read the null raw SRV, the
        // null raw UAV below it (`victory_shader::emit::null_buffer_view`).
        let null = views.saturating_sub(2);
        // Title slots past the direct ones, up to `base` (the remap's).
        let overflow = if remap && title_count > count { base.saturating_sub(count) } else { 0 };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            title_count,
            overflow,
            dynamic: extra,
            base,
            sampler,
            views,
            slots: Mutex::new(Slots::new(title_count, count, overflow)),
            // Slots from `count` to `base` stay null and hold nothing.
            kept: Mutex::new(vec![None; (base + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            pairs: Mutex::new((base + extra, Vec::new())),
            pairs_end: if !sampler && null > base + extra && null < total {
                null - 1
            } else {
                total
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            remap: std::sync::OnceLock::new(),
            viewed: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        if !sampler && null > base + extra && null < total {
            h.raw_view_at(null, None, 0, 0, false);
            h.raw_view_at(null - 1, None, 0, 0, true);
        } else if !sampler {
            tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.base + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.base + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Lowest physical index that can hold a freshly materialized root table.
    pub(crate) fn first_dynamic(&self) -> u32 {
        self.base
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.base) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        // A table made afresh: shaders reading a view's address there (e.g.
        // the BVH builder's base) index the view addresses at this slot.
        self.note_address(index, Some(view));
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None);
        self.set_mark(index, None);
        self.note_address(index, None);
        if index < self.count + self.overflow {
            self.viewed.set(index, None, false);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the integer-format
    /// bits of a typed buffer view ([`view_kind`]).
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, |v| view_stride(v) | view_kind(v)));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(self.views) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `views` (at least).
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(self.views) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `views` (at least).
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// The view-remap table, for root SRV `VIEW_REMAP`
    /// (`victory_shader::emit::Options::view_remap`), made on first use
    /// from the slots taken so far: title slot `i` (below `REMAP_VIEWS`) to
    /// its host slot, the heap's last (null) for one it has none for.
    /// None for sampler heaps.
    pub(crate) fn remap(&self) -> Option<&ID3D12Resource> {
        if self.sampler {
            return None;
        }
        if let Some(made) = self.remap.get() {
            return made.as_ref().map(|(buffer, _)| buffer);
        }
        // Filled while the slots are locked: a slot taken or given back
        // meanwhile either enters this snapshot or writes the mapped table.
        let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let made = self.remap.get_or_init(|| {
            let len = victory_shader::emit::REMAP_VIEWS as usize;
            let (buffer, data) = host()?.upload_buffer(len as u64 * 4)?;
            crate::host::name(&buffer, || "view remap".into());
            let null = self.views - 1;
            // SAFETY: `len` dwords of our fresh mapping.
            let table = unsafe { std::slice::from_raw_parts_mut(data.cast::<u32>(), len) };
            let direct = (self.count as usize).min(len);
            for (i, entry) in table[..direct].iter_mut().enumerate() {
                *entry = i as u32;
            }
            table[direct..].fill(null);
            for (&title, &phys) in &slots.to_phys {
                if let Some(entry) = table.get_mut(title as usize) {
                    *entry = phys;
                }
            }
            Some((buffer, data as usize))
        });
        made.as_ref().map(|(buffer, _)| buffer)
    }

    /// Title slot `title` now lives in host slot `phys` (the heap's last,
    /// null, for none) in the view-remap table, if it is made.
    fn write_remap(&self, title: u32, phys: u32) {
        if title < victory_shader::emit::REMAP_VIEWS
            && let Some(Some((_, data))) = self.remap.get()
        {
            // SAFETY: a dword of the mapped table, which has `REMAP_VIEWS`.
            unsafe { (*data as *mut u32).add(title as usize).write_volatile(phys) };
        }
    }

    /// The title slot host slot `phys` holds: itself among the direct ones,
    /// the one an overflow slot holds; none for a free overflow slot or a
    /// host slot of the heap's own (dynamic, pairs, null).
    pub(crate) fn title_slot(&self, phys: u32) -> Option<u32> {
        if phys < self.count {
            return Some(phys);
        }
        if phys >= self.count + self.overflow {
            return None;
        }
        self.slots.lock().unwrap_or_else(|e| e.into_inner()).title_of(self.count, phys)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            let old = std::mem::replace(slot, resource);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                use windows::core::Interface;
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
            }
            // Title threads rewrite slots while the GPU runs a submission;
            // one of its shaders may hold the old descriptor (or read the
            // slot at an index from wrong data): what it views is freed
            // only once that work has run.
            if let Some((old, _)) = old {
                crate::host::retire(old);
            }
        }
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        if let Some(table) = self.addresses.get() {
            return table.as_ref().map(|t| t.2);
        }
        // Publish while the mirrored slots are still locked. A concurrent
        // replacement either enters this snapshot or updates the mapped table.
        let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let made = self.addresses.get_or_init(|| {
            // The view is a root view, which has no bounds: an entry for
            // every slot a table can lie in, the dynamic ones too (a table
            // made afresh on a host whose heap is smaller than the title's
            // read past the title slots' entries: whatever followed the
            // buffer, or a GPU fault).
            let bytes = u64::from(self.address_slots()) * 16;
            let (buffer, data) = host()?.upload_buffer(bytes)?;
            crate::host::name(&buffer, || "view addresses".into());
            // SAFETY: this is the entire fresh mapping. Unused and reserved
            // slots must read zero even if the allocator reused a buffer.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let data = data as usize;
            // Entries by host slot: a title slot's where it lives.
            for (i, &key) in slots.made.iter().enumerate().filter(|(_, key)| **key != 0) {
                if let Some(phys) = slots.phys_of(self.count, i as u32)
                    && phys < self.address_slots()
                {
                    let view = view_at(key as u32, (key >> 32) as u32);
                    write_address(data, phys as usize, view.as_ref());
                }
            }
            // Dynamic slots are entered as they are made ([`Self::make_at`],
            // [`Self::make_null`], raw views); until then they read zero.
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// The slots the view addresses have an entry for: every slot a shader
    /// can index (indices clamped below `views`, as the stride and mark
    /// tables cover them) and the dynamic ones, which start at `base`.
    fn address_slots(&self) -> u32 {
        self.views.max(self.base + self.dynamic)
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if index < self.address_slots()
            && let Some(Some((_, data, _))) = self.addresses.get()
        {
            write_address(*data, index as usize, view);
        }
    }

    /// The view address entered for slot `index`; none for a slot past the
    /// view addresses' buffer (tests).
    #[cfg(test)]
    fn address_at(&self, index: u32) -> Option<u64> {
        let (buffer, data, _) = self.addresses.get()?.as_ref()?;
        // SAFETY: a live buffer.
        if 16 * u64::from(index) + 16 > unsafe { buffer.GetDesc() }.Width {
            return None;
        }
        // SAFETY: an entry of the mapped view addresses (checked).
        let [lo, hi] = unsafe { ((data + 16 * index as usize) as *const [u32; 2]).read_unaligned() };
        // Word 1 also carries stride in bits 16..29.
        Some(u64::from(hi & 0xffff) << 32 | u64::from(lo))
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
        self.note_address(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// A heap sync of slots `first..last`, whose records are at `start` +
    /// index × `step`: [`HostHeap::mirror`] of each slot whose record
    /// differs from what the slot holds, the rest passed by as `mirror`
    /// would, with one lock of each kind for them all instead of two per slot
    /// (a world frame looks at ~700,000 slots, run perf3: ~75 ms).
    pub(crate) fn sync_slots(&self, first: u32, last: u32, start: u64, step: u64) {
        let at = |i: u32| start + u64::from(i) * step;
        // SAFETY: records in title heap memory the caller found mapped.
        let records: Vec<[u32; 3]> =
            (first..last).map(|i| unsafe { (at(i) as *const [u32; 3]).read_unaligned() }).collect();
        let current = views_current(&records);
        let stale: Vec<u32> = {
            let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            (first..last)
                .zip(records.iter().zip(current))
                .filter(|&(i, (&[_, view, generation], current))| match slots.made.get(i as usize) {
                    Some(&m) if current => m != u64::from(generation) << 32 | u64::from(view),
                    Some(&m) => m != 0,
                    None => true,
                })
                .map(|(i, _)| i)
                .collect()
        };
        for i in stale {
            self.mirror(i, at(i), false);
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.title_count {
            return None;
        }
        if index >= self.count && (self.overflow == 0 || index >= victory_shader::emit::REMAP_VIEWS) {
            // A title slot the host heap has none for (a capped heap,
            // NVIDIA): shaders taking it from the heap itself read the null
            // last slot (`victory_shader::emit::title_views`), tables
            // reaching it are made afresh. That is only right while the
            // title keeps no view there, so those it writes are counted
            // (the title also writes null and stale records over its whole
            // heap, which say nothing).
            // SAFETY: the handle is a descriptor slot in title memory.
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            if tag == VIEW_TAG && view_at(i, generation).is_some() {
                static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let writes = WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if writes.is_power_of_two() {
                    tracing::warn!(
                        index,
                        count = self.count,
                        for_draw,
                        writes,
                        "a title view past the slots the host heap mirrors: shaders indexing the heap itself read nothing there (the title's slots need remapping)"
                    );
                }
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read. A title slot past the
            // direct ones gives its overflow slot back, which shaders then
            // stop reaching (the remap's null); what it viewed is retired
            // with the work that may still read it (`keep`).
            if slots.made[index as usize] != 0 {
                slots.made[index as usize] = 0;
                let phys = if index < self.count {
                    Some(index)
                } else {
                    let freed = slots.free_overflow(self.count, index);
                    self.write_remap(index, self.views - 1);
                    freed
                };
                if let Some(phys) = phys {
                    self.make_null(phys);
                }
            }
            return None;
        };
        // A placed texture's view left null below is made when a draw reads
        // the slot through a table; until then the slot is not made again
        // (heap syncs revisit it at every submission). Also re-mirror if a
        // later unordered or renderable view upgraded the memory's HostTexture.
        let needs_remake = |phys: Option<u32>| {
            // A title slot past the direct ones without an overflow slot
            // yet (all were taken when it was made): try again.
            let Some(phys) = phys else {
                return true;
            };
            if !for_draw || !is_texture(&view) {
                return false;
            }
            let held = self.kept.lock().unwrap_or_else(|e| e.into_inner())[phys as usize]
                .as_ref()
                .map(|(res, _)| windows::core::Interface::as_raw(res));
            let Some(held) = held else {
                return true;
            };
            is_placed_texture(&view)
                && placed_texture(&view).is_some_and(|p| {
                    let (View::Shader(r, _) | View::Unordered(r, ..)) = p else {
                        return false;
                    };
                    !crate::resource::host_copy(r as crate::object::This).is_some_and(|copy| {
                        windows::core::Interface::as_raw(&copy.texture) == held
                            || copy.twin.get().and_then(Option::as_ref).is_some_and(|t| {
                                t.planes
                                    .iter()
                                    .any(|p| windows::core::Interface::as_raw(&p.texture) == held)
                            })
                    })
                })
        };
        if slots.made[index as usize] == key && !needs_remake(slots.phys_of(self.count, index)) {
            return Some(view);
        }
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        // shortcut: views of textures the title placed itself are made
        // when a draw reads them through a table: a host texture (and a
        // staging copy) for each of the ~20,000 the title has views of
        // ran the host out of memory (run b2).
        let later = !for_draw && is_placed_texture(&view);
        if !later {
            // What the view needs (the texture it views, the copy of the
            // memory) is made without the slot lock, which title threads
            // binding and writing views wait on (run evict5).
            drop(slots);
            prepare(&view);
            slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            // The title may have written another record there meanwhile (and
            // another thread made it): that one is the slot's.
            // SAFETY: as above.
            if unsafe { (handle as *const [u32; 3]).read_unaligned() } != [tag, i, generation] {
                drop(slots);
                return self.mirror(index, handle, for_draw);
            }
            if slots.made[index as usize] == key && !needs_remake(slots.phys_of(self.count, index)) {
                return Some(view);
            }
        }
        // The host slot: the title slot itself, or past the direct ones an
        // overflow slot, entered in the view-remap table when taken.
        let index_title = index;
        let index = if index < self.count {
            index
        } else {
            let Some((phys, taken)) = slots.take_overflow(self.count, index) else {
                static FULL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = FULL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        index,
                        overflow = self.overflow,
                        n,
                        "every overflow slot of the host view heap is taken: this title slot reads nothing (VICTORY_VIEW_OVERFLOW)"
                    );
                }
                return None;
            };
            if taken {
                self.write_remap(index, phys);
            }
            phys
        };
        {
            self.set_stride(index, Some(&view));
            self.note_address(index, Some(&view));
            if later {
                // Made once a shader reads it (`crate::shadow::take_reads`).
                crate::shadow::forget_read(index);
            }
            slots.made[index_title as usize] = key;
            let (kept, mark) = if later {
                (None, None)
            } else {
                make_marked(&view, self.cpu(index))
            };
            self.set_mark(index, mark);
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            if !self.sampler {
                self.viewed.set(index, Some(&view), kept.is_some());
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let (key, phys) = {
            let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            (*slots.made.get(index as usize)?, slots.phys_of(self.count, index))
        };
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = phys.is_some_and(|p| {
            self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(p as usize).is_some_and(Option::is_some)
        });
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Slots viewing one of the host textures `textures` (raw pointers),
    /// which are given up: each gets a null view of its kind and holds the
    /// texture no more (retired until the running submission has run), its
    /// record stays noted so heap syncs leave it, and the next draw reading
    /// it through a table or shader reading it through the heap (its read
    /// mark, cleared) makes it again.
    /// The slots are found by a walk of the heap, once per pass that gives
    /// textures up: a map of each texture's slots, kept at every slot the
    /// title rewrote, cost ~25 us a slot (a texture viewed in thousands of
    /// slots had its list searched each time; 62,000 slots a frame, run
    /// evict3).
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        let slots: Vec<u32> = {
            let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            (0..kept.len() as u32)
                .filter(|&i| {
                    kept[i as usize]
                        .as_ref()
                        .is_some_and(|k| k.1.is_none() && textures.contains(&(windows::core::Interface::as_raw(&k.0) as usize)))
                })
                .collect()
        };
        let made = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        for index in slots {
            // The record of the title slot this host slot holds.
            let key = made
                .title_of(self.count, index)
                .filter(|_| index < self.count + self.overflow)
                .and_then(|t| made.made.get(t as usize).copied())
                .unwrap_or(0);
            let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
            match &view {
                Some(v) => null_for(v, self.cpu(index)),
                None => null_view(self.cpu(index)),
            }
            self.keep(index, None);
            if index < self.count + self.overflow {
                self.viewed.set(index, view.as_ref(), false);
                crate::shadow::forget_read(index);
            }
        }
    }

    /// The host textures (raw pointers) slots `slots` view.
    pub(crate) fn textures_at(&self, slots: &[u32]) -> Vec<usize> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        slots
            .iter()
            .filter_map(|&i| kept.get(i as usize)?.as_ref().filter(|k| k.1.is_none()))
            .map(|k| windows::core::Interface::as_raw(&k.0) as usize)
            .collect()
    }

    /// The view of host slot `index` if it is a placed texture's left
    /// without a host view (made when read).
    pub(crate) fn unmade_placed(&self, index: u32) -> Option<View> {
        let title = self.title_slot(index)?;
        let key = *self.slots.lock().unwrap_or_else(|e| e.into_inner()).made.get(title as usize)?;
        let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten()?;
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some_and(Option::is_some);
        (is_placed_texture(&view) && !kept).then_some(view)
    }

    /// A shader read host slot `index` through the heap: the texture it
    /// views was used now.
    pub(crate) fn stamp_read(&self, index: u32) {
        let Some(title) = self.title_slot(index) else {
            return;
        };
        let key = self.slots.lock().unwrap_or_else(|e| e.into_inner()).made.get(title as usize).copied().unwrap_or(0);
        let Some(view) = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten() else {
            return;
        };
        let view = placed_texture(&view).unwrap_or(view);
        if let View::Shader(r, _) = view
            && let Some(t) = crate::resource::made_copy(r as crate::object::This)
        {
            t.stamp();
        }
    }

    /// Whether title slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.slots.lock().unwrap_or_else(|e| e.into_inner()).made.get(index as usize).is_some_and(|&key| key != 0)
    }

    /// Whether title slot `index` holds a host view of a resource where it
    /// lives (tests).
    #[cfg(test)]
    pub(crate) fn kept(&self, index: u32) -> bool {
        let phys = self.slots.lock().unwrap_or_else(|e| e.into_inner()).phys_of(self.count, index);
        phys.is_some_and(|p| self.kept.lock().unwrap_or_else(|e| e.into_inner())[p as usize].is_some())
    }

    /// The host slot title slot `index` lives in (tests).
    #[cfg(test)]
    pub(crate) fn phys(&self, index: u32) -> Option<u32> {
        self.slots.lock().unwrap_or_else(|e| e.into_inner()).phys_of(self.count, index)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// The view-stride table's format-kind bits of a buffer view: a typed view
/// (no structure stride, not raw) of an integer format gets
/// `VIEW_INTEGER`, a signed one also `VIEW_SIGNED`; anything else none.
/// Translated shaders pass 16-bit typed data of such views as bits, not as
/// halves (`victory_shader::emit`, "d16").
fn view_kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let (unordered, words): (bool, &[u32; 10]) = match view {
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => (*unordered, words),
            Err(_) => return 0,
        },
        _ => return 0,
    };
    // Buffer views only (dimension 1); a structured or raw one is no typed view.
    let (format, typed) = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (desc.Format.0 as u32, b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 == 0)
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (desc.Format.0 as u32, b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0)
    };
    match (words[1] == 1 && typed).then(|| integer_format(format)).flatten() {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    }
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// A shared copy must place a buffer SRV on an element boundary. Raw
/// SRVs, including structured SRVs lowered to raw, also start on 16 bytes.
fn srv_alignment(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC, element: u64) -> u64 {
    // SAFETY: called for buffer SRVs.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride == 0 && buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0 {
        return element;
    }
    raw_alignment(element)
}

/// Alignment of a mirrored buffer UAV in a shared host resource.
fn uav_alignment(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC, element: u64) -> u64 {
    // SAFETY: called for buffer UAVs.
    if unsafe { desc.Anonymous.Buffer.Flags }.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        raw_alignment(element)
    } else {
        element
    }
}

fn raw_alignment(element: u64) -> u64 {
    let alignment = u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT);
    // The alignment is a power of two. Divide out its common factor with the stride.
    let common = 1u64 << element.trailing_zeros().min(alignment.trailing_zeros());
    element * (alignment / common)
}

/// Writes slot `index`'s words into the view addresses at `data` (the
/// caller keeps `index` below [`HostHeap::address_slots`]), 16 bytes a
/// slot: a texture view's size fields ([`texture_words`]), a buffer view's
/// buffer descriptor fields ([`BufferView::words`]), zeros for anything
/// else (and for a buffer view whose range the host cannot tell).
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let words = view
        .and_then(|v| texture_words(v).or_else(|| buffer_view(v).map(|b| b.words())))
        .unwrap_or([0; 4]);
    // SAFETY: the mapped view addresses hold 16 bytes per shader-visible
    // and dynamic slot (below `address_slots`).
    unsafe { ((data + 16 * index) as *mut [u32; 4]).write_unaligned(words) };
}

/// A texture view's console image descriptor dwords 0-3 as far as its size
/// goes: width - 1 and height - 1 of the resource's level 0 (dword 1 bits
/// 30-31, dword 2 bits 0-13 and 14-29), dword 3 an identity swizzle, the
/// view's first and last level (bits 12-15, 16-19) and the image type (bits
/// 28-31). `None` for buffers and samplers.
// shortcut: the address, format and tiling fields read 0 (no translated
// shader reads them: the shader refuses such reads where it can tell),
// and the swizzle is the identity whatever the view's component mapping.
fn texture_words(view: &View) -> Option<[u32; 4]> {
    let (resource, desc, unordered) = match view {
        View::Shader(r, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, false),
        View::Unordered(r, _, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, true),
        View::Placed(unordered, resource, desc) if resource[0] != 1 => {
            (*resource, desc[..10].try_into().ok(), *unordered)
        }
        _ => return None,
    };
    Some(image_words(&resource, desc.as_ref(), unordered))
}

/// [`texture_words`] of a texture of resource description `resource` (as
/// words) through a view of description `desc` (none: the resource's own).
fn image_words(resource: &[u32; 14], desc: Option<&[u32; 10]>, unordered: bool) -> [u32; 4] {
    let width = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let height = resource[6];
    let (depth, mips) = (resource[7] & 0xFFFF, resource[7] >> 16);
    let (samples, dimension) = (resource[9], resource[0]);
    // Mip count 0: the whole chain.
    let deepest = if dimension == 4 { depth } else { 1 };
    let chain = 64 - (width.max(u64::from(height)).max(u64::from(deepest)) | 1).leading_zeros();
    let mips = if mips == 0 { chain } else { mips };
    // Image types: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array,
    // 14 2D MSAA, 15 2D MSAA array.
    let (ty, first, count) = match desc {
        // Shader and unordered view dimensions number alike up to 3D (8).
        Some(d) => {
            let ty = [0, 0, 8, 12, 9, 13, 14, 15, 10, 11, 11].get(d[1] as usize).copied().unwrap_or(0);
            match (unordered, d[1]) {
                (_, 6 | 7) => (ty, 0, 1),
                (true, _) => (ty, d[4], 1),
                (false, _) => (ty, d[4], if d[5] == u32::MAX { mips.saturating_sub(d[4]) } else { d[5] }),
            }
        }
        None => {
            let array = depth > 1;
            let ty = match dimension {
                2 if array => 12,
                2 => 8,
                3 if samples > 1 => 14 + u32::from(array),
                3 => 9 + 4 * u32::from(array),
                _ => 10,
            };
            (ty, 0, if unordered { 1 } else { mips })
        }
    };
    let (w, h) = (width.saturating_sub(1) as u32, height.saturating_sub(1));
    let last = first + count.max(1) - 1;
    [
        0,
        (w & 3) << 30,
        (w >> 2 & 0x3FFF) | (h & 0xFFFF) << 14,
        0xFAC | (first & 0xF) << 12 | (last & 0xF) << 16 | ty << 28,
    ]
}

/// A buffer shader or unordered access view decoded from its API
/// description: the title memory it reads and the console buffer
/// descriptor fields the host models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BufferView {
    /// The title address of its first element.
    address: u64,
    /// Its length in bytes.
    bytes: u64,
    /// NUM_RECORDS: bytes for raw views, elements for typed and structured
    /// ones (as the console's driver counts them); a raw range past
    /// `u32::MAX` bytes saturates.
    records: u32,
    /// STRIDE: bytes per record, the element size of typed views (packed
    /// formats and 12- and 16-byte ones included) and the structure stride
    /// of structured ones; 0 for raw views, whose records are bytes.
    stride: u32,
    /// BUF_FMT ([`buffer_format`]).
    format: u8,
}

impl BufferView {
    /// The console buffer descriptor's dwords 0-3.
    /// Modeled: BASE_ADDRESS (dword 0, dword 1 bits 0-15), STRIDE (dword 1
    /// bits 16-29), NUM_RECORDS (dword 2) and FORMAT (dword 3 bits 12-18).
    /// Not modeled, left 0: the rest of dword 1 (CACHE_SWIZZLE,
    /// SWIZZLE_ENABLE) and of dword 3 (DST_SEL_XYZW, INDEX_STRIDE,
    /// ADD_TID_ENABLE, OOB_SELECT, TYPE). No word of the API description
    /// (an SRV's padding, the high half of a UAV's FirstElement) is passed
    /// through. A shader taking dwords 0-1 as an address drops bits 48 and
    /// up, as the hardware does (`victory_shader::emit`, "Title memory").
    fn words(&self) -> [u32; 4] {
        [
            self.address as u32,
            (self.address >> 32) as u32 & 0xFFFF | (self.stride & 0x3FFF) << 16,
            self.records,
            u32::from(self.format & 0x7F) << 12,
        ]
    }
}

/// GFX10 BUF_FMT of a buffer view of API format `format`, structure stride
/// `stride` and raw or not, as AMD PAL's gfx10_3.yaml and
/// gfx9_plus_merged_enum.h number them. Raw and structured views take 32_UINT
/// (PAL's CreateUntypedBufferViewSrds). BUF_FMT_INVALID (0) for formats
/// without a known buffer encoding (sRGB, shared exponent, the console's
/// own) and for descriptions the API does not allow.
fn buffer_format(format: DXGI_FORMAT, stride: u32, raw: bool) -> u8 {
    let format = match (format, stride, raw) {
        (DXGI_FORMAT_R32_TYPELESS, 0, true) | (DXGI_FORMAT_UNKNOWN, 1.., false) => DXGI_FORMAT_R32_UINT,
        (typed, 0, false) => typed,
        _ => return 0,
    };
    match format {
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_A8_UNORM => 1,
        DXGI_FORMAT_R8_SNORM => 2,
        DXGI_FORMAT_R8_UINT => 5,
        DXGI_FORMAT_R8_SINT => 6,
        DXGI_FORMAT_R16_UNORM => 7,
        DXGI_FORMAT_R16_SNORM => 8,
        DXGI_FORMAT_R16_UINT => 11,
        DXGI_FORMAT_R16_SINT => 12,
        DXGI_FORMAT_R16_FLOAT => 13,
        DXGI_FORMAT_R8G8_UNORM => 14,
        DXGI_FORMAT_R8G8_SNORM => 15,
        DXGI_FORMAT_R8G8_UINT => 18,
        DXGI_FORMAT_R8G8_SINT => 19,
        DXGI_FORMAT_R32_UINT => 20,
        DXGI_FORMAT_R32_SINT => 21,
        DXGI_FORMAT_R32_FLOAT => 22,
        DXGI_FORMAT_R16G16_UNORM => 23,
        DXGI_FORMAT_R16G16_SNORM => 24,
        DXGI_FORMAT_R16G16_UINT => 27,
        DXGI_FORMAT_R16G16_SINT => 28,
        DXGI_FORMAT_R16G16_FLOAT => 29,
        // PAL names packed components from the high bits down.
        DXGI_FORMAT_R11G11B10_FLOAT => 36,
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM => 50,
        DXGI_FORMAT_R10G10B10A2_UINT => 54,
        // The component order of BGRA is DST_SEL's, not modeled.
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8X8_UNORM => 56,
        DXGI_FORMAT_R8G8B8A8_SNORM => 57,
        DXGI_FORMAT_R8G8B8A8_UINT => 60,
        DXGI_FORMAT_R8G8B8A8_SINT => 61,
        DXGI_FORMAT_R32G32_UINT => 62,
        DXGI_FORMAT_R32G32_SINT => 63,
        DXGI_FORMAT_R32G32_FLOAT => 64,
        DXGI_FORMAT_R16G16B16A16_UNORM => 65,
        DXGI_FORMAT_R16G16B16A16_SNORM => 66,
        DXGI_FORMAT_R16G16B16A16_UINT => 69,
        DXGI_FORMAT_R16G16B16A16_SINT => 70,
        DXGI_FORMAT_R16G16B16A16_FLOAT => 71,
        DXGI_FORMAT_R32G32B32_UINT => 72,
        DXGI_FORMAT_R32G32B32_SINT => 73,
        DXGI_FORMAT_R32G32B32_FLOAT => 74,
        DXGI_FORMAT_R32G32B32A32_UINT => 75,
        DXGI_FORMAT_R32G32B32A32_SINT => 76,
        DXGI_FORMAT_R32G32B32A32_FLOAT => 77,
        _ => 0,
    }
}

/// A buffer shader or unordered access view decoded ([`BufferView`]);
/// `None` for views of textures, of no resource, of another dimension than
/// buffer, of elements without a size, or whose range overflows the
/// address space.
fn buffer_view(view: &View) -> Option<BufferView> {
    let memory = |r: usize| crate::resource::buffer_bytes(r as crate::object::This).map(|(data, _)| data as u64);
    let (base, unordered, words): (u64, bool, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (memory(*r)?, false, desc),
        View::Unordered(r, _, Some(desc)) => (memory(*r)?, true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            *unordered,
            desc[..10].try_into().ok()?,
        ),
        _ => return None,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return None;
    }
    // The buffer member sits at dword 4 of a shader view's description and
    // at dword 2 of an unordered one's.
    let (first, count, stride, raw, element) = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        let element = if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if raw {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride, raw, element)
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride, raw, element_size(&desc))
    };
    if element == 0 {
        return None;
    }
    // At most 2^32 elements of at most 2^32 bytes: no overflow.
    let bytes = u64::from(count) * element;
    // Callers take `address + bytes` as the range's end.
    let address = first
        .checked_mul(element)
        .and_then(|offset| base.checked_add(offset))
        .filter(|address| address.checked_add(bytes).is_some())?;
    Some(BufferView {
        address,
        bytes,
        // D3D12 counts raw views in dwords, the console in bytes.
        records: if raw { count.saturating_mul(4) } else { count },
        // As the record count: a raw flag wins over a structure stride the
        // API does not allow beside it. Strides past 14 bits (the API allows
        // 2048 at most) keep their low bits, as the field takes them.
        stride: if raw { 0 } else { element as u32 },
        // The title's own format, before the host's substitutions (`srv_desc`).
        format: buffer_format(DXGI_FORMAT(words[0] as i32), stride, raw),
    })
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    buffer_view(view).map(|v| (v.address, v.bytes))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let element = u64::from(view_stride(view));
    let alignment = match view {
        View::Shader(_, Some(d)) => srv_alignment(&srv_desc(d), element),
        View::Placed(false, _, d) => srv_alignment(&srv_desc(d[..10].try_into().ok()?), element),
        View::Unordered(_, _, Some(d)) => uav_alignment(&uav_desc(d), element),
        View::Placed(true, _, d) => uav_alignment(&uav_desc(d[..10].try_into().ok()?), element),
        _ => element,
    };
    let (shadow, _) = crate::shadow::for_view(address, len, alignment)?;
    Some((shadow, address, len))
}

/// The host resource a slot made by [`make`] views, if any, and the view's
/// own copy of title memory ([`crate::shadow::for_view`]): the slot holds
/// it, so that draws reading the view find and refresh that copy.
type Kept = Option<(ID3D12Resource, Option<std::sync::Arc<crate::shadow::Shadow>>)>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    let alignment = srv_alignment(&desc, element);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let start = address + buffer.FirstElement * element;
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, alignment)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    buffer.FirstElement = shadow.offset(start) / element;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    raw_structured(&mut desc)?;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

/// Makes a structured buffer shader view (stride a multiple of 4) the raw
/// view of the same bytes: translated shaders read structured shader views
/// raw, at element Ãƒâ€” stride + offset (`victory_shader::emit`, "Buffer views
/// in the heap"); a raw read of a structured view is addressed in the
/// view's elements on some hosts (NVIDIA). The view-stride table keeps the
/// title's stride ([`view_stride`]). Typed and raw views are left alone.
fn raw_structured(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) -> Option<()> {
    // SAFETY: the caller checked the view's buffer dimension.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let stride = u64::from(buffer.StructureByteStride);
    if stride == 0 {
        return Some(());
    }
    let dwords = u64::from(buffer.NumElements) * stride / 4;
    if !stride.is_multiple_of(4) || dwords > 1 << 27 {
        once!(stride, count = buffer.NumElements, "structured buffer exceeds raw SRV limits");
        return None;
    }
    buffer.FirstElement = buffer.FirstElement.checked_mul(stride / 4)?;
    buffer.NumElements = dwords as u32;
    buffer.StructureByteStride = 0;
    buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    Some(())
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let bytes = u64::from(buffer.NumElements) * element;
    let alignment = uav_alignment(&desc, element);
    let (shadow, own) = crate::shadow::for_view(start, bytes, alignment)?;
    // SAFETY: the buffer descriptor is still the active union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take))
}

/// A texture view `desc` on a host texture that leaves out the title's
/// `drop` largest levels (`crate::resource::drop_levels`): its levels shift
/// down by as many ([`crate::resource::shift_levels`]); its array slices
/// stay. 2D textures, arrays and cubes only, the dimensions such host
/// textures have.
fn shift_view_levels(drop: u32, desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) {
    if drop == 0 {
        return;
    }
    let shift = crate::resource::shift_levels;
    // SAFETY: each arm reads and writes the union member its dimension names.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let t = &mut desc.Anonymous.Texture2D;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let t = &mut desc.Anonymous.Texture2DArray;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let t = &mut desc.Anonymous.TextureCube;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let t = &mut desc.Anonymous.TextureCubeArray;
                shift(drop, &mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
            }
            _ => {}
        }
    }
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    fits_desc(&d, castable, format, unordered)
}

/// Applies the host resource's view-casting rules, including explicitly
/// supported casts for placed resources which share the same storage.
pub(crate) fn fits_desc(d: &D3D12_RESOURCE_DESC, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let is_typeless = |f: u32| family(f) == f && family(f + 1) == f;
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // A typed unordered view in a format the host cannot store to
    // (B5G5R5A1_UNORM on NVIDIA) removes the device when made, in the
    // texture's own format too: the slot stays null, its writes dropped.
    // The format checked is the one the view is made in: a typed
    // texture's own unless the requested one is castable (below).
    let typed = if view == 0 || (!typeless && !castable.contains(&view)) { own } else { view };
    if unordered && !(view == 0 && typeless) && !typed_store(typed) {
        once!(view = typed, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) || is_typeless(view) {
        once!(unordered, view, own, "view the host does not allow");
        return false;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let effective = if typeless || cast { view } else { own };
    if unordered && !typed_store(effective) {
        once!(
            view = effective,
            own,
            "an unordered view in a format the host cannot store"
        );
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if let Some(desc) = desc.as_mut()
                    && desc.Format.0 == 0
                    && let Some(format) = crate::resource::view_format(resource)
                {
                    desc.Format = DXGI_FORMAT(format as i32);
                }
                if let Some(desc) = desc.as_mut() {
                    shift_view_levels(crate::resource::dropped(resource), desc);
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if let Some(desc) = desc.as_mut()
                && desc.Format.0 == 0
                && let Some(format) = crate::resource::view_format(resource)
            {
                desc.Format = DXGI_FORMAT(format as i32);
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// Makes what a host view of `view` needs, if it is not made yet: the
/// texture it views (made on first use) or the host copy of the buffer
/// memory it views. The slot's view is then quick to make.
fn prepare(view: &View) {
    #[cfg(test)]
    tests::while_preparing();
    match view {
        View::Placed(..) if is_placed_texture(view) => {
            let _ = placed_texture(view);
        }
        View::Shader(r, _) | View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some() => {
            let _ = crate::resource::host_resource(*r as crate::object::This);
        }
        _ => {
            let _ = view_copy(view);
        }
    }
}

/// Whether `view` is a shader view of a texture (one placed in memory, or a
/// resource object's): such a slot left null is made when read.
fn is_texture(view: &View) -> bool {
    is_placed_texture(view)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// A view of a texture the title placed in memory itself, as a view of the
/// resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    let view_desc = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture as usize, 0, view_desc)
    } else {
        View::Shader(texture as usize, view_desc)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shader views of a host texture without the title's two largest
    /// levels, of every dimension such a texture has, with partial windows:
    /// a window past the levels left out moves down by two; one that
    /// starts in them keeps the levels of it the host has; one wholly in
    /// them shows the host's largest level; all levels stay all; the clamp
    /// moves down and stops at zero; array slices and cube counts stay.
    #[test]
    fn partial_level_windows_of_every_dimension_shift_and_keep_their_slices() {
        let views = [
            D3D12_SRV_DIMENSION_TEXTURE2D,
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            D3D12_SRV_DIMENSION_TEXTURECUBE,
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
        ];
        // (first, levels, clamp) asked, then as made, two levels left out.
        let windows = [
            ((3, 2, 0.0), (1, 2, 0.0)),
            ((1, 3, 0.0), (0, 2, 0.0)),
            ((0, 2, 0.0), (0, 1, 0.0)),
            ((0, u32::MAX, 0.0), (0, u32::MAX, 0.0)),
            ((4, u32::MAX, 5.5), (2, u32::MAX, 3.5)),
            ((2, 1, 1.0), (0, 1, 0.0)),
        ];
        for dimension in views {
            for ((first, levels, clamp), want) in windows {
                let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC { ViewDimension: dimension, ..Default::default() };
                match dimension {
                    D3D12_SRV_DIMENSION_TEXTURE2D => {
                        desc.Anonymous.Texture2D =
                            D3D12_TEX2D_SRV { MostDetailedMip: first, MipLevels: levels, PlaneSlice: 0, ResourceMinLODClamp: clamp }
                    }
                    D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                        desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                            MostDetailedMip: first,
                            MipLevels: levels,
                            FirstArraySlice: 5,
                            ArraySize: 3,
                            PlaneSlice: 0,
                            ResourceMinLODClamp: clamp,
                        }
                    }
                    D3D12_SRV_DIMENSION_TEXTURECUBE => {
                        desc.Anonymous.TextureCube =
                            D3D12_TEXCUBE_SRV { MostDetailedMip: first, MipLevels: levels, ResourceMinLODClamp: clamp }
                    }
                    _ => {
                        desc.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV {
                            MostDetailedMip: first,
                            MipLevels: levels,
                            First2DArrayFace: 6,
                            NumCubes: 2,
                            ResourceMinLODClamp: clamp,
                        }
                    }
                }
                shift_view_levels(2, &mut desc);
                // SAFETY: the union member of the dimension set above.
                let (got, slices) = unsafe {
                    match dimension {
                        D3D12_SRV_DIMENSION_TEXTURE2D => {
                            let t = desc.Anonymous.Texture2D;
                            ((t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp), (0, 0))
                        }
                        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                            let t = desc.Anonymous.Texture2DArray;
                            ((t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp), (t.FirstArraySlice, t.ArraySize))
                        }
                        D3D12_SRV_DIMENSION_TEXTURECUBE => {
                            let t = desc.Anonymous.TextureCube;
                            ((t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp), (0, 0))
                        }
                        _ => {
                            let t = desc.Anonymous.TextureCubeArray;
                            ((t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp), (t.First2DArrayFace, t.NumCubes))
                        }
                    }
                };
                assert_eq!(got, want, "{dimension:?} window {first} {levels}");
                let slices_asked = match dimension {
                    D3D12_SRV_DIMENSION_TEXTURE2DARRAY => (5, 3),
                    D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => (6, 2),
                    _ => (0, 0),
                };
                assert_eq!(slices, slices_asked, "{dimension:?}: slices stay");
            }
        }
        // Nothing left out: nothing moves, and other dimensions are untouched.
        let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC { ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, ..Default::default() };
        desc.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 3, MipLevels: 2, PlaneSlice: 0, ResourceMinLODClamp: 1.0 };
        shift_view_levels(0, &mut desc);
        // SAFETY: the 2D member, set above.
        assert_eq!(unsafe { desc.Anonymous.Texture2D.MostDetailedMip }, 3);
        let mut volume = D3D12_SHADER_RESOURCE_VIEW_DESC { ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D, ..Default::default() };
        volume.Anonymous.Texture3D = D3D12_TEX3D_SRV { MostDetailedMip: 3, MipLevels: 2, ResourceMinLODClamp: 0.0 };
        shift_view_levels(2, &mut volume);
        // SAFETY: the 3D member, set above.
        assert_eq!(unsafe { volume.Anonymous.Texture3D.MostDetailedMip }, 3);
    }

    /// Typed buffer views of integer formats carry their kind in the
    /// view-stride table; float, normalized, raw and structured ones none.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(57, 0, 0)), VIEW_INTEGER, "R16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(64, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R8_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_kind(&srv), VIEW_INTEGER, "R8_UINT");
        assert_eq!(view_stride(&uav(12, 0, 0)) | view_kind(&uav(12, 0, 0)), 8 | VIEW_INTEGER);
    }

    /// A placed texture's view the title writes into a heap is left null
    /// and noted once: heap syncs (every submission) pass it by, a draw
    /// reading it through a table makes it.
    #[test]
    fn a_placed_texture_slot_left_null_is_made_once_a_draw_reads_it() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        for _ in 0..2 {
            assert!(host.mirror(0, title_heap.start, false).is_some());
            assert!(host.made(0), "the record is noted");
            assert!(!host.kept(0), "no host texture for it yet");
        }
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
    }

    /// A slot viewing a texture given up (`crate::resource::evict`) holds it
    /// no more: heap syncs pass it by, a draw reading it through a table
    /// makes it again.
    #[test]
    fn a_slot_viewing_a_texture_given_up_is_made_again_when_read() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!host.kept(0), "the slot holds it no more");
        assert!(host.made(0), "its record stays noted");
        assert!(host.mirror(0, title_heap.start, false).is_some());
        assert!(!host.kept(0), "heap syncs pass it by");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made again for the draw");
        assert!(crate::resource::made_copy(texture).is_some(), "with a host texture again");
    }

    /// Work to run while a view's texture is prepared, once (tests).
    pub(crate) static WHILE_PREPARING: std::sync::Mutex<Option<crate::descriptor::tests::Work>> = std::sync::Mutex::new(None);

    pub(crate) fn while_preparing() {
        let work = WHILE_PREPARING.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(work) = work {
            work();
        }
    }

    /// A record the title writes into a slot while another thread prepares
    /// the slot's old record is the one the slot ends up viewing.
    #[test]
    fn a_slot_rewritten_while_its_view_is_prepared_views_the_new_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let s = 960u32;
        let at = title_heap.start + u64::from(s) * 32;
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let view_of = |address: u64| {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            view
        };
        let (first, second) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + (1 << 16));
        crate::descriptor::create_placed_srv(null, &words, &view_of(first), at);
        let host = title_heap.host().expect("a host view heap");
        let (done, waited) = std::sync::mpsc::channel();
        let second_view = view_of(second);
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            // The title writes the other record into the slot meanwhile.
            std::thread::spawn(move || {
                crate::descriptor::create_placed_srv(std::ptr::null(), &words, &second_view, at);
                done.send(()).unwrap();
            });
            assert!(waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(), "the write waits");
        }));
        // A draw reads the slot: its first record is prepared.
        assert!(host.mirror(s, at, true).is_some());
        let second_texture = crate::resource::texture_at(second, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(second_texture).expect("made").texture) as usize;
        assert_eq!(host.textures_at(&[s]), [raw], "the slot views the record written last");
    }

    /// What a view needs is made without the slot lock: another thread
    /// binding a view of the same heap meanwhile is not held up.
    #[test]
    fn views_bound_while_another_slot_gets_its_texture_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (first, second) = (950u32, 951u32);
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        for (i, at) in [(first, 0u64), (second, 1 << 16)] {
            let address = memory.as_ptr() as u64 + at;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, slot(i));
        }
        let host = title_heap.host().expect("a host view heap");
        let heap_at = title_heap.start;
        let heap = heap as usize;
        let (done, waited) = std::sync::mpsc::channel();
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            std::thread::spawn(move || {
                let h = crate::descriptor::heap(heap as crate::object::This);
                h.host().unwrap().mirror(second, heap_at + u64::from(second) * 32, true);
                done.send(()).unwrap();
            });
            assert!(
                waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                "a view bound while another slot's texture is made waits"
            );
        }));
        assert!(host.mirror(first, slot(first), true).is_some());
        assert!(WHILE_PREPARING.lock().unwrap().is_none(), "the prepare ran");
        assert!(host.kept(first) && host.kept(second), "both made");
    }

    /// The textures of slots shaders read are made without the heaps' lock:
    /// a title thread writing a view meanwhile is not held up by the make.
    #[test]
    fn views_written_while_read_slots_get_their_textures_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (read, written) = (900u32, 901u32);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, slot(read));
        let host = title_heap.host().expect("a host view heap");
        assert!(!host.kept(read), "left null until read");
        let (done, waited) = std::sync::mpsc::channel();
        let at = slot(written);
        *crate::descriptor::tests::WHILE_MAKING.lock().unwrap() = Some((
            read,
            Box::new(move || {
                std::thread::spawn(move || {
                    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, at);
                    done.send(()).unwrap();
                });
                assert!(
                    waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                    "a view written while a read slot's texture is made waits"
                );
            }),
        ));
        crate::descriptor::make_read(&[read]);
        assert!(crate::descriptor::tests::WHILE_MAKING.lock().unwrap().is_none(), "the make ran");
        assert!(host.kept(read), "made for the read");
    }

    /// A heap sync makes the slots whose records the title changed as bytes
    /// (a copy of another slot's record, a record cleared) and passes the
    /// others by.
    #[test]
    fn heap_syncs_make_the_slots_whose_records_changed() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // A view of no memory: a view of the test's own heap memory would get
        // a host copy whose 4 MiB window the guards may cover (the process
        // heap guarded read-only hung the suite).
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start);
        let host = title_heap.host().expect("a host view heap");
        // SAFETY: slots 0 and 1 of the heap's memory, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32) };
        assert!(!host.made(1), "copied as bytes: not seen yet");
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && host.made(1) && !host.made(2));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes((start + 32) as *mut u8, 0, 32) };
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && !host.made(1), "a cleared record leaves its slot");
    }

    #[test]
    fn structured_srv_retains_its_byte_range_in_a_raw_descriptor() {
        let view = |first, count, stride| D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV {
                FirstElement: first, NumElements: count, StructureByteStride: stride,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            } },
            ..Default::default()
        };
        let mut converted = view(4, 7, 12);
        raw_structured(&mut converted).unwrap();
        assert_eq!(converted.Format, DXGI_FORMAT_R32_TYPELESS);
        // SAFETY: this is the buffer descriptor constructed above.
        let range = unsafe { converted.Anonymous.Buffer };
        assert_eq!((range.FirstElement, range.NumElements, range.StructureByteStride), (12, 21, 0));
        assert_eq!(range.Flags, D3D12_BUFFER_SRV_FLAG_RAW);
        assert!(raw_structured(&mut view(0, 1, 6)).is_none());
        assert!(raw_structured(&mut view(0, 1 << 27, 8)).is_none());
        assert!(raw_structured(&mut view(u64::MAX, 1, 12)).is_none());
    }

    /// GFX10 BUF_FMT codes: packed formats named from the high bits down,
    /// raw and structured views as 32_UINT, the rest invalid.
    #[test]
    fn buffer_formats_use_rdna2_packing_and_untyped_view_kinds() {
        let cases = [
            (DXGI_FORMAT_R11G11B10_FLOAT, 0, false, 36),
            (DXGI_FORMAT_R10G10B10A2_UNORM, 0, false, 50),
            (DXGI_FORMAT_R10G10B10A2_UINT, 0, false, 54),
            (DXGI_FORMAT_R32G32B32_FLOAT, 0, false, 74),
            (DXGI_FORMAT_R32_TYPELESS, 0, true, 20),
            (DXGI_FORMAT_UNKNOWN, 12, false, 20),
            (DXGI_FORMAT_R32_TYPELESS, 0, false, 0),
            (DXGI_FORMAT_UNKNOWN, 0, false, 0),
            (DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0, false, 0),
            (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 0, false, 0),
        ];
        for (format, stride, raw, expected) in cases {
            assert_eq!(buffer_format(format, stride, raw), expected, "{format:?}");
        }
    }

    /// Dword 3 holds the view's FORMAT alone: an SRV's padding or the high
    /// half of a UAV's FirstElement (API dword 3) is not passed through.
    #[test]
    fn descriptor_word_3_uses_the_format_instead_of_api_padding_or_first_element() {
        let mut resource = [0; 14];
        resource[0] = 1;
        for (unordered, format, expected) in [
            (false, DXGI_FORMAT_R8G8_SNORM, 15),
            (true, DXGI_FORMAT_R16G16B16A16_FLOAT, 71),
            (false, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0),
        ] {
            let mut words = [0; 16];
            words[0] = format.0 as u32;
            words[1] = 1; // buffer dimension
            words[3] = 0x1234;
            words[if unordered { 4 } else { 6 }] = 3;
            let mut entry = [u32::MAX; 4];
            let view = View::Placed(unordered, resource, words);
            write_address(entry.as_mut_ptr() as usize, 0, Some(&view));
            assert_eq!(entry[2], 3);
            assert_eq!(entry[3], expected << 12);
        }
    }

    /// A buffer view the host cannot size (not of buffer dimension, or
    /// typed without a format) clears its entry.
    #[test]
    fn views_without_a_buffer_element_clear_the_table_entry() {
        let mut resource = [0; 14];
        resource[0] = 1;
        for (dimension, format) in [
            (D3D12_SRV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_UINT),
            (D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_UNKNOWN),
        ] {
            let mut words = [0; 16];
            words[0] = format.0 as u32;
            words[1] = dimension.0 as u32;
            words[6] = 37;
            let view = View::Placed(false, resource, words);
            let mut entry = [u32::MAX; 4];
            write_address(entry.as_mut_ptr() as usize, 0, Some(&view));
            assert_eq!(entry, [0; 4]);
            assert_eq!(buffer_range(&view), None);
        }
    }

    /// A raw view of 4 GiB keeps its byte range; its record count saturates.
    #[test]
    fn raw_record_bound_keeps_the_byte_range_beyond_u32() {
        let mut resource = [0; 14];
        resource[0] = 1;
        let mut words = [0; 16];
        words[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        words[6] = 1 << 30; // 4 GiB in dwords
        words[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        let view = buffer_view(&View::Placed(false, resource, words)).unwrap();
        assert_eq!(view.bytes, 1u64 << 32);
        assert_eq!(view.records, u32::MAX);
    }

    /// Dword 1 holds address bits 32-47 and, beside them, STRIDE (bits
    /// 16-29): the element size of typed views, the structure stride of
    /// structured ones, 0 for raw ones. Neither field changes the other, nor
    /// the byte range and record count.
    #[test]
    fn descriptor_word_1_holds_the_high_address_and_the_stride() {
        let mut resource = [0; 14];
        resource[0] = 1;
        let base = 0x7654_3210_0000u64;
        let raw = (D3D12_BUFFER_SRV_FLAG_RAW.0 | D3D12_BUFFER_UAV_FLAG_RAW.0) as u32;
        // Unordered, format, structure stride, flags; stride, record count
        // and bytes expected for 6 elements from element 1.
        for (unordered, format, structure, flags, stride, records, bytes) in [
            (false, DXGI_FORMAT_R8_UINT, 0, 0, 1, 6, 6),
            (false, DXGI_FORMAT_R11G11B10_FLOAT, 0, 0, 4, 6, 24),
            (true, DXGI_FORMAT_R10G10B10A2_UINT, 0, 0, 4, 6, 24),
            (true, DXGI_FORMAT_R16G16B16A16_FLOAT, 0, 0, 8, 6, 48),
            (false, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, 12, 6, 72),
            (true, DXGI_FORMAT_R32G32B32A32_UINT, 0, 0, 16, 6, 96),
            (false, DXGI_FORMAT_UNKNOWN, 24, 0, 24, 6, 144),
            (true, DXGI_FORMAT_UNKNOWN, 2048, 0, 2048, 6, 12288),
            (false, DXGI_FORMAT_R32_TYPELESS, 0, raw, 0, 24, 24),
            (true, DXGI_FORMAT_R32_TYPELESS, 0, raw, 0, 24, 24),
            // Not allowed by the API, kept as before: a raw flag beside a
            // structure stride counts bytes, a stride past 14 bits keeps
            // its low bits and its byte range.
            (false, DXGI_FORMAT_UNKNOWN, 12, raw, 0, 24, 72),
            (true, DXGI_FORMAT_UNKNOWN, 0x4004, 0, 4, 6, 6 * 0x4004),
        ] {
            let mut words = [0; 16];
            words[0] = format.0 as u32;
            words[1] = 1; // buffer dimension
            let (first, count, stride_at) = if unordered { (2, 4, 5) } else { (4, 6, 7) };
            (words[first], words[count], words[stride_at], words[8]) = (1, 6, structure, flags);
            (words[10], words[11]) = (base as u32, (base >> 32) as u32);
            let view = buffer_view(&View::Placed(unordered, resource, words)).unwrap();
            let element = if structure > 0 { u64::from(structure) } else { bytes / 6 };
            let element = if flags != 0 && structure == 0 { 4 } else { element };
            assert_eq!((view.address, view.bytes, view.records), (base + element, bytes, records), "{format:?}");
            let word = view.words()[1];
            assert_eq!(word & 0xFFFF, 0x7654, "{format:?}: address bits 32-47");
            assert_eq!(word >> 16, stride, "{format:?}: STRIDE, bits 30-31 clear");
        }
    }

    /// Ranges past the address space are refused rather than wrapped; the
    /// same views placed low keep their range.
    #[test]
    fn buffer_ranges_past_the_address_space_are_refused() {
        let mut resource = [0; 14];
        resource[0] = 1;
        let view = |unordered: bool, base: u64, first: u32| {
            let mut words = [0; 16];
            words[0] = DXGI_FORMAT_R32_FLOAT.0 as u32;
            words[1] = 1; // buffer dimension
            let (first_at, count_at) = if unordered { (2, 4) } else { (4, 6) };
            (words[first_at], words[count_at]) = (first, 4);
            (words[10], words[11]) = (base as u32, (base >> 32) as u32);
            View::Placed(unordered, resource, words)
        };
        for unordered in [false, true] {
            assert_eq!(buffer_range(&view(unordered, 0x1000, 2)), Some((0x1008, 16)));
            assert_eq!(buffer_range(&view(unordered, u64::MAX - 7, 2)), None, "first element past the end");
            assert_eq!(buffer_range(&view(unordered, u64::MAX - 15, 0)), None, "last element past the end");
        }
    }

    /// One slot of the view addresses rewritten from a buffer view to a
    /// texture view and back: it holds the words of the view written last
    /// each time (image size fields, then record count and format), and the
    /// slots beside it keep theirs.
    #[test]
    fn a_buffer_slot_turned_texture_and_back_keeps_its_neighbours_words() {
        use windows::core::{IUnknown, Interface};
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        // SAFETY: create_heap handed out one reference.
        let _heap_owner = unsafe { IUnknown::from_raw(heap.cast_mut().cast()) };
        let title_heap = crate::descriptor::heap(heap);
        let slots = title_heap.host().expect("a host view heap");
        // The table made first: every write below rewrites its entries.
        slots.view_addresses().expect("the view addresses");
        let Some(Some((_, data, _))) = slots.addresses.get() else {
            unreachable!("made above");
        };
        // SAFETY: the mapped view addresses hold 16 bytes per title slot.
        let entry = |i: usize| unsafe { ((data + 16 * i) as *const [u32; 4]).read_unaligned() };
        let input = crate::resource::test_buffer(256);
        // SAFETY: test_buffer handed out one reference.
        let _input_owner = unsafe { IUnknown::from_raw(input.cast_mut().cast()) };
        let base = crate::resource::buffer_bytes(input).unwrap().0 as u64;
        let at = |i: u64| title_heap.start + 32 * i;
        // Format, dimension (buffer), mapping, padding (FORMAT 127 if it
        // leaked), first element, count, stride, flags.
        let srv = |format: DXGI_FORMAT, first, count, flags| {
            [format.0 as u32, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 0x0007_F000, first, 0, count, 0, flags, 0]
        };
        // Dword 1: address bits 32-47, STRIDE at bits 16-29.
        let buffer = |address: u64, stride: u32, records: u32, format: u32| {
            [address as u32, (address >> 32) as u32 & 0xFFFF | stride << 16, records, format << 12]
        };
        // Slot 1, the guard beside the slot rewritten: 3 R32_FLOAT from
        // element 2. Slots 0 and 3 stay empty.
        crate::descriptor::create_srv(null, input, &srv(DXGI_FORMAT_R32_FLOAT, 2, 3, 0), at(1));
        let guard = buffer(base + 8, 4, 3, 22);
        // Slot 2: 6 RGBA8 elements from element 4.
        crate::descriptor::create_srv(null, input, &srv(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 6, 0), at(2));
        let table = || [entry(0), entry(1), entry(2), entry(3)];
        assert_eq!(table(), [[0; 4], guard, buffer(base + 16, 4, 6, 56), [0; 4]], "a typed buffer view");
        // Slot 2: a 2D view of level 0 of a 4x1 RGBA8 texture placed in
        // memory of its own (left null until a draw reads it; its words
        // are written all the same).
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let texture: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &texture, &view, at(2));
        // Width - 1 = 3 (dword 1 bits 30-31), height - 1 = 0, levels 0-0,
        // identity swizzle, a 2D image (type 9); no address, no format.
        let image = [0, 3 << 30, 0, 0xFAC | 9 << 28];
        assert_eq!(table(), [[0; 4], guard, image, [0; 4]], "a texture view");
        // Slot 2: a raw view of 5 dwords from dword 2, counted in bytes.
        let raw = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        crate::descriptor::create_srv(null, input, &srv(DXGI_FORMAT_R32_TYPELESS, 2, 5, raw), at(2));
        assert_eq!(table(), [[0; 4], guard, buffer(base + 8, 0, 20, 20), [0; 4]], "a raw buffer view again");
        drop(memory);
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A table made afresh lies in dynamic slots, and a shader reading a
    /// view's address (e.g. the BVH builder's base) indexes the view
    /// addresses there: they have an entry for each dynamic slot, entered
    /// whenever the slot is made (a view, null, or a raw view of ours).
    /// Giving a slot back leaves its entry: every table made afresh makes
    /// all its slots. (Hosts whose view heaps are smaller than the title's,
    /// NVIDIA, make such tables; the shader read past the buffer.)
    ///
    /// Any heap size the host takes will do: a title heap of 4 slots leaves
    /// dynamic slots past it either way ([`DYNAMIC`] or [`MIN_DYNAMIC`]).
    #[test]
    fn view_addresses_cover_tables_made_afresh() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // The fixed mirror can reserve slots between the title's last
        // descriptor and the first dynamic descriptor on this host.
        let end = heap.base + heap.dynamic;
        assert!(heap.view_addresses().is_some());
        let first = heap.dynamic(2).unwrap();
        assert!(first >= heap.base && first + 1 < end);
        // A buffer the title placed at 0xd_6013_e000: 16 elements of 4 bytes.
        let address = 0xd_6013_e000u64;
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 64);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 4);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let view = View::Placed(false, resource, desc);
        assert_eq!(buffer_range(&view), Some((address, 64)));
        assert_eq!(heap.address_at(first), Some(0), "not made yet");
        heap.make_at(first, &view);
        assert_eq!(heap.address_at(first), Some(address));
        assert_eq!(heap.address_at(first + 1), Some(0), "not made yet");
        // The last dynamic slot has its entry too.
        heap.make_at(end - 1, &view);
        assert_eq!(heap.address_at(end - 1), Some(address));
        // A slot given back keeps its entry until it is made again; made
        // null, it has its entry cleared.
        heap.release(first, 2);
        assert_eq!(
            heap.address_at(first),
            Some(address),
            "given back, not made"
        );
        heap.make_null(first);
        assert_eq!(heap.address_at(first), Some(0));
        assert_eq!(heap.address_at(end - 1), Some(address), "still made");
        // Raw views of ours (root views, host copies' view pairs) in dynamic
        // slots clear what an earlier table entered there.
        heap.make_at(first, &view);
        heap.raw_view_at(first, None, 0, 0, false);
        assert_eq!(heap.address_at(first), Some(0), "a root view");
        heap.make_at(first, &view);
        heap.make_at(first + 1, &view);
        heap.raw_pair_at(first, None, 0);
        assert_eq!(heap.address_at(first), Some(0), "a view pair");
        assert_eq!(heap.address_at(first + 1), Some(0), "a view pair");
        // Every slot a shader can index has an entry (host copies' view
        // pairs and the null slots too, below `views`); nothing is written
        // past the last: a mark there (the buffer's allocation reaches past
        // it) stays.
        assert!(heap.address_slots() >= end);
        let end = heap.address_slots();
        assert_eq!(heap.address_at(end), None);
        let host = host().unwrap();
        let (buffer, data, _) = heap.addresses.get().unwrap().as_ref().unwrap();
        // SAFETY: plain queries of a live buffer.
        let (desc, size) = unsafe {
            let desc = buffer.GetDesc();
            let info = host.device.GetResourceAllocationInfo(0, &[desc]);
            (desc, info.SizeInBytes)
        };
        assert_eq!(desc.Width, 16 * u64::from(end));
        // A table of every shader-visible slot may end on an allocation
        // boundary: the mark needs room past it.
        if size >= desc.Width + 8 {
            let past = (data + 16 * end as usize) as *mut u64;
            let mark = 0x5a5a_5a5a_5a5a_5a5a;
            // SAFETY: mapped memory of the buffer's allocation (checked).
            unsafe { past.write_unaligned(mark) };
            heap.note_address(end, Some(&view));
            // SAFETY: as above.
            assert_eq!(unsafe { past.read_unaligned() }, mark);
        } else {
            // Nothing to observe; it must still not write (or fault).
            heap.note_address(end, Some(&view));
        }
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A view heap as large as shaders index: where the host caps it
    /// (NVIDIA), the title's slots give way to the dynamic slots and the
    /// null buffer views, which stay below the clamped index.
    #[test]
    fn a_capped_view_heap_keeps_the_null_buffer_views() {
        let views = victory_shader::emit::heap_views();
        let Some(heap) = HostHeap::new(false, views) else {
            return; // no host GPU
        };
        // SAFETY: a plain query of a live heap.
        let total = unsafe { heap.heap.GetDesc().NumDescriptors };
        assert!(total >= views);
        if heap.count == views {
            return; // the host took every title slot
        }
        assert_eq!(total, views, "capped at what shaders index");
        let null = victory_shader::emit::null_buffer_view();
        // The null UAV (at `null - 1`) is the first slot past the dynamic ones.
        assert!(heap.base + heap.dynamic < null, "room for the null buffer views");
        assert_eq!(heap.pairs_end, null - 1, "host copies' pairs stop below them");
        assert_eq!(null + 2, views, "the clamped index past them");
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `heap_views()`, which is what this device makes, and
        // `HEAP_SAMPLERS`), with the full dynamic range also where the
        // device caps its heaps (NVIDIA).
        // SAFETY: plain queries of live heaps.
        unsafe {
            let views = victory_shader::emit::heap_views();
            let limit = victory_shader::emit::host::Host::from_device(host().unwrap().device.clone()).view_heap_limit();
            assert_eq!(views, limit, "translated for what the device makes");
            assert!(heap.heap.GetDesc().NumDescriptors >= views);
            assert_eq!(heap.dynamic, DYNAMIC);
            assert_eq!(heap.views, views);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.base + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            // Released on a thread of its own.
            let since = std::time::Instant::now();
            while references(&buffer) > 1 && since.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (heap.base, heap.base + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(heap.base));
        assert_eq!(heap.dynamic(40), Some(heap.base + heap.dynamic - 40));
    }

    /// A heap on a host that makes none larger (as NVIDIA's 1,000,000
    /// slots, here 4,096) has every slot a shader translated for it can
    /// index: the title's slots that fit, the dynamic range, slots for view
    /// pairs, then the null buffer views and the clamp's slot.
    #[test]
    fn a_capped_heap_keeps_dynamic_slots_pairs_and_null_views() {
        use victory_shader::emit::HEAP_VIEWS_PORTABLE;
        // What a title heap larger than the host's is cut to: the full
        // dynamic range and dedicated pairs on a 1,000,000-slot host.
        assert_eq!(capped_layout(HEAP_VIEWS_PORTABLE), (721_469, DYNAMIC, CAPPED_PAIRS));
        assert_eq!(capped_layout(4096), (1537, 2046, 510));
        assert_eq!(capped_layout(2), (0, 0, 0));
        let Some(host) = host() else {
            return; // no host GPU
        };
        let views = 4096;
        let heap = HostHeap::sized(false, 10_000, views).unwrap();
        // SAFETY: a plain query of a live heap.
        assert_eq!(unsafe { heap.heap.GetDesc() }.NumDescriptors, views);
        assert_eq!((heap.count(), heap.dynamic), (1537, 2046));
        // The whole dynamic range in one run, then nothing until released.
        assert_eq!(heap.dynamic(2046), Some(1537));
        assert_eq!(heap.dynamic(1), None);
        // Pairs past it, below the null views; then from the dynamic slots.
        let pairs: Vec<u32> = (0..255).map(|_| heap.pair().unwrap()).collect();
        assert_eq!((pairs[0], pairs[254]), (1537 + 2046, views - 5));
        assert_eq!(heap.pair(), None);
        heap.release(1537, 2046);
        assert_eq!(heap.pair(), Some(1537));
        // The null buffer views (raw: 4-byte elements) are where shaders
        // translated for such a heap read them, the last slot left null.
        let (_, strides) = heap.strides.as_ref().unwrap();
        // SAFETY: dwords of the mapped table, which covers these indices.
        let stride = |index: u32| unsafe { (*strides as *const u32).add(index as usize).read_volatile() };
        assert_eq!([stride(views - 3), stride(views - 2), stride(views - 1)], [4, 4, 0]);
        // A smaller title heap keeps its slots and shares the same host-only boundary.
        let small = HostHeap::sized(false, 4, views).unwrap();
        assert_eq!((small.count(), small.dynamic, small.pair()), (4, 2046, Some(1537 + 2046)));
        // A title heap that fits with the full dynamic range on a capped
        // host keeps every slot, in a heap of exactly the capped size.
        let fits = HostHeap::sized(false, 10_000, HEAP_VIEWS_PORTABLE).unwrap();
        // SAFETY: a plain query of a live heap.
        assert_eq!(unsafe { fits.heap.GetDesc() }.NumDescriptors, HEAP_VIEWS_PORTABLE);
        assert_eq!((fits.count(), fits.dynamic, fits.pair()), (10_000, DYNAMIC, Some(721_469 + DYNAMIC)));
        assert_eq!(fits.pairs_end, HEAP_VIEWS_PORTABLE - 3);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (heap.base, heap.base + 1, heap.base + 3, heap.base + 5));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// A texture view's size words are the console image descriptor's
    /// fields: level 0's size of the resource, the view's levels, its type.
    #[test]
    fn a_texture_view_gives_its_image_descriptor_size_fields() {
        // A 1000x600 2D texture of 10 levels, then an array of 6 with the
        // whole chain (mip count 0, 1024 wide: 11 levels).
        let mut texture = [0u32; 14];
        (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 1000, 600, 1 | 10 << 16, 1);
        let mut array = texture;
        (array[4], array[7]) = (1024, 6);
        let mut view = [0u32; 10];
        // Levels 2-4 of the texture: width - 1 = 999 (0b11_1110_0111).
        (view[1], view[4], view[5]) = (4, 2, 3);
        let words = image_words(&texture, Some(&view), false);
        assert_eq!(words, [0, 3 << 30, 249 | 599 << 14, 0xFAC | 2 << 12 | 4 << 16 | 9 << 28]);
        // Its whole chain from level 1.
        (view[4], view[5]) = (1, u32::MAX);
        assert_eq!(image_words(&texture, Some(&view), false)[3], 0xFAC | 1 << 12 | 9 << 16 | 9 << 28);
        // An unordered view of level 3: that level alone.
        let uav = [0, 4, 0, 0, 3, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&texture, Some(&uav), true)[3], 0xFAC | 3 << 12 | 3 << 16 | 9 << 28);
        // The array's own view: a 2D array, levels 0-10.
        let words = image_words(&array, None, false);
        assert_eq!(words[1..], [3 << 30, 255 | 599 << 14, 0xFAC | 10 << 16 | 13 << 28]);
    }

    fn placed_memory() -> u64 {
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: non-zero size and power-of-two alignment; leaked for the test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    fn placed_tex_desc(width: u64, height: u32, format: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width as u32,
            (width >> 32) as u32,
            height,
            1 | (1 << 16),
            format,
            1,
            0,
            0,
            flags,
            0,
        ]
    }

    fn placed_tex_uav_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn placed_tex_srv_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_SRV_DIMENSION_TEXTURE2D
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1; // MipLevels
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    #[test]
    fn placed_uav_textures_create_real_host_uavs() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let _ = host.log_messages();
        let bgrx_uav = typed_store(88);
        let cases = [
            (28u32, 28u32, 4u32, true),     // R8G8B8A8_UNORM
            (27u32, 28u32, 4u32, true),     // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
            (29u32, 29u32, 4u32, true),     // R8G8B8A8_UNORM_SRGB -> linear R8G8B8A8_UNORM
            (29u32, 0u32, 4u32, true),      // R8G8B8A8_UNORM_SRGB + UNKNOWN -> linear 28
            (91u32, 91u32, 4u32, true),     // B8G8R8A8_UNORM_SRGB -> linear B8G8R8A8_UNORM
            (91u32, 0u32, 4u32, true),      // B8G8R8A8_UNORM_SRGB + UNKNOWN -> linear 87
            (28u32, 28u32, 0u32, false),    // missing ALLOW_UNORDERED_ACCESS -> null fallback
            (93u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_UNORM_SRGB -> linear 88 if supported
            (93u32, 0u32, 4u32, bgrx_uav),  // B8G8R8X8_UNORM_SRGB + UNKNOWN -> linear 88
            (92u32, 88u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM
            (92u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM_SRGB
            (40u32, 40u32, 4u32, false),    // D32_FLOAT + UAV flag (no depth flag) -> null fallback
            (55u32, 55u32, 4u32, false),    // D16_UNORM + UAV flag (no depth flag) -> null fallback
            (39u32, 40u32, 4u32, false),    // R32_TYPELESS + D32_FLOAT view -> null fallback
            (53u32, 55u32, 4u32, false),    // R16_TYPELESS + D16_UNORM view -> null fallback
            (27u32, 27u32, 4u32, false),    // R8G8B8A8_TYPELESS + typeless view -> null fallback
        ];
        let mut slots = [[0u32; 8]; 24];
        for (idx, (res_fmt, view_fmt, flags, expect_real)) in cases.into_iter().enumerate() {
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, flags);
            let desc = placed_tex_uav_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, slot_addr);
            let view = crate::descriptor::view(slot_addr).unwrap();
            assert!(matches!(
                placed_texture(&view),
                Some(View::Unordered(_, 0, Some(_)))
            ));
            assert!(heap.mirror(idx as u32, slot_addr, false).is_some());
            assert!(
                heap.kept.lock().unwrap()[idx].is_none(),
                "case {idx} mirror(false) should defer host creation"
            );
            let mirrored = heap.mirror(idx as u32, slot_addr, true).unwrap();
            let dyn_slot = heap.count + idx as u32;
            heap.make_at(dyn_slot, &mirrored);
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept[idx].is_some(), expect_real, "case {idx} mirror");
            assert_eq!(
                kept[dyn_slot as usize].is_some(),
                expect_real,
                "case {idx} make_at"
            );
        }
        for (offset, (res_fmt, view_fmt, expect_real)) in [
            (39u32, 40u32, false), // R32_TYPELESS + D32_FLOAT SRV -> null fallback
            (53u32, 55u32, false), // R16_TYPELESS + D16_UNORM SRV -> null fallback
            (29u32, 0u32, true),   // R8G8B8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 29
            (91u32, 0u32, true),   // B8G8R8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 91
        ]
        .into_iter()
        .enumerate()
        {
            let idx = cases.len() + offset;
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, 4);
            let desc = placed_tex_srv_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot_addr);
            heap.mirror(idx as u32, slot_addr, true).unwrap();
            assert_eq!(
                heap.kept.lock().unwrap()[idx].is_some(),
                expect_real,
                "srv case {offset}"
            );
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn generation_records_cover_only_reachable_logical_slots() {
        assert_eq!(Slots::new(4096, 16, 0).made.len(), 16);
        assert_eq!(Slots::new(256, 16, 8).made.len(), 256);
        let limit = victory_shader::emit::REMAP_VIEWS;
        assert_eq!(Slots::new(limit + 8, 16, 8).made.len(), limit as usize);
    }

    #[test]
    fn capped_view_heap_remaps_high_slots_and_recycles_overflow_entries() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        if !host.view_remap {
            return; // uncapped hosts keep every title slot direct
        }
        let null = std::ptr::null();
        // 1,245,184 title view slots (> 1,000,000 `PORTABLE_VIEWS`).
        let total_title_slots: u32 = 1_245_184;
        let heap_desc: [u32; 8] = [0, total_title_slots, 1, 0, 0, 0, 0, 0];
        let mut raw_heap = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                heap_desc.as_ptr().cast(),
                &crate::descriptor::HEAP_IIDS[0],
                &mut raw_heap
            ),
            crate::object::S_OK
        );
        let title_heap = crate::descriptor::heap(raw_heap);
        let h = title_heap.host().expect("capped host view heap");
        assert_eq!(h.count(), capped_layout(h.views).0 - overflow_slots(capped_layout(h.views).0));
        assert_eq!(h.overflow, overflow_slots(capped_layout(h.views).0));
        assert_eq!(h.dynamic, capped_layout(h.views).1);

        // Initialize `remap` and `view_addresses` so live updates are verified too.
        assert!(h.remap().is_some());
        assert!(h.view_addresses().is_some());
        let remap_ptr = h.remap.get().unwrap().as_ref().unwrap().1 as *const u32;
        let stride_ptr = h.strides.as_ref().unwrap().1 as *const u32;

        // Register a 64 KiB page of title memory so placed buffer views build real SRVs.
        let pool = crate::watch::tests::pool(1);
        let base_va = pool as u64;
        crate::shadow::register(pool, base_va, crate::watch::PAGE as u64);

        let mut buf_res = [0u32; 14];
        (buf_res[0], buf_res[4]) = (1, 256);
        let mut buf_desc = [0u32; 16];
        (buf_desc[1], buf_desc[2], buf_desc[6], buf_desc[7]) =
            (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 16);
        (buf_desc[10], buf_desc[11]) = (base_va as u32, (base_va >> 32) as u32);

        let high_slots = [737_856u32, 950_000, 1_200_000];
        let mut phys_assigned = Vec::new();
        for &slot_idx in &high_slots {
            let handle = title_heap.start + u64::from(slot_idx) * 32;
            crate::descriptor::create_placed_srv(null, &buf_res, &buf_desc, handle);
            assert!(h.made(slot_idx), "high slot {slot_idx} recorded");
            assert!(h.kept(slot_idx), "high slot {slot_idx} holds host SRV");

            // SAFETY: `slot_idx < REMAP_VIEWS` in mapped upload buffer.
            let phys = unsafe { remap_ptr.add(slot_idx as usize).read_volatile() };
            assert!(
                phys >= h.count() && phys < h.count() + h.overflow,
                "high slot {slot_idx} mapped to overflow slot {phys}"
            );
            assert_eq!(h.title_slot(phys), Some(slot_idx));
            assert_eq!(h.phys(slot_idx), Some(phys));
            assert_eq!(h.address_at(phys), Some(base_va));
            // SAFETY: `phys < HEAP_VIEWS` in mapped stride table.
            let stride = unsafe { stride_ptr.add(phys as usize).read_volatile() };
            assert_eq!(stride, 16);
            phys_assigned.push(phys);
        }

        // Placed texture in a high slot (`1,210,000`): deferred on `mirror(false)`,
        // materialized when `make_read(&[phys])` runs with the physical overflow slot!
        let tex_slot: u32 = 1_210_000;
        let tex_va = placed_memory();
        let tex_res = placed_tex_desc(4, 4, 28, 0);
        let tex_desc = placed_tex_srv_desc(tex_va, 28);
        let tex_handle = title_heap.start + u64::from(tex_slot) * 32;
        crate::descriptor::create_placed_srv(null, &tex_res, &tex_desc, tex_handle);
        assert!(h.made(tex_slot));
        assert!(!h.kept(tex_slot), "placed texture deferred until read");
        // SAFETY: `tex_slot < REMAP_VIEWS`.
        let tex_phys = unsafe { remap_ptr.add(tex_slot as usize).read_volatile() };
        assert_eq!(h.title_slot(tex_phys), Some(tex_slot));
        crate::descriptor::make_read(&[tex_phys]);
        assert!(h.kept(tex_slot), "make_read materialized high-slot placed texture");

        // Clearing a high slot resets `remap[slot]` to `CAPPED_HEAP_VIEWS - 1` and returns
        // its physical overflow slot to the free list for reuse.
        let cleared = high_slots[1];
        let freed_phys = phys_assigned[1];
        let cleared_handle = title_heap.start + u64::from(cleared) * 32;
        // SAFETY: valid descriptor slot in `title_heap`.
        unsafe { std::ptr::write_bytes(cleared_handle as *mut u8, 0, 32) };
        h.sync_slots(cleared, cleared + 1, title_heap.start, 32);
        assert!(!h.made(cleared) && !h.kept(cleared));
        // SAFETY: `cleared < REMAP_VIEWS`.
        let unmapped = unsafe { remap_ptr.add(cleared as usize).read_volatile() };
        assert_eq!(unmapped, h.views - 1);
        assert_eq!(h.title_slot(freed_phys), None);

        // Next high slot reuses `freed_phys` (LIFO free list).
        let next_high: u32 = 1_220_000;
        let next_handle = title_heap.start + u64::from(next_high) * 32;
        crate::descriptor::create_placed_srv(null, &buf_res, &buf_desc, next_handle);
        // SAFETY: `next_high < REMAP_VIEWS`.
        let reused_phys = unsafe { remap_ptr.add(next_high as usize).read_volatile() };
        assert_eq!(reused_phys, freed_phys);
        assert_eq!(h.title_slot(reused_phys), Some(next_high));

        // SAFETY: plain device check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        eprintln!("high texture descriptors mapped, materialized, cleared and reused on the capped adapter");
        crate::descriptor::release(raw_heap);
        crate::shadow::unregister(pool, base_va);
        crate::watch::tests::free(pool);
    }
}
