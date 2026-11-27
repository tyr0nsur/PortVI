//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6; validator 1.6,
//! or 1.8 for compute derivatives).

use super::bitcode::{Attr, Md, Module, Ty, V, pred};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;
/// The root SRV slot (`t255` in [`ROOT_CBV_SPACE`], past every title root
/// parameter) of the view-stride table: dword `i` is the byte stride of
/// the buffer view at view-heap index `i` ([`VIEW_STRIDE_BITS`]), with
/// [`VIEW_INTEGER`] and [`VIEW_SIGNED`] for typed views of integer formats.
pub const VIEW_STRIDES: u8 = 255;
/// The root SRV slot (`t254` in [`ROOT_CBV_SPACE`], past every title root
/// parameter) of the view-remap table ([`super::Options::view_remap`]):
/// dword `i` (below [`REMAP_VIEWS`]) is the host heap slot of the title's
/// view-heap slot `i`, the heap's last slot (null) for one it has none for.
pub const VIEW_REMAP: u8 = 254;
/// Title view-heap slots the view-remap table covers ([`VIEW_REMAP`]): the
/// console's view heap limit (2^21); an index past it reads the last entry.
pub const REMAP_VIEWS: u32 = 1 << 21;
/// The bits of a view-stride table dword holding the stride.
pub const VIEW_STRIDE_BITS: u32 = (1 << 30) - 1;
/// In a view-stride table dword: a typed buffer view of an integer format
/// (`UINT` or `SINT`), whose 16-bit typed accesses (`_d16`) pass bits.
pub const VIEW_INTEGER: u32 = 1 << 31;
/// With [`VIEW_INTEGER`]: a signed (`SINT`) format.
pub const VIEW_SIGNED: u32 = 1 << 30;
/// Where the title-memory page table is bound: `t0` (a raw root SRV) in
/// this space (see `emit` "Title memory").
pub const MEMORY_SPACE: u32 = 2;
/// View heap indices are clamped below [`heap_views`], which is this where
/// the host device makes a shader-visible view heap this large.
pub const HEAP_VIEWS_MAX: u32 = 1 << 21;
/// The largest shader-visible view heap every device makes (resource
/// binding tiers 1 and 2); NVIDIA makes no larger one on tier 3 either.
pub const HEAP_VIEWS_PORTABLE: u32 = 1_000_000;
/// Sampler heap indices are clamped below this: the host's sampler heaps
/// are at least this large, the slots past their own null.
pub const HEAP_SAMPLERS: u32 = 2048;

/// The device size and the title-visible bound published as a single profile.
static VIEW_PROFILE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
    ((HEAP_VIEWS_MAX as u64) << 32) | HEAP_VIEWS_MAX as u64,
);

thread_local! {
    /// [`with_heap_views`], [`with_view_heap`]: this thread's size and
    /// title bound, over the process-wide ones.
    static SCOPED_HEAP_VIEWS: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// The slots of the host's shader-visible view heaps, which view heap
/// indices are clamped below (`emit`, "Heap indices"): every host view heap
/// has at least this many, the last three null (the null raw UAV, the null
/// raw SRV at [`null_buffer_view`], and the slot clamped indices land on).
/// [`HEAP_VIEWS_MAX`] unless the host device said otherwise
/// ([`set_heap_views`], with what `host::Host::view_heap_limit` measured
/// on that device). The value is part of the translation, so a shader
/// cache must key on it (`victory-gpu`'s does).
pub fn heap_views() -> u32 {
    view_heap().0
}

/// The title slots a host view heap mirrors at most (`emit`, "Title
/// slots"): a heap index a shader takes from the resource heap itself
/// (not through a table) at or past it reads the heap's last slot, which
/// stays null, instead of whatever the host keeps there (dynamic tables,
/// view pairs). [`heap_views`] (no bound past the clamp) unless the host
/// said otherwise ([`set_view_heap`]). Part of the translation, as
/// [`heap_views`] is.
pub fn title_views() -> u32 {
    view_heap().1
}

/// [`heap_views`] and [`title_views`], the latter at most the former.
pub fn view_heap() -> (u32, u32) {
    SCOPED_HEAP_VIEWS.with(std::cell::Cell::get).unwrap_or_else(|| {
        let profile = VIEW_PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        ((profile >> 32) as u32, profile as u32)
    })
}

fn clamp_heap_views(views: u32) -> u32 {
    views.clamp(HEAP_SAMPLERS, HEAP_VIEWS_MAX)
}

/// Translations from now on are for view heaps of `views` slots (clamped
/// to [`HEAP_SAMPLERS`]..=[`HEAP_VIEWS_MAX`]) mirroring every title slot
/// below them (no [`title_views`] bound): what the host device makes, said
/// once when it is made, before the first translation.
pub fn set_heap_views(views: u32) {
    set_view_heap(views, u32::MAX);
}

/// [`set_heap_views`] for a host whose view heaps mirror title slots below
/// `title` only (clamped to 1..=`views`), the host's own slots (dynamic
/// tables, view pairs, the null views) after them: every host view heap
/// must then keep its title slots below `title` and nothing else there.
pub fn set_view_heap(views: u32, title: u32) {
    let views = clamp_heap_views(views);
    let profile = (u64::from(views) << 32) | u64::from(title.clamp(1, views));
    VIEW_PROFILE.store(profile, std::sync::atomic::Ordering::Relaxed);
}

/// `f`, with translations and probe-host heaps on this thread for view
/// heaps of `views` slots (no [`title_views`] bound): for checks that run
/// on more than one device (hardware, WARP) in one process.
pub fn with_heap_views<R>(views: u32, f: impl FnOnce() -> R) -> R {
    with_view_heap(views, u32::MAX, f)
}

/// [`with_heap_views`] with title slots below `title` only (as
/// [`set_view_heap`]).
pub fn with_view_heap<R>(views: u32, title: u32, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<(u32, u32)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPED_HEAP_VIEWS.with(|s| s.set(self.0));
        }
    }
    let views = clamp_heap_views(views);
    let scoped = Some((views, title.clamp(1, views)));
    let _restore = Restore(SCOPED_HEAP_VIEWS.with(|s| s.replace(scoped)));
    f()
}

/// The heap index of the host's null raw buffer view (the null raw
/// unordered access view at the index below), read instead of a slot that
/// holds no buffer view where a shader reads one at an index it computes
/// (`emit`, "Buffer views at computed heap indices"): two below the end of
/// the host's view heaps ([`heap_views`]), whose last slot stays null.
pub fn null_buffer_view() -> u32 {
    heap_views() - 2
}
/// The view-mark table's register: `t2` (a raw root SRV) in
/// [`MEMORY_SPACE`] (`emit`, "Write marks").
pub const VIEW_MARKS: u32 = 2;
/// The written-pages bitmap of buffer views: `u1` (a raw root UAV) in
/// [`MEMORY_SPACE`].
pub const VIEW_WRITES: u32 = 1;
/// A view-mark table entry's base for a slot whose writes are not marked.
pub const VIEW_MARK_NONE: u32 = u32::MAX;
/// Bits in the written-pages bitmap of buffer views; marks past it go to
/// the last bit.
pub const VIEW_WRITE_BITS: u32 = 1 << 22;
/// The first bit of the read marks of texture views (`emit`, "Read
/// marks"), after the written pages in the same bitmap: one bit per
/// resource-heap index, [`HEAP_VIEWS_MAX`] of them.
pub const VIEW_READS: u32 = VIEW_WRITE_BITS;

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    /// UAVs in a vertex, geometry or mesh program.
    pub const UAVS_EVERY_STAGE: u64 = 1 << 16;
    pub const ROVS: u64 = 1 << 18;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const INT64: u64 = 1 << 20;
    pub const BARYCENTRICS: u64 = 1 << 22;
    /// 64-bit floating point (`fptrunc` from a `double`).
    pub const FLOAT64: u64 = 1 << 2;
    pub const ATOMIC64_TYPED: u64 = 1 << 27;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
    pub const ATOMIC64_HEAP: u64 = 1 << 32;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader taking triangles and emitting triangle strips of
    /// at most this many vertices (`rect`).
    pub gs: Option<u32>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Immediate samplers uniform control flow chooses between: handles
    /// `a` and `b` and when `a` is taken. An operation on `a` runs on both,
    /// and the result is chosen.
    pub sampler_choices: Vec<(V, V, V)>,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    /// The title-memory page table's handle, once read.
    pub memory: Option<V>,
    /// The written-pages bitmap's handle, once title memory is written.
    pub memory_dirty: Option<V>,
    /// The view addresses' handle, once a descriptor's address is read.
    pub view_addresses: Option<V>,
    /// The view-mark table's and the buffer views' written-pages bitmap's
    /// handles, once a write through a view is marked.
    pub view_marks: Option<V>,
    pub view_writes: Option<V>,
    /// The heap index of each handle of an unordered buffer view (writes
    /// through it are marked by the index's view-mark entry).
    pub view_index: std::collections::HashMap<V, V>,
    /// The heap index of each handle of a texture shader view (its reads
    /// are marked, `emit` "Read marks"), and those the instruction being
    /// lowered uses.
    pub texture_index: std::collections::HashMap<V, V>,
    pub texture_reads: Vec<V>,
    pub flags: u64,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Compute derivatives require validator 1.8 rather than 1.6 or 1.7.
    compute_derivatives: bool,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            gs: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            sampler_choices: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            memory: None,
            memory_dirty: None,
            view_addresses: None,
            view_marks: None,
            view_writes: None,
            view_index: std::collections::HashMap::new(),
            texture_index: std::collections::HashMap::new(),
            texture_reads: Vec::new(),
            flags: 0,
            nonuniform_handles: 0,
            threads: None,
            wave_size: None,
            compute_derivatives: false,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        let g = self.m.global("lds", arr, 3);
        self.lds = Some(g);
        g
    }

    /// More thread-group shared memory beside [`Shader::lds`]: `[dwords x
    /// i32]` in address space 3, named `name`.
    pub fn shared(&mut self, name: &str, dwords: u32) -> V {
        let arr = self.m.array(dwords, self.t.i32);
        self.m.global(name, arr, 3)
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        // Validators before 1.8 reject gradient operations outside the
        // pixel stage. This set matches DXC's gradient classification:
        // the implicit-derivative samples (60/61/64) and the LOD query
        // (81), plus the explicit derivatives (83..=86). Bfi and the dot
        // products (53..=56) are arithmetic and stay on 1.6, and
        // SampleGrad/SampleLevel (62/63) take explicit gradients, so the
        // validator already accepts them in compute.
        if self.threads.is_some()
            && self.mesh.is_none()
            && matches!(opcode, 60 | 61 | 64 | 81 | 83..=86)
        {
            self.compute_derivatives = true;
        }
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        let choice = self
            .sampler_choices
            .iter()
            .copied()
            .find(|&(a, _, _)| args.contains(&a));
        let r = self.m.call(block, f, all.clone());
        let Some((a, b, cond)) = choice else {
            return r;
        };
        for x in &mut all {
            if *x == a {
                *x = b;
            }
        }
        let other = self.m.call(block, f, all);
        if ret == self.t.res_ret {
            // Drivers miscompile a select of the result structure (AMD
            // read zeros): its fields are chosen where they are read.
            self.m.chosen.insert(r, (cond, other));
            return r;
        }
        self.m.select(block, cond, r, other)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`.
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of the title-memory page table (`t0` in [`MEMORY_SPACE`],
    /// a raw buffer), made in the entry block.
    pub fn memory_handle(&mut self) -> V {
        if let Some(h) = self.memory {
            return h;
        }
        let r = self.m.i32(0);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.memory = Some(h);
        h
    }

    /// The handle of the view addresses (`t1` in [`MEMORY_SPACE`], a raw
    /// buffer: per resource-heap index, a descriptor's dwords 0-3 as far as
    /// the host knows them, `emit` "View addresses"), made in the entry
    /// block.
    pub fn view_addresses_handle(&mut self) -> V {
        if let Some(h) = self.view_addresses {
            return h;
        }
        let r = self.m.i32(1);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.view_addresses = Some(h);
        h
    }

    /// The handle of the bitmap of title-memory pages written (`u0` in
    /// [`MEMORY_SPACE`], a raw buffer), made in the entry block.
    pub fn memory_dirty_handle(&mut self) -> V {
        if let Some(h) = self.memory_dirty {
            return h;
        }
        let r = self.m.i32(0);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.memory_dirty = Some(h);
        h
    }

    /// The handle of the view-mark table (`t2` in [`MEMORY_SPACE`], a raw
    /// buffer: per resource-heap index, two dwords), made in the entry block.
    pub fn view_marks_handle(&mut self) -> V {
        if let Some(h) = self.view_marks {
            return h;
        }
        let h = self.memory_space_handle(VIEW_MARKS, false);
        self.view_marks = Some(h);
        h
    }

    /// The handle of the buffer views' written-pages bitmap (`u1` in
    /// [`MEMORY_SPACE`], a raw buffer), made in the entry block.
    pub fn view_writes_handle(&mut self) -> V {
        if let Some(h) = self.view_writes {
            return h;
        }
        let h = self.memory_space_handle(VIEW_WRITES, true);
        self.view_writes = Some(h);
        h
    }

    /// A raw buffer bound at `register` in [`MEMORY_SPACE`] (`uav`: `u`,
    /// else `t`), made in the entry block.
    fn memory_space_handle(&mut self, register: u32, uav: bool) -> V {
        let r = self.m.i32(register);
        let bind = [r, r, self.m.i32(MEMORY_SPACE), self.m.i8(u8::from(uav))];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, if uav { 11 | 0x1000 } else { 11 }, 0);
        self.flags |= flags::RAW_BUFFERS;
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        self.nonuniform_handles += u32::from(nonuniform);
        // An index past the host's heap faults the device (a console shader
        // reads whatever memory follows its heap): the last slot instead.
        let last = self.m.i32(if sampler { HEAP_SAMPLERS } else { heap_views() } - 1);
        let over = self.m.cmp(block, pred::IUGT, index, last);
        let index = self.m.select(block, over, last, index);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        let h = self.annotate(block, h, props.0, props.1);
        // Unordered typed, raw and structured buffers (kinds 10-12), and
        // typed buffers read (their format kind, `Lower::view_word`).
        let kind = props.0 & 0xFF;
        if !sampler && (kind == 10 || props.0 & 0x1000 != 0 && (11..=12).contains(&kind)) {
            self.view_index.insert(h, index);
        }
        // Texture shader views (kinds 1-9).
        if !sampler && props.0 & 0x1000 == 0 && (1..=9).contains(&kind) {
            self.texture_index.insert(h, index);
        }
        h
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The flags of what lowering made, as the module writes it: what only
        // unread values or never-taken edges used is not written.
        let uses = self.m.uses();
        for (flag, used) in [
            (flags::RESOURCE_HEAP, uses.heap),
            (flags::SAMPLER_HEAP, uses.sampler_heap),
            (flags::WAVE_OPS, uses.wave),
            (flags::INT64, uses.int64),
        ] {
            self.flags = if used { self.flags | flag } else { self.flags & !flag };
        }
        // UAVs outside pixel and compute programs (the write and read marks
        // of vertex, geometry and mesh programs), as DXC counts them: UAV
        // ranges declared below, not UAVs indexed from the heap.
        let compute = self.threads.is_some() && self.mesh.is_none() && self.gs.is_none();
        let declared_uavs =
            !self.root_uavs.is_empty() || self.memory_dirty.is_some() || self.view_writes.is_some();
        if declared_uavs && !self.pixel && !compute {
            self.flags |= flags::UAVS_EVERY_STAGE;
        }
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || self.memory.is_some()
            || self.memory_dirty.is_some()
            || self.view_addresses.is_some()
            || self.view_marks.is_some()
            || self.view_writes.is_some()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty()
                || self.memory.is_some()
                || self.view_addresses.is_some()
                || self.view_marks.is_some()
            {
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                let all: Vec<(u32, u32)> = self
                    .root_srvs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .chain(self.memory.map(|_| (MEMORY_SPACE, 0)))
                    .chain(self.view_addresses.map(|_| (MEMORY_SPACE, 1)))
                    .chain(self.view_marks.map(|_| (MEMORY_SPACE, VIEW_MARKS)))
                    .collect();
                for (n, (space, slot)) in all.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, slot)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space,
                        register: slot,
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if declared_uavs {
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let no = self.m.bool(false);
                let no = self.m.md_value(no);
                let mut list = Vec::new();
                let all: Vec<(u32, u32)> = self
                    .root_uavs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .chain(self.memory_dirty.map(|_| (MEMORY_SPACE, 0)))
                    .chain(self.view_writes.map(|_| (MEMORY_SPACE, VIEW_WRITES)))
                    .collect();
                for (n, (space, slot)) in all.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, slot)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(no),
                        Some(no),
                        Some(no),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space,
                        register: slot,
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // A geometry shader has four streams, each its output count and
            // table; the other three empty.
            if self.gs.is_some() {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        if self.pixel {
            container::same_mode_per_row(&mut self.inputs, true);
        } else {
            container::same_mode_per_row(&mut self.outputs, false);
        }
        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), thread-group size (4),
        // wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        // Geometry state (tag 1): triangles in, maximum vertices, stream
        // mask, triangle strips out, one instance.
        if let Some(max) = self.gs {
            props.push(i32c(&mut self, 1));
            let state = [3, max, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        let val_minor = if self.compute_derivatives { 8 } else { 6 };
        let val_minor = i32c(&mut self, val_minor);
        let val_version = self.m.md_node(vec![Some(one), Some(val_minor)]);
        self.m.named_md("dx.valver", vec![val_version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.gs.is_some() => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let bitcode = self.m.write();
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::ROVS != 0 {
            features |= 0x1000;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::FLOAT64 != 0 {
            features |= 0x1;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC64_TYPED != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC64_HEAP != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            gs: self.gs,
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
            validator18: self.compute_derivatives,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::Term;
    use super::super::validator;
    use super::*;

    /// A program that sets a bit through the write marks' UAV, or (`heap`)
    /// through a UAV indexed from the heap: a vertex (`pixel` false, no
    /// thread-group size), pixel or compute program.
    fn marked(pixel: bool, threads: Option<[u32; 3]>, heap: bool) -> Vec<u8> {
        let mut s = Shader::new(pixel);
        s.threads = threads;
        let b = s.entry;
        let h = if heap {
            let index = s.m.i32(0);
            s.heap_handle(b, index, false, false, (11 | 0x1000, 0))
        } else {
            s.view_writes_handle()
        };
        let (or, zero, one) = (s.m.i32(2), s.m.i32(0), s.m.i32(1));
        let iu = s.m.undef(s.t.i32);
        let i32t = s.t.i32;
        s.op(b, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[h, or, zero, iu, iu, one]);
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// A vertex program writing a declared UAV declares UAVs at every stage
    /// (the validator refuses it otherwise: "Flags must match usage", the
    /// 5,046 programs of the title's list); pixel and compute programs need
    /// not, nor a vertex program whose UAVs all come from the heap (the
    /// validator counts declared ranges only: 26 mesh and vertex programs).
    #[test]
    fn uavs_outside_pixel_and_compute_programs_are_declared() {
        let mut validator = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        )
        .ok();
        for (name, pixel, threads, heap) in [
            ("vertex", false, None, false),
            ("pixel", true, None, false),
            ("compute", false, Some([1, 1, 1]), false),
            ("vertex, heap", false, None, true),
        ] {
            let mut dxil = marked(pixel, threads, heap);
            if let Some(v) = validator.as_mut() {
                v.validate(&mut dxil)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }

    // Keep the tested operation live through the emitter's dead-code pass.
    fn store_scalar(s: &mut Shader, value: V) {
        let buffer = s.root_uav_handle(0);
        let at = s.m.i32(0);
        let offset = s.m.undef(s.t.i32);
        let unused = s.m.undef(s.t.f32);
        let mask = s.m.i8(1);
        let alignment = s.m.i32(4);
        s.op(s.entry, "rawBufferStore.f32", Attr::NoUnwind, s.t.void, 140,
             &[buffer, at, offset, value, unused, unused, unused, mask, alignment]);
    }

    /// Builds a compute shader calling `opcode` and returns its container
    /// plus the derivative flag as detected by `op`. `Some(v)` overrides
    /// the flag before the container is written.
    fn compute_with_op(
        opcode: u32,
        name: &str,
        attr: Attr,
        args: usize,
        validator18: Option<bool>,
    ) -> (Vec<u8>, bool) {
        let mut s = Shader::new(false);
        s.threads = Some([32, 1, 1]);
        s.wave_size = Some(32);
        let block = s.entry;
        let zero = s.m.f32(0);
        let args = vec![zero; args];
        let ret = s.t.f32;
        let value = s.op(block, name, attr, ret, opcode, &args);
        store_scalar(&mut s, value);
        let detected = s.compute_derivatives;
        if let Some(v) = validator18 {
            s.compute_derivatives = v;
        }
        s.m.terminate(block, Term::Ret);
        (s.finish(), detected)
    }

    #[test]
    fn compute_implicit_sample_validates() {
        let shader = |legacy: bool| {
            let mut s = Shader::new(false);
            s.threads = Some([32, 1, 1]);
            s.wave_size = Some(32);
            s.samplers = 1;
            let block = s.entry;
            let index = s.m.i32(0);
            let texture = s.heap_handle(block, index, false, false, (2, 1033));
            let sampler = s.sampler_handle(block, 0);
            let zero = s.m.f32(0);
            let fu = s.m.undef(s.t.f32);
            let iu = s.m.undef(s.t.i32);
            let args = [texture, sampler, zero, zero, fu, fu, iu, iu, iu, fu];
            let ret = s.t.res_ret;
            let sample = s.op(block, "sample.f32", Attr::ReadOnly, ret, 60, &args);
            let value = s.m.extract(block, sample, 0);
            store_scalar(&mut s, value);
            assert!(s.compute_derivatives);
            if legacy {
                s.compute_derivatives = false;
            }
            s.m.terminate(block, Term::Ret);
            s.finish()
        };
        let path = validator::Validator::default_path();
        let Ok(mut validator) = validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        validator.validate(&mut shader(false)).unwrap();
        assert!(validator.validate(&mut shader(true)).is_err());
    }

    #[test]
    fn compute_explicit_derivative_validates() {
        // DerivCoarseX (83): rejected before validator 1.8 like the
        // implicit-derivative samples.
        let (mut dxil, detected) = compute_with_op(83, "unary.f32", Attr::ReadNone, 1, Some(true));
        assert!(detected);
        let (mut legacy, _) = compute_with_op(83, "unary.f32", Attr::ReadNone, 1, Some(false));
        let path = validator::Validator::default_path();
        let Ok(mut validator) = validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        validator.validate(&mut dxil).unwrap();
        assert!(validator.validate(&mut legacy).is_err());
    }

    #[test]
    fn compute_dot3_stays_on_validator_1_6() {
        // Dot3 (55) is arithmetic, not a gradient op: it must not bump the
        // validator version, and the resulting 1.6 module validates.
        let (mut dxil, detected) = compute_with_op(55, "dot3.f32", Attr::ReadNone, 6, None);
        assert!(!detected);
        let path = validator::Validator::default_path();
        let Ok(mut validator) = validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        validator.validate(&mut dxil).unwrap();
    }
}
