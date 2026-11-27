//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, Options, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in a folder of `dir` from now on: the folder
/// of this translator, `None` if it cannot be opened.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    let dir = dir.join(&blake3::Hash::from(*build()).to_hex()[..16]);
    match Cache::open(&dir) {
        Ok(cache) => {
            tracing::info!(dir = %dir.display(), "shader cache");
            let _ = CACHE.set(cache);
            Some(dir)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no shader cache");
            None
        }
    }
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// The translation options of this run: writes through buffer views mark
/// the pages for the renderer's copy-back, and wave64 programs run in
/// halves where the host's waves are 32 lanes at most ([`host_wave32`]).
fn options() -> Options {
    Options {
        view_marks: true,
        host_wave32: host_wave32(),
        view_remap: view_remap(),
    }
}

/// Whether translations read title view-heap slots through the view-remap
/// table: the host GPU's view heaps are capped below the title's and the
/// remap is on (`crate::host::Host::view_remap`); none without a device.
fn view_remap() -> bool {
    crate::host::host().is_some_and(|h| h.view_remap)
}

/// Whether translations are for a host of 32-lane waves: the host GPU's
/// (made here if it is not yet: the cache belongs to the GPU it was filled
/// on, as the pipeline library does); none without one.
fn host_wave32() -> bool {
    crate::host::host().is_some_and(crate::host::Host::wave32)
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// loop limit and translation options passed to the emitter (among them
/// whether the host's waves are 32 lanes at most, [`host_wave32`]).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(victory_shader::SOURCE_HASH, ENCODING, limit, options())
    })
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(translator: &str, encoding: &[u8], limit: u32, options: Options) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    // Destructure all options so a new emitter flag requires a cache-key decision.
    let Options {
        view_marks,
        host_wave32,
        view_remap,
    } = options;
    h.update(&[u8::from(view_marks)]);
    h.update(&[u8::from(host_wave32)]);
    h.update(&[u8::from(view_remap)]);
    *h.finalize().as_bytes()
}

/// [`build`] as this host's entries are keyed: a translation also depends
/// on the slots of the host device's view heaps
/// (`victory_shader::emit::heap_views`), which differ between devices
/// sharing one cache folder (an NVIDIA GPU, WARP, another vendor's).
fn host_build(build: &[u8; 32], (views, title): (u32, u32)) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&views.to_le_bytes());
    h.update(&title.to_le_bytes());
    *h.finalize().as_bytes()
}

/// The translation of a console shader container, from the cache when
/// there is one, and whether it came from there.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    // The host device first: making it says how large its view heaps are,
    // which translations depend on (once per process, not per shader).
    // Without a device they are as most hosts make them.
    let _ = crate::host::host();
    let profile = victory_shader::emit::view_heap();
    // Writes through views mark their pages: copy-backs compare only those
    // (`crate::shadow::copy_back_marks`).
    let options = options();
    let translate = |c: &[u8]| victory_shader::emit::with_view_heap(profile.0, profile.1, || {
        victory_shader::emit::translate_with(c, options).map_err(|e| e.to_string())
    });
    match CACHE.get() {
        Some(cache) => cached(cache, &host_build(build(), profile), container, translate),
        None => (translate(container), false),
    }
}

/// What a linked mesh/pixel translation ([`translate_linked`]) depends on
/// besides its containers, snapshotted once for both the key and the
/// translation: the build ([`build`]: translator, loop limit, options) and
/// the host device's view heap size and title bound, each on its own
/// ([`host_build`]). The same as [`host_shader`] keys standalone
/// translations by.
#[cfg(windows)]
pub(crate) fn linked_build() -> ([u8; 32], (u32, u32)) {
    let _ = crate::host::host();
    let profile = victory_shader::emit::view_heap();
    (host_build(build(), profile), profile)
}

/// The linked translation of a mesh container and its pixel container (or
/// none), made as [`host_shader`] makes each standalone one: the same
/// options, under the view heap `profile` [`linked_build`] took.
#[cfg(windows)]
pub(crate) fn translate_linked(
    profile: (u32, u32),
    mesh: &[u8],
    pixel: Option<&[u8]>,
) -> Result<(HostShader, Option<HostShader>), String> {
    victory_shader::emit::with_view_heap(profile.0, profile.1, || {
        victory_shader::emit::translate_linked(mesh, pixel, options())
    })
    .map_err(|e| e.to_string())
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit;
/// `made` keeps a miss's translation, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms here).
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let made = (self.translate)(&input[32..]);
        let bytes = encode(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return (shader, false);
        }
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    (
        made().unwrap_or_else(|| (converter.translate)(container)),
        false,
    )
}

const MAGIC: &[u8; 4] = b"VSH1";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let memory = r.word()?;
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        view_marks: memory & 32 != 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &[7; 32], b"a container", translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the options: view marks and
    /// the host's wave width) makes another key, so no entry written before
    /// it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        const OPTIONS: Options = Options {
            view_marks: true,
            host_wave32: false,
            view_remap: false,
        };
        let k = key("0123456789abcdef", b"encoding", 1 << 20, OPTIONS);
        assert_eq!(k, key("0123456789abcdef", b"encoding", 1 << 20, OPTIONS));
        assert_ne!(k, key("0123456789abcdee", b"encoding", 1 << 20, OPTIONS));
        assert_ne!(k, key("0123456789abcdef", b"encoding.", 1 << 20, OPTIONS));
        assert_ne!(k, key("0123456789abcdef", b"encoding", 1 << 19, OPTIONS));
        assert_eq!(
            *build(),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                options(),
            )
        );
        assert_eq!(options().host_wave32, host_wave32());
        assert!(options().view_marks);
        assert_ne!(
            k,
            key(
                "0123456789abcdef",
                b"encoding",
                1 << 20,
                Options {
                    view_marks: false,
                    ..OPTIONS
                }
            )
        );
        let wave32 = key(
            "0123456789abcdef",
            b"encoding",
            1 << 20,
            Options {
                host_wave32: true,
                ..OPTIONS
            },
        );
        assert_ne!(k, wave32);
        let remapped = key(
            "0123456789abcdef",
            b"encoding",
            1 << 20,
            Options {
                view_remap: true,
                ..OPTIONS
            },
        );
        assert_ne!(k, remapped, "the view remap makes other translations");
        assert_eq!(options().view_remap, view_remap());
        // Heap caps and wave width key separately (below): neither stands
        // in for the other.
        // Each view heap size a device makes has entries of its own.
        let (max, portable) = (
            victory_shader::emit::HEAP_VIEWS_MAX,
            victory_shader::emit::HEAP_VIEWS_PORTABLE,
        );
        assert_eq!(host_build(&k, (max, max)), host_build(&k, (max, max)));
        assert_ne!(host_build(&k, (max, max)), host_build(&k, (portable, portable)));
        assert_ne!(host_build(&k, (portable, portable)), host_build(&k, (portable, 721_469)));
        assert_ne!(host_build(&k, (max, max)), k);
        assert_ne!(host_build(&k, (max, max)), host_build(&key("0123456789abcdee", b"encoding", 1 << 20, OPTIONS), (max, max)));
        assert_ne!(host_build(&k, (max, max)), host_build(&wave32, (max, max)));
        assert_ne!(host_build(&wave32, (max, max)), host_build(&wave32, (portable, portable)));
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    /// Linked translations key on the view heap size and the title bound
    /// each (as standalone ones do), taken from the calling thread's
    /// profile at the time, plus the build (translator, loop limit, options).
    #[cfg(windows)]
    #[test]
    fn linked_translations_key_on_the_snapshotted_heap_size_and_title_bound() {
        use victory_shader::emit::{HEAP_VIEWS_MAX, HEAP_VIEWS_PORTABLE, with_view_heap};
        let a = with_view_heap(HEAP_VIEWS_MAX, u32::MAX, linked_build);
        let b = with_view_heap(HEAP_VIEWS_MAX, 4096, linked_build);
        let c = with_view_heap(HEAP_VIEWS_PORTABLE, u32::MAX, linked_build);
        assert_eq!(a.1, (HEAP_VIEWS_MAX, HEAP_VIEWS_MAX));
        assert_eq!(b.1, (HEAP_VIEWS_MAX, 4096));
        assert_eq!(c.1, (HEAP_VIEWS_PORTABLE, HEAP_VIEWS_PORTABLE));
        for (key, profile) in [a, b, c] {
            assert_eq!(key, host_build(build(), profile));
        }
        assert_ne!(a.0, b.0, "the title bound alone changes the key");
        assert_ne!(a.0, c.0, "the heap size changes the key");
        assert_ne!(b.0, c.0);
    }

    #[test]
    fn an_unwritable_entry_keeps_the_translated_shader_and_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let calls = AtomicU64::new(0);
        let run = || cached(&cache, &[9; 32], b"publication fixture", |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        });
        assert_eq!(run(), (Ok(shader()), false));
        let entries = walk(dir.path());
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        std::fs::remove_file(entry).unwrap();
        std::fs::create_dir(entry).unwrap();
        assert_eq!(run(), (Ok(shader()), false), "the shader is usable even if publishing fails");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert!(entry.is_dir());
        std::fs::remove_dir(entry).unwrap();
        assert_eq!(run(), (Ok(shader()), false));
        assert!(entry.is_file());
        assert_eq!(run(), (Ok(shader()), true));
        assert_eq!(calls.load(Ordering::Relaxed), 3, "the final call is a warm hit");
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
