//! Conversion at the CPU/D3D12 boundary of writable RGB9E5 resources.
//! Finite components of both packed formats are integral multiples of 2^-24.

const MAX_SHARED: u64 = 511 << 31;

/// Rounds a positive fixed-point integer to nearest, with ties rounded to even.
fn rounded(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    (value + (1 << (shift - 1)) - 1 + ((value >> shift) & 1)) >> shift
}

/// Shared-exponent units to an unsigned 5-exponent-bit float. Saturate
/// channels at their largest finite value and preserve representable subnormals.
fn packed_channel(units: u64, mantissa_bits: u32) -> u32 {
    let min_shift = 10 - mantissa_bits;
    // DirectXMath clamps R/G below the smallest R11 subnormal; B rounds to R10.
    if mantissa_bits == 6 && units < (1 << min_shift) {
        return 0;
    }
    if units < 1024 {
        return rounded(units, min_shift) as u32;
    }
    let highest = 63 - units.leading_zeros();
    let mut exponent = highest - 9;
    let mut mantissa = rounded(units, highest - mantissa_bits) as u32;
    if mantissa == 2 << mantissa_bits {
        mantissa >>= 1;
        exponent += 1;
    }
    if exponent >= 31 {
        return (31 << mantissa_bits) - 1;
    }
    (exponent << mantissa_bits) | (mantissa - (1 << mantissa_bits))
}

pub(crate) fn to_host(word: u32) -> u32 {
    let exponent = word >> 27;
    let channel = |offset: u32, mantissa_bits| {
        packed_channel(u64::from((word >> offset) & 511) << exponent, mantissa_bits)
    };
    channel(0, 6) | channel(9, 6) << 11 | channel(18, 5) << 22
}

fn shared_units(bits: u32, mantissa_bits: u32) -> u64 {
    let mantissa = bits & ((1 << mantissa_bits) - 1);
    let exponent = bits >> mantissa_bits;
    match exponent {
        0 => u64::from(mantissa) << (10 - mantissa_bits),
        31 if mantissa != 0 => 0, // RGB9E5 has no NaN.
        31 => MAX_SHARED,
        _ => u64::from(mantissa | (1 << mantissa_bits)) << (exponent + 9 - mantissa_bits),
    }
}

pub(crate) fn to_title(word: u32) -> u32 {
    let channels = [
        shared_units(word & 2047, 6),
        shared_units((word >> 11) & 2047, 6),
        shared_units(word >> 22, 5),
    ];
    let peak = channels[0].max(channels[1]).max(channels[2]);
    let mut exponent = (64 - peak.leading_zeros()).saturating_sub(9);
    // Rounding the largest channel up to 512 carries into the shared exponent.
    if rounded(peak, exponent) > 511 {
        exponent += 1;
    }
    let [r, g, b] = channels.map(|v| rounded(v, exponent) as u32);
    r | (g << 9) | (b << 18) | (exponent << 27)
}

/// Converts visible words in a single row; callers exclude footprint padding.
pub(crate) fn row(source: &[u8], destination: &mut [u8], convert: fn(u32) -> u32) {
    debug_assert_eq!(source.len(), destination.len());
    debug_assert!(source.len().is_multiple_of(4));
    for (src, dst) in source.chunks_exact(4).zip(destination.chunks_exact_mut(4)) {
        let word = u32::from_le_bytes(src.try_into().unwrap());
        dst.copy_from_slice(&convert(word).to_le_bytes());
    }
}

/// Converts a full upload footprint. The last D3D12 row may omit padding.
pub(crate) fn footprint(source: &[u8], destination: &mut [u8], width: u32, pitch: usize) -> bool {
    let Some(visible) = (width as usize).checked_mul(4) else { return false };
    if pitch == 0 || visible == 0 || pitch < visible || source.len() != destination.len() {
        return false;
    }
    let tail = source.len() % pitch;
    if source.len() < visible || (tail != 0 && tail < visible) {
        return false;
    }
    for (src, dst) in source.chunks(pitch).zip(destination.chunks_mut(pitch)) {
        row(&src[..visible], &mut dst[..visible], to_host);
    }
    true
}

/// Whether a GPU copy from host format `source` into `destination` must
/// convert, and which way: `Some(true)` packs stand-in R11G11B10 into
/// RGB9E5 ([`to_title`]), `Some(false)` unpacks ([`to_host`]). Equal formats
/// (both native where the adapter writes RGB9E5) copy bits as they are.
pub(crate) fn converts(source: u32, destination: u32) -> Option<bool> {
    use crate::resource::{RGB9E5, RGB9E5_STAND_IN};
    match (source, destination) {
        (RGB9E5_STAND_IN, RGB9E5) => Some(true),
        (RGB9E5, RGB9E5_STAND_IN) => Some(false),
        _ => None,
    }
}

/// A region of a converted copy: the source box (`area`, left, top, front,
/// right, bottom, back; the whole `source` subresource if `None`) and the
/// row pitch of its rows in the buffer, which D3D12 aligns to 256 bytes (the
/// rest of each row is padding nothing reads). `None` for an empty box or
/// one outside the `source` extent, or one that, placed at `at`, leaves the
/// `room` extent of the destination subresource: D3D12 would not copy it.
fn plan(area: Option<[u32; 6]>, at: [u32; 3], source: [u32; 3], room: [u32; 3]) -> Option<([u32; 6], u32)> {
    let [w, h, d] = source;
    let area = area.unwrap_or([0, 0, 0, w, h, d]);
    let [left, top, front, right, bottom, back] = area;
    if right <= left || bottom <= top || back <= front || right > w || bottom > h || back > d {
        return None;
    }
    let size = [right - left, bottom - top, back - front];
    if (0..3).any(|i| u64::from(at[i]) + u64::from(size[i]) > u64::from(room[i])) {
        return None;
    }
    Some((area, (size[0] * 4).next_multiple_of(256)))
}

/// Numeric GPU copies between a native RGB9E5 host texture and an
/// R11G11B10 stand-in (`crate::resource::RGB9E5_STAND_IN`). The two formats
/// are in different copy-compatibility groups, so D3D12 cannot copy one into
/// the other: each region goes to a buffer in the source's format, a compute
/// pass of ours converts its words in place by the rules of [`to_host`] and
/// [`to_title`], and the buffer goes on to the destination in its format.
///
/// The texture-to-buffer-to-texture route, a 5.1 shader with its root
/// signature inside and root views needing no descriptor heap follow the
/// FP16-to-RGB9E5 bridge by marquezini (upstream `1f76bd2`, on
/// `rgb-fx-pack-1868`); the conversion here is the integer one of this file.
#[cfg(windows)]
mod gpu {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::s;

    use crate::host::{HostList, host, transition};

    /// The integer conversions of the CPU path in 32-bit words: a value of
    /// either format is `sig << sh` units of 2^-24, `sig` below 2^10.
    const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), UAV(u0)"
cbuffer C : register(b0) { uint width; uint height; uint rowPitch; uint slicePitch; uint toTitle; };
RWByteAddressBuffer words : register(u0);
// v / 2^s to nearest, ties to even; v below 2^31.
uint rounded(uint v, uint s) {
    uint r = v;
    if (s > 31) r = 0;
    else if (s != 0) r = (v + (1u << (s - 1)) - 1 + ((v >> s) & 1)) >> s;
    return r;
}
// A 9-bit mantissa m over shared exponent e as an unsigned float with mb
// mantissa bits and 5 exponent bits.
uint toFloat(uint m, uint e, uint mb) {
    uint top = firstbithigh(m);
    uint highest = top + e;
    uint r = 0;
    // Zero, and R/G below R11's smallest subnormal, stay 0.
    if (m == 0 || (mb == 6 && highest < 4)) {
        r = 0;
    } else if (highest < 10) {
        r = rounded(m << e, 10 - mb);
    } else {
        uint exponent = highest - 9;
        uint mantissa = top >= mb ? rounded(m, top - mb) : m << (mb - top);
        if (mantissa == (2u << mb)) { mantissa >>= 1; exponent += 1; }
        r = exponent >= 31 ? (31u << mb) - 1 : (exponent << mb) | (mantissa - (1u << mb));
    }
    return r;
}
uint toHost(uint w) {
    uint e = w >> 27;
    return toFloat(w & 511, e, 6) | toFloat((w >> 9) & 511, e, 6) << 11 | toFloat((w >> 18) & 511, e, 5) << 22;
}
// An unsigned float channel as (sig, sh); NaN is 0, infinity the largest.
uint2 units(uint bits, uint mb) {
    uint mantissa = bits & ((1u << mb) - 1);
    uint exponent = bits >> mb;
    uint2 r = uint2(mantissa | (1u << mb), exponent + 9 - mb);
    if (exponent == 0) r = uint2(mantissa << (10 - mb), 0);
    if (exponent == 31) r = mantissa != 0 ? uint2(0, 0) : uint2(511, 31);
    return r;
}
uint bitLength(uint2 v) { return v.x == 0 ? 0 : firstbithigh(v.x) + 1 + v.y; }
uint scaled(uint2 v, uint e) { return e <= v.y ? v.x << (v.y - e) : rounded(v.x, e - v.y); }
uint toShared(uint w) {
    uint2 r = units(w & 2047, 6), g = units((w >> 11) & 2047, 6), b = units(w >> 22, 5);
    uint peak = max(bitLength(r), max(bitLength(g), bitLength(b)));
    uint e = peak > 9 ? peak - 9 : 0;
    if (max(scaled(r, e), max(scaled(g, e), scaled(b, e))) > 511) e += 1;
    return scaled(r, e) | scaled(g, e) << 9 | scaled(b, e) << 18 | e << 27;
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    uint at = id.z * slicePitch + id.y * rowPitch + id.x * 4;
    uint w = words.Load(at);
    words.Store(at, toTitle != 0 ? toShared(w) : toHost(w));
}
"#;

    struct Converter {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
    }

    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Converter {}
    // SAFETY: as above.
    unsafe impl Sync for Converter {}

    fn converter() -> Option<&'static Converter> {
        static MADE: OnceLock<Option<Converter>> = OnceLock::new();
        MADE.get_or_init(|| match compile() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no RGB9E5 copy conversion: mixed-format copies keep their destination");
                None
            }
        })
        .as_ref()
    }

    fn compile() -> windows::core::Result<Converter> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("rgb9e5"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
        };
        if let Err(e) = compiled {
            if let Some(errors) = errors {
                // SAFETY: the blob's own bytes.
                let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
                tracing::warn!("RGB9E5 copy conversion: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the blob's own bytes; the root signature is inside the shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            Ok(Converter { root, pso })
        }
    }

    /// Whether mixed copies can be converted (the pass compiled).
    pub(crate) fn available() -> bool {
        converter().is_some()
    }

    /// Converted copies recorded (tests).
    pub(crate) static CONVERTED: AtomicU64 = AtomicU64::new(0);

    /// One copy: host subresource `from` (the texels in `area`, left, top,
    /// front, right, bottom, back; all of it if `None`) into host
    /// subresource `to` at `at`.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Region {
        pub(crate) from: u32,
        pub(crate) to: u32,
        pub(crate) area: Option<[u32; 6]>,
        pub(crate) at: [u32; 3],
    }

    fn subresources(d: &D3D12_RESOURCE_DESC) -> u32 {
        let mips = u32::from(d.MipLevels).max(1);
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { mips } else { mips * u32::from(d.DepthOrArraySize).max(1) }
    }

    /// The size of subresource `index` of a texture described by `d`.
    fn extent(d: &D3D12_RESOURCE_DESC, index: u32) -> [u32; 3] {
        let mip = index % u32::from(d.MipLevels).max(1);
        let depth = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u32::from(d.DepthOrArraySize) } else { 1 };
        let at = |v: u32| (v >> mip).max(1);
        [at(d.Width as u32), at(d.Height), at(depth)]
    }

    /// A default-heap buffer shaders may write, in `COPY_DEST`.
    fn scratch(size: u64) -> Option<ID3D12Resource> {
        let host = host()?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
        };
        made.ok().and(buffer)
    }

    /// Records in `list` the numeric copy of `regions` from host texture
    /// `from` into host texture `to`, one in RGB9E5 and the other in its
    /// stand-in, both already in their copy states. Texels of `to` outside
    /// the regions keep their values; a region outside either texture is
    /// left out, as D3D12 would not copy it. The list keeps its buffer until
    /// it is reset, and sets the title's compute root signature and
    /// pipeline again at its next draw or dispatch ([`HostList::need_rebind`]).
    /// False if nothing could be recorded.
    pub(crate) fn copy(list: &HostList, to: &ID3D12Resource, from: &ID3D12Resource, regions: &[Region]) -> bool {
        let Some(c) = converter() else { return false };
        // SAFETY: live resources.
        let (a, b) = unsafe { (from.GetDesc(), to.GetDesc()) };
        let Some(to_title) = super::converts(a.Format.0 as u32, b.Format.0 as u32) else { return false };
        let to_title = u32::from(to_title);
        let mut planned = Vec::with_capacity(regions.len());
        let mut total = 0u64;
        for r in regions {
            if r.from >= subresources(&a) || r.to >= subresources(&b) {
                continue;
            }
            let Some(([left, top, front, right, bottom, back], pitch)) =
                super::plan(r.area, r.at, extent(&a, r.from), extent(&b, r.to))
            else {
                continue;
            };
            let size = [right - left, bottom - top, back - front];
            total = total.next_multiple_of(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: total,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: a.Format,
                    Width: size[0],
                    Height: size[1],
                    Depth: size[2],
                    RowPitch: pitch,
                },
            };
            total += u64::from(pitch) * u64::from(size[1]) * u64::from(size[2]);
            let area = D3D12_BOX { left, top, front, right, bottom, back };
            planned.push((*r, area, footprint));
        }
        if planned.is_empty() {
            return true;
        }
        let Some(buffer) = scratch(total) else { return false };
        crate::host::name(&buffer, || format!("RGB9E5 copy conversion {total:#x}"));
        list.keep(&buffer);
        let place = |f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        let sub = |t: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(t) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
        };
        let l = &list.list;
        // SAFETY: an open list; our pipeline and buffer, and live textures
        // in the copy states, the regions inside both (checked above).
        unsafe {
            for (r, area, f) in &planned {
                l.CopyTextureRegion(&place(*f), 0, 0, 0, &sub(from, r.from), Some(area));
            }
            l.ResourceBarrier(&[transition(&buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            let gpu = buffer.GetGPUVirtualAddress();
            for (_, _, f) in &planned {
                let s = f.Footprint;
                let constants = [s.Width, s.Height, s.RowPitch, s.RowPitch * s.Height, to_title];
                l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                l.SetComputeRootUnorderedAccessView(1, gpu + f.Offset);
                l.Dispatch(s.Width.div_ceil(8), s.Height.div_ceil(8), s.Depth);
            }
            l.ResourceBarrier(&[transition(&buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (r, _, f) in &planned {
                let mut f = *f;
                f.Footprint.Format = b.Format;
                l.CopyTextureRegion(&sub(to, r.to), r.at[0], r.at[1], r.at[2], &place(f), None);
            }
        }
        list.need_rebind();
        CONVERTED.fetch_add(1, Ordering::Relaxed);
        true
    }
}

#[cfg(all(windows, test))]
pub(crate) use gpu::CONVERTED;
#[cfg(windows)]
pub(crate) use gpu::{Region, available, copy};

/// Three source-authored colours which both formats represent exactly.
#[cfg(test)]
pub(crate) const TEST_COLOURS: [(u32, [f32; 4]); 3] = [
    ((18 << 27) | (8 << 18) | (32 << 9) | 256, [4.0, 0.5, 0.125, 1.0]),
    ((17 << 27) | (64 << 18) | (256 << 9) | 32, [0.25, 2.0, 0.5, 1.0]),
    ((19 << 27) | (256 << 18) | (16 << 9) | 2, [0.0625, 0.5, 8.0, 1.0]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb9e5_footprint_converts_visible_words_only() {
        let mut src = [0xcc; 32];
        let mut dst = [0xa5; 32];
        let rows = [TEST_COLOURS[0].0, TEST_COLOURS[1].0];
        for (y, word) in rows.iter().enumerate() {
            src[y * 16..y * 16 + 4].copy_from_slice(&word.to_le_bytes());
        }
        assert!(footprint(&src[..20], &mut dst[..20], 1, 16));
        let mut expected = [0xa5; 32];
        for (y, word) in rows.iter().enumerate() {
            expected[y * 16..y * 16 + 4].copy_from_slice(&to_host(*word).to_le_bytes());
        }
        assert_eq!(dst, expected);
        assert!(!footprint(&src[..18], &mut dst[..18], 1, 16));
        assert!(!footprint(&src, &mut dst, 2, 4));
    }

    #[test]
    fn rgb9e5_converts_independent_channels_and_packs_one_shared_exponent() {
        // Colours are (4, 0.5, 0.125), (0.25, 2, 0.5), and (1/16, 0.5, 8).
        // All components have an exact representation in both formats.
        let cases: [(u32, u32); 3] = [
            ((18 << 27) | (8 << 18) | (32 << 9) | 256, (12 << 27) | (14 << 17) | (17 << 6)),
            ((17 << 27) | (64 << 18) | (256 << 9) | 32, (14 << 27) | (16 << 17) | (13 << 6)),
            ((19 << 27) | (256 << 18) | (16 << 9) | 2, (18 << 27) | (14 << 17) | (11 << 6)),
        ];
        for (shared, native) in cases {
            assert_eq!(to_host(shared), native);
            assert_eq!(to_title(native), shared);
        }
        assert_eq!(to_host(0), 0);
        assert_eq!(to_title(0), 0);
        assert_eq!(to_title(0x7c1), 0); // One R11 NaN, zero green and blue.
        assert_eq!(to_title(0x7c0), (31 << 27) | 511); // Red infinity saturates.
        let max = to_host(u32::MAX);
        assert_eq!((max & 2047, (max >> 11) & 2047, max >> 22), (1983, 1983, 991));
        assert_eq!(to_host(16), 1); // Smallest nonzero red in R11.
        assert_eq!(to_title(1), 16);
        // 257/64 quantizes to 4 in R11; the canonical packed result records that loss.
        assert_eq!(to_title(to_host((18 << 27) | 257)), (18 << 27) | 256);
    }

    /// Converted copies take the boxes D3D12 copies: whole subresources,
    /// sub-rectangles to an offset (rows padded to 256 bytes), and nothing
    /// for empty boxes or ones leaving either subresource.
    #[test]
    fn rgb9e5_copy_regions_keep_their_box_offset_and_padded_rows() {
        let level = [8, 8, 1];
        assert_eq!(plan(None, [0; 3], level, level), Some(([0, 0, 0, 8, 8, 1], 256)));
        assert_eq!(plan(None, [0; 3], [65, 3, 1], [65, 3, 1]), Some(([0, 0, 0, 65, 3, 1], 512)));
        assert_eq!(plan(None, [0; 3], [64, 1, 1], [64, 1, 1]).unwrap().1, 256, "64 texels fill a row");
        let corner = [2, 4, 0, 5, 8, 1];
        assert_eq!(plan(Some(corner), [5, 3, 0], level, level), Some((corner, 256)), "3x4 into the far corner");
        assert_eq!(plan(Some(corner), [6, 3, 0], level, level), None, "one column past the destination");
        assert_eq!(plan(Some(corner), [5, 5, 0], level, level), None, "one row past the destination");
        assert_eq!(plan(Some([0, 0, 0, 9, 1, 1]), [0; 3], level, [16, 16, 1]), None, "past the source");
        assert_eq!(plan(Some([3, 0, 0, 3, 1, 1]), [0; 3], level, level), None, "empty");
        assert_eq!(plan(Some([0, 0, 1, 1, 1, 2]), [0, 0, 3], [4, 4, 4], [4, 4, 4]).unwrap().0[2], 1, "last slice");
        assert_eq!(plan(Some([0, 0, 1, 1, 1, 2]), [0, 0, 4], [4, 4, 4], [4, 4, 4]), None, "past a 3D destination");
        assert_eq!(plan(Some([0, 0, 1, 1, 1, 3]), [0, 0, 2], [4, 4, 4], [4, 4, 4]), Some(([0, 0, 1, 1, 1, 3], 256)));
        assert_eq!(plan(None, [u32::MAX, 0, 0], level, level), None, "offsets do not wrap");
    }

    #[test]
    fn rgb9e5_copies_convert_only_between_the_two_packed_formats() {
        use crate::resource::{RGB9E5, RGB9E5_STAND_IN};
        assert_eq!(converts(RGB9E5_STAND_IN, RGB9E5), Some(true));
        assert_eq!(converts(RGB9E5, RGB9E5_STAND_IN), Some(false));
        for (a, b) in [(RGB9E5, RGB9E5), (RGB9E5_STAND_IN, RGB9E5_STAND_IN), (42, RGB9E5), (RGB9E5_STAND_IN, 41)] {
            assert_eq!(converts(a, b), None, "{a} into {b}");
        }
    }

    /// The GPU pass of [`copy`] word for word against the CPU oracle, in
    /// both directions: every (mantissa, exponent) of each shared-exponent
    /// channel and every 11- and 10-bit float of each RGB channel (NaN,
    /// infinity and subnormals among them), then pseudo-random words. The
    /// pass leaves the list to set the title's compute state again.
    #[cfg(windows)]
    #[test]
    fn gpu_rgb9e5_copy_conversion_matches_the_cpu_oracle_both_ways() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let host = crate::host::host().expect("RGB9E5 copy conversion requires a D3D12 adapter");
        const W: u32 = 256;
        const H: u32 = 128;
        let n = (W * H) as usize;
        let mut seed = 0x9e37_79b9u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let half = n as u32 / 2;
        let shared: Vec<u32> = (0..n as u32)
            .map(|i| match i < half {
                true => {
                    let (m, e) = (i % 512, i / 512 % 32);
                    (e << 27) | m | (((m * 7 + 3) % 512) << 9) | (((m * 13 + 5) % 512) << 18)
                }
                false => random(),
            })
            .collect();
        let floats: Vec<u32> = (0..n as u32)
            .map(|i| match i < half {
                true => (i % 2048) | (((i * 7 + 3) % 2048) << 11) | (((i * 13 + 5) % 1024) << 22),
                false => random(),
            })
            .collect();
        let texture = |format: DXGI_FORMAT| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(W),
                Height: H,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut t: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut t) }
                .unwrap();
            t.unwrap()
        };
        let (packed, native) = (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R11G11B10_FLOAT);
        // (source, destination) for each direction.
        let pairs = [(texture(packed), texture(native)), (texture(native), texture(packed))];
        let pitch = W * 4;
        let bytes = u64::from(pitch * H);
        let list = host.new_list().unwrap();
        let (upload, data) = list.upload(host, 2 * bytes).unwrap();
        let (readback, read) = list.readback(host, 2 * bytes).unwrap();
        let words = data.cast::<u32>();
        for (i, word) in shared.iter().chain(&floats).enumerate() {
            // SAFETY: 2 * n words of mapped upload memory.
            unsafe { words.add(i).write(*word) };
        }
        let place = |buffer: &ID3D12Resource, k: u64, format| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: k * bytes,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: W, Height: H, Depth: 1, RowPitch: pitch },
                },
            },
        };
        let sub = |t: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(t) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        use D3D12_RESOURCE_STATE_COPY_DEST as DEST;
        use D3D12_RESOURCE_STATE_COPY_SOURCE as SOURCE;
        let _ = host.log_messages();
        let whole = [Region { from: 0, to: 0, area: None, at: [0; 3] }];
        for (k, (from, to)) in pairs.iter().enumerate() {
            // SAFETY: an open list; live resources in the states named.
            unsafe {
                let (a, b) = (from.GetDesc().Format, to.GetDesc().Format);
                list.list.CopyTextureRegion(&sub(from), 0, 0, 0, &place(&upload, k as u64, a), None);
                list.list.ResourceBarrier(&[crate::host::transition(from, DEST, SOURCE)]);
                assert!(!list.take_rebind());
                assert!(copy(&list, to, from, &whole), "direction {k} recorded");
                assert!(list.take_rebind(), "the title's compute state is set again after the pass");
                list.list.ResourceBarrier(&[crate::host::transition(to, DEST, SOURCE)]);
                list.list.CopyTextureRegion(&place(&readback, k as u64, b), 0, 0, 0, &sub(to), None);
            }
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None, "debug layer");
        // SAFETY: the list ran; 2 * n words of mapped readback memory.
        let out = unsafe { std::slice::from_raw_parts(read.cast::<u32>(), 2 * n) };
        for (i, &word) in shared.iter().enumerate() {
            assert_eq!(out[i], to_host(word), "to_host({word:#010x}) at {i}");
        }
        for (i, &word) in floats.iter().enumerate() {
            assert_eq!(out[n + i], to_title(word), "to_title({word:#010x}) at {i}");
        }
    }
}
