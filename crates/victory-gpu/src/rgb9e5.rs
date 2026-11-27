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
}
