//! Hamming distance for equal-length strings.
//!
//! Uses SIMD (SSE2/AVX2/NEON) packed u32 comparison when available,
//! with scalar fallback.

use crate::error::{ReclinkError, Result};
use crate::metrics::DistanceMetric;

/// Hamming distance counts the number of positions where corresponding
/// characters differ. Requires strings of equal length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Hamming;

impl DistanceMetric for Hamming {
    fn distance(&self, a: &str, b: &str) -> Result<usize> {
        hamming_distance(a, b)
    }
}

/// Scalar Hamming distance (fallback for non-SIMD platforms).
#[allow(dead_code)]
fn hamming_scalar(a: &[char], b: &[char]) -> usize {
    a.iter().zip(b.iter()).filter(|(a, b)| a != b).count()
}

/// SIMD Hamming using SSE2: compare 4 u32s at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hamming_sse2(a: &[char], b: &[char]) -> usize {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut mismatches = 0usize;
    let mut i = 0;

    // Process 4 chars (u32) at a time
    let a_ptr = a.as_ptr() as *const u32;
    let b_ptr = b.as_ptr() as *const u32;

    while i + 4 <= n {
        let va = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b_ptr.add(i) as *const __m128i);
        let eq = _mm_cmpeq_epi32(va, vb);
        let mask = _mm_movemask_ps(_mm_castsi128_ps(eq)) as u32;
        mismatches += (4 - mask.count_ones()) as usize;
        i += 4;
    }

    // Scalar tail
    for j in i..n {
        if a[j] != b[j] {
            mismatches += 1;
        }
    }

    mismatches
}

/// SIMD Hamming using AVX2: compare 8 u32s at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(a: &[char], b: &[char]) -> usize {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut mismatches = 0usize;
    let mut i = 0;

    let a_ptr = a.as_ptr() as *const u32;
    let b_ptr = b.as_ptr() as *const u32;

    // Process 8 chars (u32) at a time with AVX2
    while i + 8 <= n {
        let va = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(i) as *const __m256i);
        let eq = _mm256_cmpeq_epi32(va, vb);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32;
        mismatches += (8 - mask.count_ones()) as usize;
        i += 8;
    }

    // SSE2 tail: 4 at a time
    while i + 4 <= n {
        let va = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b_ptr.add(i) as *const __m128i);
        let eq = _mm_cmpeq_epi32(va, vb);
        let mask = _mm_movemask_ps(_mm_castsi128_ps(eq)) as u32;
        mismatches += (4 - mask.count_ones()) as usize;
        i += 4;
    }

    // Scalar tail
    for j in i..n {
        if a[j] != b[j] {
            mismatches += 1;
        }
    }

    mismatches
}

/// SIMD Hamming using NEON: compare 4 u32s at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &[char], b: &[char]) -> usize {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut mismatches = 0usize;
    let mut i = 0;

    let a_ptr = a.as_ptr() as *const u32;
    let b_ptr = b.as_ptr() as *const u32;

    while i + 4 <= n {
        let va = vld1q_u32(a_ptr.add(i));
        let vb = vld1q_u32(b_ptr.add(i));
        // vceqq_u32 returns 0xFFFFFFFF for equal, 0 for not equal
        let eq = vceqq_u32(va, vb);
        // Invert: 0xFFFFFFFF for not equal
        let neq = vmvnq_u32(eq);
        // Shift right by 31 to get 1 for not equal, 0 for equal
        let ones = vshrq_n_u32(neq, 31);
        mismatches += vaddvq_u32(ones) as usize;
        i += 4;
    }

    for j in i..n {
        if a[j] != b[j] {
            mismatches += 1;
        }
    }

    mismatches
}

/// Computes Hamming distance between two equal-length strings.
///
/// Returns an error if the strings have different lengths.
/// Uses SIMD when available (AVX2/SSE2/NEON).
pub fn hamming_distance(a: &str, b: &str) -> Result<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.len() != b_chars.len() {
        return Err(ReclinkError::UnequalLength {
            a: a_chars.len(),
            b: b_chars.len(),
        });
    }

    let result = crate::metrics::simd_util::dispatch_simd!(
        avx2: unsafe { hamming_avx2(&a_chars, &b_chars) },
        sse2: unsafe { hamming_sse2(&a_chars, &b_chars) },
        neon: unsafe { hamming_neon(&a_chars, &b_chars) },
        scalar: hamming_scalar(&a_chars, &b_chars),
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert_eq!(hamming_distance("hello", "hello").unwrap(), 0);
    }

    #[test]
    fn empty() {
        assert_eq!(hamming_distance("", "").unwrap(), 0);
    }

    #[test]
    fn known_values() {
        assert_eq!(hamming_distance("karolin", "kathrin").unwrap(), 3);
        assert_eq!(hamming_distance("1011101", "1001001").unwrap(), 2);
    }

    #[test]
    fn unequal_length_error() {
        assert!(hamming_distance("abc", "ab").is_err());
    }

    #[test]
    fn unicode() {
        assert_eq!(hamming_distance("café", "cafè").unwrap(), 1);
    }

    #[test]
    fn long_strings_simd() {
        let a: String = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 10 || i == 50 || i == 99 {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        assert_eq!(hamming_distance(&a, &b).unwrap(), 3);
    }

    #[test]
    fn all_different() {
        let a: String = "a".repeat(200);
        let b: String = "b".repeat(200);
        assert_eq!(hamming_distance(&a, &b).unwrap(), 200);
    }

    #[test]
    fn simd_vs_scalar() {
        // Verify SIMD and scalar agree at various lengths
        for len in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100] {
            let a: Vec<char> = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let b: Vec<char> = a
                .iter()
                .enumerate()
                .map(|(i, &c)| if i % 7 == 0 { 'Z' } else { c })
                .collect();
            let expected = hamming_scalar(&a, &b);
            let a_str: String = a.into_iter().collect();
            let b_str: String = b.into_iter().collect();
            assert_eq!(
                hamming_distance(&a_str, &b_str).unwrap(),
                expected,
                "Mismatch at length {len}"
            );
        }
    }
}
