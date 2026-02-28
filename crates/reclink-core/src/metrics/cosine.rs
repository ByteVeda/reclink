//! Cosine similarity on character n-grams.
//!
//! After hash map lookup, extracts matching count pairs into aligned vectors,
//! then uses SIMD dot product (SSE2/AVX2/NEON) for the inner product and norms.

use ahash::AHashMap;

use crate::metrics::SimilarityMetric;

/// Cosine similarity computes the cosine of the angle between two n-gram
/// frequency vectors derived from the input strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cosine {
    /// Size of character n-grams. Default: 2 (bigrams).
    pub n: usize,
}

impl Default for Cosine {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl SimilarityMetric for Cosine {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        cosine_similarity(a, b, self.n)
    }
}

/// Builds a character n-gram frequency map.
fn ngram_freq(s: &str, n: usize) -> AHashMap<Vec<char>, u32> {
    let chars: Vec<char> = s.chars().collect();
    let mut freq = AHashMap::new();
    if chars.len() < n {
        return freq;
    }
    for window in chars.windows(n) {
        *freq.entry(window.to_vec()).or_insert(0) += 1;
    }
    freq
}

/// Scalar dot product and norms computation.
fn dot_norms_scalar(
    freq_a: &AHashMap<Vec<char>, u32>,
    freq_b: &AHashMap<Vec<char>, u32>,
) -> (u64, u64, u64) {
    let mut dot = 0u64;
    let mut norm_a = 0u64;
    let mut norm_b = 0u64;

    for (ngram, &count_a) in freq_a {
        norm_a += (count_a as u64) * (count_a as u64);
        if let Some(&count_b) = freq_b.get(ngram) {
            dot += (count_a as u64) * (count_b as u64);
        }
    }
    for &count_b in freq_b.values() {
        norm_b += (count_b as u64) * (count_b as u64);
    }

    (dot, norm_a, norm_b)
}

/// SIMD dot product for paired count vectors (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(a_counts: &[u32], b_counts: &[u32]) -> u64 {
    use std::arch::x86_64::*;

    let n = a_counts.len();
    let mut i = 0;
    let mut sum = _mm_setzero_si128();

    // Process 4 u32 pairs at a time using _mm_mul_epu32 (multiplies lanes 0,2)
    while i + 4 <= n {
        let va = _mm_loadu_si128(a_counts.as_ptr().add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b_counts.as_ptr().add(i) as *const __m128i);

        // _mm_mul_epu32 multiplies lanes 0 and 2 (32-bit → 64-bit)
        let prod_02 = _mm_mul_epu32(va, vb);
        // Shift right by 4 bytes to get lanes 1,3 into position 0,2
        let va_13 = _mm_srli_si128(va, 4);
        let vb_13 = _mm_srli_si128(vb, 4);
        let prod_13 = _mm_mul_epu32(va_13, vb_13);

        sum = _mm_add_epi64(sum, prod_02);
        sum = _mm_add_epi64(sum, prod_13);
        i += 4;
    }

    // Extract sum
    let mut result: u64 = 0;
    let mut tmp = [0u64; 2];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum);
    result += tmp[0] + tmp[1];

    // Scalar tail
    for j in i..n {
        result += (a_counts[j] as u64) * (b_counts[j] as u64);
    }

    result
}

/// SIMD sum of squares for a count vector (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sum_squares_sse2(counts: &[u32]) -> u64 {
    use std::arch::x86_64::*;

    let n = counts.len();
    let mut i = 0;
    let mut sum = _mm_setzero_si128();

    while i + 4 <= n {
        let v = _mm_loadu_si128(counts.as_ptr().add(i) as *const __m128i);
        let prod_02 = _mm_mul_epu32(v, v);
        let v_13 = _mm_srli_si128(v, 4);
        let prod_13 = _mm_mul_epu32(v_13, v_13);
        sum = _mm_add_epi64(sum, prod_02);
        sum = _mm_add_epi64(sum, prod_13);
        i += 4;
    }

    let mut result: u64 = 0;
    let mut tmp = [0u64; 2];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum);
    result += tmp[0] + tmp[1];

    for &c in &counts[i..n] {
        result += (c as u64) * (c as u64);
    }

    result
}

/// Computes cosine similarity between character n-gram vectors of two strings.
#[must_use]
pub fn cosine_similarity(a: &str, b: &str, n: usize) -> f64 {
    let freq_a = ngram_freq(a, n);
    let freq_b = ngram_freq(b, n);

    if freq_a.is_empty() && freq_b.is_empty() {
        return 1.0;
    }
    if freq_a.is_empty() || freq_b.is_empty() {
        return 0.0;
    }

    // For small frequency maps, use scalar path directly (SIMD overhead not worth it)
    let use_simd = freq_a.len() >= 16;

    let (dot, norm_a, norm_b) = if use_simd {
        cosine_simd_path(&freq_a, &freq_b)
    } else {
        dot_norms_scalar(&freq_a, &freq_b)
    };

    if norm_a == 0 || norm_b == 0 {
        return 0.0;
    }

    dot as f64 / ((norm_a as f64).sqrt() * (norm_b as f64).sqrt())
}

/// SIMD-accelerated path: extract count pairs into aligned vectors, then SIMD dot/norm.
fn cosine_simd_path(
    freq_a: &AHashMap<Vec<char>, u32>,
    freq_b: &AHashMap<Vec<char>, u32>,
) -> (u64, u64, u64) {
    use crate::metrics::simd_util::AlignedVec;

    // Extract paired counts for shared n-grams into 32-byte aligned buffers
    let mut a_shared = AlignedVec::<u32>::with_capacity(freq_a.len());
    let mut b_shared = AlignedVec::<u32>::with_capacity(freq_a.len());
    let mut a_all = AlignedVec::<u32>::with_capacity(freq_a.len());
    let mut b_all = AlignedVec::<u32>::with_capacity(freq_b.len());

    for (ngram, &count_a) in freq_a {
        a_all.push(count_a);
        if let Some(&count_b) = freq_b.get(ngram) {
            a_shared.push(count_a);
            b_shared.push(count_b);
        }
    }
    for &count_b in freq_b.values() {
        b_all.push(count_b);
    }

    let (dot, norm_a, norm_b) = crate::metrics::simd_util::dispatch_simd!(
        avx2: unsafe {
            (
                dot_product_sse2(&a_shared, &b_shared),
                sum_squares_sse2(&a_all),
                sum_squares_sse2(&b_all),
            )
        },
        sse2: unsafe {
            (
                dot_product_sse2(&a_shared, &b_shared),
                sum_squares_sse2(&a_all),
                sum_squares_sse2(&b_all),
            )
        },
        neon: {
            let dot = a_shared.iter().zip(b_shared.iter())
                .map(|(&a, &b)| (a as u64) * (b as u64)).sum::<u64>();
            let na = a_all.iter().map(|&c| (c as u64) * (c as u64)).sum::<u64>();
            let nb = b_all.iter().map(|&c| (c as u64) * (c as u64)).sum::<u64>();
            (dot, na, nb)
        },
        scalar: {
            let dot = a_shared.iter().zip(b_shared.iter())
                .map(|(&a, &b)| (a as u64) * (b as u64)).sum::<u64>();
            let na = a_all.iter().map(|&c| (c as u64) * (c as u64)).sum::<u64>();
            let nb = b_all.iter().map(|&c| (c as u64) * (c as u64)).sum::<u64>();
            (dot, na, nb)
        },
    );

    (dot, norm_a, norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(cosine_similarity("hello", "hello", 2), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(cosine_similarity("", "", 2), 1.0));
        assert!(approx_eq(cosine_similarity("abc", "", 2), 0.0));
    }

    #[test]
    fn completely_different() {
        assert!(approx_eq(cosine_similarity("ab", "cd", 2), 0.0));
    }

    #[test]
    fn partial_overlap() {
        let sim = cosine_similarity("night", "nacht", 2);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn symmetry() {
        let a = cosine_similarity("abc", "bcd", 2);
        let b = cosine_similarity("bcd", "abc", 2);
        assert!(approx_eq(a, b));
    }

    #[test]
    fn short_strings() {
        // Single chars produce no bigrams, so both-empty returns 1.0
        assert!(approx_eq(cosine_similarity("a", "a", 2), 1.0));
        // Both "a" and "b" produce empty bigram sets -> treated as equal (both empty)
        assert!(approx_eq(cosine_similarity("a", "b", 2), 1.0));
    }

    #[test]
    fn long_strings() {
        // Exercise SIMD path (>= 16 n-grams)
        let a: String = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = (0..100).map(|i| (b'a' + ((i + 1) % 26)) as char).collect();
        let sim = cosine_similarity(&a, &b, 2);
        assert!(sim > 0.0 && sim < 1.0);
    }
}
