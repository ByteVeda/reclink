//! Longest Common Subsequence (LCS) metric.
//!
//! - ≤64 chars: Allison-Dix bit-parallel O(n·⌈m/64⌉)
//! - >64 chars: multi-block bit-parallel with carry propagation

use ahash::AHashMap;

use crate::metrics::SimilarityMetric;

/// LCS computes the longest common subsequence length between two strings
/// and derives a normalized similarity score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lcs;

impl SimilarityMetric for Lcs {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        lcs_similarity(a, b)
    }
}

/// Allison-Dix bit-parallel LCS for patterns ≤ 64 chars.
///
/// Time: O(n), Space: O(|Σ|) where Σ is the alphabet.
fn lcs_bit_parallel(a_chars: &[char], b_chars: &[char]) -> usize {
    let m = a_chars.len();
    debug_assert!(m <= 64);

    // Build pattern-match bitmask: PM[c] has bit i set iff a[i] == c
    let mut pm: AHashMap<char, u64> = AHashMap::new();
    for (i, &c) in a_chars.iter().enumerate() {
        *pm.entry(c).or_insert(0u64) |= 1u64 << i;
    }

    let mut s: u64 = 0;

    for &bc in b_chars {
        let x = pm.get(&bc).copied().unwrap_or(0) | s;
        let y = x.wrapping_sub((s << 1) | 1);
        s = x & (x ^ y);
    }

    s.count_ones() as usize
}

/// Multi-block bit-parallel LCS for patterns > 64 chars.
///
/// Splits pattern into ⌈m/64⌉ blocks with carry propagation for the `(S << 1) | 1`
/// operation across block boundaries.
///
/// Time: O(n · ⌈m/64⌉)
#[allow(clippy::needless_range_loop)]
fn lcs_multi_block(a_chars: &[char], b_chars: &[char]) -> usize {
    use crate::metrics::scratch::LCS_SCRATCH;

    let m = a_chars.len();
    let num_blocks = m.div_ceil(64);

    LCS_SCRATCH.with_borrow_mut(|scratch| {
        scratch.reset(num_blocks);

        // Build pattern-match bitmask per block
        for (i, &c) in a_chars.iter().enumerate() {
            let block = i / 64;
            let bit = i % 64;
            let entry = scratch
                .pm
                .entry(c)
                .or_insert_with(|| vec![0u64; num_blocks]);
            entry[block] |= 1u64 << bit;
        }

        for &bc in b_chars {
            let pm_bc = scratch.pm.get(&bc).unwrap_or(&scratch.empty_pm);

            let mut carry_shift: u64 = 1;
            let mut carry_sub: u64 = 0;

            for k in 0..num_blocks {
                let x = pm_bc[k] | scratch.s[k];

                let s_shifted = (scratch.s[k] << 1) | carry_shift;
                carry_shift = scratch.s[k] >> 63;

                let y = x.wrapping_sub(s_shifted).wrapping_sub(carry_sub);
                carry_sub = if x < s_shifted || (carry_sub > 0 && x.wrapping_sub(s_shifted) == 0) {
                    1
                } else {
                    0
                };

                scratch.s[k] = x & (x ^ y);
            }
        }

        scratch.s.iter().map(|&v| v.count_ones() as usize).sum()
    })
}

/// Computes the length of the longest common subsequence.
///
/// Uses Allison-Dix bit-parallel: O(n) for ≤64 chars, O(n·⌈m/64⌉) for longer.
#[must_use]
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() || b_chars.is_empty() {
        return 0;
    }

    // Use shorter string as the pattern (for PM bitmask)
    let (pattern, text) = if a_chars.len() <= b_chars.len() {
        (a_chars, b_chars)
    } else {
        (b_chars, a_chars)
    };

    if pattern.len() <= 64 {
        lcs_bit_parallel(&pattern, &text)
    } else {
        lcs_multi_block(&pattern, &text)
    }
}

/// Computes normalized LCS similarity: `2 * lcs_len / (len_a + len_b)`.
///
/// Returns 1.0 for two empty strings, 0.0 if one is empty.
#[must_use]
pub fn lcs_similarity(a: &str, b: &str) -> f64 {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    let total = a_len + b_len;

    if total == 0 {
        return 1.0;
    }

    let lcs = lcs_length(a, b);
    (2 * lcs) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert_eq!(lcs_length("hello", "hello"), 5);
        assert!(approx_eq(lcs_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert_eq!(lcs_length("", ""), 0);
        assert_eq!(lcs_length("abc", ""), 0);
        assert_eq!(lcs_length("", "abc"), 0);
        assert!(approx_eq(lcs_similarity("", ""), 1.0));
    }

    #[test]
    fn known_values() {
        // "abcde" and "ace" -> LCS is "ace" = 3
        assert_eq!(lcs_length("abcde", "ace"), 3);
        // similarity = 2*3 / (5+3) = 0.75
        assert!(approx_eq(lcs_similarity("abcde", "ace"), 0.75));
    }

    #[test]
    fn symmetry() {
        assert_eq!(lcs_length("abc", "bca"), lcs_length("bca", "abc"));
    }

    #[test]
    fn no_common() {
        assert_eq!(lcs_length("abc", "xyz"), 0);
        assert!(approx_eq(lcs_similarity("abc", "xyz"), 0.0));
    }

    #[test]
    fn unicode() {
        assert_eq!(lcs_length("cafe\u{0301}", "cafe"), 4);
    }

    #[test]
    fn boundary_64_chars() {
        let a: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();
        // LCS should be 63 (all except the changed char)
        assert_eq!(lcs_length(&a, &b), 63);
    }

    #[test]
    fn multi_block_100_chars() {
        let a: String = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 20 || i == 50 || i == 80 {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        // LCS should be 97 (all except the 3 changed chars)
        assert_eq!(lcs_length(&a, &b), 97);
    }

    #[test]
    fn multi_block_identical() {
        let a: String = (0..200).map(|i| (b'a' + (i % 26)) as char).collect();
        assert_eq!(lcs_length(&a, &a), 200);
    }

    #[test]
    fn multi_block_no_common() {
        let a: String = "a".repeat(100);
        let b: String = "b".repeat(100);
        assert_eq!(lcs_length(&a, &b), 0);
    }

    #[test]
    fn boundary_consistency() {
        // 64-char (single block) vs 65-char (multi-block) with same prefix
        let base64: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let mod64: String = base64
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();

        let bp = lcs_length(&base64, &mod64);

        let base65 = base64.clone() + "x";
        let mod65 = mod64.clone() + "x";
        let mb = lcs_length(&base65, &mod65);

        assert_eq!(mb, bp + 1); // multi-block has one more matching char
    }
}
