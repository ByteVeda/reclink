//! Levenshtein edit distance with Myers' bit-vector acceleration.
//!
//! - ≤64 chars: single-block Myers O(n)
//! - >64 chars: multi-block Myers O(n·⌈m/64⌉)

use ahash::AHashMap;

use crate::error::Result;
use crate::metrics::DistanceMetric;

/// Levenshtein distance computes the minimum number of single-character edits
/// (insertions, deletions, substitutions) needed to transform one string into another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Levenshtein;

impl DistanceMetric for Levenshtein {
    fn distance(&self, a: &str, b: &str) -> Result<usize> {
        Ok(levenshtein_distance(a, b))
    }
}

/// Myers' bit-parallel algorithm for edit distance (single u64 block).
///
/// Processes one column per text character using bitwise operations on u64,
/// giving O(n) time for patterns up to 64 characters.
///
/// Reference: Gene Myers, "A Fast Bit-Vector Algorithm for Approximate String
/// Matching Based on Dynamic Programming" (1999).
///
/// Requires: `pattern.len() <= 64` and `pattern.len() <= text.len()`.
fn myers_bit_parallel(pattern: &[char], text: &[char]) -> usize {
    let m = pattern.len();
    debug_assert!(m <= 64);
    debug_assert!(m <= text.len());

    // Precompute pattern-match bitmasks: PM[c] has bit i set iff pattern[i] == c
    let mut pm = AHashMap::new();
    for (i, &c) in pattern.iter().enumerate() {
        *pm.entry(c).or_insert(0u64) |= 1u64 << i;
    }

    let mask = 1u64 << (m - 1);
    let mut vp: u64 = !0u64;
    let mut vn: u64 = 0u64;
    let mut score = m;

    for &tc in text {
        let pm_j = pm.get(&tc).copied().unwrap_or(0);
        let d0 = (((pm_j & vp).wrapping_add(vp)) ^ vp) | pm_j | vn;
        let hp = vn | !(d0 | vp);
        let hn = d0 & vp;

        if hp & mask != 0 {
            score += 1;
        }
        if hn & mask != 0 {
            score -= 1;
        }

        let x = (hp << 1) | 1;
        vp = (hn << 1) | !(d0 | x);
        vn = d0 & x;
    }

    score
}

/// Multi-block Myers' bit-parallel algorithm for patterns > 64 chars.
///
/// Splits the pattern into ⌈m/64⌉ blocks, each with its own VP/VN/PM bitmasks.
/// Three carry chains propagate between blocks:
/// - Addition carry from `(X & VP) + VP`
/// - HP shift carry from `HP << 1`
/// - HN shift carry from `HN << 1`
///
/// Time: O(n · ⌈m/64⌉).
#[allow(clippy::needless_range_loop)]
fn myers_multi_block(pattern: &[char], text: &[char]) -> usize {
    use crate::metrics::scratch::LEV_SCRATCH;

    let m = pattern.len();
    let num_blocks = m.div_ceil(64);

    LEV_SCRATCH.with_borrow_mut(|scratch| {
        scratch.reset(num_blocks);

        // Precompute pattern-match bitmasks per block
        for (i, &c) in pattern.iter().enumerate() {
            let block = i / 64;
            let bit = i % 64;
            let entry = scratch
                .pm
                .entry(c)
                .or_insert_with(|| vec![0u64; num_blocks]);
            entry[block] |= 1u64 << bit;
        }

        // Last block mask
        let last_block_bits = m % 64;
        let last_mask = if last_block_bits == 0 {
            1u64 << 63
        } else {
            1u64 << (last_block_bits - 1)
        };

        let mut score = m;

        for &tc in text {
            let pm_j = scratch.pm.get(&tc).unwrap_or(&scratch.empty_pm);

            let mut carry_add: u64 = 0;
            let mut carry_hp: u64 = 1;
            let mut carry_hn: u64 = 0;

            for k in 0..num_blocks {
                let eq = pm_j[k];
                let xv = eq | scratch.vn[k];
                let xh = xv & scratch.vp[k];

                let sum = xh.wrapping_add(scratch.vp[k]);
                let c1 = (sum < xh) as u64;
                let sum2 = sum.wrapping_add(carry_add);
                let c2 = (sum2 < sum) as u64;
                carry_add = c1 + c2;

                let d0 = (sum2 ^ scratch.vp[k]) | eq | scratch.vn[k];
                let hp = scratch.vn[k] | !(d0 | scratch.vp[k]);
                let hn = d0 & scratch.vp[k];

                if k == num_blocks - 1 {
                    if hp & last_mask != 0 {
                        score += 1;
                    }
                    if hn & last_mask != 0 {
                        score -= 1;
                    }
                }

                let next_carry_hp = hp >> 63;
                let next_carry_hn = hn >> 63;

                let hp_shifted = (hp << 1) | carry_hp;
                let hn_shifted = (hn << 1) | carry_hn;

                scratch.vp[k] = hn_shifted | !(d0 | hp_shifted);
                scratch.vn[k] = d0 & hp_shifted;

                carry_hp = next_carry_hp;
                carry_hn = next_carry_hn;
            }
        }

        score
    })
}

/// Computes Levenshtein distance using Myers' bit-parallel algorithm for strings ≤ 64 chars,
/// multi-block Myers for longer strings.
///
/// Time: O(n) for pattern ≤ 64, O(n·⌈m/64⌉) otherwise. Space: O(⌈m/64⌉)
#[must_use]
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    if short.is_empty() {
        return long.len();
    }

    if short.len() <= 64 {
        return myers_bit_parallel(short, long);
    }

    myers_multi_block(short, long)
}

/// Computes Levenshtein distance with early termination.
///
/// Returns `None` if the distance exceeds `max_distance`, `Some(distance)` otherwise.
/// Uses Myers' bit-parallel for short strings, length filtering and row-minimum pruning otherwise.
#[must_use]
pub fn levenshtein_distance_threshold(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    // Length filter: if length difference alone exceeds threshold, bail out
    if long_len - short_len > max_distance {
        return None;
    }

    if short_len == 0 {
        return Some(long_len);
    }

    if short_len <= 64 {
        let dist = myers_bit_parallel(short, long);
        return if dist > max_distance {
            None
        } else {
            Some(dist)
        };
    }

    // For longer strings, try multi-block Myers first (no threshold, but fast)
    let dist = myers_multi_block(short, long);
    if dist > max_distance {
        None
    } else {
        Some(dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn known_values() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("flaw", "lawn"), 2);
    }

    #[test]
    fn symmetry() {
        assert_eq!(
            levenshtein_distance("abc", "def"),
            levenshtein_distance("def", "abc")
        );
    }

    #[test]
    fn unicode() {
        assert_eq!(levenshtein_distance("café", "cafe"), 1);
        assert_eq!(levenshtein_distance("日本語", "日本人"), 1);
    }

    #[test]
    fn normalized_similarity() {
        let m = Levenshtein;
        let sim = m.normalized_similarity("kitten", "sitting").unwrap();
        assert!((sim - (1.0 - 3.0 / 7.0)).abs() < 1e-10);
    }

    #[test]
    fn threshold_within() {
        assert_eq!(
            levenshtein_distance_threshold("kitten", "sitting", 3),
            Some(3)
        );
        assert_eq!(
            levenshtein_distance_threshold("kitten", "sitting", 5),
            Some(3)
        );
    }

    #[test]
    fn threshold_exceeded() {
        assert_eq!(levenshtein_distance_threshold("kitten", "sitting", 2), None);
        assert_eq!(levenshtein_distance_threshold("kitten", "sitting", 1), None);
    }

    #[test]
    fn threshold_length_filter() {
        assert_eq!(levenshtein_distance_threshold("a", "abcdef", 2), None);
    }

    #[test]
    fn threshold_empty() {
        assert_eq!(levenshtein_distance_threshold("", "", 0), Some(0));
        assert_eq!(levenshtein_distance_threshold("abc", "", 3), Some(3));
        assert_eq!(levenshtein_distance_threshold("abc", "", 2), None);
    }

    #[test]
    fn threshold_identical() {
        assert_eq!(levenshtein_distance_threshold("hello", "hello", 0), Some(0));
    }

    #[test]
    fn boundary_63_chars() {
        let a: String = "a".repeat(63);
        let b: String = "a".repeat(62) + "b";
        assert_eq!(levenshtein_distance(&a, &b), 1);
        // Symmetric
        assert_eq!(levenshtein_distance(&b, &a), 1);
    }

    #[test]
    fn boundary_64_chars() {
        let a: String = "a".repeat(64);
        let b: String = "a".repeat(63) + "b";
        assert_eq!(levenshtein_distance(&a, &b), 1);
        assert_eq!(levenshtein_distance(&b, &a), 1);
    }

    #[test]
    fn boundary_65_chars_fallback() {
        // 65 chars: should use multi-block Myers
        let a: String = "a".repeat(65);
        let b: String = "a".repeat(64) + "b";
        assert_eq!(levenshtein_distance(&a, &b), 1);
        assert_eq!(levenshtein_distance(&b, &a), 1);
    }

    #[test]
    fn boundary_consistency() {
        // Verify both paths give the same result for a string that spans the boundary
        let base: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let modified: String = base
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();

        let bp_result = levenshtein_distance(&base, &modified);

        // Force multi-block by padding to 65 chars
        let base_long = base.clone() + "x";
        let modified_long = modified.clone() + "x";
        let wf_result = levenshtein_distance(&base_long, &modified_long);

        assert_eq!(bp_result, wf_result);
    }

    #[test]
    fn threshold_boundary_64() {
        let a: String = "a".repeat(64);
        let b: String = "a".repeat(63) + "b";
        assert_eq!(levenshtein_distance_threshold(&a, &b, 1), Some(1));
        assert_eq!(levenshtein_distance_threshold(&a, &b, 0), None);
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
        assert_eq!(levenshtein_distance(&a, &b), 3);
    }

    #[test]
    fn multi_block_200_chars() {
        let a: String = (0..200).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 50 || i == 100 || i == 150 || i == 199 {
                    'Z'
                } else {
                    c
                }
            })
            .collect();
        assert_eq!(levenshtein_distance(&a, &b), 4);
    }

    #[test]
    fn multi_block_identical() {
        let a: String = (0..200).map(|i| (b'a' + (i % 26)) as char).collect();
        assert_eq!(levenshtein_distance(&a, &a), 0);
    }

    #[test]
    fn multi_block_completely_different() {
        let a: String = "a".repeat(100);
        let b: String = "b".repeat(100);
        assert_eq!(levenshtein_distance(&a, &b), 100);
    }

    #[test]
    fn multi_block_vs_single_block_at_boundary() {
        // 64 chars (single block) vs 65 chars (multi-block): same prefix, different suffix
        for len in [65, 100, 128, 129, 200, 256, 257] {
            let a: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
            let b: String = a
                .chars()
                .enumerate()
                .map(|(i, c)| if i == len / 2 { 'Z' } else { c })
                .collect();
            assert_eq!(levenshtein_distance(&a, &b), 1, "Failed for length {len}");
        }
    }
}
