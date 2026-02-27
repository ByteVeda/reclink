//! Levenshtein edit distance with Myers' bit-vector acceleration for strings ≤ 64 chars.

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

/// Myers' bit-parallel algorithm for edit distance.
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

/// Wagner-Fischer single-row DP algorithm for edit distance.
///
/// Fallback for strings longer than 64 characters.
fn wagner_fischer(short: &[char], long: &[char]) -> usize {
    let short_len = short.len();
    let long_len = long.len();

    let mut prev_row: Vec<usize> = (0..=short_len).collect();

    for i in 1..=long_len {
        let mut prev_diag = prev_row[0];
        prev_row[0] = i;

        for j in 1..=short_len {
            let old_diag = prev_row[j];
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };

            prev_row[j] = (prev_row[j] + 1)
                .min(prev_row[j - 1] + 1)
                .min(prev_diag + cost);

            prev_diag = old_diag;
        }
    }

    prev_row[short_len]
}

/// Wagner-Fischer DP with early termination and row-minimum pruning.
fn wagner_fischer_threshold(
    short: &[char],
    long: &[char],
    max_distance: usize,
) -> Option<usize> {
    let short_len = short.len();
    let long_len = long.len();

    let mut prev_row: Vec<usize> = (0..=short_len).collect();

    for i in 1..=long_len {
        let mut prev_diag = prev_row[0];
        prev_row[0] = i;
        let mut row_min = prev_row[0];

        for j in 1..=short_len {
            let old_diag = prev_row[j];
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };

            prev_row[j] = (prev_row[j] + 1)
                .min(prev_row[j - 1] + 1)
                .min(prev_diag + cost);

            row_min = row_min.min(prev_row[j]);
            prev_diag = old_diag;
        }

        if row_min > max_distance {
            return None;
        }
    }

    let result = prev_row[short_len];
    if result > max_distance {
        None
    } else {
        Some(result)
    }
}

/// Computes Levenshtein distance using Myers' bit-parallel algorithm for strings ≤ 64 chars,
/// falling back to single-row Wagner-Fischer for longer strings.
///
/// Time: O(n) for pattern ≤ 64, O(m*n) otherwise. Space: O(min(m,n))
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

    wagner_fischer(short, long)
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
        return if dist > max_distance { None } else { Some(dist) };
    }

    wagner_fischer_threshold(short, long, max_distance)
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
        // 65 chars: should use Wagner-Fischer fallback
        let a: String = "a".repeat(65);
        let b: String = "a".repeat(64) + "b";
        assert_eq!(levenshtein_distance(&a, &b), 1);
        assert_eq!(levenshtein_distance(&b, &a), 1);
    }

    #[test]
    fn boundary_consistency() {
        // Verify both paths give the same result for a string that spans the boundary
        let base: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let modified: String = base.chars().enumerate().map(|(i, c)| {
            if i == 32 { 'Z' } else { c }
        }).collect();

        let bp_result = levenshtein_distance(&base, &modified);

        // Force Wagner-Fischer by padding to 65 chars
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
}
