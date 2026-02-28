//! Jaro similarity metric with bit-parallel acceleration.
//!
//! - Both strings ≤ 64 chars: single-block bit-parallel
//! - Either string > 64 chars: multi-block bit-parallel with Vec<u64> blocks

use ahash::AHashMap;

use crate::metrics::SimilarityMetric;

/// Jaro similarity measures the edit distance between two strings,
/// accounting for character matches within a sliding window and transpositions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Jaro;

impl SimilarityMetric for Jaro {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaro_similarity(a, b)
    }
}

/// Bit-parallel Jaro matching for strings where both fit in a u64 bitmask.
///
/// Uses bitmask-based matching and transposition counting instead of boolean vectors.
fn jaro_bit_parallel(a_chars: &[char], b_chars: &[char]) -> f64 {
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    debug_assert!(a_len <= 64 && b_len <= 64);

    // Build pattern-match bitmask for string b
    let mut pm: AHashMap<char, u64> = AHashMap::new();
    for (j, &c) in b_chars.iter().enumerate() {
        *pm.entry(c).or_insert(0u64) |= 1u64 << j;
    }

    let match_window = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_match_bits: u64 = 0; // which positions in a are matched
    let mut b_match_bits: u64 = 0; // which positions in b are matched
    let mut matches: usize = 0;

    for (i, &ac) in a_chars.iter().enumerate() {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(b_len);

        // Build window mask: bits [start..end) set
        let window_mask = if end >= 64 {
            !0u64 << start
        } else {
            ((1u64 << end) - 1) & !((1u64 << start) - 1)
        };

        let candidates = pm.get(&ac).copied().unwrap_or(0) & window_mask & !b_match_bits;

        if candidates != 0 {
            // Take the lowest set bit (leftmost eligible match in b)
            let lowest = candidates & candidates.wrapping_neg();
            b_match_bits |= lowest;
            a_match_bits |= 1u64 << i;
            matches += 1;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions by iterating matched positions in order
    let mut transpositions: usize = 0;
    let mut b_remaining = b_match_bits;

    let mut a_remaining = a_match_bits;
    while a_remaining != 0 {
        let a_pos = a_remaining.trailing_zeros() as usize;
        a_remaining &= a_remaining - 1; // clear lowest bit

        let b_pos = b_remaining.trailing_zeros() as usize;
        b_remaining &= b_remaining - 1; // clear lowest bit

        if a_chars[a_pos] != b_chars[b_pos] {
            transpositions += 1;
        }
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Multi-block bit-parallel Jaro for strings > 64 chars.
///
/// Uses `Vec<u64>` blocks for the PM and match bitmasks, extending the
/// bit-parallel approach beyond the u64 limit.
fn jaro_multi_block(a_chars: &[char], b_chars: &[char]) -> f64 {
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    let num_blocks = b_len.div_ceil(64);

    // Build multi-block PM for string b
    let mut pm: AHashMap<char, Vec<u64>> = AHashMap::new();
    for (j, &c) in b_chars.iter().enumerate() {
        let block = j / 64;
        let bit = j % 64;
        let entry = pm.entry(c).or_insert_with(|| vec![0u64; num_blocks]);
        entry[block] |= 1u64 << bit;
    }

    let match_window = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_match_positions: Vec<usize> = Vec::new();
    let mut b_match_bits = vec![0u64; num_blocks];
    let mut matches: usize = 0;

    let empty_pm = vec![0u64; num_blocks];

    for (i, &ac) in a_chars.iter().enumerate() {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(b_len);

        let pm_char = pm.get(&ac).unwrap_or(&empty_pm);

        // Find first available match in b within the window
        let start_block = start / 64;
        let end_block = (end.saturating_sub(1)) / 64;

        for k in start_block..=end_block.min(num_blocks - 1) {
            let block_start = k * 64;
            let block_end = block_start + 64;

            // Window mask for this block
            let lo = start.saturating_sub(block_start);
            let hi = if end < block_end {
                end - block_start
            } else {
                64
            };

            if lo >= 64 || hi == 0 || lo >= hi {
                continue;
            }

            let window_mask = if hi >= 64 {
                !0u64 << lo
            } else {
                ((1u64 << hi) - 1) & !((1u64 << lo) - 1)
            };

            let candidates = pm_char[k] & window_mask & !b_match_bits[k];

            if candidates != 0 {
                let lowest = candidates & candidates.wrapping_neg();
                b_match_bits[k] |= lowest;
                a_match_positions.push(i);
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Collect matched b positions in order
    let mut b_matched_positions: Vec<usize> = Vec::with_capacity(matches);
    for (k, &block) in b_match_bits.iter().enumerate() {
        let mut bits = block;
        while bits != 0 {
            let bit_pos = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            b_matched_positions.push(k * 64 + bit_pos);
        }
    }

    // Count transpositions
    let mut transpositions: usize = 0;
    for (idx, &a_pos) in a_match_positions.iter().enumerate() {
        if a_chars[a_pos] != b_chars[b_matched_positions[idx]] {
            transpositions += 1;
        }
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Computes the Jaro similarity between two strings.
///
/// Uses bit-parallel matching when both strings fit in 64 bits,
/// multi-block bit-parallel otherwise.
///
/// Returns a value in \[0, 1\] where 1 means the strings are identical.
#[must_use]
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 && b_len == 0 {
        return 1.0;
    }
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    if a_len <= 64 && b_len <= 64 {
        return jaro_bit_parallel(&a_chars, &b_chars);
    }

    jaro_multi_block(&a_chars, &b_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(jaro_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(jaro_similarity("", ""), 1.0));
        assert!(approx_eq(jaro_similarity("abc", ""), 0.0));
        assert!(approx_eq(jaro_similarity("", "abc"), 0.0));
    }

    #[test]
    fn known_values() {
        assert!(approx_eq(jaro_similarity("martha", "marhta"), 0.9444));
        assert!(approx_eq(jaro_similarity("dwayne", "duane"), 0.8222));
        assert!(approx_eq(jaro_similarity("dixon", "dicksonx"), 0.7667));
    }

    #[test]
    fn symmetry() {
        let a = jaro_similarity("abc", "bac");
        let b = jaro_similarity("bac", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn completely_different() {
        assert!(approx_eq(jaro_similarity("abc", "xyz"), 0.0));
    }

    #[test]
    fn boundary_63_chars() {
        let a: String = (0..63).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 31 { 'Z' } else { c })
            .collect();
        let sim = jaro_similarity(&a, &b);
        assert!(sim > 0.9 && sim < 1.0);
    }

    #[test]
    fn boundary_64_chars() {
        let a: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();
        let sim = jaro_similarity(&a, &b);
        assert!(sim > 0.9 && sim < 1.0);
    }

    #[test]
    fn boundary_65_chars_multiblock() {
        let a: String = (0..65).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();
        let sim = jaro_similarity(&a, &b);
        assert!(sim > 0.9 && sim < 1.0);
    }

    #[test]
    fn boundary_consistency() {
        // Compare bit-parallel (64 chars) vs multi-block (65 chars, same prefix)
        let base64: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
        let mod64: String = base64
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 32 { 'Z' } else { c })
            .collect();

        let base65 = base64.clone() + "x";
        let mod65 = mod64.clone() + "x";

        let bp_result = jaro_similarity(&base64, &mod64);
        let mb_result = jaro_similarity(&base65, &mod65);

        // Results should be very close (not identical due to different string lengths)
        assert!((bp_result - mb_result).abs() < 0.02);
    }

    #[test]
    fn multi_block_identical() {
        let a: String = (0..200).map(|i| (b'a' + (i % 26)) as char).collect();
        assert!(approx_eq(jaro_similarity(&a, &a), 1.0));
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
        let sim = jaro_similarity(&a, &b);
        assert!(
            sim > 0.8 && sim < 1.0,
            "Expected 0.8 < sim < 1.0, got {sim}"
        );
    }
}
