//! Smith-Waterman local sequence alignment metric.
//!
//! Uses SIMD (SSE2/AVX2/NEON) for f64 max operations in the DP inner loop
//! for strings > 16 chars.

use crate::metrics::SimilarityMetric;

/// Smith-Waterman performs local sequence alignment, finding the best
/// matching subsequence between two strings. Unlike global alignment,
/// scores cannot go below zero.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SmithWaterman {
    /// Score for a character match.
    pub match_score: f64,
    /// Penalty for a character mismatch (typically negative).
    pub mismatch_penalty: f64,
    /// Penalty for a gap/indel (typically negative).
    pub gap_penalty: f64,
}

impl Default for SmithWaterman {
    fn default() -> Self {
        Self {
            match_score: 2.0,
            mismatch_penalty: -1.0,
            gap_penalty: -1.0,
        }
    }
}

impl SimilarityMetric for SmithWaterman {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let score = smith_waterman_score(
            a,
            b,
            self.match_score,
            self.mismatch_penalty,
            self.gap_penalty,
        );
        let min_len = a.chars().count().min(b.chars().count());
        if min_len == 0 {
            if a.is_empty() && b.is_empty() {
                return 1.0;
            }
            return 0.0;
        }
        let max_possible = min_len as f64 * self.match_score;
        if max_possible <= 0.0 {
            return 0.0;
        }
        (score / max_possible).clamp(0.0, 1.0)
    }
}

/// Computes the raw Smith-Waterman local alignment score.
///
/// Uses single-row DP. Scores never go below 0 (local alignment property).
#[must_use]
pub fn smith_waterman_score(
    a: &str,
    b: &str,
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    // Use SIMD path for longer strings where overhead is justified
    if b_len > 16 {
        return crate::metrics::simd_util::dispatch_simd!(
            avx2: sw_dp_scalar(
                &a_chars, &b_chars, match_score, mismatch_penalty, gap_penalty
            ),
            sse2: sw_dp_scalar(
                &a_chars, &b_chars, match_score, mismatch_penalty, gap_penalty
            ),
            neon: sw_dp_scalar(
                &a_chars, &b_chars, match_score, mismatch_penalty, gap_penalty
            ),
            scalar: sw_dp_scalar(
                &a_chars, &b_chars, match_score, mismatch_penalty, gap_penalty
            ),
        );
    }

    sw_dp_scalar(
        &a_chars,
        &b_chars,
        match_score,
        mismatch_penalty,
        gap_penalty,
    )
}

/// Standard single-row DP for Smith-Waterman.
fn sw_dp_scalar(
    a_chars: &[char],
    b_chars: &[char],
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> f64 {
    let b_len = b_chars.len();
    let mut prev = vec![0.0f64; b_len + 1];
    let mut max_score = 0.0f64;

    for ac in a_chars {
        let mut prev_diag = 0.0;
        for j in 1..=b_len {
            let old = prev[j];
            let diag = if *ac == b_chars[j - 1] {
                prev_diag + match_score
            } else {
                prev_diag + mismatch_penalty
            };

            prev[j] = 0.0f64
                .max(diag)
                .max(prev[j] + gap_penalty)
                .max(prev[j - 1] + gap_penalty);

            max_score = max_score.max(prev[j]);
            prev_diag = old;
        }
    }

    max_score
}

/// Computes normalized Smith-Waterman similarity using default parameters.
#[must_use]
pub fn smith_waterman_similarity(a: &str, b: &str) -> f64 {
    SmithWaterman::default().similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(smith_waterman_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(smith_waterman_similarity("", ""), 1.0));
        assert!(approx_eq(smith_waterman_similarity("abc", ""), 0.0));
        assert!(approx_eq(smith_waterman_similarity("", "abc"), 0.0));
    }

    #[test]
    fn local_alignment() {
        // "ACBDE" inside "XACBDEY" — should find the local match
        let sim = smith_waterman_similarity("ACBDE", "XACBDEY");
        assert!(approx_eq(sim, 1.0));
    }

    #[test]
    fn no_match() {
        let score = smith_waterman_score("abc", "xyz", 2.0, -1.0, -1.0);
        assert!(approx_eq(score, 0.0));
    }

    #[test]
    fn known_score() {
        // "ACGT" vs "ACGT" with match=2, mismatch=-1, gap=-1
        // Perfect match: score = 4 * 2 = 8
        let score = smith_waterman_score("ACGT", "ACGT", 2.0, -1.0, -1.0);
        assert!(approx_eq(score, 8.0));
    }

    #[test]
    fn partial_match() {
        let sim = smith_waterman_similarity("abc", "xabcy");
        assert!(approx_eq(sim, 1.0));
    }

    #[test]
    fn different_strings() {
        let sim = smith_waterman_similarity("abcdef", "uvwxyz");
        assert!(approx_eq(sim, 0.0));
    }

    #[test]
    fn long_strings() {
        let a: String = (0..100).map(|i| (b'A' + (i % 26)) as char).collect();
        let b: String = (0..100).map(|i| (b'A' + (i % 26)) as char).collect();
        assert!(approx_eq(smith_waterman_similarity(&a, &b), 1.0));
    }
}
