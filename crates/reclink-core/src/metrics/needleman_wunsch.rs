//! Needleman-Wunsch global sequence alignment metric.
//!
//! Uses dynamic programming for global alignment. Unlike Smith-Waterman
//! (local alignment), scores can go negative and the entire sequence
//! must be aligned. Uses single-row DP with thread-local scratch buffers.

use crate::metrics::scratch::NW_SCRATCH;
use crate::metrics::SimilarityMetric;

/// Needleman-Wunsch global alignment metric.
///
/// Computes the optimal global alignment score between two strings
/// using configurable match, mismatch, and gap penalties.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NeedlemanWunsch {
    /// Score for a character match.
    pub match_score: f64,
    /// Penalty for a character mismatch (typically negative).
    pub mismatch_penalty: f64,
    /// Penalty for a gap/indel (typically negative).
    pub gap_penalty: f64,
}

impl Default for NeedlemanWunsch {
    fn default() -> Self {
        Self {
            match_score: 2.0,
            mismatch_penalty: -1.0,
            gap_penalty: -1.0,
        }
    }
}

impl SimilarityMetric for NeedlemanWunsch {
    fn similarity(&self, a: &str, b: &str) -> f64 {
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

        let score = nw_dp(
            &a_chars,
            &b_chars,
            self.match_score,
            self.mismatch_penalty,
            self.gap_penalty,
        );
        let max_possible = a_len.min(b_len) as f64 * self.match_score;
        if max_possible <= 0.0 {
            return 0.0;
        }
        (score / max_possible).clamp(0.0, 1.0)
    }
}

/// Computes the raw Needleman-Wunsch global alignment score.
#[must_use]
pub fn needleman_wunsch_score(
    a: &str,
    b: &str,
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() || b_chars.is_empty() {
        return if a_chars.is_empty() && b_chars.is_empty() {
            0.0
        } else {
            a_chars.len().max(b_chars.len()) as f64 * gap_penalty
        };
    }

    nw_dp(
        &a_chars,
        &b_chars,
        match_score,
        mismatch_penalty,
        gap_penalty,
    )
}

/// Single-row DP for Needleman-Wunsch global alignment.
fn nw_dp(
    a_chars: &[char],
    b_chars: &[char],
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> f64 {
    let b_len = b_chars.len();

    NW_SCRATCH.with_borrow_mut(|scratch| {
        scratch.reset_nw(b_len, gap_penalty);

        for (i, ac) in a_chars.iter().enumerate() {
            let mut prev_diag = scratch.prev[0];
            scratch.prev[0] = (i + 1) as f64 * gap_penalty;

            for j in 1..=b_len {
                let old = scratch.prev[j];
                let diag = if *ac == b_chars[j - 1] {
                    prev_diag + match_score
                } else {
                    prev_diag + mismatch_penalty
                };

                scratch.prev[j] = diag
                    .max(scratch.prev[j] + gap_penalty)
                    .max(scratch.prev[j - 1] + gap_penalty);

                prev_diag = old;
            }
        }

        scratch.prev[b_len]
    })
}

/// Computes normalized Needleman-Wunsch similarity using default parameters.
#[must_use]
pub fn needleman_wunsch_similarity(a: &str, b: &str) -> f64 {
    NeedlemanWunsch::default().similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(
            needleman_wunsch_similarity("hello", "hello"),
            1.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(needleman_wunsch_similarity("", ""), 1.0));
        assert!(approx_eq(needleman_wunsch_similarity("abc", ""), 0.0));
        assert!(approx_eq(needleman_wunsch_similarity("", "abc"), 0.0));
    }

    #[test]
    fn completely_different() {
        let sim = needleman_wunsch_similarity("abc", "xyz");
        assert!(approx_eq(sim, 0.0));
    }

    #[test]
    fn partial_match() {
        let sim = needleman_wunsch_similarity("kitten", "sitting");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn symmetry() {
        let a = needleman_wunsch_similarity("abc", "bcd");
        let b = needleman_wunsch_similarity("bcd", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn known_score() {
        // "ACGT" vs "ACGT" with default params: match=2, mismatch=-1, gap=-1
        // Perfect match: score = 4 * 2 = 8
        let score = needleman_wunsch_score("ACGT", "ACGT", 2.0, -1.0, -1.0);
        assert!(approx_eq(score, 8.0));
    }

    #[test]
    fn global_alignment_penalty() {
        // Unlike Smith-Waterman, NW penalizes unmatched ends
        // "ABC" vs "XABCY" — must align the full strings
        let sim = needleman_wunsch_similarity("ABC", "XABCY");
        assert!(sim < 1.0);
    }

    #[test]
    fn single_char() {
        assert!(approx_eq(needleman_wunsch_similarity("a", "a"), 1.0));
    }
}
