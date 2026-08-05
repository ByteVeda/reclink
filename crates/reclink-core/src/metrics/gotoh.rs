//! Gotoh (affine gap penalty) global alignment metric.
//!
//! Extends Needleman-Wunsch with separate gap-open and gap-extend costs,
//! modeled with three DP recurrences (D, P, Q). Uses two-row optimization
//! with thread-local scratch buffers.

use crate::metrics::scratch::GOTOH_SCRATCH;
use crate::metrics::SimilarityMetric;

/// Gotoh global alignment with affine gap penalties.
///
/// Uses separate gap-open and gap-extend costs, which better models
/// biological insertions/deletions where opening a gap is costly
/// but extending it is cheaper.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Gotoh {
    /// Score for a character match.
    pub match_score: f64,
    /// Penalty for a character mismatch (typically negative).
    pub mismatch_penalty: f64,
    /// Cost to open a new gap (typically negative, larger magnitude than extend).
    pub gap_open: f64,
    /// Cost to extend an existing gap (typically negative).
    pub gap_extend: f64,
}

impl Default for Gotoh {
    fn default() -> Self {
        Self {
            match_score: 1.0,
            mismatch_penalty: -1.0,
            gap_open: -1.0,
            gap_extend: -0.5,
        }
    }
}

impl SimilarityMetric for Gotoh {
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

        let score = gotoh_dp(
            &a_chars,
            &b_chars,
            self.match_score,
            self.mismatch_penalty,
            self.gap_open,
            self.gap_extend,
        );
        let max_possible = a_len.min(b_len) as f64 * self.match_score;
        if max_possible <= 0.0 {
            return 0.0;
        }
        (score / max_possible).clamp(0.0, 1.0)
    }
}

/// Computes the raw Gotoh global alignment score.
#[must_use]
pub fn gotoh_score(
    a: &str,
    b: &str,
    match_score: f64,
    mismatch_penalty: f64,
    gap_open: f64,
    gap_extend: f64,
) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() && b_chars.is_empty() {
        return 0.0;
    }
    if a_chars.is_empty() || b_chars.is_empty() {
        let len = a_chars.len().max(b_chars.len());
        return gap_open + (len.saturating_sub(1)) as f64 * gap_extend;
    }

    gotoh_dp(
        &a_chars,
        &b_chars,
        match_score,
        mismatch_penalty,
        gap_open,
        gap_extend,
    )
}

/// Three-recurrence DP for Gotoh's affine gap alignment.
///
/// D[i][j] = best score ending with a match/mismatch
/// P[i][j] = best score ending with a gap in B (deletion from A)
/// Q[i][j] = best score ending with a gap in A (insertion from B)
fn gotoh_dp(
    a_chars: &[char],
    b_chars: &[char],
    match_score: f64,
    mismatch_penalty: f64,
    gap_open: f64,
    gap_extend: f64,
) -> f64 {
    let b_len = b_chars.len();

    GOTOH_SCRATCH.with_borrow_mut(|scratch| {
        scratch.reset(b_len, gap_open, gap_extend);

        for (i, ac) in a_chars.iter().enumerate() {
            let mut d_prev_diag = scratch.d_prev[0];
            // First column: gap in B for i+1 rows
            scratch.d_prev[0] = gap_open + i as f64 * gap_extend;
            scratch.p_prev[0] = f64::NEG_INFINITY;

            let mut q_left = f64::NEG_INFINITY;

            for j in 1..=b_len {
                let old_d = scratch.d_prev[j];

                // P[i][j]: gap in B (deletion from A)
                let p_val = (scratch.d_prev[j] + gap_open).max(scratch.p_prev[j] + gap_extend);

                // Q[i][j]: gap in A (insertion from B)
                let q_val = (scratch.d_prev[j - 1] + gap_open).max(q_left + gap_extend);

                // D[i][j]: match/mismatch
                let sub = if *ac == b_chars[j - 1] {
                    match_score
                } else {
                    mismatch_penalty
                };
                let d_val = (d_prev_diag + sub).max(p_val).max(q_val);

                d_prev_diag = old_d;
                scratch.d_prev[j] = d_val;
                scratch.p_prev[j] = p_val;
                q_left = q_val;
            }
        }

        scratch.d_prev[b_len]
    })
}

/// Computes normalized Gotoh similarity using default parameters.
#[must_use]
pub fn gotoh_similarity(a: &str, b: &str) -> f64 {
    Gotoh::default().similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(gotoh_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(gotoh_similarity("", ""), 1.0));
        assert!(approx_eq(gotoh_similarity("abc", ""), 0.0));
        assert!(approx_eq(gotoh_similarity("", "abc"), 0.0));
    }

    #[test]
    fn completely_different() {
        let sim = gotoh_similarity("abc", "xyz");
        assert!(sim < 0.5);
    }

    #[test]
    fn partial_match() {
        let sim = gotoh_similarity("kitten", "sitting");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn symmetry() {
        let a = gotoh_similarity("abc", "bcd");
        let b = gotoh_similarity("bcd", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn affine_gap_benefit() {
        // Affine gaps should score a single long gap better than many short gaps
        let gotoh = Gotoh::default();
        // "abcdef" vs "abef" — single gap of 2 (cd deleted)
        let single_gap = gotoh.similarity("abcdef", "abef");
        // "abcdef" vs "acef" — two separate single gaps (b, d deleted)
        let two_gaps = gotoh.similarity("abcdef", "acef");
        // Single contiguous gap should score at least as well
        assert!(single_gap >= two_gaps - 1e-4);
    }

    #[test]
    fn single_char() {
        assert!(approx_eq(gotoh_similarity("a", "a"), 1.0));
    }

    #[test]
    fn known_score() {
        // "ACGT" vs "ACGT" with match=1 -> score = 4
        let score = gotoh_score("ACGT", "ACGT", 1.0, -1.0, -1.0, -0.5);
        assert!(approx_eq(score, 4.0));
    }
}
