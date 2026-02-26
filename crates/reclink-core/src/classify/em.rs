//! EM algorithm for estimating Fellegi-Sunter m/u probabilities.

/// Configuration for the EM estimation algorithm.
#[derive(Debug, Clone)]
pub struct EmConfig {
    /// Maximum number of EM iterations.
    pub max_iterations: usize,
    /// Convergence threshold for parameter changes between iterations.
    pub convergence_threshold: f64,
    /// Initial prior probability that a random pair is a match.
    pub initial_p_match: f64,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            initial_p_match: 0.1,
        }
    }
}

/// Result of EM estimation.
#[derive(Debug, Clone)]
pub struct EmResult {
    /// Estimated m-probabilities: P(agree on field k | match).
    pub m_probs: Vec<f64>,
    /// Estimated u-probabilities: P(agree on field k | non-match).
    pub u_probs: Vec<f64>,
    /// Estimated overall proportion of matches.
    pub p_match: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged before reaching max_iterations.
    pub converged: bool,
}

/// Estimate Fellegi-Sunter m/u probabilities from comparison vectors using EM.
///
/// Each vector is a set of comparison scores (one per field). Scores >= 0.5
/// are treated as "agree", < 0.5 as "disagree".
#[must_use]
pub fn estimate_fellegi_sunter(vectors: &[Vec<f64>], config: &EmConfig) -> EmResult {
    if vectors.is_empty() {
        return EmResult {
            m_probs: Vec::new(),
            u_probs: Vec::new(),
            p_match: config.initial_p_match,
            iterations: 0,
            converged: true,
        };
    }

    let n = vectors.len();
    let k = vectors[0].len();

    // Binarize: agree (true) / disagree (false) for each field
    let agreements: Vec<Vec<bool>> = vectors
        .iter()
        .map(|v| v.iter().map(|&s| s >= 0.5).collect())
        .collect();

    // Initialize parameters
    let mut p_match = config.initial_p_match;
    let mut m_probs: Vec<f64> = vec![0.9; k];
    let mut u_probs: Vec<f64> = vec![0.1; k];

    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // E-step: compute P(match | vector_i) for each vector
        let mut weights = Vec::with_capacity(n);
        for agrees in &agreements {
            let mut log_match = p_match.ln();
            let mut log_non_match = (1.0 - p_match).ln();

            for (j, &agree) in agrees.iter().enumerate() {
                if agree {
                    log_match += m_probs[j].max(1e-10).ln();
                    log_non_match += u_probs[j].max(1e-10).ln();
                } else {
                    log_match += (1.0 - m_probs[j]).max(1e-10).ln();
                    log_non_match += (1.0 - u_probs[j]).max(1e-10).ln();
                }
            }

            // Use log-sum-exp trick for numerical stability
            let max_log = log_match.max(log_non_match);
            let w = (log_match - max_log).exp()
                / ((log_match - max_log).exp() + (log_non_match - max_log).exp());
            weights.push(w);
        }

        // M-step: update parameters
        let sum_w: f64 = weights.iter().sum();
        let new_p_match = sum_w / n as f64;

        let mut new_m_probs = vec![0.0; k];
        let mut new_u_probs = vec![0.0; k];

        for (i, agrees) in agreements.iter().enumerate() {
            for (j, &agree) in agrees.iter().enumerate() {
                if agree {
                    new_m_probs[j] += weights[i];
                    new_u_probs[j] += 1.0 - weights[i];
                }
            }
        }

        for j in 0..k {
            new_m_probs[j] = (new_m_probs[j] / sum_w.max(1e-10)).clamp(1e-6, 1.0 - 1e-6);
            new_u_probs[j] =
                (new_u_probs[j] / (n as f64 - sum_w).max(1e-10)).clamp(1e-6, 1.0 - 1e-6);
        }

        // Check convergence
        let mut max_diff = (new_p_match - p_match).abs();
        for j in 0..k {
            max_diff = max_diff.max((new_m_probs[j] - m_probs[j]).abs());
            max_diff = max_diff.max((new_u_probs[j] - u_probs[j]).abs());
        }

        p_match = new_p_match;
        m_probs = new_m_probs;
        u_probs = new_u_probs;

        if max_diff < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    EmResult {
        m_probs,
        u_probs,
        p_match,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn em_estimates_known_distribution() {
        // Create synthetic data: ~30% matches with known m/u probs
        // m = [0.9, 0.8], u = [0.1, 0.2]
        let mut vectors = Vec::new();
        // Matches: both fields agree with high probability
        for _ in 0..30 {
            vectors.push(vec![0.95, 0.85]);
        }
        // Non-matches: both fields disagree with high probability
        for _ in 0..70 {
            vectors.push(vec![0.1, 0.2]);
        }

        let config = EmConfig {
            max_iterations: 200,
            convergence_threshold: 1e-8,
            initial_p_match: 0.1,
        };
        let result = estimate_fellegi_sunter(&vectors, &config);

        // m-probs should be high (close to 1.0 since all matches agree)
        assert!(result.m_probs[0] > 0.8, "m[0] = {}", result.m_probs[0]);
        assert!(result.m_probs[1] > 0.8, "m[1] = {}", result.m_probs[1]);
        // u-probs should be low
        assert!(result.u_probs[0] < 0.3, "u[0] = {}", result.u_probs[0]);
        assert!(result.u_probs[1] < 0.4, "u[1] = {}", result.u_probs[1]);
        // p_match should be close to 0.3
        assert!(
            (result.p_match - 0.3).abs() < 0.1,
            "p_match = {}",
            result.p_match
        );
    }

    #[test]
    fn em_converges() {
        let vectors: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                if i < 20 {
                    vec![0.9, 0.8, 0.9]
                } else {
                    vec![0.1, 0.2, 0.1]
                }
            })
            .collect();

        let config = EmConfig::default();
        let result = estimate_fellegi_sunter(&vectors, &config);

        assert!(result.converged);
        assert!(result.iterations < config.max_iterations);
    }

    #[test]
    fn em_empty_vectors() {
        let config = EmConfig::default();
        let result = estimate_fellegi_sunter(&[], &config);

        assert!(result.m_probs.is_empty());
        assert!(result.u_probs.is_empty());
        assert_eq!(result.iterations, 0);
        assert!(result.converged);
    }
}
