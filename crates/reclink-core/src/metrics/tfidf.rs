//! TF-IDF weighted string matching.
//!
//! Provides corpus-aware similarity scoring that down-weights common tokens.

use ahash::{AHashMap, AHashSet};

use crate::metrics::batch::MatchResult;

/// A TF-IDF matcher that uses pre-computed IDF weights from a corpus.
///
/// Unlike stateless `SimilarityMetric` implementations, this struct requires fitting
/// on a corpus before use, which makes it stateful.
#[derive(Debug, Clone)]
pub struct TfIdfMatcher {
    idf: AHashMap<String, f64>,
    n_docs: usize,
}

impl TfIdfMatcher {
    /// Build IDF weights from a corpus of strings.
    ///
    /// Each string is tokenized on whitespace, lowercased, and the document frequency
    /// of each token is computed. IDF is `ln(N / df(t))` where `N` is the number of
    /// documents and `df(t)` is the number of documents containing token `t`.
    #[must_use]
    pub fn fit(corpus: &[&str]) -> Self {
        let n_docs = corpus.len();
        let mut df: AHashMap<String, usize> = AHashMap::new();

        for doc in corpus {
            let unique_tokens: AHashSet<String> =
                doc.split_whitespace().map(|t| t.to_lowercase()).collect();
            for token in unique_tokens {
                *df.entry(token).or_insert(0) += 1;
            }
        }

        let n = n_docs.max(1) as f64;
        let idf: AHashMap<String, f64> = df
            .into_iter()
            .map(|(token, count)| (token, (n / count as f64).ln()))
            .collect();

        Self { idf, n_docs }
    }

    /// Compute the TF-IDF cosine similarity between two strings.
    ///
    /// For each string, a TF-IDF vector is computed using the pre-fitted IDF weights.
    /// The similarity is the cosine of the angle between these two vectors.
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        let vec_a = self.tfidf_vector(a);
        let vec_b = self.tfidf_vector(b);
        cosine_similarity(&vec_a, &vec_b)
    }

    /// Batch: one query vs many candidates, using pre-computed IDF.
    ///
    /// Returns results sorted by descending score, optionally filtered by threshold.
    #[must_use]
    pub fn match_batch(
        &self,
        query: &str,
        candidates: &[&str],
        threshold: Option<f64>,
    ) -> Vec<MatchResult> {
        let mut results: Vec<MatchResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| MatchResult {
                index: i,
                score: self.similarity(query, c),
            })
            .filter(|r| threshold.is_none_or(|t| r.score >= t))
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Returns the number of documents the matcher was fitted on.
    #[must_use]
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    fn tfidf_vector(&self, s: &str) -> AHashMap<String, f64> {
        let tokens: Vec<String> = s.split_whitespace().map(|t| t.to_lowercase()).collect();
        let n_tokens = tokens.len() as f64;

        if n_tokens == 0.0 {
            return AHashMap::new();
        }

        // Compute term frequency
        let mut tf: AHashMap<String, f64> = AHashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Multiply by IDF
        let mut tfidf = AHashMap::new();
        for (token, count) in tf {
            let idf = self.idf.get(&token).copied().unwrap_or(0.0);
            tfidf.insert(token, (count / n_tokens) * idf);
        }
        tfidf
    }
}

fn cosine_similarity(a: &AHashMap<String, f64>, b: &AHashMap<String, f64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (token, va) in a {
        norm_a += va * va;
        if let Some(vb) = b.get(token) {
            dot += va * vb;
        }
    }
    for vb in b.values() {
        norm_b += vb * vb;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_similarity() {
        let corpus = vec!["apple inc", "banana corp", "cherry inc"];
        let matcher = TfIdfMatcher::fit(&corpus);
        // "apple" is unique to doc 0, "inc" appears in 2/3 docs
        let sim = matcher.similarity("apple inc", "apple corp");
        assert!(sim > 0.0);
    }

    #[test]
    fn common_tokens_downweighted() {
        let corpus = vec!["the cat", "the dog", "the bird", "unique phrase"];
        let matcher = TfIdfMatcher::fit(&corpus);
        // "the" appears in 3/4 docs → low IDF
        let sim = matcher.similarity("the cat", "the dog");
        assert!(sim < 0.5);
    }

    #[test]
    fn identical_strings() {
        let corpus = vec!["hello world", "foo bar"];
        let matcher = TfIdfMatcher::fit(&corpus);
        let sim = matcher.similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_strings() {
        let corpus = vec!["hello world"];
        let matcher = TfIdfMatcher::fit(&corpus);
        assert_eq!(matcher.similarity("", "hello"), 0.0);
        assert_eq!(matcher.similarity("hello", ""), 0.0);
    }

    #[test]
    fn match_batch_sorted() {
        let corpus = vec!["apple inc", "apple corp", "banana inc", "cherry co"];
        let matcher = TfIdfMatcher::fit(&corpus);
        let candidates = vec!["apple inc", "banana inc", "cherry co"];
        let results = matcher.match_batch("apple inc", &candidates, None);
        assert_eq!(results.len(), 3);
        // First result should be the exact match
        assert_eq!(results[0].index, 0);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn n_docs() {
        let corpus = vec!["a", "b", "c"];
        let matcher = TfIdfMatcher::fit(&corpus);
        assert_eq!(matcher.n_docs(), 3);
    }
}
