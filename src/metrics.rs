use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use reclink_core::metrics::{
    self, Cosine, DamerauLevenshtein, DistanceMetric, Hamming, Jaccard, Jaro, JaroWinkler, Lcs,
    Levenshtein, LongestCommonSubstring, NgramSimilarity, PartialRatio, PhoneticHybrid,
    SimilarityMetric, SmithWaterman, SorensenDice, TokenSet, TokenSort, WeightedLevenshtein,
};

/// Compute Levenshtein edit distance between two strings.
#[pyfunction]
fn levenshtein(a: &str, b: &str) -> usize {
    reclink_core::metrics::levenshtein::levenshtein_distance(a, b)
}

/// Compute normalized Levenshtein similarity in [0, 1].
#[pyfunction]
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    Levenshtein.normalized_similarity(a, b).unwrap_or(0.0)
}

/// Compute Damerau-Levenshtein distance between two strings.
#[pyfunction]
fn damerau_levenshtein(a: &str, b: &str) -> usize {
    reclink_core::metrics::damerau_levenshtein::damerau_levenshtein_distance(a, b)
}

/// Compute Damerau-Levenshtein normalized similarity in [0, 1].
#[pyfunction]
fn damerau_levenshtein_similarity(a: &str, b: &str) -> f64 {
    DamerauLevenshtein
        .normalized_similarity(a, b)
        .unwrap_or(0.0)
}

/// Compute Hamming distance between two equal-length strings.
#[pyfunction]
fn hamming(a: &str, b: &str) -> PyResult<usize> {
    reclink_core::metrics::hamming::hamming_distance(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute Hamming normalized similarity in [0, 1].
#[pyfunction]
fn hamming_similarity(a: &str, b: &str) -> PyResult<f64> {
    Hamming
        .normalized_similarity(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute Jaro similarity between two strings.
#[pyfunction]
fn jaro(a: &str, b: &str) -> f64 {
    Jaro.similarity(a, b)
}

/// Compute Jaro-Winkler similarity between two strings.
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1))]
fn jaro_winkler(a: &str, b: &str, prefix_weight: f64) -> f64 {
    JaroWinkler { prefix_weight }.similarity(a, b)
}

/// Compute cosine similarity between character n-gram vectors.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn cosine(a: &str, b: &str, n: usize) -> f64 {
    Cosine { n }.similarity(a, b)
}

/// Compute Jaccard similarity between whitespace-tokenized sets.
#[pyfunction]
fn jaccard(a: &str, b: &str) -> f64 {
    Jaccard.similarity(a, b)
}

/// Compute Sorensen-Dice coefficient between character bigrams.
#[pyfunction]
fn sorensen_dice(a: &str, b: &str) -> f64 {
    SorensenDice.similarity(a, b)
}

/// Compute weighted Levenshtein distance with configurable operation costs.
#[pyfunction]
#[pyo3(signature = (a, b, insert_cost=1.0, delete_cost=1.0, substitute_cost=1.0, transpose_cost=1.0))]
fn weighted_levenshtein(
    a: &str,
    b: &str,
    insert_cost: f64,
    delete_cost: f64,
    substitute_cost: f64,
    transpose_cost: f64,
) -> f64 {
    reclink_core::metrics::weighted_levenshtein::weighted_levenshtein_distance(
        a,
        b,
        insert_cost,
        delete_cost,
        substitute_cost,
        transpose_cost,
    )
}

/// Compute weighted Levenshtein similarity in [0, 1].
#[pyfunction]
#[pyo3(signature = (a, b, insert_cost=1.0, delete_cost=1.0, substitute_cost=1.0, transpose_cost=1.0))]
fn weighted_levenshtein_similarity(
    a: &str,
    b: &str,
    insert_cost: f64,
    delete_cost: f64,
    substitute_cost: f64,
    transpose_cost: f64,
) -> f64 {
    WeightedLevenshtein {
        insert_cost,
        delete_cost,
        substitute_cost,
        transpose_cost,
    }
    .similarity(a, b)
}

/// Compute token sort ratio between two strings.
#[pyfunction]
fn token_sort_ratio(a: &str, b: &str) -> f64 {
    TokenSort.similarity(a, b)
}

/// Compute token set ratio between two strings.
#[pyfunction]
fn token_set_ratio(a: &str, b: &str) -> f64 {
    TokenSet.similarity(a, b)
}

/// Compute partial ratio (best substring match) between two strings.
#[pyfunction]
fn partial_ratio(a: &str, b: &str) -> f64 {
    PartialRatio.similarity(a, b)
}

/// Compute the length of the longest common subsequence.
#[pyfunction]
fn lcs_length(a: &str, b: &str) -> usize {
    reclink_core::metrics::lcs::lcs_length(a, b)
}

/// Compute normalized LCS similarity in [0, 1].
#[pyfunction]
fn lcs_similarity(a: &str, b: &str) -> f64 {
    Lcs.similarity(a, b)
}

/// Compute the length of the longest common substring.
#[pyfunction]
fn longest_common_substring_length(a: &str, b: &str) -> usize {
    reclink_core::metrics::longest_common_substring::longest_common_substring_length(a, b)
}

/// Compute normalized longest common substring similarity in [0, 1].
#[pyfunction]
fn longest_common_substring_similarity(a: &str, b: &str) -> f64 {
    LongestCommonSubstring.similarity(a, b)
}

/// Compute n-gram Jaccard similarity between two strings.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn ngram_similarity(a: &str, b: &str, n: usize) -> f64 {
    NgramSimilarity { n }.similarity(a, b)
}

/// Compute raw Smith-Waterman local alignment score.
#[pyfunction]
#[pyo3(signature = (a, b, match_score=2.0, mismatch_penalty=-1.0, gap_penalty=-1.0))]
fn smith_waterman(
    a: &str,
    b: &str,
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> f64 {
    reclink_core::metrics::smith_waterman::smith_waterman_score(
        a,
        b,
        match_score,
        mismatch_penalty,
        gap_penalty,
    )
}

/// Compute normalized Smith-Waterman similarity in [0, 1].
#[pyfunction]
fn smith_waterman_similarity(a: &str, b: &str) -> f64 {
    SmithWaterman::default().similarity(a, b)
}

/// Compute phonetic + edit distance hybrid similarity.
#[pyfunction]
#[pyo3(signature = (a, b, phonetic="soundex", metric="jaro_winkler", phonetic_weight=0.3))]
fn phonetic_hybrid(
    a: &str,
    b: &str,
    phonetic: &str,
    metric: &str,
    phonetic_weight: f64,
) -> PyResult<f64> {
    let algo = crate::parsers::parse_phonetic_algorithm(phonetic)?;
    let edit_metric =
        metrics::metric_from_name(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let hybrid = PhoneticHybrid {
        phonetic_algorithm: algo,
        edit_metric: Box::new(edit_metric),
        phonetic_weight,
    };
    Ok(hybrid.similarity(a, b))
}

/// Compute Levenshtein distance with early termination.
///
/// Returns None if the distance exceeds max_distance.
#[pyfunction]
fn levenshtein_threshold(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    reclink_core::metrics::levenshtein::levenshtein_distance_threshold(a, b, max_distance)
}

/// Compute Damerau-Levenshtein distance with early termination.
///
/// Returns None if the distance exceeds max_distance.
#[pyfunction]
fn damerau_levenshtein_threshold(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    reclink_core::metrics::damerau_levenshtein::damerau_levenshtein_distance_threshold(
        a,
        b,
        max_distance,
    )
}

/// Find the best match for a query among candidates.
///
/// Returns (matched_string, score, index) or None if no match meets threshold.
#[pyfunction]
#[pyo3(signature = (query, candidates, scorer="jaro_winkler", threshold=None, workers=None))]
fn match_best(
    query: &str,
    candidates: Vec<String>,
    scorer: &str,
    threshold: Option<f64>,
    workers: Option<usize>,
) -> PyResult<Option<(String, f64, usize)>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(n) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    Ok(
        reclink_core::metrics::batch::match_best(query, &refs, &metric, threshold)
            .map(|r| (candidates[r.index].clone(), r.score, r.index)),
    )
}

/// Find all matches for a query among candidates, sorted by descending score.
///
/// Returns list of (matched_string, score, index) tuples.
#[pyfunction]
#[pyo3(signature = (query, candidates, scorer="jaro_winkler", threshold=None, limit=None, workers=None))]
fn match_batch(
    query: &str,
    candidates: Vec<String>,
    scorer: &str,
    threshold: Option<f64>,
    limit: Option<usize>,
    workers: Option<usize>,
) -> PyResult<Vec<(String, f64, usize)>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(n) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    let mut results = reclink_core::metrics::batch::match_batch(query, &refs, &metric, threshold);

    if let Some(lim) = limit {
        results.truncate(lim);
    }

    Ok(results
        .into_iter()
        .map(|r| (candidates[r.index].clone(), r.score, r.index))
        .collect())
}

/// Compute pairwise similarity matrix between two lists of strings.
///
/// Returns a numpy 2D array of shape (len(a), len(b)).
#[pyfunction]
#[pyo3(signature = (a, b, scorer="jaro_winkler", workers=None))]
fn cdist<'py>(
    py: Python<'py>,
    a: Vec<String>,
    b: Vec<String>,
    scorer: &str,
    workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(n) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore if already initialized
    }

    let rows = a.len();
    let cols = b.len();

    let b_ref = &b;
    let flat: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|i| {
            let metric = &metric;
            let a_str = &a[i];
            (0..cols)
                .map(move |j| metric.similarity(a_str, &b_ref[j]))
                .collect::<Vec<f64>>()
        })
        .collect();

    let array = numpy::PyArray1::from_vec(py, flat)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(array)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(sorensen_dice, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_length, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_length, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(smith_waterman, m)?)?;
    m.add_function(wrap_pyfunction!(smith_waterman_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(phonetic_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(match_best, m)?)?;
    m.add_function(wrap_pyfunction!(match_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    Ok(())
}
