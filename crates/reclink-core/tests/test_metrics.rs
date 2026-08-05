use reclink_core::error::ReclinkError;
use reclink_core::metrics::{
    metric_from_name, Cosine, DamerauLevenshtein, Hamming, Jaccard, Jaro, JaroWinkler, Levenshtein,
    Metric, SimilarityMetric, SorensenDice,
};

#[test]
fn metric_from_name_all_valid() {
    let names = [
        "levenshtein",
        "damerau_levenshtein",
        "hamming",
        "jaro",
        "jaro_winkler",
        "cosine",
        "jaccard",
        "sorensen_dice",
    ];
    for name in &names {
        assert!(
            metric_from_name(name).is_ok(),
            "metric_from_name should accept '{name}'"
        );
    }
}

#[test]
fn metric_from_name_unknown() {
    let result = metric_from_name("unknown_metric");
    assert!(result.is_err());
    match result.unwrap_err() {
        ReclinkError::InvalidConfig(msg) => {
            assert!(msg.contains("unknown metric"));
        }
        other => panic!("expected InvalidConfig, got: {other:?}"),
    }
}

#[test]
fn metric_enum_similarity_dispatch() {
    let cases: Vec<(Metric, &str, &str, f64)> = vec![
        (Metric::Levenshtein(Levenshtein), "kitten", "kitten", 1.0),
        (
            Metric::DamerauLevenshtein(DamerauLevenshtein),
            "abc",
            "abc",
            1.0,
        ),
        (Metric::Hamming(Hamming), "abc", "abc", 1.0),
        (Metric::Jaro(Jaro), "abc", "abc", 1.0),
        (
            Metric::JaroWinkler(JaroWinkler::default()),
            "abc",
            "abc",
            1.0,
        ),
        (Metric::Cosine(Cosine::default()), "abc", "abc", 1.0),
        (Metric::Jaccard(Jaccard), "abc", "abc", 1.0),
        (Metric::SorensenDice(SorensenDice), "abc", "abc", 1.0),
    ];
    for (metric, a, b, expected) in cases {
        let score = metric.similarity(a, b);
        assert!(
            (score - expected).abs() < 1e-10,
            "{metric:?}: expected {expected}, got {score}"
        );
    }
}

#[test]
fn metric_default_is_jaro_winkler() {
    let m = Metric::default();
    assert!(matches!(m, Metric::JaroWinkler(_)));
}

#[test]
fn similarity_metric_dissimilarity() {
    let jaro = Jaro;
    let sim = jaro.similarity("abc", "abc");
    let dissim = jaro.dissimilarity("abc", "abc");
    assert!((sim + dissim - 1.0).abs() < 1e-10);

    let sim2 = jaro.similarity("abc", "xyz");
    let dissim2 = jaro.dissimilarity("abc", "xyz");
    assert!((sim2 + dissim2 - 1.0).abs() < 1e-10);
}

#[test]
fn hamming_unequal_length_through_metric_enum() {
    // Hamming is the only metric that can Err (unequal lengths).
    // Metric::similarity swallows the error with unwrap_or(0.0).
    let m = Metric::Hamming(Hamming);
    let score = m.similarity("abc", "abcdef");
    assert_eq!(
        score, 0.0,
        "unequal lengths should return 0.0 via unwrap_or"
    );
}

#[test]
fn hamming_equal_length_through_metric_enum() {
    let m = Metric::Hamming(Hamming);
    let score = m.similarity("abc", "axc");
    // Hamming distance = 1, normalized_similarity = 1 - 1/3 = 2/3
    assert!((score - 2.0 / 3.0).abs() < 1e-10);
}
