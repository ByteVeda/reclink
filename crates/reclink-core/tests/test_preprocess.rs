use reclink_core::blocking::ExactBlocking;
use reclink_core::classify::ThresholdClassifier;
use reclink_core::compare::StringComparator;
use reclink_core::metrics::Metric;
use reclink_core::pipeline::ReclinkPipeline;
use reclink_core::preprocess::{
    apply_ops, fold_case, normalize_unicode, normalize_whitespace, preprocess_batch,
    standardize_name, strip_punctuation, whitespace_tokenize, NormalizationForm, PreprocessOp,
};
use reclink_core::record::{FieldValue, Record, RecordBatch};

/// Applies a standard preprocessing chain to a string.
fn preprocess(s: &str) -> String {
    let s = normalize_unicode(s, NormalizationForm::Nfkc);
    let s = fold_case(&s);
    let s = strip_punctuation(&s);
    normalize_whitespace(&s)
}

#[test]
fn preprocess_chain_normalizes_consistently() {
    let a = preprocess("  Dr.  José   García  ");
    let b = preprocess("dr. josé garcía");
    assert_eq!(a, b);
}

#[test]
fn preprocess_into_pipeline() {
    // Records with messy input that should match after preprocessing
    let raw_pairs = [
        ("  John   SMITH  ", "john smith"),
        ("Dr. Jane Doe", "doctor jane doe"),
    ];

    let records: Vec<Record> = raw_pairs
        .iter()
        .enumerate()
        .flat_map(|(i, (a, b))| {
            let ra = Record::new(format!("r{}", i * 2))
                .with_field("name", FieldValue::Text(preprocess(a)))
                .with_field("block", FieldValue::Text("A".into()));
            let rb = Record::new(format!("r{}", i * 2 + 1))
                .with_field("name", FieldValue::Text(standardize_name(&preprocess(b))))
                .with_field("block", FieldValue::Text("A".into()));
            [ra, rb]
        })
        .collect();

    let batch = RecordBatch::new(vec!["name".into(), "block".into()], records);

    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build()
        .unwrap();

    let matches = pipeline.dedup(&batch);

    // "john smith" (r0) and "john smith" (r1) should match exactly
    assert!(
        matches
            .iter()
            .any(|m| (m.pair.left == 0 && m.pair.right == 1)
                || (m.pair.left == 1 && m.pair.right == 0)),
        "preprocessed 'John SMITH' and 'john smith' should match"
    );
}

#[test]
fn whitespace_tokenize_integration() {
    let input = "  hello   beautiful   world  ";
    let normalized = normalize_whitespace(input);
    let tokens = whitespace_tokenize(&normalized);
    assert_eq!(tokens, vec!["hello", "beautiful", "world"]);
}

#[test]
fn standardize_name_through_pipeline() {
    // "St. Louis" → "saint louis" and "Saint Louis" → "saint louis" should match
    let records = vec![
        Record::new("r0")
            .with_field("city", FieldValue::Text(standardize_name("St. Louis")))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r1")
            .with_field("city", FieldValue::Text(standardize_name("Saint Louis")))
            .with_field("block", FieldValue::Text("A".into())),
    ];
    let batch = RecordBatch::new(vec!["city".into(), "block".into()], records);

    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("city", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.9)))
        .build()
        .unwrap();

    let matches = pipeline.dedup(&batch);
    assert!(
        matches
            .iter()
            .any(|m| (m.pair.left == 0 && m.pair.right == 1)
                || (m.pair.left == 1 && m.pair.right == 0)),
        "'St. Louis' and 'Saint Louis' should match after standardization"
    );
}

#[test]
fn unicode_normalization_affects_matching() {
    // NFC and NFD forms of "café" should match after normalization
    let cafe_nfc = normalize_unicode("caf\u{00e9}", NormalizationForm::Nfc);
    let cafe_nfd = normalize_unicode("caf\u{0065}\u{0301}", NormalizationForm::Nfc);
    assert_eq!(cafe_nfc, cafe_nfd, "NFC normalization should unify forms");
}

// --- Batch preprocessing ---

#[test]
fn preprocess_batch_chain() {
    let inputs = vec!["  Hello,   WORLD!  ".to_string(), "Dr. SMITH".to_string()];
    let ops = vec![
        PreprocessOp::FoldCase,
        PreprocessOp::StripPunctuation,
        PreprocessOp::NormalizeWhitespace,
    ];
    let results = preprocess_batch(&inputs, &ops);
    assert_eq!(results[0], "hello world");
    assert_eq!(results[1], "dr smith");
}

#[test]
fn preprocess_batch_empty() {
    let inputs: Vec<String> = vec![];
    let ops = vec![PreprocessOp::FoldCase];
    let results = preprocess_batch(&inputs, &ops);
    assert!(results.is_empty());
}

#[test]
fn apply_ops_unicode_and_case() {
    let result = apply_ops(
        "CAF\u{00c9}",
        &[
            PreprocessOp::NormalizeUnicode(NormalizationForm::Nfkc),
            PreprocessOp::FoldCase,
        ],
    );
    assert_eq!(result, "café");
}
