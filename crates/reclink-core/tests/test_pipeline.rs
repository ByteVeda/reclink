use reclink_core::blocking::{ExactBlocking, QgramBlocking};
use reclink_core::classify::{ThresholdClassifier, WeightedSumClassifier};
use reclink_core::cluster::Linkage;
use reclink_core::compare::{DateComparator, ExactComparator, NumericComparator, StringComparator};
use reclink_core::error::ReclinkError;
use reclink_core::metrics::Metric;
use reclink_core::pipeline::ReclinkPipeline;
use reclink_core::record::{FieldValue, MatchClass, Record, RecordBatch};

fn make_batch(names: &[&str]) -> RecordBatch {
    let records: Vec<Record> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            Record::new(format!("r{i}"))
                .with_field("name", FieldValue::Text((*name).into()))
                .with_field("block", FieldValue::Text("A".into()))
        })
        .collect();
    RecordBatch::new(vec!["name".into(), "block".into()], records)
}

// --- Builder validation ---

#[test]
fn build_no_blocker_errors() {
    let result = ReclinkPipeline::builder()
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build();
    match result {
        Err(ReclinkError::Pipeline(msg)) => assert!(msg.contains("blocking")),
        Err(other) => panic!("expected Pipeline error, got: {other:?}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn build_no_comparator_errors() {
    let result = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build();
    match result {
        Err(ReclinkError::Pipeline(msg)) => assert!(msg.contains("comparator")),
        Err(other) => panic!("expected Pipeline error, got: {other:?}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn build_no_classifier_errors() {
    let result = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .build();
    match result {
        Err(ReclinkError::Pipeline(msg)) => assert!(msg.contains("classifier")),
        Err(other) => panic!("expected Pipeline error, got: {other:?}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn build_all_components_ok() {
    let result = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build();
    assert!(result.is_ok());
}

// --- Pipeline operations ---

fn build_pipeline() -> ReclinkPipeline {
    ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build()
        .unwrap()
}

#[test]
fn dedup_finds_similar_records() {
    let pipeline = build_pipeline();
    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let matches = pipeline.dedup(&batch);
    // Smith and Smyth should match (Jaro-Winkler > 0.8)
    assert!(
        matches
            .iter()
            .any(|m| (m.pair.left == 0 && m.pair.right == 1)
                || (m.pair.left == 1 && m.pair.right == 0)),
        "Smith and Smyth should be matched"
    );
    // All matches should be Match or Possible
    for m in &matches {
        assert!(m.class == MatchClass::Match || m.class == MatchClass::Possible);
    }
}

#[test]
fn dedup_cluster_with_clustering() {
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .with_clustering()
        .build()
        .unwrap();

    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let clusters = pipeline.dedup_cluster(&batch);
    // Should produce clusters (connected components)
    assert!(
        !clusters.is_empty(),
        "should produce at least one cluster for Smith/Smyth"
    );
    // Each cluster should have at least 2 members
    for cluster in &clusters {
        assert!(cluster.len() >= 2);
    }
}

#[test]
fn dedup_cluster_without_clustering() {
    let pipeline = build_pipeline();
    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let groups = pipeline.dedup_cluster(&batch);
    // Without clustering, each match becomes a pair [left, right]
    for group in &groups {
        assert_eq!(group.len(), 2, "without clustering, groups should be pairs");
    }
}

#[test]
fn link_matches_across_datasets() {
    let pipeline = build_pipeline();
    let left = make_batch(&["Smith", "Jones"]);
    let right = make_batch(&["Smyth", "Brown"]);
    let matches = pipeline.link(&left, &right);
    // Smith (left 0) and Smyth (right 0) should match
    assert!(
        matches
            .iter()
            .any(|m| m.pair.left == 0 && m.pair.right == 0),
        "Smith and Smyth should match across datasets"
    );
}

// --- Edge cases ---

#[test]
fn dedup_empty_batch() {
    let pipeline = build_pipeline();
    let batch = RecordBatch::new(vec!["name".into(), "block".into()], vec![]);
    let matches = pipeline.dedup(&batch);
    assert!(matches.is_empty());
}

#[test]
fn dedup_single_record() {
    let pipeline = build_pipeline();
    let batch = make_batch(&["Smith"]);
    let matches = pipeline.dedup(&batch);
    assert!(
        matches.is_empty(),
        "single record should produce no matches"
    );
}

#[test]
fn dedup_all_different() {
    let pipeline = build_pipeline();
    let batch = make_batch(&["Alice", "Zephyr", "Quantum"]);
    let matches = pipeline.dedup(&batch);
    assert!(
        matches.is_empty(),
        "completely different names should produce no matches"
    );
}

#[test]
fn dedup_records_with_missing_fields() {
    let pipeline = build_pipeline();
    // Record 1 has name + block, record 2 has only block (missing "name"),
    // record 3 has name + block
    let records = vec![
        Record::new("r0")
            .with_field("name", FieldValue::Text("Smith".into()))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r1").with_field("block", FieldValue::Text("A".into())),
        Record::new("r2")
            .with_field("name", FieldValue::Text("Smyth".into()))
            .with_field("block", FieldValue::Text("A".into())),
    ];
    let batch = RecordBatch::new(vec!["name".into(), "block".into()], records);
    let matches = pipeline.dedup(&batch);
    // Should not panic; r0 and r2 should still match
    assert!(
        matches
            .iter()
            .any(|m| (m.pair.left == 0 && m.pair.right == 2)
                || (m.pair.left == 2 && m.pair.right == 0)),
        "Smith and Smyth should match despite missing field on r1"
    );
}

#[test]
fn link_empty_left() {
    let pipeline = build_pipeline();
    let left = RecordBatch::new(vec!["name".into(), "block".into()], vec![]);
    let right = make_batch(&["Smith"]);
    let matches = pipeline.link(&left, &right);
    assert!(matches.is_empty());
}

#[test]
fn link_empty_right() {
    let pipeline = build_pipeline();
    let left = make_batch(&["Smith"]);
    let right = RecordBatch::new(vec!["name".into(), "block".into()], vec![]);
    let matches = pipeline.link(&left, &right);
    assert!(matches.is_empty());
}

// --- Multiple blockers and comparators ---

#[test]
fn multiple_blockers_union_candidates() {
    // ExactBlocking on "block" won't pair records in different blocks,
    // but QgramBlocking on "name" will find them via shared q-grams.
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_blocker(Box::new(QgramBlocking::new("name", 2, 1)))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .build()
        .unwrap();

    // Smith in block A, Smyth in block B — ExactBlocking won't pair them,
    // but QgramBlocking will via shared bigrams like "th"
    let records = vec![
        Record::new("r0")
            .with_field("name", FieldValue::Text("Smith".into()))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r1")
            .with_field("name", FieldValue::Text("Smyth".into()))
            .with_field("block", FieldValue::Text("B".into())),
    ];
    let batch = RecordBatch::new(vec!["name".into(), "block".into()], records);
    let matches = pipeline.dedup(&batch);
    assert!(
        matches
            .iter()
            .any(|m| (m.pair.left == 0 && m.pair.right == 1)
                || (m.pair.left == 1 && m.pair.right == 0)),
        "second blocker should find Smith/Smyth across different blocks"
    );
}

#[test]
fn multiple_comparators_weighted() {
    // Pipeline with two comparators (name + city) and a weighted classifier
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .add_comparator(Box::new(ExactComparator::new("city")))
        .set_classifier(Box::new(WeightedSumClassifier::new(vec![0.7, 0.3], 0.7)))
        .build()
        .unwrap();

    let records = vec![
        Record::new("r0")
            .with_field("name", FieldValue::Text("Smith".into()))
            .with_field("city", FieldValue::Text("NYC".into()))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r1")
            .with_field("name", FieldValue::Text("Smyth".into()))
            .with_field("city", FieldValue::Text("NYC".into()))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r2")
            .with_field("name", FieldValue::Text("Smyth".into()))
            .with_field("city", FieldValue::Text("LA".into()))
            .with_field("block", FieldValue::Text("A".into())),
    ];
    let batch = RecordBatch::new(vec!["name".into(), "city".into(), "block".into()], records);
    let matches = pipeline.dedup(&batch);

    // r0-r1: name ~0.83 * 0.7 + city 1.0 * 0.3 = ~0.88 → Match
    let r0_r1 = matches.iter().any(|m| {
        (m.pair.left == 0 && m.pair.right == 1) || (m.pair.left == 1 && m.pair.right == 0)
    });
    assert!(r0_r1, "Smith/Smyth in same city should match");
}

// --- Mixed-type comparators through pipeline ---

#[test]
fn pipeline_with_date_and_numeric_comparators() {
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .add_comparator(Box::new(DateComparator::new("dob")))
        .add_comparator(Box::new(NumericComparator::new("age", 5.0)))
        .set_classifier(Box::new(ThresholdClassifier::new(0.7)))
        .build()
        .unwrap();

    let records = vec![
        Record::new("r0")
            .with_field("name", FieldValue::Text("John Smith".into()))
            .with_field("dob", FieldValue::Date("1990-05-15".into()))
            .with_field("age", FieldValue::Integer(34))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r1")
            .with_field("name", FieldValue::Text("Jon Smith".into()))
            .with_field("dob", FieldValue::Date("1990-05-15".into()))
            .with_field("age", FieldValue::Integer(34))
            .with_field("block", FieldValue::Text("A".into())),
        Record::new("r2")
            .with_field("name", FieldValue::Text("Jane Doe".into()))
            .with_field("dob", FieldValue::Date("2000-12-01".into()))
            .with_field("age", FieldValue::Integer(24))
            .with_field("block", FieldValue::Text("A".into())),
    ];
    let batch = RecordBatch::new(
        vec!["name".into(), "dob".into(), "age".into(), "block".into()],
        records,
    );
    let matches = pipeline.dedup(&batch);

    // r0-r1: high name sim + exact date + exact age → Match
    let r0_r1 = matches.iter().any(|m| {
        (m.pair.left == 0 && m.pair.right == 1) || (m.pair.left == 1 && m.pair.right == 0)
    });
    assert!(
        r0_r1,
        "John Smith / Jon Smith with same DOB+age should match"
    );

    // r0-r2 or r1-r2: different name + different date + different age → no match
    let r_r2 = matches
        .iter()
        .any(|m| m.pair.left == 2 || m.pair.right == 2);
    assert!(!r_r2, "Jane Doe should not match either John/Jon Smith");
}

// --- Hierarchical clustering ---

#[test]
fn dedup_cluster_hierarchical_single_linkage() {
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .with_hierarchical_clustering(Linkage::Single, 0.5)
        .build()
        .unwrap();

    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let clusters = pipeline.dedup_cluster(&batch);
    assert!(
        !clusters.is_empty(),
        "hierarchical single-linkage should produce clusters for Smith/Smyth"
    );
    // Smith and Smyth should be in the same cluster
    let together = clusters.iter().any(|c| c.contains(&0) && c.contains(&1));
    assert!(together, "Smith and Smyth should be clustered together");
}

#[test]
fn dedup_cluster_hierarchical_average_linkage() {
    let pipeline = ReclinkPipeline::builder()
        .add_blocker(Box::new(ExactBlocking::new("block")))
        .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
        .set_classifier(Box::new(ThresholdClassifier::new(0.8)))
        .with_hierarchical_clustering(Linkage::Average, 0.5)
        .build()
        .unwrap();

    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let clusters = pipeline.dedup_cluster(&batch);
    assert!(
        !clusters.is_empty(),
        "hierarchical average-linkage should produce clusters"
    );
}

#[test]
fn dedup_cluster_config_none_returns_pairs() {
    // Default ClusterConfig::None should produce pair-wise groups
    let pipeline = build_pipeline();
    let batch = make_batch(&["Smith", "Smyth", "Jones"]);
    let groups = pipeline.dedup_cluster(&batch);
    for group in &groups {
        assert_eq!(
            group.len(),
            2,
            "ClusterConfig::None should return pair-wise groups"
        );
    }
}
