use reclink_core::classify::{Classifier, FellegiSunterClassifier, WeightedSumClassifier};
use reclink_core::record::{CandidatePair, ComparisonVector, MatchClass};

#[test]
fn weighted_non_match() {
    let classifier = WeightedSumClassifier::new(vec![0.5, 0.5], 0.8);
    let vector = ComparisonVector {
        pair: CandidatePair { left: 0, right: 1 },
        scores: vec![0.3, 0.4],
    };
    let result = classifier.classify(&vector);
    assert_eq!(result.class, MatchClass::NonMatch);
    // Weighted sum = 0.3*0.5 + 0.4*0.5 = 0.35, below 0.8
    assert!((result.aggregate_score - 0.35).abs() < 1e-10);
}

#[test]
fn fellegi_sunter_possible() {
    // m-probs: high agreement likelihood for matches
    // u-probs: low agreement likelihood for non-matches
    // upper=5.0, lower=-5.0 to create a wide "possible" zone
    let classifier = FellegiSunterClassifier::new(
        vec![0.9, 0.9],
        vec![0.1, 0.1],
        5.0,  // upper threshold (high to force Possible)
        -5.0, // lower threshold (low to avoid NonMatch)
    );
    let vector = ComparisonVector {
        pair: CandidatePair { left: 0, right: 1 },
        scores: vec![0.6, 0.6], // above agreement_threshold (0.5) so agreements
    };
    let result = classifier.classify(&vector);
    // log(0.9/0.1)*2 = log(9)*2 ≈ 4.39, which is between -5.0 and 5.0
    assert_eq!(result.class, MatchClass::Possible);
}

#[test]
fn fellegi_sunter_custom_agreement_threshold() {
    let mut classifier = FellegiSunterClassifier::new(vec![0.95], vec![0.05], 1.0, -1.0);
    // Set a high agreement threshold so 0.6 counts as disagreement
    classifier.agreement_threshold = 0.8;

    let vector = ComparisonVector {
        pair: CandidatePair { left: 0, right: 1 },
        scores: vec![0.6], // below 0.8, so disagreement
    };
    let result = classifier.classify(&vector);
    // Disagreement weight = ln((1-0.95)/(1-0.05)) = ln(0.05/0.95) ≈ -2.944
    // This is below lower_threshold of -1.0 → NonMatch
    assert_eq!(result.class, MatchClass::NonMatch);
}
