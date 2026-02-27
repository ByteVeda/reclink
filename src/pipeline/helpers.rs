use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::config::PyClusterConfig;

pub fn generate_dedup_candidates(
    blockers: &[Box<dyn reclink_core::blocking::BlockingStrategy>],
    records: &reclink_core::record::RecordBatch,
) -> Vec<reclink_core::record::CandidatePair> {
    let all: Vec<Vec<reclink_core::record::CandidatePair>> = blockers
        .par_iter()
        .map(|b| b.block_dedup(records))
        .collect();
    dedup_candidate_pairs(all)
}

pub fn generate_link_candidates(
    blockers: &[Box<dyn reclink_core::blocking::BlockingStrategy>],
    left: &reclink_core::record::RecordBatch,
    right: &reclink_core::record::RecordBatch,
) -> Vec<reclink_core::record::CandidatePair> {
    let all: Vec<Vec<reclink_core::record::CandidatePair>> = blockers
        .par_iter()
        .map(|b| b.block_link(left, right))
        .collect();
    dedup_candidate_pairs(all)
}

pub fn compare_pairs(
    comparators: &[Box<dyn reclink_core::compare::FieldComparator>],
    left: &reclink_core::record::RecordBatch,
    right: &reclink_core::record::RecordBatch,
    candidates: &[reclink_core::record::CandidatePair],
) -> Vec<reclink_core::record::ComparisonVector> {
    candidates
        .par_iter()
        .map(|pair| {
            let scores: Vec<f64> = comparators
                .iter()
                .map(|cmp| {
                    let left_val = left.records[pair.left]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(reclink_core::record::FieldValue::Null);
                    let right_val = right.records[pair.right]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(reclink_core::record::FieldValue::Null);
                    cmp.compare(&left_val, &right_val)
                })
                .collect();
            reclink_core::record::ComparisonVector {
                pair: *pair,
                scores,
            }
        })
        .collect()
}

pub fn dedup_candidate_pairs(
    all: Vec<Vec<reclink_core::record::CandidatePair>>,
) -> Vec<reclink_core::record::CandidatePair> {
    let mut seen = ahash::AHashSet::new();
    let mut result = Vec::new();
    for pairs in all {
        for pair in pairs {
            let key = if pair.left <= pair.right {
                (pair.left, pair.right)
            } else {
                (pair.right, pair.left)
            };
            if seen.insert(key) {
                result.push(pair);
            }
        }
    }
    result
}

pub fn cluster_matches(
    cluster_config: &PyClusterConfig,
    matches: &[reclink_core::record::ClassifiedPair],
    n_records: usize,
) -> PyResult<Vec<Vec<usize>>> {
    use reclink_core::cluster::{ConnectedComponents, HierarchicalClustering};

    match cluster_config {
        PyClusterConfig::None => Ok(matches
            .iter()
            .map(|m| vec![m.pair.left, m.pair.right])
            .collect()),
        PyClusterConfig::ConnectedComponents => {
            let edges: Vec<(usize, usize)> = matches
                .iter()
                .map(|m| (m.pair.left, m.pair.right))
                .collect();
            Ok(ConnectedComponents::find(n_records, &edges))
        }
        PyClusterConfig::Hierarchical { linkage, threshold } => {
            let l = match linkage.as_str() {
                "single" => reclink_core::cluster::Linkage::Single,
                "complete" => reclink_core::cluster::Linkage::Complete,
                "average" => reclink_core::cluster::Linkage::Average,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown linkage: {linkage}. Expected: single, complete, average"
                    )));
                }
            };
            let similarities: Vec<(usize, usize, f64)> = matches
                .iter()
                .map(|m| (m.pair.left, m.pair.right, m.aggregate_score))
                .collect();
            let hc = HierarchicalClustering::new(l, *threshold);
            Ok(hc.cluster(n_records, &similarities))
        }
    }
}
