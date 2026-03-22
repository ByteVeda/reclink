use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::blocking;
use reclink_core::compare;
use reclink_core::metrics;
use reclink_core::preprocess;

use super::config::{PyBlockerConfig, PyClassifierConfig, PyClusterConfig, PyComparatorConfig};
use super::PyPipeline;
use super::PyRecord;
use crate::parsers::{parse_date_resolution, parse_phonetic_algorithm, parse_preprocess_ops};

impl PyPipeline {
    pub fn build_record_batch(
        &self,
        records: &[PyRef<PyRecord>],
    ) -> PyResult<reclink_core::record::RecordBatch> {
        // Pre-parse preprocess ops for each field
        let mut parsed_ops: ahash::AHashMap<String, Vec<preprocess::PreprocessOp>> =
            ahash::AHashMap::new();
        for (field, ops) in &self.preprocess_ops {
            parsed_ops.insert(field.clone(), parse_preprocess_ops(ops)?);
        }

        let mut field_names: Vec<String> = Vec::new();
        if let Some(first) = records.first() {
            field_names = first.fields.keys().cloned().collect();
            field_names.sort();
        }

        let core_records: Vec<reclink_core::record::Record> = records
            .iter()
            .map(|r| {
                let mut rec = reclink_core::record::Record::new(r.id.clone());
                for (k, v) in &r.fields {
                    let field_value = if self.numeric_fields.contains(k) {
                        // Try to parse as numeric
                        if let Ok(f) = v.parse::<f64>() {
                            reclink_core::record::FieldValue::Float(f)
                        } else {
                            reclink_core::record::FieldValue::Text(v.clone())
                        }
                    } else if self.date_fields.contains(k) {
                        reclink_core::record::FieldValue::Date(v.clone())
                    } else {
                        // Apply per-field preprocess ops
                        let mut value = v.clone();
                        if let Some(ops) = parsed_ops.get(k) {
                            value = preprocess::apply_ops(&value, ops).unwrap_or(value);
                        }
                        // Backward compat: preprocess_lowercase
                        if self.preprocess_lowercase.contains(k) {
                            value = value.to_lowercase();
                        }
                        reclink_core::record::FieldValue::Text(value)
                    };
                    rec.fields.insert(k.clone(), field_value);
                }
                rec
            })
            .collect();

        Ok(reclink_core::record::RecordBatch::new(
            field_names,
            core_records,
        ))
    }

    pub fn build_blockers(
        &self,
    ) -> PyResult<Vec<Box<dyn reclink_core::blocking::BlockingStrategy>>> {
        use reclink_core::blocking::*;

        let mut blockers: Vec<Box<dyn BlockingStrategy>> = Vec::new();
        for blocker_cfg in &self.blockers {
            let blocker: Box<dyn BlockingStrategy> = match blocker_cfg {
                PyBlockerConfig::Exact { field } => Box::new(ExactBlocking::new(field.clone())),
                PyBlockerConfig::Phonetic { field, algorithm } => {
                    let algo = parse_phonetic_algorithm(algorithm)?;
                    Box::new(PhoneticBlocking::new(field.clone(), algo))
                }
                PyBlockerConfig::SortedNeighborhood { field, window } => {
                    Box::new(SortedNeighborhood::new(field.clone(), *window))
                }
                PyBlockerConfig::Qgram {
                    field,
                    q,
                    threshold,
                } => Box::new(QgramBlocking::new(field.clone(), *q, *threshold)),
                PyBlockerConfig::Lsh {
                    field,
                    num_hashes,
                    num_bands,
                } => Box::new(LshBlocking::new(field.clone(), *num_hashes, *num_bands)),
                PyBlockerConfig::Canopy {
                    field,
                    t_tight,
                    t_loose,
                    metric,
                } => {
                    let m = metrics::metric_from_name(metric)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Box::new(CanopyClustering::new(field.clone(), *t_tight, *t_loose, m))
                }
                PyBlockerConfig::Numeric { field, bucket_size } => {
                    Box::new(NumericBlocking::new(field.clone(), *bucket_size))
                }
                PyBlockerConfig::DateBlock { field, resolution } => {
                    let res = parse_date_resolution(resolution)?;
                    Box::new(DateBlocking::new(field.clone(), res))
                }
                PyBlockerConfig::Trie {
                    field,
                    min_prefix_len,
                    max_frequency,
                } => Box::new(TrieBlocking::new(
                    field.clone(),
                    *min_prefix_len,
                    *max_frequency,
                )),
                PyBlockerConfig::Custom { name } => Box::new(
                    blocking::custom_blocker_from_name(name)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
            };
            blockers.push(blocker);
        }
        Ok(blockers)
    }

    pub fn build_comparators(
        &self,
    ) -> PyResult<Vec<Box<dyn reclink_core::compare::FieldComparator>>> {
        use reclink_core::compare::*;

        let mut comparators: Vec<Box<dyn FieldComparator>> = Vec::new();
        for comp_cfg in &self.comparators {
            let comparator: Box<dyn FieldComparator> = match comp_cfg {
                PyComparatorConfig::String { field, metric } => {
                    let m = metrics::metric_from_name(metric)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Box::new(StringComparator::new(field.clone(), m))
                }
                PyComparatorConfig::Exact { field } => {
                    Box::new(ExactComparator::new(field.clone()))
                }
                PyComparatorConfig::Numeric { field, max_diff } => {
                    Box::new(NumericComparator::new(field.clone(), *max_diff))
                }
                PyComparatorConfig::Date { field } => Box::new(DateComparator::new(field.clone())),
                PyComparatorConfig::Phonetic { field, algorithm } => {
                    let algo = parse_phonetic_algorithm(algorithm)?;
                    Box::new(PhoneticComparator::new(field.clone(), algo))
                }
                PyComparatorConfig::Custom { field, name } => Box::new(
                    compare::custom_comparator_from_name(field, name)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
            };
            comparators.push(comparator);
        }
        Ok(comparators)
    }

    pub fn build_pipeline(&self) -> PyResult<reclink_core::pipeline::ReclinkPipeline> {
        use reclink_core::classify::*;

        let mut builder = reclink_core::pipeline::PipelineBuilder::new();

        for blocker in self.build_blockers()? {
            builder = builder.add_blocker(blocker);
        }

        for comparator in self.build_comparators()? {
            builder = builder.add_comparator(comparator);
        }

        if let Some(ref cls_cfg) = self.classifier {
            let classifier: Box<dyn Classifier> = match cls_cfg {
                PyClassifierConfig::Threshold { threshold } => {
                    Box::new(ThresholdClassifier::new(*threshold))
                }
                PyClassifierConfig::Weighted { weights, threshold } => {
                    Box::new(WeightedSumClassifier::new(weights.clone(), *threshold))
                }
                PyClassifierConfig::FellegiSunter {
                    m_probs,
                    u_probs,
                    upper,
                    lower,
                } => Box::new(FellegiSunterClassifier::new(
                    m_probs.clone(),
                    u_probs.clone(),
                    *upper,
                    *lower,
                )),
                PyClassifierConfig::ThresholdBands { upper, lower } => {
                    Box::new(ThresholdBandsClassifier::new(*upper, *lower))
                }
                PyClassifierConfig::WeightedBands {
                    weights,
                    upper,
                    lower,
                } => Box::new(WeightedSumBandsClassifier::new(
                    weights.clone(),
                    *upper,
                    *lower,
                )),
                PyClassifierConfig::FellegiSunterAuto { .. } => {
                    // Handled in dedup/link methods directly
                    unreachable!("FellegiSunterAuto should be handled before build_pipeline")
                }
                PyClassifierConfig::Custom { name } => Box::new(
                    reclink_core::classify::custom_classifier_from_name(name)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
            };
            builder = builder.set_classifier(classifier);
        }

        match &self.cluster {
            PyClusterConfig::None => {}
            PyClusterConfig::ConnectedComponents => {
                builder = builder.with_clustering();
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
                builder = builder.with_hierarchical_clustering(l, *threshold);
            }
        }

        builder
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
