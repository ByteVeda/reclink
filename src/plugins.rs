//! PyO3 registration functions for custom blockers, comparators, classifiers,
//! and preprocessors.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::blocking;
use reclink_core::classify;
use reclink_core::compare;
use reclink_core::preprocess;
use reclink_core::record::{CandidatePair, RecordBatch};

// ---------------------------------------------------------------------------
// Blocker wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python blocker object with `block_dedup` and `block_link` methods.
struct PyBlockerWrapper {
    obj: PyObject,
}

unsafe impl Send for PyBlockerWrapper {}
unsafe impl Sync for PyBlockerWrapper {}

impl PyBlockerWrapper {
    /// Convert a `RecordBatch` into `list[dict[str, str]]` for Python.
    fn batch_to_py(py: Python<'_>, batch: &RecordBatch) -> PyResult<PyObject> {
        use pyo3::types::{PyDict, PyList};

        let mut list = Vec::with_capacity(batch.records.len());
        for record in &batch.records {
            let dict = PyDict::new(py);
            for (k, v) in &record.fields {
                dict.set_item(k, v.to_string())?;
            }
            list.push(dict.into_any().unbind());
        }
        Ok(PyList::new(py, &list)?.into_any().unbind())
    }

    /// Extract `list[tuple[int, int]]` from a Python result.
    fn extract_pairs(py: Python<'_>, result: &PyObject) -> PyResult<Vec<CandidatePair>> {
        let pairs: Vec<(usize, usize)> = result.extract(py)?;
        Ok(pairs
            .into_iter()
            .map(|(left, right)| CandidatePair { left, right })
            .collect())
    }

    fn call_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        Python::with_gil(|py| {
            let py_records = Self::batch_to_py(py, records).ok()?;
            let result = self
                .obj
                .call_method1(py, "block_dedup", (py_records,))
                .ok()?;
            Self::extract_pairs(py, &result).ok()
        })
        .unwrap_or_default()
    }

    fn call_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        Python::with_gil(|py| {
            let py_left = Self::batch_to_py(py, left).ok()?;
            let py_right = Self::batch_to_py(py, right).ok()?;
            let result = self
                .obj
                .call_method1(py, "block_link", (py_left, py_right))
                .ok()?;
            Self::extract_pairs(py, &result).ok()
        })
        .unwrap_or_default()
    }
}

/// Register a custom blocker object.
///
/// The object must have ``block_dedup(records)`` and ``block_link(left, right)``
/// methods, where records/left/right are ``list[dict[str, str]]`` and the return
/// value is ``list[tuple[int, int]]``.
#[pyfunction]
fn register_blocker(name: String, obj: PyObject) -> PyResult<()> {
    // Validate by test-calling block_dedup
    Python::with_gil(|py| -> PyResult<()> {
        use pyo3::types::{PyDict, PyList};

        let test_dict = PyDict::new(py);
        test_dict.set_item("id", "1")?;
        test_dict.set_item("name", "test")?;
        let test_list = PyList::new(py, &[test_dict.into_any()])?;
        let result = obj.call_method1(py, "block_dedup", (test_list,))?;
        let _: Vec<(usize, usize)> = result
            .extract(py)
            .map_err(|_| PyValueError::new_err("block_dedup must return list[tuple[int, int]]"))?;
        Ok(())
    })?;

    let wrapper = Arc::new(PyBlockerWrapper { obj });

    let dedup_wrapper = Arc::clone(&wrapper);
    let dedup_fn: blocking::CustomBlockerDedupFn =
        Arc::new(move |records| dedup_wrapper.call_dedup(records));

    let link_wrapper = Arc::clone(&wrapper);
    let link_fn: blocking::CustomBlockerLinkFn =
        Arc::new(move |left, right| link_wrapper.call_link(left, right));

    blocking::register_custom_blocker(&name, dedup_fn, link_fn)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Unregister a custom blocker. Returns True if it existed.
#[pyfunction]
fn unregister_blocker(name: &str) -> bool {
    blocking::unregister_custom_blocker(name)
}

/// List all registered custom blocker names.
#[pyfunction]
fn list_custom_blockers() -> Vec<String> {
    blocking::list_custom_blockers()
}

// ---------------------------------------------------------------------------
// Comparator wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python callable `(str, str) -> float`.
struct PyComparatorWrapper {
    func: PyObject,
}

unsafe impl Send for PyComparatorWrapper {}
unsafe impl Sync for PyComparatorWrapper {}

impl PyComparatorWrapper {
    fn call(&self, a: &str, b: &str) -> f64 {
        Python::with_gil(|py| {
            self.func
                .call1(py, (a, b))
                .and_then(|r| r.extract::<f64>(py))
                .unwrap_or(0.0)
        })
    }
}

/// Register a custom comparator function.
///
/// The function must accept two strings and return a float in [0, 1].
#[pyfunction]
fn register_comparator(name: String, func: PyObject) -> PyResult<()> {
    // Validate by test-calling
    Python::with_gil(|py| -> PyResult<()> {
        let result = func.call1(py, ("test", "test"))?;
        let _: f64 = result
            .extract(py)
            .map_err(|_| PyValueError::new_err("comparator function must return a float"))?;
        Ok(())
    })?;

    let wrapper = Arc::new(PyComparatorWrapper { func });
    let cmp_fn: compare::CustomComparatorFn = Arc::new(move |a, b| wrapper.call(a, b));

    compare::register_custom_comparator(&name, cmp_fn)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Unregister a custom comparator. Returns True if it existed.
#[pyfunction]
fn unregister_comparator(name: &str) -> bool {
    compare::unregister_custom_comparator(name)
}

/// List all registered custom comparator names.
#[pyfunction]
fn list_custom_comparators() -> Vec<String> {
    compare::list_custom_comparators()
}

// ---------------------------------------------------------------------------
// Classifier wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python callable `(list[float]) -> tuple[float, str]`.
struct PyClassifierWrapper {
    func: PyObject,
}

unsafe impl Send for PyClassifierWrapper {}
unsafe impl Sync for PyClassifierWrapper {}

impl PyClassifierWrapper {
    fn call(&self, scores: &[f64]) -> (f64, reclink_core::record::MatchClass) {
        Python::with_gil(|py| {
            let result = self.func.call1(py, (scores.to_vec(),)).ok()?;
            let (score, class_str): (f64, String) = result.extract(py).ok()?;
            let class = match class_str.as_str() {
                "match" => reclink_core::record::MatchClass::Match,
                "possible" => reclink_core::record::MatchClass::Possible,
                _ => reclink_core::record::MatchClass::NonMatch,
            };
            Some((score, class))
        })
        .unwrap_or((0.0, reclink_core::record::MatchClass::NonMatch))
    }
}

/// Register a custom classifier function.
///
/// The function must accept a list of floats (comparison scores) and return
/// ``(aggregate_score, match_class)`` where ``match_class`` is one of
/// ``"match"``, ``"possible"``, or ``"non_match"``.
#[pyfunction]
fn register_classifier(name: String, func: PyObject) -> PyResult<()> {
    // Validate by test-calling
    Python::with_gil(|py| -> PyResult<()> {
        let result = func.call1(py, (vec![0.5_f64, 0.5_f64],))?;
        let (score, class_str): (f64, String) = result.extract(py).map_err(|_| {
            PyValueError::new_err("classifier function must return tuple[float, str]")
        })?;
        if !["match", "possible", "non_match"].contains(&class_str.as_str()) {
            return Err(PyValueError::new_err(
                "match_class must be 'match', 'possible', or 'non_match'",
            ));
        }
        let _ = score;
        Ok(())
    })?;

    let wrapper = Arc::new(PyClassifierWrapper { func });
    let cls_fn: classify::CustomClassifierFn = Arc::new(move |scores| wrapper.call(scores));

    classify::register_custom_classifier(&name, cls_fn)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Unregister a custom classifier. Returns True if it existed.
#[pyfunction]
fn unregister_classifier(name: &str) -> bool {
    classify::unregister_custom_classifier(name)
}

/// List all registered custom classifier names.
#[pyfunction]
fn list_custom_classifiers() -> Vec<String> {
    classify::list_custom_classifiers()
}

// ---------------------------------------------------------------------------
// Preprocessor wrapper
// ---------------------------------------------------------------------------

/// Wraps a Python callable `(str) -> str`.
struct PyPreprocessorWrapper {
    func: PyObject,
}

unsafe impl Send for PyPreprocessorWrapper {}
unsafe impl Sync for PyPreprocessorWrapper {}

impl PyPreprocessorWrapper {
    fn call(&self, input: &str) -> String {
        Python::with_gil(|py| {
            self.func
                .call1(py, (input,))
                .and_then(|r| r.extract::<String>(py))
                .unwrap_or_else(|_| input.to_string())
        })
    }
}

/// Register a custom preprocessor function.
///
/// The function must accept a string and return a string.
#[pyfunction]
fn register_preprocessor(name: String, func: PyObject) -> PyResult<()> {
    // Validate by test-calling
    Python::with_gil(|py| -> PyResult<()> {
        let result = func.call1(py, ("test",))?;
        let _: String = result
            .extract(py)
            .map_err(|_| PyValueError::new_err("preprocessor function must return a string"))?;
        Ok(())
    })?;

    let wrapper = Arc::new(PyPreprocessorWrapper { func });
    let prep_fn: preprocess::custom::CustomPreprocessFn =
        Arc::new(move |input| wrapper.call(input));

    preprocess::custom::register_custom_preprocessor(&name, prep_fn)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Unregister a custom preprocessor. Returns True if it existed.
#[pyfunction]
fn unregister_preprocessor(name: &str) -> bool {
    preprocess::custom::unregister_custom_preprocessor(name)
}

/// List all registered custom preprocessor names.
#[pyfunction]
fn list_custom_preprocessors() -> Vec<String> {
    preprocess::custom::list_custom_preprocessors()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Blockers
    m.add_function(wrap_pyfunction!(register_blocker, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_blocker, m)?)?;
    m.add_function(wrap_pyfunction!(list_custom_blockers, m)?)?;
    // Comparators
    m.add_function(wrap_pyfunction!(register_comparator, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_comparator, m)?)?;
    m.add_function(wrap_pyfunction!(list_custom_comparators, m)?)?;
    // Classifiers
    m.add_function(wrap_pyfunction!(register_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(list_custom_classifiers, m)?)?;
    // Preprocessors
    m.add_function(wrap_pyfunction!(register_preprocessor, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_preprocessor, m)?)?;
    m.add_function(wrap_pyfunction!(list_custom_preprocessors, m)?)?;
    Ok(())
}
