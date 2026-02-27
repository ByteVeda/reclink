use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::metrics;

/// Compare two strings with multiple algorithms and return the score breakdown.
///
/// Returns a dict mapping algorithm names to similarity scores.
/// If `algorithms` is None, runs all 17 metrics.
#[pyfunction]
#[pyo3(signature = (a, b, algorithms=None))]
fn explain(
    a: &str,
    b: &str,
    algorithms: Option<Vec<String>>,
) -> PyResult<std::collections::HashMap<String, f64>> {
    let result = match algorithms {
        None => reclink_core::metrics::explain::explain(a, b),
        Some(names) => {
            let metrics: Vec<reclink_core::metrics::Metric> = names
                .iter()
                .map(|n| {
                    metrics::metric_from_name(n).map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            reclink_core::metrics::explain::explain_with(a, b, &metrics)
        }
    };

    Ok(result
        .scores
        .into_iter()
        .map(|s| (s.algorithm, s.score))
        .collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(explain, m)?)?;
    Ok(())
}
