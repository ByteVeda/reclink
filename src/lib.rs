//! PyO3 bindings for the reclink library.
//!
//! Exposes string similarity metrics, phonetic algorithms, preprocessing,
//! and a record linkage pipeline to Python.

mod arrow_interop;
mod explain;
mod index;
mod metrics;
mod parsers;
mod phonetic;
mod pipeline;
#[cfg(feature = "polars-plugin")]
mod polars_plugin;
mod preprocess;
mod scoring;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    metrics::register(m)?;
    phonetic::register(m)?;
    preprocess::register(m)?;
    explain::register(m)?;
    index::register(m)?;
    scoring::register(m)?;
    pipeline::register(m)?;
    arrow_interop::register(m)?;
    Ok(())
}
