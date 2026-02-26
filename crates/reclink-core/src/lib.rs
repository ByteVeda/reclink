//! # reclink-core
//!
//! Pure Rust library for blazing-fast fuzzy string matching and record linkage.
//!
//! This crate provides:
//! - 8 string similarity/distance metrics
//! - 4 phonetic encoding algorithms
//! - Text preprocessing and normalization
//! - Blocking strategies for candidate pair generation
//! - Field comparators and classifiers
//! - A builder-pattern pipeline for end-to-end record linkage

pub mod blocking;
pub mod classify;
pub mod cluster;
pub mod compare;
pub mod error;
pub mod metrics;
pub mod phonetic;
pub mod pipeline;
pub mod preprocess;
pub mod record;
