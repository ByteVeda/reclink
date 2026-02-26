//! Clustering algorithms for grouping matched records.

mod connected_components;
mod hierarchical;

pub use connected_components::ConnectedComponents;
pub use hierarchical::{HierarchicalClustering, Linkage};
