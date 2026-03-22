//! Clustering algorithms for grouping matched records.

mod connected_components;
mod hierarchical;
pub mod incremental;

pub use connected_components::ConnectedComponents;
pub use hierarchical::{HierarchicalClustering, Linkage};
pub use incremental::{ClusterAssignment, IncrementalCluster};
