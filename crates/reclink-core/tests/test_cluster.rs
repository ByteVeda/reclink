use reclink_core::cluster::{HierarchicalClustering, Linkage};

#[test]
fn average_linkage() {
    let hc = HierarchicalClustering::new(Linkage::Average, 0.5);
    // Three nodes: 0-1 very similar, 1-2 moderately similar, 3-4 similar
    let sims = vec![(0, 1, 0.9), (1, 2, 0.7), (3, 4, 0.85)];
    let clusters = hc.cluster(5, &sims);
    assert!(!clusters.is_empty(), "should produce at least one cluster");
    // nodes 3 and 4 should definitely be clustered (distance=0.15 < threshold=0.5)
    let has_34 = clusters.iter().any(|c| c.contains(&3) && c.contains(&4));
    assert!(has_34, "nodes 3 and 4 should be in the same cluster");
}

#[test]
fn single_linkage_membership() {
    let hc = HierarchicalClustering::new(Linkage::Single, 0.5);
    // Chain: 0-1 (dist=0.1), 1-2 (dist=0.3) — single linkage should merge all three
    let sims = vec![(0, 1, 0.9), (1, 2, 0.7)];
    let clusters = hc.cluster(3, &sims);
    assert_eq!(clusters.len(), 1, "single linkage should merge the chain");
    let cluster = &clusters[0];
    assert!(cluster.contains(&0));
    assert!(cluster.contains(&1));
    assert!(cluster.contains(&2));
}
