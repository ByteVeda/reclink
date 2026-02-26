//! Connected components clustering using Union-Find.

/// Finds connected components in a graph of matched record pairs using Union-Find.
#[derive(Debug)]
pub struct ConnectedComponents;

impl ConnectedComponents {
    /// Finds connected components given the number of nodes and edges.
    ///
    /// Returns a vector of components, where each component is a vector of node indices.
    #[must_use]
    pub fn find(num_nodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut uf = UnionFind::new(num_nodes);

        for &(a, b) in edges {
            uf.union(a, b);
        }

        let mut components: ahash::AHashMap<usize, Vec<usize>> = ahash::AHashMap::new();
        for i in 0..num_nodes {
            let root = uf.find(i);
            components.entry(root).or_default().push(i);
        }

        components.into_values().filter(|c| c.len() > 1).collect()
    }
}

/// Union-Find data structure with path compression and union by rank.
#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_components() {
        let components = ConnectedComponents::find(5, &[(0, 1), (2, 3)]);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn transitive_closure() {
        let components = ConnectedComponents::find(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);
    }

    #[test]
    fn no_edges() {
        let components = ConnectedComponents::find(3, &[]);
        assert!(components.is_empty()); // No multi-node components
    }
}
