#[derive(Debug, Clone)]
pub enum PyBlockerConfig {
    Exact {
        field: String,
    },
    Phonetic {
        field: String,
        algorithm: String,
    },
    SortedNeighborhood {
        field: String,
        window: usize,
    },
    Qgram {
        field: String,
        q: usize,
        threshold: usize,
    },
    Lsh {
        field: String,
        num_hashes: usize,
        num_bands: usize,
    },
    Canopy {
        field: String,
        t_tight: f64,
        t_loose: f64,
        metric: String,
    },
    Numeric {
        field: String,
        bucket_size: f64,
    },
    DateBlock {
        field: String,
        resolution: String,
    },
}

#[derive(Debug, Clone)]
pub enum PyClusterConfig {
    None,
    ConnectedComponents,
    Hierarchical { linkage: String, threshold: f64 },
}

#[derive(Debug, Clone)]
pub enum PyComparatorConfig {
    String { field: String, metric: String },
    Exact { field: String },
    Numeric { field: String, max_diff: f64 },
    Date { field: String },
    Phonetic { field: String, algorithm: String },
}

#[derive(Debug, Clone)]
pub enum PyClassifierConfig {
    Threshold {
        threshold: f64,
    },
    Weighted {
        weights: Vec<f64>,
        threshold: f64,
    },
    FellegiSunter {
        m_probs: Vec<f64>,
        u_probs: Vec<f64>,
        upper: f64,
        lower: f64,
    },
    FellegiSunterAuto {
        max_iterations: usize,
        convergence_threshold: f64,
        initial_p_match: f64,
    },
}
