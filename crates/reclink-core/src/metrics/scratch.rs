//! Thread-local scratch buffers for amortizing allocations across repeated metric calls.
//!
//! Each Rayon worker thread gets its own set of buffers via `thread_local!`,
//! so there is zero contention. The buffers grow as needed and retain their
//! capacity across calls.

use std::cell::RefCell;

use ahash::AHashMap;

// ─── Levenshtein (multi-block Myers) ─────────────────────────────────────────

pub(crate) struct LevenshteinScratch {
    pub pm: AHashMap<char, Vec<u64>>,
    pub vp: Vec<u64>,
    pub vn: Vec<u64>,
    pub empty_pm: Vec<u64>,
}

impl LevenshteinScratch {
    fn new() -> Self {
        Self {
            pm: AHashMap::new(),
            vp: Vec::new(),
            vn: Vec::new(),
            empty_pm: Vec::new(),
        }
    }

    pub fn reset(&mut self, num_blocks: usize) {
        self.pm.clear();
        self.vp.clear();
        self.vp.resize(num_blocks, !0u64);
        self.vn.clear();
        self.vn.resize(num_blocks, 0);
        self.empty_pm.clear();
        self.empty_pm.resize(num_blocks, 0);
    }
}

thread_local! {
    pub(crate) static LEV_SCRATCH: RefCell<LevenshteinScratch> =
        RefCell::new(LevenshteinScratch::new());
}

// ─── Damerau-Levenshtein (3-row DP) ─────────────────────────────────────────

pub(crate) struct DamerauLevenshteinScratch {
    pub prev_prev: Vec<usize>,
    pub prev: Vec<usize>,
    pub curr: Vec<usize>,
}

impl DamerauLevenshteinScratch {
    fn new() -> Self {
        Self {
            prev_prev: Vec::new(),
            prev: Vec::new(),
            curr: Vec::new(),
        }
    }

    pub fn reset(&mut self, cols: usize) {
        self.prev_prev.clear();
        self.prev_prev.resize(cols + 1, 0);
        self.prev.clear();
        self.prev = (0..=cols).collect();
        self.curr.clear();
        self.curr.resize(cols + 1, 0);
    }
}

thread_local! {
    pub(crate) static DL_SCRATCH: RefCell<DamerauLevenshteinScratch> =
        RefCell::new(DamerauLevenshteinScratch::new());
}

// ─── LCS (multi-block bit-parallel) ─────────────────────────────────────────

pub(crate) struct LcsScratch {
    pub pm: AHashMap<char, Vec<u64>>,
    pub s: Vec<u64>,
    pub empty_pm: Vec<u64>,
}

impl LcsScratch {
    fn new() -> Self {
        Self {
            pm: AHashMap::new(),
            s: Vec::new(),
            empty_pm: Vec::new(),
        }
    }

    pub fn reset(&mut self, num_blocks: usize) {
        self.pm.clear();
        self.s.clear();
        self.s.resize(num_blocks, 0);
        self.empty_pm.clear();
        self.empty_pm.resize(num_blocks, 0);
    }
}

thread_local! {
    pub(crate) static LCS_SCRATCH: RefCell<LcsScratch> =
        RefCell::new(LcsScratch::new());
}

// ─── Smith-Waterman (single-row DP) ─────────────────────────────────────────

pub(crate) struct SmithWatermanScratch {
    pub prev: Vec<f64>,
}

impl SmithWatermanScratch {
    fn new() -> Self {
        Self { prev: Vec::new() }
    }

    pub fn reset(&mut self, b_len: usize) {
        self.prev.clear();
        self.prev.resize(b_len + 1, 0.0);
    }
}

thread_local! {
    pub(crate) static SW_SCRATCH: RefCell<SmithWatermanScratch> =
        RefCell::new(SmithWatermanScratch::new());
}
