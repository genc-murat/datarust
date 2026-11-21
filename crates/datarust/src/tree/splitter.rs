//! Exact port of scikit-learn 1.6.1's dense `BestSplitter` and
//! `RandomSplitter` machinery.
//!
//! This is a faithful transliteration of sklearn's Cython implementation
//! (`tree/_splitter.pyx` `node_split_best` / `node_split_random`,
//! `tree/_partitioner.pyx`
//! `DensePartitioner`, `tree/_criterion.pyx`, and the MT19937/`rand_r`
//! primitives from `utils/_random.pxd`). The goal is **bit-exact parity**:
//! given the same `X`, `y`, parameters, and integer `random_state`, the
//! grown tree matches sklearn's node-for-node (features, thresholds,
//! impurities, importances).
//!
//! Ported behaviors that matter for parity:
//!
//! - **Feature draws** (Fisher-Yates over `features`, preserving
//!   `constant_features` across sibling/child nodes) — sklearn permutes
//!   features at *every* split even with `splitter="best"`; ties therefore
//!   resolve by the RNG stream, not feature order.
//! - **Seeding chain**: `np.random.RandomState(seed).randint(0, 2³¹−1)`
//!   (legacy MT19937, masked rejection) → the xorshift `rand_r` seed.
//! - **Random thresholds** (`node_split_random`, sklearn's
//!   `splitter="random"` used by ExtraTrees): one `rand_uniform` draw per
//!   non-constant feature on the *same* `rand_r` stream as the feature
//!   draws; `find_min_max` + `partition_samples` replace the sort + scan.
//! - **float32 feature quantization**: sklearn converts `X` to float32;
//!   thresholds are computed as `f32(v_prev)/2 + f32(v)/2` in float64.
//! - **Introsort** (median-of-3, 3-way partition, heapsort fallback) — the
//!   sample order after sorting feeds the criterion's floating-point
//!   accumulation order, so the sort must be reproduced exactly.
//! - **Criterion arithmetic**: running `sum_left` updates choose between
//!   forward accumulation and reverse subtraction by sklearn's
//!   "fewer moves" rule; regression `children_impurity` re-derives the left
//!   square-sum by scanning `[start, pos)`; `proxy_impurity_improvement` and
//!   `impurity_improvement` use sklearn's exact expressions.
//! - Weighted samples supported via the internal weighted-fit path (the
//!   forest's bootstrap multiplicities); no missing-value support (`X` is
//!   validated finite upstream, so `n_missing == 0` paths are hardcoded).
//!
//! The builder (`tree::grow_tree`) mirrors `DepthFirstTreeBuilder.build`.

use crate::matrix::Matrix;

use super::node::ValueSummary;
use super::{Criterion, MaxFeatures};

// ---------------------------------------------------------------------------
// RNG: sklearn's rand_r (xorshift) + numpy legacy MT19937 seeding
// ---------------------------------------------------------------------------

/// `RAND_R_MAX` from `sklearn/tree/_utils.pxd`.
const RAND_R_MAX: u32 = 2147483647;

/// `our_rand_r` from `sklearn/utils/_random.pxd`: a 32-bit xorshift whose
/// output is clamped to `2³¹ − 1`.
#[inline]
fn our_rand_r(seed: &mut u32) -> u32 {
    if *seed == 0 {
        *seed = 1;
    }
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    // RAND_R_MAX cast to u32 before +1: 2³¹ (no overflow).
    *seed % (RAND_R_MAX.wrapping_add(1))
}

/// `rand_int(low, high, random_state)` from `_utils.pyx` (exclusive `high`).
#[inline]
fn rand_int(low: usize, high: usize, seed: &mut u32) -> usize {
    debug_assert!(high > low, "rand_int requires high > low");
    low + (our_rand_r(seed) as usize) % (high - low)
}

/// `rand_uniform(low, high, random_state)` from `tree/_utils.pyx`: the f32
/// bounds are widened to f64, and the division precedes the addition (no
/// FMA contraction).
#[inline]
fn rand_uniform(low: f32, high: f32, seed: &mut u32) -> f64 {
    ((high as f64 - low as f64) * our_rand_r(seed) as f64 / RAND_R_MAX as f64) + low as f64
}

/// Minimal legacy MT19937 (matches `numpy.random.RandomState`).
struct Mt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl Mt19937 {
    /// `mt19937_seed` — scalar seeding (`init_genrand`).
    fn new(seed: u64) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed as u32;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, mti: 624 }
    }

    /// Next tempered 32-bit output.
    fn next_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        if self.mti >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % N] & LOWER);
                self.mt[i] = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    self.mt[i] ^= MATRIX_A;
                }
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Emulates `RandomState(seed).randint(0, RAND_R_MAX)` with masked
    /// rejection (legacy `use_masked=True` path): draw `next_u32() &
    /// mask` until the value is within `[0, high)` where
    /// `high = RAND_R_MAX` and `mask = 2³¹ − 1`.
    fn randint_half_open(&mut self, high: u32) -> u32 {
        debug_assert!(high > 0);
        let rng = high - 1; // inclusive bound
        let mask = u32::MAX >> (rng.leading_zeros()); // smallest 2^k−1 ≥ rng
        loop {
            let val = self.next_u32() & mask;
            if val <= rng {
                return val;
            }
        }
    }
}

/// Reproduces sklearn's `Splitter.init` seeding:
/// `self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)`.
fn rand_r_seed_from(random_state: u64) -> u32 {
    Mt19937::new(random_state).randint_half_open(RAND_R_MAX)
}

/// `FEATURE_THRESHOLD` from `_partitioner.pxd` (single precision).
const FEATURE_THRESHOLD: f32 = 1e-7;

// ---------------------------------------------------------------------------
// Introsort (exact port of `_partitioner.pyx::sort`)
// ---------------------------------------------------------------------------

/// Sorts `feature_values[off..off+n]` ascending while permuting
/// `samples[off..off+n]` identically. Unstable (introsort), exactly like
/// sklearn — the resulting order determines float accumulation order.
fn sort_rf(fv: &mut [f32], sa: &mut [usize], off: usize, n: usize) {
    if n == 0 {
        return;
    }
    let maxd = 2 * (n as f64).log2().floor() as i64;
    introsort(fv, sa, off, n, maxd);
}

fn swap_rf(fv: &mut [f32], sa: &mut [usize], off: usize, i: usize, j: usize) {
    fv.swap(off + i, off + j);
    sa.swap(off + i, off + j);
}

/// Median of three pivot selection (Bentley & McIlroy).
fn median3(fv: &[f32], off: usize, n: usize) -> f32 {
    let a = fv[off];
    let b = fv[off + n / 2];
    let c = fv[off + n - 1];
    if a < b {
        if b < c {
            b
        } else if a < c {
            c
        } else {
            a
        }
    } else if b < c {
        if a < c {
            a
        } else {
            c
        }
    } else {
        b
    }
}

fn introsort(fv: &mut [f32], sa: &mut [usize], mut off: usize, mut n: usize, mut maxd: i64) {
    while n > 1 {
        if maxd <= 0 {
            heapsort_rf(fv, sa, off, n);
            return;
        }
        maxd -= 1;

        let pivot = median3(fv, off, n);

        // Three-way partition.
        let mut i = 0usize;
        let mut l = 0usize;
        let mut r = n;
        while i < r {
            if fv[off + i] < pivot {
                swap_rf(fv, sa, off, i, l);
                i += 1;
                l += 1;
            } else if fv[off + i] > pivot {
                r -= 1;
                swap_rf(fv, sa, off, i, r);
            } else {
                i += 1;
            }
        }

        introsort(fv, sa, off, l, maxd);
        off += r;
        n -= r;
    }
}

fn sift_down(fv: &mut [f32], sa: &mut [usize], off: usize, start: usize, end: usize) {
    let mut root = start;
    loop {
        let child = root * 2 + 1;
        let mut maxind = root;
        if child < end && fv[off + maxind] < fv[off + child] {
            maxind = child;
        }
        if child + 1 < end && fv[off + maxind] < fv[off + child + 1] {
            maxind = child + 1;
        }
        if maxind == root {
            break;
        }
        swap_rf(fv, sa, off, root, maxind);
        root = maxind;
    }
}

fn heapsort_rf(fv: &mut [f32], sa: &mut [usize], off: usize, n: usize) {
    // heapify
    let mut start = (n as i64 - 2) / 2; // C truncation semantics; n ≥ 2 here
    let end = n;
    loop {
        sift_down(fv, sa, off, start as usize, end);
        if start == 0 {
            break;
        }
        start -= 1;
    }
    // sort by shrinking the heap
    let mut end = n - 1;
    while end > 0 {
        swap_rf(fv, sa, off, 0, end);
        sift_down(fv, sa, off, 0, end);
        end -= 1;
    }
}

// ---------------------------------------------------------------------------
// Criterion (exact port of `_criterion.pyx`, single output, uniform weights)
// ---------------------------------------------------------------------------

/// Running split statistics for one node. Mirrors sklearn's `Criterion`
/// subclass state (`sum_total/left/right`, `sq_sum_total`, weighted counts).
struct CriterionState {
    kind: Criterion,
    n_classes: usize,
    /// Sample window of the current node within `samples`.
    start: usize,
    end: usize,
    pos: usize,
    sum_total: Vec<f64>,
    sum_left: Vec<f64>,
    sum_right: Vec<f64>,
    sq_sum_total: f64,
    weighted_n_node_samples: f64,
    weighted_n_left: f64,
    weighted_n_right: f64,
    weighted_n_samples: f64,
    sample_weights: Vec<f64>,
}

impl CriterionState {
    #[cfg(test)]
    fn new(kind: Criterion, n_classes: usize, weighted_n_samples: f64) -> Self {
        Self::with_weights(
            kind,
            n_classes,
            vec![1.0; weighted_n_samples.max(0.0) as usize],
        )
    }

    fn with_weights(kind: Criterion, n_classes: usize, sample_weights: Vec<f64>) -> Self {
        let weighted_n_samples = sample_weights.iter().sum();
        let width = match kind {
            Criterion::SquaredError => 1,
            _ => n_classes,
        };
        CriterionState {
            kind,
            n_classes: width,
            start: 0,
            end: 0,
            pos: 0,
            sum_total: vec![0.0; width],
            sum_left: vec![0.0; width],
            sum_right: vec![0.0; width],
            sq_sum_total: 0.0,
            weighted_n_node_samples: 0.0,
            weighted_n_left: 0.0,
            weighted_n_right: 0.0,
            weighted_n_samples,
            sample_weights,
        }
    }

    /// `ClassificationCriterion.init` / `RegressionCriterion.init`.
    ///
    /// Accumulates statistics over `samples[start..end]` **in that order**
    /// (the samples array is the partitioner-maintained order — this is what
    /// fixes the floating-point accumulation order).
    fn init(
        &mut self,
        y: &[f64],
        labels: Option<&[usize]>,
        samples: &[usize],
        start: usize,
        end: usize,
    ) {
        self.start = start;
        self.end = end;
        self.sum_total.fill(0.0);
        self.sq_sum_total = 0.0;
        self.weighted_n_node_samples = 0.0;
        for &i in &samples[start..end] {
            let weight = self.sample_weights[i];
            match self.kind {
                Criterion::SquaredError => {
                    let y_ik = y[i];
                    self.sum_total[0] = weight.mul_add(y_ik, self.sum_total[0]);
                    self.sq_sum_total += weight * y_ik * y_ik;
                }
                _ => {
                    let c = labels.expect("classification criterion without labels")[i];
                    self.sum_total[c] += weight;
                }
            }
            self.weighted_n_node_samples += weight;
        }
        self.reset();
    }

    /// `reset`: criterion at `pos = start`.
    fn reset(&mut self) {
        self.pos = self.start;
        let (n, st) = (self.n_classes, self.weighted_n_node_samples);
        let (sl, sr, wl, wr) = (
            &mut self.sum_left,
            &mut self.sum_right,
            &mut self.weighted_n_left,
            &mut self.weighted_n_right,
        );
        sl[..n].fill(0.0);
        sr[..n].copy_from_slice(&self.sum_total[..n]);
        *wl = 0.0;
        *wr = st;
    }

    /// `reverse_reset`: criterion at `pos = end`.
    fn reverse_reset(&mut self) {
        self.pos = self.end;
        let (n, st) = (self.n_classes, self.weighted_n_node_samples);
        let (sl, sr, wl, wr) = (
            &mut self.sum_left,
            &mut self.sum_right,
            &mut self.weighted_n_left,
            &mut self.weighted_n_right,
        );
        sl[..n].copy_from_slice(&self.sum_total[..n]);
        sr[..n].fill(0.0);
        *wl = st;
        *wr = 0.0;
    }

    /// `update(new_pos)`: move `samples[pos..new_pos]` to the left child,
    /// choosing the cheaper direction exactly like sklearn.
    fn update(&mut self, y: &[f64], labels: Option<&[usize]>, samples: &[usize], new_pos: usize) {
        let pos = self.pos;
        let end = self.end;
        if new_pos - pos <= end - new_pos {
            // Forward: accumulate samples[pos..new_pos] into the left child.
            for &i in &samples[pos..new_pos] {
                let weight = self.sample_weights[i];
                match self.kind {
                    Criterion::SquaredError => {
                        self.sum_left[0] = weight.mul_add(y[i], self.sum_left[0])
                    }
                    _ => {
                        let c = labels.expect("labels")[i];
                        self.sum_left[c] += weight;
                    }
                }
                self.weighted_n_left += weight;
            }
        } else {
            // Reverse: start from `pos = end` and subtract
            // samples[end-1..new_pos] (reverse order matters for floats).
            self.reverse_reset();
            let mut p = end;
            while p > new_pos {
                p -= 1;
                let i = samples[p];
                let weight = self.sample_weights[i];
                match self.kind {
                    Criterion::SquaredError => {
                        self.sum_left[0] = (-weight).mul_add(y[i], self.sum_left[0])
                    }
                    _ => {
                        let c = labels.expect("labels")[i];
                        self.sum_left[c] -= weight;
                    }
                }
                self.weighted_n_left -= weight;
            }
        }
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left;
        for k in 0..self.n_classes {
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k];
        }
        self.pos = new_pos;
    }

    /// `node_impurity` for the root node.
    fn node_impurity(&self) -> f64 {
        let w = self.weighted_n_node_samples;
        match self.kind {
            Criterion::Gini => {
                let mut sq_count = 0.0;
                for &c in &self.sum_total {
                    sq_count += c * c;
                }
                1.0 - sq_count / (w * w)
            }
            Criterion::Entropy => {
                let mut entropy = 0.0;
                for &c in &self.sum_total {
                    if c > 0.0 {
                        let p = c / w;
                        // sklearn: `entropy -= count_k * log(count_k)` where
                        // `log(x) = ln(x) / ln(2.0)`. clang contracts the
                        // product into the subtraction (FMA); bit parity
                        // requires the same single-rounding form.
                        entropy = f64::mul_add(-p, p.ln() / std::f64::consts::LN_2, entropy);
                    }
                }
                entropy
            }
            Criterion::SquaredError => {
                let mut impurity = self.sq_sum_total / w;
                impurity -= (self.sum_total[0] / w).powi(2);
                impurity
            }
        }
    }

    /// `children_impurity` at the current `pos`.
    fn children_impurity(&self, y: &[f64], samples: &[usize]) -> (f64, f64) {
        let pos = self.pos;
        let start = self.start;
        let wnl = self.weighted_n_left;
        let wnr = self.weighted_n_right;
        match self.kind {
            Criterion::Gini => {
                let mut sq_left = 0.0;
                let mut sq_right = 0.0;
                for k in 0..self.n_classes {
                    sq_left += self.sum_left[k] * self.sum_left[k];
                    sq_right += self.sum_right[k] * self.sum_right[k];
                }
                (1.0 - sq_left / (wnl * wnl), 1.0 - sq_right / (wnr * wnr))
            }
            Criterion::Entropy => {
                let mut el = 0.0;
                let mut er = 0.0;
                for k in 0..self.n_classes {
                    if self.sum_left[k] > 0.0 {
                        let p = self.sum_left[k] / wnl;
                        el = f64::mul_add(-p, p.ln() / std::f64::consts::LN_2, el);
                    }
                    if self.sum_right[k] > 0.0 {
                        let p = self.sum_right[k] / wnr;
                        er = f64::mul_add(-p, p.ln() / std::f64::consts::LN_2, er);
                    }
                }
                (el, er)
            }
            Criterion::SquaredError => {
                // MSE re-derives the left square-sum by scanning
                // samples[start..pos] — exact sklearn order.
                let mut sq_sum_left = 0.0;
                for &i in &samples[start..pos] {
                    let weight = self.sample_weights[i];
                    sq_sum_left += weight * y[i] * y[i];
                }
                let sq_sum_right = self.sq_sum_total - sq_sum_left;
                let mut il = sq_sum_left / wnl;
                let mut ir = sq_sum_right / wnr;
                il -= (self.sum_left[0] / wnl).powi(2);
                ir -= (self.sum_right[0] / wnr).powi(2);
                (il, ir)
            }
        }
    }

    /// `proxy_impurity_improvement` (sklearn's exact expressions).
    fn proxy_impurity_improvement(&self, y: &[f64], samples: &[usize]) -> f64 {
        match self.kind {
            Criterion::SquaredError => {
                let (mut pl, mut pr) = (0.0, 0.0);
                pl += self.sum_left[0] * self.sum_left[0];
                pr += self.sum_right[0] * self.sum_right[0];
                pl / self.weighted_n_left + pr / self.weighted_n_right
            }
            _ => {
                let (il, ir) = self.children_impurity(y, samples);
                // sklearn's Cython compiles `-wr*ir - wl*il` with FP
                // contraction on arm64: the right product rounds first and
                // the left one fuses into the subtraction. Plain two-step
                // rounding ties mirror-image splits (n_L/n_R swapped with
                // matching counts) that sklearn breaks by 1 ulp.
                f64::mul_add(-self.weighted_n_right, ir, -(self.weighted_n_left * il))
            }
        }
    }

    /// `impurity_improvement` (the value stored on nodes).
    fn impurity_improvement(
        &self,
        impurity_parent: f64,
        impurity_left: f64,
        impurity_right: f64,
    ) -> f64 {
        (self.weighted_n_node_samples / self.weighted_n_samples)
            * (impurity_parent
                - (self.weighted_n_right / self.weighted_n_node_samples * impurity_right)
                - (self.weighted_n_left / self.weighted_n_node_samples * impurity_left))
    }

    /// `node_value`: the node's own statistics (independent of `pos`).
    fn node_value(&self) -> ValueSummary {
        match self.kind {
            Criterion::SquaredError => ValueSummary::Regression {
                sum: self.sum_total[0],
                sum_sq: self.sq_sum_total,
                n: self.weighted_n_node_samples,
            },
            _ => ValueSummary::Classification(super::node::ClassHistogram {
                counts: self.sum_total.clone(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Splitter
// ---------------------------------------------------------------------------

/// A candidate split (mirror of sklearn's `SplitRecord`, `n_missing` fixed 0).
#[derive(Debug, Clone)]
pub(crate) struct SplitRecord {
    pub feature: usize,
    pub threshold: f64,
    pub impurity_left: f64,
    pub impurity_right: f64,
    pub improvement: f64,
    pub pos: usize,
}

impl SplitRecord {
    pub(crate) fn init(pos: usize) -> Self {
        SplitRecord {
            feature: 0,
            threshold: 0.0,
            impurity_left: f64::INFINITY,
            impurity_right: f64::INFINITY,
            improvement: f64::NEG_INFINITY,
            pos,
        }
    }
}

/// Exact port of `BestSplitter` + `DensePartitioner` for dense, uniformly
/// weighted, finite data (no missing values).
pub(crate) struct SkBestSplitter<'a> {
    /// float32-quantized features, row-major (sklearn converts X to f32).
    x32: Vec<f32>,
    ncols: usize,
    y: &'a [f64],
    labels: Option<&'a [usize]>,
    samples: Vec<usize>,
    feature_values: Vec<f32>,
    features: Vec<usize>,
    constant_features: Vec<usize>,
    rand_r_state: u32,
    criterion: CriterionState,
    start: usize,
    end: usize,
    max_features: usize,
    n_features: usize,
    min_samples_leaf: usize,
    min_weight_leaf: f64,
}

impl<'a> SkBestSplitter<'a> {
    /// Creates a splitter with unit weights.
    #[cfg(test)]
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: &Matrix,
        y: &'a [f64],
        labels: Option<&'a [usize]>,
        n_classes: usize,
        criterion: Criterion,
        max_features: MaxFeatures,
        min_samples_leaf: usize,
        min_weight_leaf: f64,
        random_state: u64,
    ) -> Self {
        Self::new_with_weights(
            x,
            y,
            labels,
            n_classes,
            criterion,
            max_features,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_weights(
        x: &Matrix,
        y: &'a [f64],
        labels: Option<&'a [usize]>,
        n_classes: usize,
        criterion: Criterion,
        max_features: MaxFeatures,
        min_samples_leaf: usize,
        min_weight_leaf: f64,
        random_state: u64,
        sample_weights: Option<Vec<f64>>,
    ) -> Self {
        let (nrows, ncols) = (x.nrows(), x.ncols());
        let mut x32 = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                x32.push(x.get(i, j) as f32);
            }
        }
        let n_features = ncols;
        let sample_weights = sample_weights.unwrap_or_else(|| vec![1.0; nrows]);
        if sample_weights.len() != nrows {
            panic!("sample_weights length must match the number of rows");
        }
        // sklearn's splitter keeps only positively weighted samples in its
        // working array. The total weight still includes every input row.
        let samples: Vec<usize> = (0..nrows)
            .filter(|&index| sample_weights[index] != 0.0)
            .collect();
        SkBestSplitter {
            x32,
            ncols,
            y,
            labels,
            samples,

            feature_values: vec![0.0; nrows],
            features: (0..n_features).collect(),
            constant_features: vec![0; n_features],
            rand_r_state: rand_r_seed_from(random_state),
            criterion: CriterionState::with_weights(criterion, n_classes, sample_weights),
            start: 0,
            end: 0,
            max_features: max_features.resolve(n_features),
            min_samples_leaf,
            min_weight_leaf,
            n_features,
        }
    }

    #[inline]
    fn xv(&self, sample: usize, feature: usize) -> f32 {
        self.x32[sample * self.ncols + feature]
    }

    /// `node_reset`: (re)initialize the criterion on `samples[start..end]`;
    /// returns the node's weighted sample count.
    pub fn node_reset(&mut self, start: usize, end: usize) -> f64 {
        self.start = start;
        self.end = end;
        let (y, labels, samples) = (self.y, self.labels, &self.samples);
        self.criterion.init(y, labels, samples, start, end);
        self.criterion.weighted_n_node_samples
    }

    /// Total sample count (the persistent `samples` array's length).
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// `node_impurity` (root only).
    pub fn node_impurity(&self) -> f64 {
        self.criterion.node_impurity()
    }

    /// `node_value` for the node most recently `node_reset`-ed.
    pub fn node_value(&self) -> ValueSummary {
        self.criterion.node_value()
    }

    /// `DensePartitioner.sort_samples_and_feature_values` (no missing
    /// values): copy the feature column into `feature_values` and
    /// introsort together with the samples array.
    fn sort_samples_and_feature_values(&mut self, current_feature: usize) {
        let (start, end) = (self.start, self.end);
        for p in start..end {
            self.feature_values[p] = self.xv(self.samples[p], current_feature);
        }
        let n = end - start;
        sort_rf(&mut self.feature_values, &mut self.samples, start, n);
    }

    /// `DensePartitioner.next_p`: advance to the next distinct-value
    /// boundary (`p_prev < p`), skipping runs of equal feature values.
    fn next_p(&self, p_prev: &mut usize, p: &mut usize) {
        let end = self.end;
        let fv = &self.feature_values;
        while *p + 1 < end && fv[*p + 1] <= fv[*p] + FEATURE_THRESHOLD {
            *p += 1;
        }
        *p_prev = *p;
        *p += 1;
    }

    /// `DensePartitioner.partition_samples_final` (no missing values):
    /// stably-in-place reorder of `samples[start..end]` so that
    /// `x[f] <= threshold` goes left.
    fn partition_samples_final(
        &mut self,
        best_pos: usize,
        best_threshold: f64,
        best_feature: usize,
    ) {
        let _ = best_pos;
        let start = self.start;
        let mut p = start;
        let mut partition_end = self.end - 1;
        while p < partition_end {
            if self.xv(self.samples[p], best_feature) as f64 <= best_threshold {
                p += 1;
            } else {
                self.samples.swap(p, partition_end);
                partition_end -= 1;
            }
        }
    }

    /// `node_split_best`: the full feature-draw loop, scan, and finalization.
    ///
    /// Takes the parent's impurity and its `n_constant_features` count;
    /// returns the best split plus the updated constant count for children.
    pub fn node_split(
        &mut self,
        parent_impurity: f64,
        n_known_constants: usize,
    ) -> (SplitRecord, usize) {
        let start = self.start;
        let end = self.end;
        let max_features = self.max_features;
        let min_samples_leaf = self.min_samples_leaf;
        let min_weight_leaf = self.min_weight_leaf;

        let mut best_split = SplitRecord::init(end);
        let mut best_proxy_improvement = f64::NEG_INFINITY;

        let mut f_i = self.n_features;
        let mut n_found_constants = 0usize;
        let mut n_drawn_constants = 0usize;
        let mut n_total_constants = n_known_constants;
        let mut n_visited_features = 0usize;

        // Fisher-Yates draw loop (exact port).
        while f_i > n_total_constants
            && (n_visited_features < max_features
                || n_visited_features <= n_found_constants + n_drawn_constants)
        {
            n_visited_features += 1;

            let mut f_j = rand_int(
                n_drawn_constants,
                f_i - n_found_constants,
                &mut self.rand_r_state,
            );

            if f_j < n_known_constants {
                // Drawn a known-constant feature: move it to the drawn
                // constants block and skip.
                self.features.swap(n_drawn_constants, f_j);
                n_drawn_constants += 1;
                continue;
            }

            f_j += n_found_constants;
            let current_feature = self.features[f_j];
            self.sort_samples_and_feature_values(current_feature);

            let fv = &self.feature_values;
            // Constant-feature test (single precision).
            if end == start || fv[end - 1] <= fv[start] + FEATURE_THRESHOLD {
                self.features.swap(f_j, n_total_constants);
                n_found_constants += 1;
                n_total_constants += 1;
                continue;
            }

            f_i -= 1;
            self.features.swap(f_i, f_j);

            // Evaluate all split positions for this feature.
            self.criterion.reset();
            let mut p = start;
            let mut p_prev = start;
            while p < end {
                self.next_p(&mut p_prev, &mut p);
                if p >= end {
                    continue;
                }
                let n_left = p - start;
                let n_right = end - p;
                if n_left < min_samples_leaf || n_right < min_samples_leaf {
                    continue;
                }
                let (y, labels, samples) = (self.y, self.labels, &self.samples);
                self.criterion.update(y, labels, samples, p);
                let current_proxy_improvement =
                    self.criterion.proxy_impurity_improvement(y, samples);

                if current_proxy_improvement > best_proxy_improvement {
                    best_proxy_improvement = current_proxy_improvement;
                    let fv = &self.feature_values;
                    // Sum of halves avoids infinite values.
                    let mut threshold = fv[p_prev] as f64 / 2.0 + fv[p] as f64 / 2.0;
                    if threshold == fv[p] as f64 || threshold.is_infinite() {
                        threshold = fv[p_prev] as f64;
                    }
                    best_split = SplitRecord {
                        feature: current_feature,
                        threshold,
                        impurity_left: f64::INFINITY,
                        impurity_right: f64::INFINITY,
                        improvement: f64::NEG_INFINITY,
                        pos: p,
                    };
                    // n_missing == 0 → missing_go_to_left = n_left > n_right;
                    // irrelevant without missing-value support.
                }
                let _ = min_weight_leaf; // always 0.0 here (uniform weights)
            }
        }

        // Finalize the best split: repartition samples, recompute the
        // criterion at best.pos, and compute the true improvement.
        if best_split.pos < end {
            self.partition_samples_final(best_split.pos, best_split.threshold, best_split.feature);
            let (y, labels, samples) = (self.y, self.labels, &self.samples);
            self.criterion.reset();
            self.criterion.update(y, labels, samples, best_split.pos);
            let (il, ir) = self.criterion.children_impurity(y, samples);
            best_split.impurity_left = il;
            best_split.impurity_right = ir;
            best_split.improvement = self.criterion.impurity_improvement(parent_impurity, il, ir);
        }

        // Preserve the constant-feature invariant for sibling/child nodes.
        self.features[0..n_known_constants]
            .copy_from_slice(&self.constant_features[0..n_known_constants]);
        let tail = self.features[n_known_constants..n_known_constants + n_found_constants].to_vec();
        self.constant_features[n_known_constants..n_known_constants + n_found_constants]
            .copy_from_slice(&tail);

        (best_split, n_total_constants)
    }

    /// `DensePartitioner.find_min_max` (no-missing path): fills
    /// `feature_values[start..end]` in samples order and returns the
    /// (min, max) of the current feature.
    fn find_min_max(&mut self, current_feature: usize) -> (f32, f32) {
        let (start, end) = (self.start, self.end);
        let mut min = self.xv(self.samples[start], current_feature);
        let mut max = min;
        self.feature_values[start] = min;
        for p in start + 1..end {
            let value = self.xv(self.samples[p], current_feature);
            self.feature_values[p] = value;
            if value < min {
                min = value;
            } else if value > max {
                max = value;
            }
        }
        (min, max)
    }

    /// `DensePartitioner.partition_samples`: partitions on the cached
    /// `feature_values` (f32 widened to f64), decrement-before-swap,
    /// swapping *both* `feature_values` and `samples`. Returns the left
    /// partition's end. Mutates sample order even for later-rejected splits.
    fn partition_samples(&mut self, threshold: f64) -> usize {
        let mut p = self.start;
        let mut partition_end = self.end;
        while p < partition_end {
            if self.feature_values[p] as f64 <= threshold {
                p += 1;
            } else {
                partition_end -= 1;
                self.feature_values.swap(p, partition_end);
                self.samples.swap(p, partition_end);
            }
        }
        partition_end
    }

    /// `node_split_random` (sklearn's `splitter="random"`, ExtraTrees):
    /// same Fisher-Yates feature draws as [`Self::node_split`], but each
    /// non-constant feature gets a single `rand_uniform` threshold between
    /// its min and max instead of a full midpoint scan.
    pub fn node_split_random(
        &mut self,
        parent_impurity: f64,
        n_known_constants: usize,
    ) -> (SplitRecord, usize) {
        let start = self.start;
        let end = self.end;
        let max_features = self.max_features;
        let min_samples_leaf = self.min_samples_leaf;
        let min_weight_leaf = self.min_weight_leaf;

        let mut best_split = SplitRecord::init(end);
        let mut best_proxy_improvement = f64::NEG_INFINITY;

        let mut f_i = self.n_features;
        let mut n_found_constants = 0usize;
        let mut n_drawn_constants = 0usize;
        let mut n_total_constants = n_known_constants;
        let mut n_visited_features = 0usize;
        // sklearn's `current_split.feature` scratch: set before the constant
        // check and persists across `continue`s; read at finalization only
        // when a split was found (>= 1 full iteration), so 0 is safe here.
        let mut current_feature = 0usize;

        while f_i > n_total_constants
            && (n_visited_features < max_features
                || n_visited_features <= n_found_constants + n_drawn_constants)
        {
            n_visited_features += 1;

            let mut f_j = rand_int(
                n_drawn_constants,
                f_i - n_found_constants,
                &mut self.rand_r_state,
            );

            if f_j < n_known_constants {
                self.features.swap(n_drawn_constants, f_j);
                n_drawn_constants += 1;
                continue;
            }

            f_j += n_found_constants;
            current_feature = self.features[f_j];
            let (min_value, max_value) = self.find_min_max(current_feature);

            // Constant-feature test (single precision, unsorted min/max).
            if end == start || max_value <= min_value + FEATURE_THRESHOLD {
                self.features.swap(f_j, n_total_constants);
                n_found_constants += 1;
                n_total_constants += 1;
                continue;
            }

            f_i -= 1;
            self.features.swap(f_i, f_j);

            // Draw one random threshold; collapse to min when it lands on
            // max (f64 threshold vs widened f32 max).
            let mut threshold = rand_uniform(min_value, max_value, &mut self.rand_r_state);
            if threshold == max_value as f64 {
                threshold = min_value as f64;
            }

            let pos = self.partition_samples(threshold);
            let n_left = pos - start;
            let n_right = end - pos;
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            // Evaluate the partitioned split (reset + update, not the
            // incremental scan used by node_split).
            let (y, labels, samples) = (self.y, self.labels, &self.samples);
            self.criterion.reset();
            self.criterion.update(y, labels, samples, pos);
            if self.criterion.weighted_n_left < min_weight_leaf
                || self.criterion.weighted_n_right < min_weight_leaf
            {
                continue;
            }
            let current_proxy_improvement = self.criterion.proxy_impurity_improvement(y, samples);

            if current_proxy_improvement > best_proxy_improvement {
                best_proxy_improvement = current_proxy_improvement;
                best_split = SplitRecord {
                    feature: current_feature,
                    threshold,
                    impurity_left: f64::INFINITY,
                    impurity_right: f64::INFINITY,
                    improvement: f64::NEG_INFINITY,
                    pos,
                };
            }
        }

        if best_split.pos < end {
            // When the last evaluated feature IS the best feature, samples
            // stay in partition_samples order (sklearn skips the final
            // repartition) — this changes FP accumulation order downstream.
            if current_feature != best_split.feature {
                self.partition_samples_final(
                    best_split.pos,
                    best_split.threshold,
                    best_split.feature,
                );
            }
            let (y, labels, samples) = (self.y, self.labels, &self.samples);
            self.criterion.reset();
            self.criterion.update(y, labels, samples, best_split.pos);
            let (il, ir) = self.criterion.children_impurity(y, samples);
            best_split.impurity_left = il;
            best_split.impurity_right = ir;
            best_split.improvement = self.criterion.impurity_improvement(parent_impurity, il, ir);
        }

        // Preserve the constant-feature invariant for sibling/child nodes.
        self.features[0..n_known_constants]
            .copy_from_slice(&self.constant_features[0..n_known_constants]);
        let tail = self.features[n_known_constants..n_known_constants + n_found_constants].to_vec();
        self.constant_features[n_known_constants..n_known_constants + n_found_constants]
            .copy_from_slice(&tail);

        (best_split, n_total_constants)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The numpy → rand_r seeding chain, pinned against
    /// `np.random.RandomState(seed).randint(0, 2147483647)` measured with
    /// numpy 2.0.2 (legacy `RandomState` is frozen by NEP 19).
    #[test]
    fn rand_r_seed_matches_numpy_randint() {
        let expected: [(u64, u32); 5] = [
            (0, 209652396),
            (7, 327741615),
            (42, 1608637542),
            (123, 843828734),
            (2026, 942082305),
        ];
        for (seed, value) in expected {
            assert_eq!(rand_r_seed_from(seed), value, "seed {seed}");
        }
    }

    /// `our_rand_r` matches sklearn's tested primitive, measured against
    /// `sklearn.utils._random._our_rand_r_py` (1.6.1):
    /// seed 1 → 270369, seed 2 → 540738.
    #[test]
    fn our_rand_r_reference_values() {
        let mut seed: u32 = 1;
        let first = our_rand_r(&mut seed);
        assert_eq!(first, 270369);
        let mut seed2: u32 = 2;
        assert_eq!(our_rand_r(&mut seed2), 540738);
        // Determinism: restarting from the returned state replays.
        let mut s2 = first;
        assert_eq!(our_rand_r(&mut s2), our_rand_r(&mut seed));
    }

    /// The iris root threshold from sklearn (rs=7): midpoint of float32
    /// (1.9, 3.0) computed in float64 — 2.4499999880790710.
    #[test]
    fn float32_midpoint_arithmetic() {
        let threshold = (1.9f32) as f64 / 2.0 + (3.0f32) as f64 / 2.0;
        assert_eq!(threshold, 2.449999988079071);
    }

    /// `next_p` skips equal-value runs (a `FEATURE_THRESHOLD` fuzz is
    /// irrelevant at these magnitudes).
    #[test]
    fn next_p_boundaries() {
        let splitter = SkBestSplitter {
            x32: Vec::new(),
            ncols: 0,
            y: &[],
            labels: None,
            samples: vec![],
            feature_values: vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
            features: vec![],
            constant_features: vec![],
            rand_r_state: 1,
            criterion: CriterionState::new(Criterion::Gini, 2, 6.0),
            start: 0,
            end: 6,
            max_features: 1,
            min_samples_leaf: 1,
            min_weight_leaf: 0.0,
            n_features: 1,
        };
        let mut p = 0usize;
        let mut boundaries = Vec::new();
        while p < 6 {
            let mut pp = 0usize;
            splitter.next_p(&mut pp, &mut p);
            boundaries.push((pp, p));
        }
        // `p_prev` is the last index of the current equal-value run:
        // [1,1] → (1,2), [2] → (2,3), [3,3,3] → (5,6).
        assert_eq!(boundaries, vec![(1, 2), (2, 3), (5, 6)]);
    }

    /// Introsort must match a reference sort's *permutation* on distinct
    /// values and behave deterministically on duplicates.
    #[test]
    fn sort_rf_orders_and_permutes() {
        let fv = &mut [3.0f32, 1.0, 2.0, 1.0, 3.0];
        let sa = &mut [10usize, 11, 12, 13, 14];
        sort_rf(fv, sa, 0, 5);
        assert_eq!(fv.to_vec(), vec![1.0, 1.0, 2.0, 3.0, 3.0]);
        // Duplicates are grouped; exact intra-group order is introsort's
        // (unstable) — pin it to catch accidental algorithm drift.
        assert_eq!(sa.to_vec(), vec![11, 13, 12, 10, 14]);
    }

    /// `rand_uniform` pins computed from sklearn's formula
    /// `((high - low) * our_rand_r(state) / RAND_R_MAX) + low` with f32
    /// bounds widened to f64 (numpy 2.0.2 / sklearn 1.6.1 semantics).
    #[test]
    fn rand_uniform_reference_values() {
        let mut s = 1u32;
        assert_eq!(
            rand_uniform(0.0, 1.0, &mut s).to_bits(),
            0.0001259003766467331f64.to_bits()
        );
        assert_eq!(s, 270369);
        let mut s = 7u32;
        assert_eq!(
            rand_uniform(4.3, 7.9, &mut s).to_bits(),
            4.303172880142314f64.to_bits()
        );
        let mut s = 42u32;
        assert_eq!(
            rand_uniform(-1.0, 1.0, &mut s).to_bits(),
            (-0.9894244298289644f64).to_bits()
        );
        let mut s = 3u32;
        assert_eq!(rand_uniform(2.0, 2.0, &mut s), 2.0);
    }

    /// `find_min_max` fills feature_values in samples order;
    /// `partition_samples` swaps both arrays with decrement-before-swap.
    #[test]
    fn random_split_partition_helpers() {
        let y = [0.0; 4];
        let mut splitter = SkBestSplitter {
            x32: vec![3.0f32, 1.0, 2.0, 0.0],
            ncols: 1,
            y: &y,
            labels: None,
            samples: vec![0, 1, 2, 3],
            feature_values: vec![0.0; 4],
            features: vec![0],
            constant_features: vec![0],
            rand_r_state: 1,
            criterion: CriterionState::new(Criterion::Gini, 2, 4.0),
            start: 0,
            end: 4,
            max_features: 1,
            min_samples_leaf: 1,
            min_weight_leaf: 0.0,
            n_features: 1,
        };
        let (min, max) = splitter.find_min_max(0);
        assert_eq!((min, max), (0.0, 3.0));
        assert_eq!(splitter.feature_values, vec![3.0, 1.0, 2.0, 0.0]);
        let pos = splitter.partition_samples(1.5);
        assert_eq!(pos, 2);
        assert_eq!(splitter.samples, vec![3, 1, 2, 0]);
        assert_eq!(splitter.feature_values, vec![0.0, 1.0, 2.0, 3.0]);
    }
}
