use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Strategy for computing bin edges in [`KBinsDiscretizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinStrategy {
    /// Equal-width bins between min and max.
    #[default]
    Uniform,
    /// Equal-frequency bins (quantiles).
    Quantile,
    /// 1-D k-means clustering to find bin centers.
    KMeans,
}

/// Output encoding for [`KBinsDiscretizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KBinsEncode {
    /// Output bin indices (integers 0..n_bins-1) in a dense matrix.
    #[default]
    Ordinal,
    /// One-hot encode bins into a dense matrix.
    OneHotDense,
}

/// Bin continuous data into intervals, mirroring
/// `sklearn.preprocessing.KBinsDiscretizer`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KBinsDiscretizer {
    n_bins: usize,
    strategy: BinStrategy,
    encode: KBinsEncode,
    /// Bin edges per column (length n_bins+1 each).
    bin_edges: Vec<Vec<f64>>,
    /// Effective number of bins per column (may be less than n_bins for
    /// constant columns).
    n_actual_bins: Vec<usize>,
    n_features: usize,
    fitted: bool,
}

impl KBinsDiscretizer {
    /// Create a new discretizer. `n_bins` must be >= 2.
    pub fn new(n_bins: usize) -> Result<Self> {
        if n_bins < 2 {
            return Err(DatarustError::InvalidConfig("n_bins must be >= 2".into()));
        }
        Ok(Self {
            n_bins,
            strategy: BinStrategy::Uniform,
            encode: KBinsEncode::Ordinal,
            bin_edges: vec![],
            n_actual_bins: vec![],
            n_features: 0,
            fitted: false,
        })
    }

    /// Builder: set the strategy used to compute bin edges.
    pub fn strategy(mut self, s: BinStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Builder: set the output encoding scheme.
    pub fn encode(mut self, e: KBinsEncode) -> Self {
        self.encode = e;
        self
    }

    /// Returns the fitted bin edges per column.
    pub fn bin_edges(&self) -> &[Vec<f64>] {
        &self.bin_edges
    }

    /// Returns the effective number of bins per column.
    pub fn n_actual_bins(&self) -> &[usize] {
        &self.n_actual_bins
    }

    /// Compute bin edges for one column.
    fn compute_edges(col: &[f64], n_bins: usize, strategy: BinStrategy) -> Vec<f64> {
        let mut sorted = col.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let lo = sorted[0];
        let hi = sorted[sorted.len() - 1];

        // Constant column: single bin containing everything.
        if (hi - lo).abs() < f64::EPSILON {
            return vec![lo];
        }

        match strategy {
            BinStrategy::Uniform => {
                let width = (hi - lo) / n_bins as f64;
                (0..=n_bins).map(|i| lo + width * i as f64).collect()
            }
            BinStrategy::Quantile => {
                // Edges at quantiles i/n_bins for i=0..=n_bins.
                (0..=n_bins)
                    .map(|i| {
                        stats::quantile(&sorted, i as f64 / n_bins as f64)
                            .expect("i in 0..=n_bins keeps q in [0,1]")
                    })
                    .collect()
            }
            BinStrategy::KMeans => kmeans_1d(&sorted, n_bins),
        }
    }

    /// Find the bin index for a value given the edges.
    fn value_to_bin(value: f64, edges: &[f64]) -> usize {
        if edges.len() <= 1 {
            return 0;
        }
        // Binary search for the right edge.
        let n_bins = edges.len() - 1;
        // Clamp to last bin if value >= last edge.
        if value >= edges[n_bins] {
            return n_bins - 1;
        }
        // Clamp to first bin if value <= first edge.
        if value <= edges[0] {
            return 0;
        }
        // Find the first edge[i] > value; bin = i-1.
        let mut lo = 0usize;
        let mut hi = edges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if edges[mid] <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the first index where edges[lo] > value; bin = lo - 1.
        (lo - 1).min(n_bins - 1)
    }

    fn total_output_cols(&self) -> usize {
        match self.encode {
            KBinsEncode::Ordinal => self.n_features,
            KBinsEncode::OneHotDense => self.n_actual_bins.iter().sum(),
        }
    }
}

impl Transformer for KBinsDiscretizer {
    fn name(&self) -> &'static str {
        "KBinsDiscretizer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let ncols = x.ncols();
        let mut edges_all = Vec::with_capacity(ncols);
        let mut actual_bins = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let col = x.col(j);
            let edges = Self::compute_edges(&col, self.n_bins, self.strategy);
            actual_bins.push(edges.len().saturating_sub(1).max(1));
            edges_all.push(edges);
        }
        self.bin_edges = edges_all;
        self.n_actual_bins = actual_bins;
        self.n_features = ncols;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("KBinsDiscretizer".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        match self.encode {
            KBinsEncode::Ordinal => {
                let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
                for (i, out_row) in out.iter_mut().enumerate() {
                    for (j, cell) in out_row.iter_mut().enumerate() {
                        *cell = Self::value_to_bin(x.get(i, j), &self.bin_edges[j]) as f64;
                    }
                }
                Matrix::new(out)
            }
            KBinsEncode::OneHotDense => {
                let n_out = self.total_output_cols();
                let mut out = vec![vec![0.0; n_out]; x.nrows()];
                for (i, out_row) in out.iter_mut().enumerate() {
                    let mut offset = 0;
                    for j in 0..x.ncols() {
                        let bin = Self::value_to_bin(x.get(i, j), &self.bin_edges[j]);
                        out_row[offset + bin] = 1.0;
                        offset += self.n_actual_bins[j];
                    }
                }
                Matrix::new(out)
            }
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for KBinsDiscretizer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features),
        };
        match self.encode {
            KBinsEncode::Ordinal => names,
            KBinsEncode::OneHotDense => {
                let mut out = Vec::new();
                for (j, name) in names.iter().enumerate() {
                    let nbins = self.n_actual_bins.get(j).copied().unwrap_or(1);
                    for b in 0..nbins {
                        out.push(format!("{}_bin{}", name, b));
                    }
                }
                out
            }
        }
    }
}

/// Simple 1-D k-means to produce sorted cluster centers, then derive bin edges
/// as midpoints between consecutive centers.
fn kmeans_1d(sorted: &[f64], k: usize) -> Vec<f64> {
    let n = sorted.len();
    let k = k.min(n);
    if k <= 1 {
        return vec![sorted[0]];
    }
    // Initialize centers at quantiles.
    let mut centers: Vec<f64> = (0..k)
        .map(|i| {
            stats::quantile(sorted, i as f64 / (k - 1) as f64).expect("i in 0..k keeps q in [0,1]")
        })
        .collect();
    centers.sort_by(|a, b| a.total_cmp(b));
    centers.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    if centers.len() < 2 {
        return vec![sorted[0]];
    }
    let k = centers.len();

    for _ in 0..50 {
        // Assign points to nearest center.
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        for &v in sorted {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (ci, &c) in centers.iter().enumerate() {
                let d = (v - c).abs();
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            sums[best] += v;
            counts[best] += 1;
        }
        // Update centers.
        let mut changed = false;
        for ci in 0..k {
            if counts[ci] > 0 {
                let new_c = sums[ci] / counts[ci] as f64;
                if (new_c - centers[ci]).abs() > 1e-12 {
                    changed = true;
                }
                centers[ci] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    centers.sort_by(|a, b| a.total_cmp(b));

    // Bin edges: first = min(sorted), last = max(sorted),
    // interior = midpoints between consecutive centers.
    let mut edges = Vec::with_capacity(k + 1);
    edges.push(sorted[0]);
    for i in 0..k - 1 {
        edges.push((centers[i] + centers[i + 1]) / 2.0);
    }
    edges.push(sorted[n - 1]);
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn uniform_ordinal() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(2)
            .unwrap()
            .strategy(BinStrategy::Uniform);
        let out = kb.fit_transform(&x).unwrap();
        // width = 2.0; edges = [0, 2, 4]
        // 0 -> bin0, 1 -> bin0, 2 -> bin1 (2>=2), 3 -> bin1, 4 -> bin1
        assert_eq!(out.col(0), vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn quantile_ordinal() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(5)
            .unwrap()
            .strategy(BinStrategy::Quantile);
        let out = kb.fit_transform(&x).unwrap();
        // 5 bins for 5 points -> each value in its own bin
        assert_eq!(out.col(0), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn onehot_dense_encode() {
        let x = Matrix::new(vec![vec![0.0], vec![2.0], vec![4.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(3)
            .unwrap()
            .strategy(BinStrategy::Uniform)
            .encode(KBinsEncode::OneHotDense);
        let out = kb.fit_transform(&x).unwrap();
        // 3 output cols, each row has exactly one 1
        assert_eq!(out.ncols(), 3);
        for i in 0..3 {
            let ones: usize = out.row(i).iter().filter(|&&v| v == 1.0).count();
            assert_eq!(ones, 1);
        }
    }

    #[test]
    fn multi_column() {
        let x = Matrix::new(vec![
            vec![0.0, 10.0],
            vec![1.0, 20.0],
            vec![2.0, 30.0],
            vec![3.0, 40.0],
        ])
        .unwrap();
        let mut kb = KBinsDiscretizer::new(2)
            .unwrap()
            .strategy(BinStrategy::Uniform);
        let out = kb.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn constant_column_single_bin() {
        let x = Matrix::new(vec![vec![5.0], vec![5.0], vec![5.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(3).unwrap();
        let out = kb.fit_transform(&x).unwrap();
        assert_eq!(out.col(0), vec![0.0, 0.0, 0.0]);
        assert_eq!(kb.n_actual_bins(), &[1]);
    }

    #[test]
    fn kmeans_strategy() {
        // Clear bimodal distribution
        let x = Matrix::new(vec![
            vec![0.0],
            vec![0.5],
            vec![1.0],
            vec![10.0],
            vec![10.5],
            vec![11.0],
        ])
        .unwrap();
        let mut kb = KBinsDiscretizer::new(2)
            .unwrap()
            .strategy(BinStrategy::KMeans);
        let out = kb.fit_transform(&x).unwrap();
        // First 3 in bin 0, last 3 in bin 1
        assert_eq!(out.col(0), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn n_bins_too_small_errors() {
        assert!(KBinsDiscretizer::new(1).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let kb = KBinsDiscretizer::new(3).unwrap();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(kb.transform(&x), Err(DatarustError::NotFitted(_))));
    }

    #[test]
    fn value_at_max_edge_goes_to_last_bin() {
        let edges = vec![0.0, 1.0, 2.0];
        assert_eq!(KBinsDiscretizer::value_to_bin(2.0, &edges), 1);
        assert_eq!(KBinsDiscretizer::value_to_bin(3.0, &edges), 1);
    }

    #[test]
    fn value_below_min_edge_goes_to_first_bin() {
        let edges = vec![0.0, 1.0, 2.0];
        assert_eq!(KBinsDiscretizer::value_to_bin(-5.0, &edges), 0);
    }

    #[test]
    fn feature_names_ordinal_preserves() {
        let x = Matrix::new(vec![vec![0.0, 10.0], vec![5.0, 40.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(2).unwrap();
        kb.fit(&x).unwrap();
        let names = kb.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn feature_names_onehot() {
        let x = Matrix::new(vec![vec![0.0], vec![4.0]]).unwrap();
        let mut kb = KBinsDiscretizer::new(2)
            .unwrap()
            .encode(KBinsEncode::OneHotDense);
        kb.fit(&x).unwrap();
        let names = kb.feature_names_out(Some(&["price".into()]));
        assert_eq!(names, vec!["price_bin0", "price_bin1"]);
    }

    #[test]
    fn uniform_edges_correct() {
        let edges =
            KBinsDiscretizer::compute_edges(&[0.0, 1.0, 2.0, 3.0, 4.0], 4, BinStrategy::Uniform);
        assert!(approx(edges[0], 0.0, 1e-12));
        assert!(approx(edges[4], 4.0, 1e-12));
        assert!(approx(edges[1], 1.0, 1e-12));
    }
}
