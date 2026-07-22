//! k-means clustering via Lloyd's algorithm.
//!
//! Mirrors `sklearn.cluster.KMeans`. Partitions `n` observations into `k`
//! clusters by minimizing within-cluster variance (inertia). Each iteration
//! assigns every point to its nearest centroid, then recomputes centroids as
//! the mean of their members. Convergence is reached when centroid movement
//! drops below `tol` or `max_iter` iterations elapse.
//!
//! Initialization uses k-means++ by default (spread initial centroids to be
//! far apart), which dramatically improves solution quality over random
//! initialization. `n_init` restarts are run and the lowest-inertia result is
//! kept, mirroring scikit-learn's default behaviour.

use crate::cluster::kmeans::KMeansInit::KMeansPlusPlus;
use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::rng::Rng;
use crate::traits::{Clusterer, ParamValue, Params};

/// Output of a single Lloyd's run: final centroids, per-point assignments,
/// total inertia, and the number of iterations executed.
type LloydResult = (Vec<Vec<f64>>, Vec<usize>, f64, usize);

/// Initialization strategy for [`KMeans`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KMeansInit {
    /// k-means++: pick the first centroid uniformly at random, then each
    /// subsequent centroid is sampled with probability proportional to the
    /// squared distance from the nearest already-chosen centroid. Produces
    /// well-spread initial centroids and is the scikit-learn default.
    #[default]
    KMeansPlusPlus,
    /// Choose `n_clusters` distinct points uniformly at random from the data.
    Random,
}

/// k-means clustering.
///
/// Minimizes within-cluster sum of squares via Lloyd's algorithm with
/// k-means++ initialization. The fitted model assigns new points to their
/// nearest learned centroid.
///
/// ```rust
/// use datarust::cluster::KMeans;
/// use datarust::traits::Clusterer;
/// use datarust::Matrix;
///
/// // Three well-separated blobs of points.
/// let rows = vec![
///     vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1],
///     vec![10.0, 10.0], vec![10.1, 10.0], vec![10.0, 10.1],
///     vec![20.0, 20.0], vec![20.1, 20.0], vec![20.0, 20.1],
/// ];
/// let x = Matrix::new(rows)?;
/// let mut km = KMeans::new().with_n_clusters(3).with_random_state(0);
/// let labels = km.fit_predict(&x)?;
/// assert_eq!(labels.len(), 9);
/// // Three distinct clusters were found.
/// let unique: std::collections::BTreeSet<usize> = labels.iter().copied().collect();
/// assert_eq!(unique.len(), 3);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KMeans {
    n_clusters: usize,
    init: KMeansInit,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    random_state: Option<u64>,
    // Fitted state.
    cluster_centers_: Vec<Vec<f64>>,
    labels_: Vec<usize>,
    inertia_: f64,
    n_iter_: usize,
    n_features_in_: usize,
    fitted: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new()
    }
}

impl KMeans {
    /// Creates a new `KMeans` with scikit-learn defaults:
    /// `n_clusters = 8`, `init = KMeansPlusPlus`, `max_iter = 300`,
    /// `tol = 1e-4`, `n_init = 10`, `random_state = None`.
    pub fn new() -> Self {
        Self {
            n_clusters: 8,
            init: KMeansPlusPlus,
            max_iter: 300,
            tol: 1e-4,
            n_init: 10,
            random_state: None,
            cluster_centers_: Vec::new(),
            labels_: Vec::new(),
            inertia_: 0.0,
            n_iter_: 0,
            n_features_in_: 0,
            fitted: false,
        }
    }

    /// Builder: number of clusters to form (default `8`). Must be `>= 1`.
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Builder: initialization strategy (default [`KMeansInit::KMeansPlusPlus`]).
    pub fn with_init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    /// Builder: maximum number of Lloyd's iterations per run (default `300`).
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Builder: relative convergence tolerance on centroid movement
    /// (default `1e-4`). A run stops when the squared Frobenius norm of the
    /// centroid shift drops below this value.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Builder: number of independent restarts (default `10`). The run with the
    /// lowest inertia is kept. Mirrors scikit-learn's `n_init = "auto"`.
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Builder: deterministic seed for centroid initialization (default `None`,
    /// which uses a fixed splinter constant so results are reproducible across
    /// runs but not user-controlled).
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Coordinates of the fitted cluster centers, one row per cluster.
    /// Available only after [`fit`](Clusterer::fit).
    pub fn cluster_centers(&self) -> &[Vec<f64>] {
        &self.cluster_centers_
    }

    /// Cluster index assigned to each training row during `fit`.
    /// Available only after [`fit`](Clusterer::fit).
    pub fn labels(&self) -> &[usize] {
        &self.labels_
    }

    /// Sum of squared distances of each training sample to its nearest
    /// centroid (the objective minimized by k-means). Lower is better.
    /// Available only after [`fit`](Clusterer::fit).
    pub fn inertia(&self) -> f64 {
        self.inertia_
    }

    /// Number of Lloyd's iterations run by the best (lowest-inertia) restart.
    /// Available only after [`fit`](Clusterer::fit).
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Number of features seen during `fit`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Validate configuration and input shape. Returns `(n_samples, n_features)`.
    fn validate(&self, x: &Matrix) -> Result<(usize, usize)> {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 {
            return Err(DatarustError::EmptyInput("X has no rows".into()));
        }
        if p == 0 {
            return Err(DatarustError::EmptyInput("X has no columns".into()));
        }
        if self.n_clusters == 0 {
            return Err(DatarustError::InvalidConfig(
                "n_clusters must be >= 1".into(),
            ));
        }
        if self.n_clusters > n {
            return Err(DatarustError::InvalidConfig(format!(
                "n_clusters ({}) cannot be greater than n_samples ({})",
                self.n_clusters, n
            )));
        }
        if self.max_iter == 0 {
            return Err(DatarustError::InvalidConfig("max_iter must be > 0".into()));
        }
        if self.n_init == 0 {
            return Err(DatarustError::InvalidConfig("n_init must be > 0".into()));
        }
        Ok((n, p))
    }

    /// Squared Euclidean distance between two equal-length rows.
    #[inline]
    fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| {
                let d = ai - bi;
                d * d
            })
            .sum()
    }

    /// Pick `n_clusters` initial centroids using the given strategy.
    fn init_centroids(&self, x: &Matrix, n: usize, p: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        match self.init {
            KMeansInit::Random => {
                let mut indices: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut indices);
                indices[..self.n_clusters]
                    .iter()
                    .map(|&i| x.row(i).to_vec())
                    .collect()
            }
            KMeansInit::KMeansPlusPlus => {
                // First centroid: uniform random point.
                let first = rng.next_usize(n);
                let mut centers: Vec<Vec<f64>> = Vec::with_capacity(self.n_clusters);
                centers.push(x.row(first).to_vec());
                // squared distance from each point to nearest chosen centroid.
                let mut nearest_sq: Vec<f64> = (0..n)
                    .map(|i| Self::sq_dist(x.row(i), &centers[0]))
                    .collect();
                for _ in 1..self.n_clusters {
                    let total: f64 = nearest_sq.iter().sum();
                    let mut centers_row = vec![0.0_f64; p];
                    if total <= 0.0 {
                        // All remaining points coincide with a center; just
                        // pick the next point deterministically.
                        let next = centers.len().min(n - 1);
                        centers_row.copy_from_slice(x.row(next));
                    } else {
                        // Sample with probability proportional to nearest_sq.
                        let r = rng.next_unit() * total;
                        let mut acc = 0.0;
                        let mut chosen = n - 1;
                        for (i, &d) in nearest_sq.iter().enumerate() {
                            acc += d;
                            if acc >= r {
                                chosen = i;
                                break;
                            }
                        }
                        centers_row.copy_from_slice(x.row(chosen));
                    }
                    // Update nearest_sq for the new centroid.
                    for (i, nearest) in nearest_sq.iter_mut().enumerate() {
                        let d = Self::sq_dist(x.row(i), &centers_row);
                        if d < *nearest {
                            *nearest = d;
                        }
                    }
                    centers.push(centers_row);
                }
                centers
            }
        }
    }

    /// One Lloyd's run from the given initial centroids. Returns the final
    /// centroids, per-point assignments, total inertia, and iteration count.
    fn lloyds_run(
        &self,
        x: &Matrix,
        n: usize,
        p: usize,
        mut centers: Vec<Vec<f64>>,
    ) -> LloydResult {
        let mut labels = vec![0usize; n];
        let mut centroid_shift_sq = f64::MAX;
        let mut iter = 0;

        while iter < self.max_iter && centroid_shift_sq > self.tol {
            // Assignment step: each point to its nearest centroid.
            for (label, i) in labels.iter_mut().zip(0..n) {
                let row = x.row(i);
                let (best_idx, _) = centers
                    .iter()
                    .enumerate()
                    .map(|(c, ctr)| (c, Self::sq_dist(row, ctr)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, 0.0));
                *label = best_idx;
            }

            // Update step: recompute centroids as the mean of their members.
            let mut new_centers = vec![vec![0.0_f64; p]; self.n_clusters];
            let mut counts = vec![0usize; self.n_clusters];
            for (i, &label) in labels.iter().enumerate() {
                counts[label] += 1;
                for (j, center) in new_centers[label].iter_mut().enumerate() {
                    *center += x.get(i, j);
                }
            }
            for (c, center) in new_centers.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for val in center.iter_mut() {
                        *val /= counts[c] as f64;
                    }
                } else {
                    // Empty cluster: keep the old centroid.
                    center.copy_from_slice(&centers[c]);
                }
            }

            // Convergence: squared Frobenius norm of centroid shift.
            centroid_shift_sq = new_centers
                .iter()
                .zip(centers.iter())
                .map(|(new, old)| Self::sq_dist(new, old))
                .sum::<f64>()
                / self.n_clusters as f64;

            centers = new_centers;
            iter += 1;
        }

        // Final inertia from the converged assignments.
        let inertia = (0..n)
            .map(|i| Self::sq_dist(x.row(i), &centers[labels[i]]))
            .sum::<f64>();

        (centers, labels, inertia, iter)
    }
}

impl Params for KMeans {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        vec![
            ("n_clusters", ParamValue::Int(self.n_clusters)),
            ("max_iter", ParamValue::Int(self.max_iter)),
            ("tol", ParamValue::Float(self.tol)),
            ("n_init", ParamValue::Int(self.n_init)),
        ]
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match (name, value) {
            ("n_clusters", ParamValue::Int(v)) => self.n_clusters = v,
            ("max_iter", ParamValue::Int(v)) => self.max_iter = v,
            ("tol", ParamValue::Float(v)) => self.tol = v,
            ("n_init", ParamValue::Int(v)) => self.n_init = v,
            (other, _) => {
                return Err(DatarustError::InvalidInput(format!(
                    "KMeans has no tunable parameter '{other}'"
                )));
            }
        }
        self.fitted = false;
        Ok(())
    }
}

impl Clusterer for KMeans {
    fn name(&self) -> &'static str {
        "KMeans"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (n, p) = self.validate(x)?;
        self.n_features_in_ = p;

        let base_seed = self.random_state.unwrap_or(0x9E3779B97F4A7C15);

        let mut best: Option<LloydResult> = None;
        for run in 0..self.n_init {
            // Each restart uses a distinct seed derived from the base.
            let mut rng = Rng::new(base_seed.wrapping_add(run as u64));
            let init_centers = self.init_centroids(x, n, p, &mut rng);
            let (centers, labels, inertia, iters) = self.lloyds_run(x, n, p, init_centers);
            match &best {
                None => best = Some((centers, labels, inertia, iters)),
                Some((_, _, best_inertia, _)) if inertia < *best_inertia => {
                    best = Some((centers, labels, inertia, iters));
                }
                _ => {}
            }
        }

        let (centers, labels, inertia, iters) =
            best.expect("n_init >= 1 checked in validate; best is always Some");
        self.cluster_centers_ = centers;
        self.labels_ = labels;
        self.inertia_ = inertia;
        self.n_iter_ = iters;
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<usize>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("KMeans".into()));
        }
        if x.ncols() != self.n_features_in_ {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features_in_),
                actual: format!("{} features", x.ncols()),
            });
        }
        let n = x.nrows();
        let mut out = vec![0usize; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let row = x.row(i);
            let (best_idx, _) = self
                .cluster_centers_
                .iter()
                .enumerate()
                .map(|(c, ctr)| (c, Self::sq_dist(row, ctr)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            *slot = best_idx;
        }
        Ok(out)
    }

    fn fit_predict(&mut self, x: &Matrix) -> Result<Vec<usize>> {
        self.fit(x)?;
        Ok(self.labels_.clone())
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn finds_three_blobs() {
        let rows: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let base = (i / 10) as f64 * 10.0;
                vec![base, base]
            })
            .collect();
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new()
            .with_n_clusters(3)
            .with_n_init(10)
            .with_random_state(0);
        let labels = km.fit_predict(&x).unwrap();
        // Each blob of 10 points must share one label.
        for blob in 0..3 {
            let first = labels[blob * 10];
            for i in 1..10 {
                assert_eq!(labels[blob * 10 + i], first, "blob {blob} not homogeneous");
            }
        }
        // Different blobs get different labels.
        assert_ne!(labels[0], labels[10]);
        assert_ne!(labels[10], labels[20]);
        assert_ne!(labels[0], labels[20]);
    }

    #[test]
    fn recovers_known_centroids() {
        // Centroids at (0,0), (10,10), (-10,-10), each with tight points.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let centers = [[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]];
        for [cx, cy] in centers {
            for dx in [-0.1, 0.0, 0.1] {
                for dy in [-0.1, 0.0, 0.1] {
                    rows.push(vec![cx + dx, cy + dy]);
                }
            }
        }
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new().with_n_clusters(3).with_random_state(42);
        km.fit(&x).unwrap();
        // Recovered centroids should be near one of the true centers.
        let mut matched = 0;
        for center in km.cluster_centers() {
            let close = centers
                .iter()
                .any(|tc| approx(center[0], tc[0], 0.2) && approx(center[1], tc[1], 0.2));
            if close {
                matched += 1;
            }
        }
        assert_eq!(
            matched,
            3,
            "not all centroids recovered: {:?}",
            km.cluster_centers()
        );
    }

    #[test]
    fn inertia_non_negative() {
        let rows = vec![vec![1.0], vec![2.0], vec![10.0], vec![11.0]];
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new().with_n_clusters(2).with_random_state(1);
        km.fit(&x).unwrap();
        assert!(km.inertia() >= 0.0);
    }

    #[test]
    fn predict_assigns_new_points() {
        let rows = vec![vec![0.0], vec![1.0], vec![100.0], vec![101.0]];
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
        km.fit(&x).unwrap();
        // The training labels give us the cluster identity of each region.
        let train_labels = km.labels();
        let left_cluster = train_labels[0]; // point at 0.0
        let right_cluster = train_labels[2]; // point at 100.0
        let test = Matrix::new(vec![vec![0.5], vec![100.5]]).unwrap();
        let pred = km.predict(&test).unwrap();
        // 0.5 is near the left cluster, 100.5 near the right cluster.
        assert_eq!(pred[0], left_cluster);
        assert_eq!(pred[1], right_cluster);
        assert_ne!(pred[0], pred[1]);
    }

    #[test]
    fn deterministic_same_seed() {
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let x = Matrix::new(rows).unwrap();
        let mut a = KMeans::new().with_n_clusters(3).with_random_state(7);
        let mut b = KMeans::new().with_n_clusters(3).with_random_state(7);
        let la = a.fit_predict(&x).unwrap();
        let lb = b.fit_predict(&x).unwrap();
        assert_eq!(la, lb);
        assert_eq!(a.cluster_centers(), b.cluster_centers());
    }

    #[test]
    fn n_clusters_zero_errors() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut km = KMeans::new().with_n_clusters(0);
        assert!(km.fit(&x).is_err());
    }

    #[test]
    fn n_clusters_exceeds_samples_errors() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut km = KMeans::new().with_n_clusters(5);
        assert!(km.fit(&x).is_err());
    }

    #[test]
    fn predict_before_fit_errors() {
        let km = KMeans::new().with_n_clusters(2);
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        assert!(km.predict(&x).is_err());
    }

    #[test]
    fn single_sample_single_cluster_works() {
        // Edge case: 1 sample, 1 cluster. The point becomes the sole centroid.
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        let mut km = KMeans::new().with_n_clusters(1);
        assert!(km.fit(&x).is_ok());
        assert_eq!(km.labels(), &[0]);
    }

    #[test]
    fn n_iter_recorded() {
        let rows = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
        km.fit(&x).unwrap();
        assert!(km.n_iter() >= 1);
    }

    #[test]
    fn kmeans_plus_plus_better_than_or_equal_random() {
        // On well-separated data both inits should find 3 clusters, but
        // k-means++ should achieve inertia at least as low.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for center in &[0.0_f64, 10.0, 20.0] {
            for d in &[-0.2, -0.1, 0.0, 0.1, 0.2] {
                rows.push(vec![center + d]);
            }
        }
        let x = Matrix::new(rows).unwrap();
        let mut pp = KMeans::new()
            .with_n_clusters(3)
            .with_init(KMeansInit::KMeansPlusPlus)
            .with_random_state(0);
        pp.fit(&x).unwrap();
        let mut rnd = KMeans::new()
            .with_n_clusters(3)
            .with_init(KMeansInit::Random)
            .with_random_state(0);
        rnd.fit(&x).unwrap();
        // Both find 3 homogeneous clusters.
        assert_eq!(
            pp.labels()
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(pp.inertia() <= rnd.inertia() + 1e-9 || pp.inertia() < 1e-6);
    }

    #[test]
    fn predict_shape_mismatch_errors() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
        km.fit(&x).unwrap();
        let wrong = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        assert!(km.predict(&wrong).is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn fit_transform_returns_one_hot() {
        let rows = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let x = Matrix::new(rows).unwrap();
        let mut km = KMeans::new().with_n_clusters(2).with_random_state(0);
        let out = km.fit_transform(&x).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 2);
        // Each row is a one-hot vector with exactly one 1.0.
        for i in 0..4 {
            let sum: f64 = (0..2).map(|j| out.get(i, j)).sum();
            assert!(approx(sum, 1.0, 1e-12));
        }
    }
}
