use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Weighting scheme for KNN imputation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KnnWeights {
    /// All neighbors contribute equally.
    Uniform,
    /// Weight by inverse Euclidean distance.
    Distance,
}

/// Impute missing values using k-Nearest Neighbors, mirroring
/// `sklearn.impute.KNNImputer`.
///
/// `fit` stores the training data as a reference set.  `transform` imputes each
/// row of the input by finding the `n_neighbors` closest reference rows and
/// aggregating their values at each missing column.
///
/// Distances are computed only over the features where **both** the target row
/// and a reference row are observed.  The raw squared distance is scaled by
/// `total_features / n_observed_features` so that pairs with many co-observed
/// features are preferred.
///
/// # Examples
///
/// ```rust,no_run
/// use datarust::matrix::Matrix;
/// use datarust::imputer::{KnnImputer, KnnWeights};
/// use datarust::Transformer;
///
/// let x = Matrix::new(vec![vec![1.0, 2.0], vec![f64::NAN, 4.0], vec![3.0, 6.0]]).unwrap();
/// let mut imputer = KnnImputer::new(2, KnnWeights::Uniform);
/// let filled = imputer.fit_transform(&x).unwrap();
/// // NAN at (1,0) imputed from nearest neighbors
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KnnImputer {
    /// Number of neighboring samples to use for imputation.
    n_neighbors: usize,
    /// Weighting scheme.
    weights: KnnWeights,
    /// Reference dataset stored during `fit`.
    reference: Option<Matrix>,
    fitted: bool,
}

impl KnnImputer {
    /// Creates a new KNN imputer with the given neighbor count and weights.
    pub fn new(n_neighbors: usize, weights: KnnWeights) -> Self {
        Self {
            n_neighbors,
            weights,
            reference: None,
            fitted: false,
        }
    }

    /// Returns the number of neighbors used for imputation.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Returns the weighting scheme.
    pub fn weights(&self) -> KnnWeights {
        self.weights
    }

    fn validate_fitted_state(&self) -> Result<&Matrix> {
        if self.n_neighbors == 0 {
            return Err(DatarustError::InvalidInput(
                "KnnImputer has inconsistent fitted state".into(),
            ));
        }
        let reference = self.reference.as_ref().ok_or_else(|| {
            DatarustError::InvalidInput("KnnImputer has inconsistent fitted state".into())
        })?;
        reference.validate_no_infinite().map_err(|_| {
            DatarustError::InvalidInput("KnnImputer has inconsistent fitted state".into())
        })?;
        Ok(reference)
    }

    /// Squared Euclidean distance between two rows, considering only features
    /// where both are not NaN. `obs` lists the features observed in the target
    /// row (precomputed once per row), so only the reference value needs a NaN
    /// check. Returns the distance scaled by `n_features / n_observed` and the
    /// number of co-observed features.
    fn nan_euclidean_sq(a: &[f64], b: &[f64], obs: &[usize]) -> Option<(f64, usize)> {
        let n = a.len();
        let mut sq_sum = 0.0;
        let mut n_obs = 0;
        for &i in obs {
            let vb = b[i];
            if !vb.is_nan() {
                let d = a[i] - vb;
                sq_sum += d * d;
                n_obs += 1;
            }
        }
        if n_obs == 0 {
            return None;
        }
        let scale = n as f64 / n_obs as f64;
        Some((sq_sum * scale, n_obs))
    }

    /// Core neighbor search. The reference matrix is borrowed as row slices, so
    /// it is never copied, and it must be validated exactly once per
    /// `transform` call by the caller.
    fn find_neighbors_from(
        ref_matrix: &Matrix,
        row: &[f64],
        n_neighbors: usize,
    ) -> Result<Vec<(f64, usize)>> {
        let obs: Vec<usize> = row
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, _)| i)
            .collect();
        #[cfg(feature = "rayon")]
        let mut distances: Vec<(f64, usize)> = (0..ref_matrix.nrows())
            .into_par_iter()
            .filter_map(|idx| {
                Self::nan_euclidean_sq(row, ref_matrix.row(idx), &obs).map(|(d, _)| (d, idx))
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let mut distances: Vec<(f64, usize)> = Vec::with_capacity(ref_matrix.nrows());
        #[cfg(not(feature = "rayon"))]
        for idx in 0..ref_matrix.nrows() {
            if let Some((d, _)) = Self::nan_euclidean_sq(row, ref_matrix.row(idx), &obs) {
                distances.push((d, idx));
            }
        }
        if distances.is_empty() {
            return Err(DatarustError::AllMissing(
                "row has no co-observed features with any reference row".into(),
            ));
        }
        // Only the k nearest neighbors are needed, so partial selection beats a
        // full sort; the aggregation is order-independent.
        let k = n_neighbors.min(distances.len());
        if k < distances.len() {
            distances.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
        }
        Ok(distances[..k].to_vec())
    }

    fn impute_row_from(
        ref_matrix: &Matrix,
        row: &[f64],
        neighbors: &[(f64, usize)],
        weights: KnnWeights,
    ) -> Vec<f64> {
        let mut out = row.to_vec();

        if neighbors.is_empty() {
            return out;
        }

        for (j, val) in out.iter_mut().enumerate() {
            if !val.is_nan() {
                continue;
            }
            match weights {
                KnnWeights::Uniform => {
                    let mut s = 0.0;
                    let mut cnt = 0;
                    for (_, idx) in neighbors {
                        let v = ref_matrix.row(*idx)[j];
                        if !v.is_nan() {
                            s += v;
                            cnt += 1;
                        }
                    }
                    *val = if cnt > 0 { s / cnt as f64 } else { *val };
                }
                KnnWeights::Distance => {
                    let mut s = 0.0;
                    let mut wsum = 0.0;
                    for (d, idx) in neighbors {
                        let v = ref_matrix.row(*idx)[j];
                        if !v.is_nan() {
                            let w = 1.0 / d.max(1e-12);
                            s += v * w;
                            wsum += w;
                        }
                    }
                    *val = if wsum > 0.0 { s / wsum } else { *val };
                }
            }
        }
        out
    }
}

impl Default for KnnImputer {
    fn default() -> Self {
        Self::new(5, KnnWeights::Uniform)
    }
}

impl FeatureNames for KnnImputer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let ncols = self.reference.as_ref().map_or(0, |m| m.ncols());
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(ncols),
        }
    }
}

impl Transformer for KnnImputer {
    fn name(&self) -> &'static str {
        "KnnImputer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        if self.n_neighbors == 0 {
            return Err(DatarustError::InvalidInput(
                "n_neighbors must be >= 1".into(),
            ));
        }
        if x.nrows() == 0 {
            return Err(DatarustError::InvalidInput(
                "KNN imputer needs at least one sample".into(),
            ));
        }
        x.validate_no_infinite()?;
        self.reference = Some(x.clone());
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("KnnImputer".into()));
        }
        let ref_matrix = self.validate_fitted_state()?;
        if ref_matrix.ncols() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", ref_matrix.ncols()),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_no_infinite()?;
        let mut out = Vec::with_capacity(x.nrows());
        for row in x.iter_rows() {
            if row.iter().any(|v| v.is_nan()) {
                // `ref_matrix` is validated once above; the per-row helpers
                // borrow it directly without re-validating or copying it.
                let neighbors = Self::find_neighbors_from(ref_matrix, row, self.n_neighbors)?;
                let imputed = Self::impute_row_from(ref_matrix, row, &neighbors, self.weights);
                out.push(imputed);
            } else {
                out.push(row.to_vec());
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nan() -> f64 {
        f64::NAN
    }

    fn sample_data() -> Matrix {
        Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ])
        .unwrap()
    }

    #[test]
    fn no_missing_passthrough() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut imp = KnnImputer::default();
        let out = imp.fit_transform(&x).unwrap();
        assert_eq!(out.rows_ref(), x.rows_ref());
    }

    #[test]
    fn impute_single_nan() {
        let mut imp = KnnImputer::new(1, KnnWeights::Uniform);
        imp.fit(&sample_data()).unwrap();
        let x = Matrix::new(vec![vec![nan(), 2.0, 3.0]]).unwrap();
        let out = imp.transform(&x).unwrap();
        // nearest neighbor by distance on cols 1,2: ref row0 (1,2,3)
        assert!((out.get(0, 0) - 1.0).abs() < 1e-9);
        assert!((out.get(0, 1) - 2.0).abs() < 1e-9);
        assert!((out.get(0, 2) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn impute_with_multiple_nans() {
        let mut imp = KnnImputer::new(1, KnnWeights::Uniform);
        imp.fit(&sample_data()).unwrap();
        let x = Matrix::new(vec![vec![nan(), nan(), 3.0]]).unwrap();
        let out = imp.transform(&x).unwrap();
        // nearest neighbor: row0 (1,2,3). impute col0->1, col1->2
        assert!((out.get(0, 0) - 1.0).abs() < 1e-9);
        assert!((out.get(0, 1) - 2.0).abs() < 1e-9);
        assert!((out.get(0, 2) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn distance_weighted_different() {
        let ref_data = Matrix::new(vec![vec![10.0, 20.0], vec![1.0, 2.0]]).unwrap();
        let mut imp = KnnImputer::new(2, KnnWeights::Distance);
        imp.fit(&ref_data).unwrap();
        let x = Matrix::new(vec![vec![nan(), 20.0]]).unwrap();
        let out = imp.transform(&x).unwrap();
        // row0 (10,20) distance on col1 = 0, weight = inf (clamped). row1 distance on col1 = 18^2 * (2/1) = 648, weight ~ 1/648
        // So nearest is row0, impute col0 -> 10.0
        assert!((out.get(0, 0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn k_larger_than_samples() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![nan(), 5.0]]).unwrap();
        let mut imp = KnnImputer::new(10, KnnWeights::Uniform);
        let out = imp.fit_transform(&x).unwrap();
        // k=10 but only 2 reference rows (excluding the query row). col0 imputed from 1.0 and 3.0 -> 2.0
        assert!((out.get(2, 0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn all_nan_row_errors() {
        let mut imp = KnnImputer::default();
        imp.fit(&sample_data()).unwrap();
        let x = Matrix::new(vec![vec![nan(), nan(), nan()]]).unwrap();
        let result = imp.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let imp = KnnImputer::default();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(
            imp.transform(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn shape_mismatch_errors() {
        let mut imp = KnnImputer::default();
        imp.fit(&sample_data()).unwrap();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(
            imp.transform(&x),
            Err(DatarustError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn feature_names_none() {
        let mut imp = KnnImputer::default();
        imp.fit(&sample_data()).unwrap();
        let names = imp.feature_names_out(None);
        assert_eq!(names, vec!["x0", "x1", "x2"]);
    }

    #[test]
    fn feature_names_some() {
        let mut imp = KnnImputer::default();
        imp.fit(&sample_data()).unwrap();
        let names = imp.feature_names_out(Some(&["a".into(), "b".into(), "c".into()]));
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn fit_empty_errors() {
        // `Matrix::new` rejects empty input at construction time, which is the
        // equivalent protection: empty data can never reach `fit`.
        assert!(Matrix::new(vec![] as Vec<Vec<f64>>).is_err());
    }

    #[test]
    fn impute_keeps_non_nan() {
        let mut imp = KnnImputer::default();
        imp.fit(&sample_data()).unwrap();
        let x = Matrix::new(vec![vec![nan(), 2.0, 3.0]]).unwrap();
        let out = imp.transform(&x).unwrap();
        assert!((out.get(0, 1) - 2.0).abs() < 1e-9);
        assert!((out.get(0, 2) - 3.0).abs() < 1e-9);
    }
}
