use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Target output distribution for [`QuantileTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OutputDistribution {
    /// Transform data to a uniform distribution on `[0, 1]`.
    #[default]
    Uniform,
    /// Transform data to a standard normal `N(0, 1)`.
    Normal,
}

/// Transform features using quantiles information, mirroring
/// `sklearn.preprocessing.QuantileTransformer`.
///
/// This method transforms the features to follow a uniform or a normal
/// distribution. It is robust to outliers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantileTransformer {
    n_quantiles: usize,
    output_distribution: OutputDistribution,
    /// Sorted reference values per column (the empirical quantile function).
    references: Vec<Vec<f64>>,
    n_features: usize,
    fitted: bool,
}

impl QuantileTransformer {
    /// Create a new transformer. `n_quantiles` must be >= 1.
    pub fn new(n_quantiles: usize) -> Result<Self> {
        if n_quantiles == 0 {
            return Err(DatarustError::InvalidConfig(
                "n_quantiles must be >= 1".into(),
            ));
        }
        Ok(Self {
            n_quantiles,
            output_distribution: OutputDistribution::Uniform,
            references: vec![],
            n_features: 0,
            fitted: false,
        })
    }

    /// Builder: set the target output distribution.
    pub fn output_distribution(mut self, d: OutputDistribution) -> Self {
        self.output_distribution = d;
        self
    }

    /// Compute reference quantiles for a sorted column.
    fn compute_references(sorted_col: &[f64], n_quantiles: usize) -> Vec<f64> {
        let n = sorted_col.len();
        // References at evenly spaced positions.
        (0..n_quantiles)
            .map(|i| {
                let q = i as f64 / (n_quantiles - 1).max(1) as f64;
                let pos = q * (n - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                if lo == hi {
                    sorted_col[lo]
                } else {
                    let frac = pos - lo as f64;
                    sorted_col[lo] * (1.0 - frac) + sorted_col[hi] * frac
                }
            })
            .collect()
    }

    /// Transform a single value through the empirical CDF.
    fn transform_value(value: f64, refs: &[f64]) -> Result<f64> {
        if value.is_nan() {
            return Err(DatarustError::InvalidInput(
                "QuantileTransformer: NaN encountered in input".into(),
            ));
        }
        let n = refs.len();
        if n == 0 {
            return Ok(0.0);
        }
        if n == 1 {
            // All values map to 0.5 percentile (the middle).
            return Ok(0.5);
        }
        // Clamp to [0, 1] percentile range.
        if value <= refs[0] {
            return Ok(0.0);
        }
        if value >= refs[n - 1] {
            return Ok(1.0);
        }
        // Binary search for position.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if refs[mid] <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is first index where refs[lo] > value.
        // Linear interpolation between refs[lo-1] and refs[lo].
        let lower = lo - 1;
        let upper = lo;
        let denom = refs[upper] - refs[lower];
        let frac = if denom.abs() < f64::EPSILON {
            0.5
        } else {
            (value - refs[lower]) / denom
        };
        // Map to percentile in [0, 1].
        Ok(lower as f64 / (n - 1) as f64 + frac / (n - 1) as f64)
    }
}

/// Default: 1000 quantiles, uniform output distribution.
impl Default for QuantileTransformer {
    fn default() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
            references: vec![],
            n_features: 0,
            fitted: false,
        }
    }
}

impl Transformer for QuantileTransformer {
    fn name(&self) -> &'static str {
        "QuantileTransformer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let ncols = x.ncols();
        let mut refs_all = Vec::with_capacity(ncols);
        let n_q = self.n_quantiles.min(x.nrows());
        for j in 0..ncols {
            let mut col = x.col(j);
            col.sort_by(|a, b| a.total_cmp(b));
            refs_all.push(Self::compute_references(&col, n_q.max(1)));
        }
        self.references = refs_all;
        self.n_features = ncols;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("QuantileTransformer".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, cell) in out_row.iter_mut().enumerate() {
                let percentile = Self::transform_value(x.get(i, j), &self.references[j])?;
                *cell = match self.output_distribution {
                    OutputDistribution::Uniform => percentile.clamp(0.0, 1.0),
                    OutputDistribution::Normal => {
                        // Clamp percentile away from 0 and 1 to avoid infinite
                        // output from the inverse normal CDF.
                        let clamped = percentile.clamp(1e-9, 1.0 - 1e-9);
                        inv_normal_cdf(clamped)
                    }
                };
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for QuantileTransformer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features),
        }
    }
}

/// Inverse of the standard normal CDF (probit function) using the
/// Acklam / Beasley-Springer-Moro approximation.
fn inv_normal_cdf(p: f64) -> f64 {
    // Coefficients for the rational approximation.
    let a = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    let b = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    let c = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    let d = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];

    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        // Rational approximation for lower region.
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= phigh {
        // Rational approximation for central region.
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region.
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn uniform_output_basic() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let mut qt = QuantileTransformer::new(5).unwrap();
        let out = qt.fit_transform(&x).unwrap();
        // Uniform output: should span [0, 1]
        for i in 0..5 {
            assert!(out.get(i, 0) >= 0.0 && out.get(i, 0) <= 1.0);
        }
        // Min maps to 0, max maps to 1
        assert!(approx(out.get(0, 0), 0.0, 1e-9));
        assert!(approx(out.get(4, 0), 1.0, 1e-9));
    }

    #[test]
    fn normal_output_approximately_standard() {
        let x = Matrix::new(vec![
            vec![-3.0],
            vec![-1.0],
            vec![0.0],
            vec![1.0],
            vec![3.0],
        ])
        .unwrap();
        let mut qt = QuantileTransformer::new(5)
            .unwrap()
            .output_distribution(OutputDistribution::Normal);
        let out = qt.fit_transform(&x).unwrap();
        // Check mean ≈ 0 and all values finite
        let mean: f64 = (0..5).map(|i| out.get(i, 0)).sum::<f64>() / 5.0;
        assert!(approx(mean, 0.0, 0.5));
        for i in 0..5 {
            assert!(out.get(i, 0).is_finite());
        }
    }

    #[test]
    fn preserves_order() {
        let x = Matrix::new(vec![vec![5.0], vec![1.0], vec![3.0], vec![2.0], vec![4.0]]).unwrap();
        let mut qt = QuantileTransformer::new(5).unwrap();
        let out = qt.fit_transform(&x).unwrap();
        // The smallest value should have the smallest output.
        let vals: Vec<f64> = (0..5).map(|i| out.get(i, 0)).collect();
        assert!(vals[1] < vals[3]); // 1 < 2
        assert!(vals[3] < vals[2]); // 2 < 3
        assert!(vals[2] < vals[4]); // 3 < 4
        assert!(vals[4] < vals[0]); // 4 < 5
    }

    #[test]
    fn multi_column_independent() {
        let x = Matrix::new(vec![vec![0.0, 100.0], vec![5.0, 200.0], vec![10.0, 300.0]]).unwrap();
        let mut qt = QuantileTransformer::new(3).unwrap();
        let out = qt.fit_transform(&x).unwrap();
        // Both columns should map independently to [0, 1]
        for j in 0..2 {
            assert!(approx(out.get(0, j), 0.0, 1e-9));
            assert!(approx(out.get(2, j), 1.0, 1e-9));
        }
    }

    #[test]
    fn transform_new_data_extrapolates() {
        let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let mut qt = QuantileTransformer::new(5).unwrap();
        qt.fit(&x).unwrap();
        let new = Matrix::new(vec![vec![-5.0], vec![5.0]]).unwrap();
        let out = qt.transform(&new).unwrap();
        // Below min -> 0, above max -> 1
        assert!(approx(out.get(0, 0), 0.0, 1e-9));
        assert!(approx(out.get(1, 0), 1.0, 1e-9));
    }

    #[test]
    fn transform_before_fit_errors() {
        let qt = QuantileTransformer::new(5).unwrap();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(qt.transform(&x), Err(DatarustError::NotFitted(_))));
    }

    #[test]
    fn n_quantiles_zero_errors() {
        assert!(QuantileTransformer::new(0).is_err());
    }

    #[test]
    fn feature_names_preserved() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut qt = QuantileTransformer::new(2).unwrap();
        qt.fit(&x).unwrap();
        let names = qt.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn inv_normal_cdf_known_values() {
        // probit(0.5) ≈ 0
        assert!(approx(inv_normal_cdf(0.5), 0.0, 1e-4));
        // probit(0.975) ≈ 1.96
        assert!(approx(inv_normal_cdf(0.975), 1.9599, 0.01));
        // probit(0.025) ≈ -1.96
        assert!(approx(inv_normal_cdf(0.025), -1.9599, 0.01));
    }
}
