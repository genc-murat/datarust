use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::stats;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Imputation strategy, mirroring `sklearn.impute.SimpleImputer`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImputeStrategy {
    /// Fill missing values with the column mean.
    Mean,
    /// Fill missing values with the column median.
    Median,
    /// Fill missing values with the column most frequent value.
    MostFrequent,
    /// Fill missing values with the given constant.
    Constant(f64),
}

/// Impute missing values (represented as `f64::NAN`) using a per-column statistic.
///
/// Mirrors `sklearn.impute.SimpleImputer`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimpleImputer {
    strategy: ImputeStrategy,
    fill_values: Vec<f64>,
    fitted: bool,
}

impl SimpleImputer {
    /// Creates a new simple imputer with the given strategy.
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill_values: vec![],
            fitted: false,
        }
    }

    /// Returns the imputation strategy.
    pub fn strategy(&self) -> &ImputeStrategy {
        &self.strategy
    }

    /// Returns the learned per-column fill values.
    pub fn fill_values(&self) -> &[f64] {
        &self.fill_values
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_fill(x: &Matrix, strategy: &ImputeStrategy) -> Result<Vec<f64>> {
        let data = x.rows_ref();
        let cols = x.ncols();
        let mut fills = Vec::with_capacity(cols);
        for j in 0..cols {
            let col: Vec<f64> = (0..x.nrows())
                .filter_map(|i| {
                    let v = data[i][j];
                    if v.is_nan() {
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();
            let fill = match strategy {
                ImputeStrategy::Mean => {
                    if col.is_empty() {
                        return Err(DatarustError::AllMissing(format!("column {}", j)));
                    }
                    let s: f64 = col.iter().sum();
                    s / col.len() as f64
                }
                ImputeStrategy::Median => {
                    if col.is_empty() {
                        return Err(DatarustError::AllMissing(format!("column {}", j)));
                    }
                    let mut c = col.clone();
                    c.sort_by(|a, b| a.total_cmp(b));
                    stats::median_sorted(&c).expect("column non-empty (checked above)")
                }
                ImputeStrategy::MostFrequent => {
                    if col.is_empty() {
                        return Err(DatarustError::AllMissing(format!("column {}", j)));
                    }
                    let mut c = col.clone();
                    c.sort_by(|a, b| a.total_cmp(b));
                    let single: Vec<Vec<f64>> = c.into_iter().map(|v| vec![v]).collect();
                    stats::mode_column(&single)[0]
                }
                ImputeStrategy::Constant(v) => *v,
            };
            fills.push(fill);
        }
        Ok(fills)
    }
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self::new(ImputeStrategy::Mean)
    }
}

impl FeatureNames for SimpleImputer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.fill_values.len()),
        }
    }
}

impl Transformer for SimpleImputer {
    fn name(&self) -> &'static str {
        "SimpleImputer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.fill_values = Self::compute_fill(x, &self.strategy)?;
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("SimpleImputer".into()));
        }
        if self.fill_values.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.fill_values.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let mut out = x.clone();
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                if out.get(i, j).is_nan() {
                    out.set(i, j, self.fill_values[j]);
                }
            }
        }
        Ok(out)
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

    fn m_missing() -> Matrix {
        Matrix::new(vec![
            vec![1.0, 10.0, nan()],
            vec![2.0, nan(), 5.0],
            vec![3.0, 30.0, 5.0],
            vec![4.0, 40.0, 5.0],
        ])
        .unwrap()
    }

    #[test]
    fn mean_strategy() {
        let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
        let out = imp.fit_transform(&m_missing()).unwrap();
        // col1 mean of (10,30,40) = 26.666
        assert!((imp.fill_values()[1] - (80.0 / 3.0)).abs() < 1e-9);
        assert!((out.get(1, 1) - (80.0 / 3.0)).abs() < 1e-9);
        // col2 mean of (5,5,5) = 5
        assert!((out.get(0, 2) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn median_strategy() {
        let mut imp = SimpleImputer::new(ImputeStrategy::Median);
        let out = imp.fit_transform(&m_missing()).unwrap();
        // col1: 10,30,40 sorted -> median 30
        assert!((imp.fill_values()[1] - 30.0).abs() < 1e-9);
        assert!((out.get(1, 1) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn most_frequent_strategy() {
        let x = Matrix::new(vec![
            vec![nan(), 5.0],
            vec![1.0, 5.0],
            vec![2.0, 9.0],
            vec![2.0, 5.0],
        ])
        .unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::MostFrequent);
        let out = imp.fit_transform(&x).unwrap();
        // col0: 1,2,2 -> mode 2
        assert!((imp.fill_values()[0] - 2.0).abs() < 1e-9);
        assert!((out.get(0, 0) - 2.0).abs() < 1e-9);
        // col1: 5,5,9,5 -> mode 5
        assert!((imp.fill_values()[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn most_frequent_tie_smallest() {
        let x = Matrix::new(vec![vec![nan()], vec![1.0], vec![2.0]]).unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::MostFrequent);
        imp.fit(&x).unwrap();
        // tie between 1 and 2 -> smallest wins
        assert!((imp.fill_values()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn constant_strategy() {
        let mut imp = SimpleImputer::new(ImputeStrategy::Constant(-99.0));
        let out = imp.fit_transform(&m_missing()).unwrap();
        assert!((out.get(0, 2) - (-99.0)).abs() < 1e-9);
        assert!((out.get(1, 1) - (-99.0)).abs() < 1e-9);
        assert!((imp.fill_values()[0] - (-99.0)).abs() < 1e-9);
    }

    #[test]
    fn no_missing_unchanged() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
        let out = imp.fit_transform(&x).unwrap();
        assert_eq!(out.rows_ref(), x.rows_ref());
    }

    #[test]
    fn all_missing_column_errors() {
        let x = Matrix::new(vec![vec![nan(), 1.0], vec![nan(), 2.0]]).unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
        let err = imp.fit(&x).unwrap_err();
        assert!(matches!(err, DatarustError::AllMissing(_)));
    }

    #[test]
    fn all_missing_median_errors() {
        let x = Matrix::new(vec![vec![nan()], vec![nan()]]).unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::Median);
        assert!(imp.fit(&x).is_err());
    }

    #[test]
    fn constant_works_with_all_missing() {
        // constant strategy fills even all-missing columns
        let x = Matrix::new(vec![vec![nan()], vec![nan()]]).unwrap();
        let mut imp = SimpleImputer::new(ImputeStrategy::Constant(0.0));
        let out = imp.fit_transform(&x).unwrap();
        for i in 0..2 {
            assert!((out.get(i, 0) - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let imp = SimpleImputer::new(ImputeStrategy::Mean);
        assert!(matches!(
            imp.transform(&m_missing()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_new_data_uses_fitted() {
        let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
        imp.fit(&m_missing()).unwrap();
        let new = Matrix::new(vec![vec![nan(), nan(), nan()]]).unwrap();
        let out = imp.transform(&new).unwrap();
        assert!((out.get(0, 1) - (80.0 / 3.0)).abs() < 1e-9);
    }
}
