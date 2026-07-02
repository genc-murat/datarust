use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{default_input_names, CategoricalTransformer, FeatureNames};

/// Strategy for handling unknown categories during transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnknownFrequency {
    /// Return an error when an unseen category is encountered.
    Error,
    /// Map unknown categories to 0.0.
    #[default]
    Zero,
}

/// Replace each category with its frequency (count or normalized proportion),
/// mirroring a common "frequency / count encoder".
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyEncoder {
    normalized: bool,
    handle_unknown: UnknownFrequency,
    /// Per-column mapping: category -> frequency value.
    mappings: Vec<HashMap<String, f64>>,
    fitted: bool,
}

impl FrequencyEncoder {
    /// Create a frequency encoder. If `normalized`, frequencies are divided by
    /// the row count (proportions); otherwise they are raw counts.
    pub fn new(normalized: bool) -> Self {
        Self {
            normalized,
            handle_unknown: UnknownFrequency::Zero,
            mappings: vec![],
            fitted: false,
        }
    }

    /// Returns whether frequencies are normalized to proportions.
    pub fn normalized(&self) -> bool {
        self.normalized
    }

    /// Sets the strategy for handling unknown categories during transform.
    pub fn handle_unknown(mut self, strategy: UnknownFrequency) -> Self {
        self.handle_unknown = strategy;
        self
    }

    /// Learns the per-column frequency of each category.
    pub fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        let ncols = x.ncols();
        let n = x.nrows();
        let mut mappings = Vec::with_capacity(ncols);
        for j in 0..ncols {
            let col = x.column(j);
            let mut counts: HashMap<String, f64> = HashMap::new();
            for cat in &col {
                *counts.entry(cat.clone()).or_insert(0.0) += 1.0;
            }
            if self.normalized {
                let inv = 1.0 / n as f64;
                for v in counts.values_mut() {
                    *v *= inv;
                }
            }
            mappings.push(counts);
        }
        self.mappings = mappings;
        self.fitted = true;
        Ok(())
    }

    /// Replaces each category with its learned frequency.
    pub fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("FrequencyEncoder".into()));
        }
        if x.ncols() != self.mappings.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} categorical columns", self.mappings.len()),
                actual: format!("{} columns", x.ncols()),
            });
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mappings = &self.mappings;
            let handle_unknown = self.handle_unknown;
            let x_data = &x.data;
            out.par_iter_mut()
                .enumerate()
                .try_for_each(|(i, out_row)| {
                    for (j, cell) in out_row.iter_mut().enumerate() {
                        let val = &x_data[i][j];
                        *cell = match mappings[j].get(val) {
                            Some(&v) => v,
                            None => match handle_unknown {
                                UnknownFrequency::Zero => 0.0,
                                UnknownFrequency::Error => {
                                    return Err(DatarustError::UnknownCategory(format!(
                                        "column {} value '{}'",
                                        j, val
                                    )))
                                }
                            },
                        };
                    }
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, out_row) in out.iter_mut().enumerate() {
                for (j, cell) in out_row.iter_mut().enumerate() {
                    let val = x.get(i, j);
                    *cell = match self.mappings[j].get(val) {
                        Some(&v) => v,
                        None => match self.handle_unknown {
                            UnknownFrequency::Zero => 0.0,
                            UnknownFrequency::Error => {
                                return Err(DatarustError::UnknownCategory(format!(
                                    "column {} value '{}'",
                                    j, val
                                )))
                            }
                        },
                    };
                }
            }
        }

        Matrix::new(out)
    }

    /// Fits the encoder and transforms the input in one step.
    pub fn fit_transform(&mut self, x: &StrMatrix) -> Result<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }
}

impl Default for FrequencyEncoder {
    fn default() -> Self {
        Self::new(false)
    }
}

impl CategoricalTransformer for FrequencyEncoder {
    fn name(&self) -> &'static str {
        "FrequencyEncoder"
    }

    fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        self.fit(x)
    }

    fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        self.transform(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for FrequencyEncoder {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n = self.mappings.len();
        match input_features {
            Some(fs) => (0..n)
                .map(|i| fs.get(i).cloned().unwrap_or_else(|| format!("x{}", i)))
                .collect(),
            None => default_input_names(n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_counts() {
        let x = StrMatrix::from_column(["A", "A", "A", "B", "B", "C"]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        let out = fe.fit_transform(&x).unwrap();
        // A -> 3, B -> 2, C -> 1
        assert_eq!(out.row(0), [3.0]);
        assert_eq!(out.row(1), [3.0]);
        assert_eq!(out.row(3), [2.0]);
        assert_eq!(out.row(5), [1.0]);
    }

    #[test]
    fn normalized_proportions() {
        let x = StrMatrix::from_column(["A", "A", "B", "C"]).unwrap();
        let mut fe = FrequencyEncoder::new(true);
        let out = fe.fit_transform(&x).unwrap();
        // n=4: A -> 0.5, B -> 0.25, C -> 0.25
        assert!((out.get(0, 0) - 0.5).abs() < 1e-12);
        assert!((out.get(2, 0) - 0.25).abs() < 1e-12);
        assert!((out.get(3, 0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn unknown_maps_to_zero() {
        let x = StrMatrix::from_column(["A", "B"]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        fe.fit(&x).unwrap();
        let x2 = StrMatrix::from_column(["A", "Z"]).unwrap();
        let out = fe.transform(&x2).unwrap();
        assert_eq!(out.get(0, 0), 1.0);
        assert_eq!(out.get(1, 0), 0.0); // unknown -> 0
    }

    #[test]
    fn multi_column_independent() {
        let x =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["a", "y"], vec!["b", "x"]]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        let out = fe.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        // col0: a=2, b=1 ; col1: x=2, y=1
        assert_eq!(out.get(0, 0), 2.0);
        assert_eq!(out.get(2, 0), 1.0);
        assert_eq!(out.get(0, 1), 2.0);
        assert_eq!(out.get(1, 1), 1.0);
    }

    #[test]
    fn transform_before_fit_errors() {
        let fe = FrequencyEncoder::new(false);
        let x = StrMatrix::from_column(["a"]).unwrap();
        assert!(matches!(fe.transform(&x), Err(DatarustError::NotFitted(_))));
    }

    #[test]
    fn column_count_mismatch() {
        let x = StrMatrix::from_strings(vec![vec!["a", "b"], vec!["c", "d"]]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        fe.fit(&x).unwrap();
        let x2 = StrMatrix::from_column(["a"]).unwrap();
        assert!(fe.transform(&x2).is_err());
    }

    #[test]
    fn counts_sum_to_n() {
        let x = StrMatrix::from_column(["A", "A", "B", "C", "C", "C"]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        let out = fe.fit_transform(&x).unwrap();
        let s: f64 = out.rows_ref().iter().flatten().sum();
        // unique counts weighted by occurrence: A(2)*2 + B(1) + C(3)*3 = 4+1+9 = 14? no
        // each row maps to its category count: rows = A,A,B,C,C,C
        // counts: A=2 -> two rows of 2 = 4 ; B=1 -> one row of 1 ; C=3 -> three rows of 3 = 9
        // sum = 4 + 1 + 9 = 14
        assert!((s - 14.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_error_mode() {
        let x = StrMatrix::from_column(["A", "B"]).unwrap();
        let mut fe = FrequencyEncoder::new(false).handle_unknown(UnknownFrequency::Error);
        fe.fit(&x).unwrap();
        let x2 = StrMatrix::from_column(["A", "Z"]).unwrap();
        assert!(matches!(
            fe.transform(&x2),
            Err(DatarustError::UnknownCategory(_))
        ));
    }

    #[test]
    fn feature_names_short_input_pads_with_synthetic() {
        let x = StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"]]).unwrap();
        let mut fe = FrequencyEncoder::new(false);
        fe.fit(&x).unwrap();
        // 2 columns but only 1 name provided
        let names = fe.feature_names_out(Some(&["city".into()]));
        assert_eq!(names, vec!["city", "x1"]);
    }
}
