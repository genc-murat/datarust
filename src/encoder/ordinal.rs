use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{default_input_names, CategoricalTransformer, FeatureNames};

/// How to determine categories for ordinal encoding.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OrdinalCategories {
    /// Infer categories from the training data (sorted lexicographically per column).
    #[default]
    Auto,
    /// Provide explicit category order per column. If used, every column's categories
    /// must be specified; each list determines the ordinal mapping.
    Manual(Vec<Vec<String>>),
}

/// Strategy for unknown categories during `transform`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OrdinalHandleUnknown {
    /// Raise an error on unknown categories (default).
    #[default]
    Error,
    /// Encode unknown categories as `-1`.
    UseNegOne,
}

/// Encode categorical features as ordinal integers (0, 1, 2, …), mirroring
/// `sklearn.preprocessing.OrdinalEncoder`.
///
/// Input is a 2-D [`StrMatrix`]; output is a numeric [`Matrix`] of the same
/// shape, where each cell is replaced by the ordinal index of its category.
///
/// Categories per column are sorted lexicographically by default (sklearn
/// default), or can be user-specified via [`OrdinalCategories::Manual`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrdinalEncoder {
    categories: OrdinalCategories,
    handle_unknown: OrdinalHandleUnknown,
    category_lists: Vec<Vec<String>>,
    category_indices: Vec<HashMap<String, usize>>,
    fitted: bool,
}

impl OrdinalEncoder {
    /// Creates a new ordinal encoder with the given category config.
    pub fn new(categories: OrdinalCategories) -> Self {
        Self {
            categories,
            handle_unknown: OrdinalHandleUnknown::default(),
            category_lists: vec![],
            category_indices: vec![],
            fitted: false,
        }
    }

    /// Sets how unknown categories are handled during transform.
    pub fn handle_unknown(mut self, h: OrdinalHandleUnknown) -> Self {
        self.handle_unknown = h;
        self
    }

    /// Returns the learned categories per column.
    pub fn categories(&self) -> &[Vec<String>] {
        &self.category_lists
    }

    /// Learns the category-to-index mapping per column.
    pub fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        let ncols = x.ncols();
        match &self.categories {
            OrdinalCategories::Auto => {
                let mut cat_lists = Vec::with_capacity(ncols);
                let mut cat_indices = Vec::with_capacity(ncols);
                for j in 0..ncols {
                    let col = x.column(j);
                    let mut set: std::collections::BTreeSet<String> =
                        std::collections::BTreeSet::new();
                    for s in &col {
                        set.insert(s.clone());
                    }
                    let list: Vec<String> = set.into_iter().collect();
                    let idx: HashMap<String, usize> = list
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (c.clone(), i))
                        .collect();
                    cat_lists.push(list);
                    cat_indices.push(idx);
                }
                self.category_lists = cat_lists;
                self.category_indices = cat_indices;
            }
            OrdinalCategories::Manual(lists) => {
                if lists.len() != ncols {
                    return Err(DatarustError::ShapeMismatch {
                        expected: format!("{} category lists", ncols),
                        actual: format!("{} lists", lists.len()),
                    });
                }
                let mut cat_indices = Vec::with_capacity(ncols);
                for (j, list) in lists.iter().enumerate() {
                    let idx: HashMap<String, usize> = list
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (c.clone(), i))
                        .collect();
                    if idx.len() != list.len() {
                        return Err(DatarustError::InvalidConfig(format!(
                            "duplicate category in column {}",
                            j
                        )));
                    }
                    cat_indices.push(idx);
                }
                self.category_lists = lists.clone();
                self.category_indices = cat_indices;
            }
        }
        self.fitted = true;
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    /// Encodes the input as ordinal integer codes.
    pub fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("OrdinalEncoder".into()));
        }
        if x.ncols() != self.category_lists.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} columns", self.category_lists.len()),
                actual: format!("{} columns", x.ncols()),
            });
        }
        let mut out = vec![vec![0.0; x.ncols()]; x.nrows()];

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let category_indices = &self.category_indices;
            let handle_unknown = self.handle_unknown;
            let x_data = &x.data;
            out.par_iter_mut().enumerate().try_for_each(|(i, row)| {
                for (j, indices) in category_indices.iter().enumerate() {
                    row[j] = match indices.get(&x_data[i][j]) {
                        Some(&idx) => idx as f64,
                        None => match handle_unknown {
                            OrdinalHandleUnknown::Error => {
                                return Err(DatarustError::UnknownCategory(format!(
                                    "column {} value '{}'",
                                    j, x_data[i][j]
                                )))
                            }
                            OrdinalHandleUnknown::UseNegOne => -1.0,
                        },
                    };
                }
                Ok(())
            })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    let val = x.get(i, j);
                    out[i][j] = match self.category_indices[j].get(val) {
                        Some(&idx) => idx as f64,
                        None => match self.handle_unknown {
                            OrdinalHandleUnknown::Error => {
                                return Err(DatarustError::UnknownCategory(format!(
                                    "column {} value '{}'",
                                    j, val
                                )))
                            }
                            OrdinalHandleUnknown::UseNegOne => -1.0,
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

    #[allow(clippy::needless_range_loop)]
    /// Decodes ordinal integer codes back to category strings.
    pub fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("OrdinalEncoder".into()));
        }
        if y.ncols() != self.category_lists.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} columns", self.category_lists.len()),
                actual: format!("{} columns", y.ncols()),
            });
        }
        let mut out: Vec<Vec<String>> = Vec::with_capacity(y.nrows());
        for i in 0..y.nrows() {
            let mut row = Vec::with_capacity(y.ncols());
            for j in 0..y.ncols() {
                let v = y.get(i, j);
                if v.is_nan() {
                    return Err(DatarustError::InvalidInput(format!(
                        "NaN value at row {}, column {} in inverse_transform input",
                        i, j
                    )));
                }
                let idx = v as isize;
                if idx == -1 {
                    // Sentinel for unknown categories (UseNegOne)
                    row.push(String::new());
                } else if idx < 0 || idx as usize >= self.category_lists[j].len() {
                    return Err(DatarustError::UnknownLabel(format!(
                        "index {} out of range for column {}",
                        idx, j
                    )));
                } else {
                    row.push(self.category_lists[j][idx as usize].clone());
                }
            }
            out.push(row);
        }
        StrMatrix::new(out)
    }
}

impl CategoricalTransformer for OrdinalEncoder {
    fn name(&self) -> &'static str {
        "OrdinalEncoder"
    }

    fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        self.fit(x)
    }

    fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        self.transform(x)
    }

    fn inverse_transform(&self, y: &Matrix) -> Result<StrMatrix> {
        self.inverse_transform(y)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self::new(OrdinalCategories::default())
    }
}

impl FeatureNames for OrdinalEncoder {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let n = self.category_lists.len();
        let names: Vec<String> = match input_features {
            Some(fs) => (0..n)
                .map(|i| fs.get(i).cloned().unwrap_or_else(|| format!("x{}", i)))
                .collect(),
            None => default_input_names(n),
        };
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_auto_fit() {
        let s = StrMatrix::from_column(["small", "medium", "large", "small"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        let out = enc.fit_transform(&s).unwrap();
        // categories sorted: large(0), medium(1), small(2)
        assert_eq!(enc.categories()[0], &["large", "medium", "small"]);
        assert_eq!(out.row(0), [2.0]); // small
        assert_eq!(out.row(1), [1.0]); // medium
        assert_eq!(out.row(2), [0.0]); // large
    }

    #[test]
    fn manual_categories() {
        let s = StrMatrix::from_column(["small", "medium", "large"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Manual(vec![vec![
            "small".into(),
            "medium".into(),
            "large".into(),
        ]]));
        let out = enc.fit_transform(&s).unwrap();
        assert_eq!(out.row(0), [0.0]);
        assert_eq!(out.row(1), [1.0]);
        assert_eq!(out.row(2), [2.0]);
    }

    #[test]
    fn inverse_round_trip() {
        let original = vec!["cat", "dog", "bird", "dog"];
        let s = StrMatrix::from_column(original.clone()).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        let encoded = enc.fit_transform(&s).unwrap();
        let decoded = enc.inverse_transform(&encoded).unwrap();
        for (i, &orig) in original.iter().enumerate() {
            assert_eq!(decoded.get(i, 0), orig);
        }
    }

    #[test]
    fn inverse_bad_index_errors() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        enc.fit(&s).unwrap();
        let bad = Matrix::new(vec![vec![0.0], vec![5.0]]).unwrap();
        assert!(enc.inverse_transform(&bad).is_err());
    }

    #[test]
    fn multi_column() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        let out = enc.fit_transform(&s).unwrap();
        assert_eq!(out.ncols(), 2);
        // col0: a(0), b(1) ; col1: x(0), y(1)
        assert_eq!(out.row(0), [0.0, 0.0]);
        assert_eq!(out.row(1), [1.0, 1.0]);
        assert_eq!(out.row(2), [0.0, 1.0]);
    }

    #[test]
    fn handle_unknown_error() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        enc.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "z"]).unwrap();
        assert!(matches!(
            enc.transform(&s2),
            Err(DatarustError::UnknownCategory(_))
        ));
    }

    #[test]
    fn handle_unknown_neg_one() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto)
            .handle_unknown(OrdinalHandleUnknown::UseNegOne);
        enc.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let out = enc.transform(&s2).unwrap();
        assert_eq!(out.row(0), [0.0]);
        assert_eq!(out.row(1), [-1.0]);
    }

    #[test]
    fn manual_column_count_mismatch_errors() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Manual(vec![
            vec!["a".into()],
            vec!["b".into()],
        ]));
        assert!(enc.fit(&s).is_err());
    }

    #[test]
    fn manual_duplicate_category_errors() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Manual(vec![vec![
            "a".into(),
            "a".into(),
        ]]));
        assert!(enc.fit(&s).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        let s = StrMatrix::from_column(["a"]).unwrap();
        assert!(matches!(
            enc.transform(&s),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn inverse_before_fit_errors() {
        let enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        let m = Matrix::new(vec![vec![0.0]]).unwrap();
        assert!(matches!(
            enc.inverse_transform(&m),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn serde_derive() {
        // compile-test: the struct must derive Serialize/Deserialize when feature is on
        let s = StrMatrix::from_column(["x", "y"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        enc.fit(&s).unwrap();
        // no explicit assertion; just ensure the type works under serde feature exists
        #[cfg(feature = "serde")]
        {
            let json = crate::serialize::to_json(&enc).unwrap();
            let _restored: OrdinalEncoder = crate::serialize::from_json(&json).unwrap();
        }
    }

    #[test]
    fn inverse_transform_sentinel_decodes_to_empty() {
        let s = StrMatrix::from_column(["cat", "dog"]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto)
            .handle_unknown(OrdinalHandleUnknown::UseNegOne);
        enc.fit(&s).unwrap();
        // unknown value 'fox' encodes to -1.0
        let x = StrMatrix::from_column(["cat", "fox"]).unwrap();
        let coded = enc.transform(&x).unwrap();
        assert_eq!(coded.get(1, 0), -1.0);
        let decoded = enc.inverse_transform(&coded).unwrap();
        assert_eq!(decoded.get(0, 0), "cat");
        assert_eq!(decoded.get(1, 0), "");
    }

    #[test]
    fn feature_names_short_input_pads_with_synthetic() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["c", "z"]]).unwrap();
        let mut enc = OrdinalEncoder::new(OrdinalCategories::Auto);
        enc.fit(&s).unwrap();
        // 2 columns but only 1 name provided
        let names = enc.feature_names_out(Some(&["city".into()]));
        assert_eq!(names, vec!["city", "x1"]);
    }
}
