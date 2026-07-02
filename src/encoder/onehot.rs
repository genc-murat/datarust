use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, SparseMatrix, StrMatrix};
use crate::traits::{default_input_names, CategoricalTransformer, FeatureNames};

/// How to handle unknown categories during `transform`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandleUnknown {
    /// Raise an error on unknown categories (default).
    #[default]
    Error,
    /// Encode unknown categories as all-zeros row (with no `drop`).
    Ignore,
}

/// Whether to drop one category per column to avoid collinearity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DropStrategy {
    /// Keep all categories (default).
    #[default]
    None,
    /// Drop the first category (in sorted order) of each column.
    First,
}

/// Encode categorical features as one-hot numeric columns, mirroring
/// `sklearn.preprocessing.OneHotEncoder`. Categories per column are
/// sorted lexicographically (sklearn default).
///
/// Input is a 2-D [`StrMatrix`]; output is a dense [`Matrix`] by default.
/// Use [`sparse_output`](OneHotEncoder::sparse_output) and
/// [`transform_sparse`](OneHotEncoder::transform_sparse) for CSR output.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OneHotEncoder {
    drop: DropStrategy,
    handle_unknown: HandleUnknown,
    sparse_output: bool,
    categories: Vec<Vec<String>>,
    category_index: Vec<HashMap<String, usize>>,
    n_output_cols: usize,
    fitted: bool,
}

impl OneHotEncoder {
    /// Creates a new one-hot encoder.
    pub fn new() -> Self {
        Self {
            drop: DropStrategy::None,
            handle_unknown: HandleUnknown::Error,
            sparse_output: false,
            categories: vec![],
            category_index: vec![],
            n_output_cols: 0,
            fitted: false,
        }
    }

    /// Sets the category drop strategy.
    pub fn drop(mut self, d: DropStrategy) -> Self {
        self.drop = d;
        self
    }

    /// Sets how unknown categories are handled during transform.
    pub fn handle_unknown(mut self, h: HandleUnknown) -> Self {
        self.handle_unknown = h;
        self
    }

    /// Enable or disable sparse (CSR) output. When `true`, use
    /// [`transform_sparse`](OneHotEncoder::transform_sparse) to obtain a
    /// [`SparseMatrix`]. Default is `false` (dense).
    pub fn sparse_output(mut self, sparse: bool) -> Self {
        self.sparse_output = sparse;
        self
    }

    /// Returns whether sparse output is enabled.
    pub fn is_sparse(&self) -> bool {
        self.sparse_output
    }

    /// Returns the learned categories per column.
    pub fn categories(&self) -> &[Vec<String>] {
        &self.categories
    }

    /// Returns the number of output columns after transform.
    pub fn n_output_cols(&self) -> usize {
        self.n_output_cols
    }

    fn build_categories(col: &[String]) -> Vec<String> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for s in col {
            set.insert(s.clone());
        }
        set.into_iter().collect()
    }

    fn kept_categories(&self, col_idx: usize) -> &[String] {
        let cats = &self.categories[col_idx];
        match self.drop {
            DropStrategy::None => cats,
            DropStrategy::First => &cats[1..],
        }
    }

    /// Validates that the encoder is fitted and the input has the correct
    /// number of columns.
    fn validate_input(&self, x: &StrMatrix) -> Result<()> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("OneHotEncoder".into()));
        }
        if x.ncols() != self.categories.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} categorical columns", self.categories.len()),
                actual: format!("{} columns", x.ncols()),
            });
        }
        Ok(())
    }

    /// Computes the output column offset for each input feature and the total
    /// number of output columns.
    fn compute_offsets(&self) -> (usize, Vec<usize>) {
        let mut offsets = Vec::with_capacity(self.categories.len());
        let mut acc = 0;
        for j in 0..self.categories.len() {
            offsets.push(acc);
            acc += self.kept_categories(j).len();
        }
        (acc, offsets)
    }

    /// Encodes a single value for a single input feature, returning the
    /// output column index if the value should be set to 1.0, or `None` if
    /// it should be skipped (unknown category with Ignore, or dropped).
    fn encode_one(&self, val: &str, j: usize, offsets: &[usize]) -> Result<Option<usize>> {
        let idx_in_full = match self.category_index[j].get(val) {
            Some(&idx) => idx,
            None => match self.handle_unknown {
                HandleUnknown::Error => {
                    return Err(DatarustError::UnknownCategory(format!(
                        "column {} value '{}'",
                        j, val
                    )));
                }
                HandleUnknown::Ignore => return Ok(None),
            },
        };
        let target = match self.drop {
            DropStrategy::None => Some(offsets[j] + idx_in_full),
            DropStrategy::First => {
                if idx_in_full == 0 {
                    None
                } else {
                    Some(offsets[j] + (idx_in_full - 1))
                }
            }
        };
        Ok(target)
    }

    /// Learns the sorted categories per column.
    pub fn fit(&mut self, x: &StrMatrix) -> Result<()> {
        let ncols = x.ncols();
        let mut categories = Vec::with_capacity(ncols);
        let mut category_index = Vec::with_capacity(ncols);
        let mut total = 0;
        for j in 0..ncols {
            let col = x.column(j);
            let cats = Self::build_categories(&col);
            let idx: HashMap<String, usize> = cats
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i))
                .collect();
            let kept = match self.drop {
                DropStrategy::None => cats.len(),
                DropStrategy::First => cats.len().saturating_sub(1),
            };
            total += kept;
            categories.push(cats);
            category_index.push(idx);
        }
        self.categories = categories;
        self.category_index = category_index;
        self.n_output_cols = total;
        self.fitted = true;
        Ok(())
    }

    /// Encodes the input as a dense one-hot matrix.
    pub fn transform(&self, x: &StrMatrix) -> Result<Matrix> {
        self.validate_input(x)?;
        let (n_out, offsets) = self.compute_offsets();
        let nrows = x.nrows();
        let mut out = vec![vec![0.0; n_out]; nrows];

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            out.par_iter_mut()
                .enumerate()
                .try_for_each(|(i, out_row)| {
                    for j in 0..x.ncols() {
                        if let Some(col) = self.encode_one(x.get(i, j), j, &offsets)? {
                            out_row[col] = 1.0;
                        }
                    }
                    Ok::<(), DatarustError>(())
                })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, out_row) in out.iter_mut().enumerate() {
                for j in 0..x.ncols() {
                    if let Some(col) = self.encode_one(x.get(i, j), j, &offsets)? {
                        out_row[col] = 1.0;
                    }
                }
            }
        }

        Matrix::new(out)
    }

    /// Fits the encoder and transforms the input into a dense matrix.
    pub fn fit_transform(&mut self, x: &StrMatrix) -> Result<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Transform using sparse (CSR) output, regardless of the `sparse_output`
    /// flag. Useful when only some callers need sparse data.
    pub fn transform_sparse(&self, x: &StrMatrix) -> Result<SparseMatrix> {
        self.validate_input(x)?;
        let (n_out, offsets) = self.compute_offsets();
        let nrows = x.nrows();

        let mut row_triplets: Vec<Vec<(usize, usize, f64)>> = vec![vec![]; nrows];

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            row_triplets
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, trip)| {
                    for j in 0..x.ncols() {
                        if let Some(col) = self.encode_one(x.get(i, j), j, &offsets)? {
                            trip.push((i, col, 1.0));
                        }
                    }
                    Ok::<(), DatarustError>(())
                })?;
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, trip) in row_triplets.iter_mut().enumerate() {
                for j in 0..x.ncols() {
                    if let Some(col) = self.encode_one(x.get(i, j), j, &offsets)? {
                        trip.push((i, col, 1.0));
                    }
                }
            }
        }

        let mut triplets: Vec<(usize, usize, f64)> =
            Vec::with_capacity(row_triplets.iter().map(|t| t.len()).sum());
        for t in row_triplets {
            triplets.extend(t);
        }
        SparseMatrix::from_triplets(nrows, n_out, &triplets)
    }

    /// Convenience: fit then transform into sparse output.
    pub fn fit_transform_sparse(&mut self, x: &StrMatrix) -> Result<SparseMatrix> {
        self.fit(x)?;
        self.transform_sparse(x)
    }

    /// Convert one-hot encoded dense data back to categorical strings.
    ///
    /// For each row, the active (1.0) column within each feature block is
    /// decoded to its category.  When `drop = First`, an all-zero block
    /// decodes to the dropped (first) category.  All-zero blocks with
    /// `drop = None` decode to an empty string (matching sklearn's `None`).
    pub fn inverse_transform(&self, x: &Matrix) -> Result<StrMatrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("OneHotEncoder".into()));
        }
        let expected_cols = self.n_output_cols;
        if x.ncols() != expected_cols {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} output columns", expected_cols),
                actual: format!("{} columns", x.ncols()),
            });
        }
        let (_n_out, offsets) = self.compute_offsets();
        let n_features = self.categories.len();

        let decode_row = |i: usize| -> Vec<String> {
            let mut row_cats = Vec::with_capacity(n_features);
            for (j, &block_start) in offsets.iter().enumerate() {
                let block_len = self.kept_categories(j).len();
                let block_end = block_start + block_len;
                let mut found: Option<usize> = None;
                for c in block_start..block_end {
                    if x.get(i, c) >= 0.5 {
                        found = Some(c - block_start);
                        break;
                    }
                }
                let cat = match (self.drop, found) {
                    (DropStrategy::None, Some(k)) => self.categories[j][k].clone(),
                    (DropStrategy::None, None) => String::new(),
                    (DropStrategy::First, Some(k)) => self.categories[j][k + 1].clone(),
                    (DropStrategy::First, None) => self.categories[j][0].clone(),
                };
                row_cats.push(cat);
            }
            row_cats
        };

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let out: Vec<Vec<String>> = (0..x.nrows()).into_par_iter().map(decode_row).collect();
            StrMatrix::new(out)
        }

        #[cfg(not(feature = "rayon"))]
        {
            let out: Vec<Vec<String>> = (0..x.nrows()).map(decode_row).collect();
            StrMatrix::new(out)
        }
    }

    /// Inverse transform from sparse (CSR) input.
    pub fn inverse_transform_sparse(&self, x: &SparseMatrix) -> Result<StrMatrix> {
        let dense = x.to_dense()?;
        self.inverse_transform(&dense)
    }

    /// Transform returning sparse if `sparse_output` is set, dense otherwise.
    pub fn transform_auto(&self, x: &StrMatrix) -> Result<OneHotOutput> {
        if self.sparse_output {
            Ok(OneHotOutput::Sparse(self.transform_sparse(x)?))
        } else {
            Ok(OneHotOutput::Dense(self.transform(x)?))
        }
    }
}

/// Output of [`OneHotEncoder::transform_auto`].
#[derive(Debug, Clone, PartialEq)]
pub enum OneHotOutput {
    /// Dense encoded output.
    Dense(Matrix),
    /// Sparse (CSR) encoded output.
    Sparse(SparseMatrix),
}

impl CategoricalTransformer for OneHotEncoder {
    fn name(&self) -> &'static str {
        "OneHotEncoder"
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

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for OneHotEncoder {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.categories.len()),
        };
        let mut out = Vec::new();
        for (j, cats) in self.categories.iter().enumerate() {
            let col_name = names.get(j).cloned().unwrap_or_else(|| format!("x{}", j));
            let kept: Vec<&String> = match self.drop {
                DropStrategy::None => cats.iter().collect(),
                DropStrategy::First => cats.iter().skip(1).collect(),
            };
            for cat in kept {
                out.push(format!("{}_{}", col_name, cat));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_column_basic() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let out = ohe.fit_transform(&s).unwrap();
        // categories sorted: Blue, Green, Red
        assert_eq!(ohe.categories()[0], &["Blue", "Green", "Red"]);
        // Red -> [0,0,1]
        assert_eq!(out.row(0), [0.0, 0.0, 1.0]);
        assert_eq!(out.row(1), [1.0, 0.0, 0.0]);
        assert_eq!(out.row(2), [0.0, 1.0, 0.0]);
        assert_eq!(out.row(3), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn multiple_columns() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let out = ohe.fit_transform(&s).unwrap();
        // col0 cats: a,b  ; col1 cats: x,y  -> 4 output cols
        assert_eq!(out.ncols(), 4);
        // row0: a,x -> a=1,x=1 -> [1,0,1,0]
        assert_eq!(out.row(0), [1.0, 0.0, 1.0, 0.0]);
        // row1: b,y -> [0,1,0,1]
        assert_eq!(out.row(1), [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn drop_first() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green"]).unwrap();
        let mut ohe = OneHotEncoder::new().drop(DropStrategy::First);
        let out = ohe.fit_transform(&s).unwrap();
        // kept: Green, Red (Blue dropped)
        assert_eq!(out.ncols(), 2);
        // Red -> [0,1]
        assert_eq!(out.row(0), [0.0, 1.0]);
        // Blue -> [0,0]
        assert_eq!(out.row(1), [0.0, 0.0]);
        // Green -> [1,0]
        assert_eq!(out.row(2), [1.0, 0.0]);
    }

    #[test]
    fn handle_unknown_error() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new(); // default error
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "c"]).unwrap();
        assert!(matches!(
            ohe.transform(&s2),
            Err(DatarustError::UnknownCategory(_))
        ));
    }

    #[test]
    fn handle_unknown_ignore_zeros() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore);
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let out = ohe.transform(&s2).unwrap();
        // a -> [1,0]; z -> [0,0]
        assert_eq!(out.row(0), [1.0, 0.0]);
        assert_eq!(out.row(1), [0.0, 0.0]);
    }

    #[test]
    fn transform_new_data_uses_fitted_categories() {
        let s = StrMatrix::from_column(["a", "b", "c"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["b", "a"]).unwrap();
        let out = ohe.transform(&s2).unwrap();
        assert_eq!(out.row(0), [0.0, 1.0, 0.0]);
        assert_eq!(out.row(1), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn categories_sorted() {
        let s = StrMatrix::from_column(["zebra", "apple", "mango"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        assert_eq!(ohe.categories()[0], &["apple", "mango", "zebra"]);
    }

    #[test]
    fn transform_before_fit_errors() {
        let ohe = OneHotEncoder::new();
        let s = StrMatrix::from_column(["a"]).unwrap();
        assert!(matches!(
            ohe.transform(&s),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn column_count_mismatch() {
        let s = StrMatrix::from_strings(vec![vec!["a", "b"], vec!["c", "d"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a"]).unwrap(); // 1 col vs 2 expected
        assert!(ohe.transform(&s2).is_err());
    }

    #[test]
    fn duplicate_rows() {
        let s = StrMatrix::from_column(["a", "a", "a"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let out = ohe.fit_transform(&s).unwrap();
        for i in 0..3 {
            assert_eq!(out.row(i), [1.0]);
        }
    }

    #[test]
    fn feature_names_single_col() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        // default input name x0, categories sorted: Blue, Green, Red
        let names = ohe.feature_names_out(None);
        assert_eq!(names, vec!["x0_Blue", "x0_Green", "x0_Red"]);
    }

    #[test]
    fn feature_names_multi_col_custom() {
        let s = StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        let names = ohe.feature_names_out(Some(&["c1".to_string(), "c2".to_string()]));
        // c1 cats: a,b ; c2 cats: x,y
        assert_eq!(names, vec!["c1_a", "c1_b", "c2_x", "c2_y"]);
    }

    #[test]
    fn feature_names_with_drop() {
        let s = StrMatrix::from_column(["a", "b", "c"]).unwrap();
        let mut ohe = OneHotEncoder::new().drop(DropStrategy::First);
        ohe.fit(&s).unwrap();
        let names = ohe.feature_names_out(Some(&["city".to_string()]));
        // 'a' dropped -> city_b, city_c
        assert_eq!(names, vec!["city_b", "city_c"]);
    }

    #[test]
    fn sparse_single_column() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let sp = ohe.fit_transform_sparse(&s).unwrap();
        assert_eq!(sp.nrows(), 4);
        assert_eq!(sp.ncols(), 3);
        assert_eq!(sp.nnz(), 4);
        // cats sorted: Blue, Green, Red
        // Red -> col 2
        assert_eq!(sp.get(0, 2), 1.0);
        // Blue -> col 0
        assert_eq!(sp.get(1, 0), 1.0);
        // Green -> col 1
        assert_eq!(sp.get(2, 1), 1.0);
        // zeros
        assert_eq!(sp.get(0, 0), 0.0);
        assert_eq!(sp.get(1, 2), 0.0);
    }

    #[test]
    fn sparse_multiple_columns() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let sp = ohe.fit_transform_sparse(&s).unwrap();
        assert_eq!(sp.ncols(), 4);
        assert_eq!(sp.nnz(), 6); // 3 rows * 2 active columns
                                 // row0: a,x -> cols 0,2
        assert_eq!(sp.get(0, 0), 1.0);
        assert_eq!(sp.get(0, 2), 1.0);
        // row1: b,y -> cols 1,3
        assert_eq!(sp.get(1, 1), 1.0);
        assert_eq!(sp.get(1, 3), 1.0);
    }

    #[test]
    fn sparse_with_drop_first() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green"]).unwrap();
        let mut ohe = OneHotEncoder::new().drop(DropStrategy::First);
        let sp = ohe.fit_transform_sparse(&s).unwrap();
        assert_eq!(sp.ncols(), 2);
        // kept: Green(0), Red(1); Blue dropped
        assert_eq!(sp.get(0, 1), 1.0); // Red -> col 1
        assert_eq!(sp.get(1, 0), 0.0); // Blue -> all zeros
        assert_eq!(sp.get(1, 1), 0.0);
        assert_eq!(sp.get(2, 0), 1.0); // Green -> col 0
    }

    #[test]
    fn sparse_matches_dense() {
        let s = StrMatrix::from_strings(vec![
            vec!["a", "x"],
            vec!["b", "y"],
            vec!["a", "y"],
            vec!["c", "x"],
        ])
        .unwrap();
        let mut ohe = OneHotEncoder::new();
        let dense = ohe.fit_transform(&s).unwrap();
        let mut ohe2 = OneHotEncoder::new();
        let sp = ohe2.fit_transform_sparse(&s).unwrap();
        for i in 0..s.nrows() {
            for j in 0..dense.ncols() {
                assert_eq!(sp.get(i, j), dense.get(i, j));
            }
        }
    }

    #[test]
    fn sparse_handle_unknown_ignore() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore);
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let sp = ohe.transform_sparse(&s2).unwrap();
        // a -> [1,0]; z -> [0,0]
        assert_eq!(sp.get(0, 0), 1.0);
        assert_eq!(sp.get(1, 0), 0.0);
        assert_eq!(sp.get(1, 1), 0.0);
        assert_eq!(sp.nnz(), 1);
    }

    #[test]
    fn sparse_transform_before_fit_errors() {
        let ohe = OneHotEncoder::new();
        let s = StrMatrix::from_column(["a"]).unwrap();
        assert!(matches!(
            ohe.transform_sparse(&s),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_auto_dense_by_default() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        match ohe.transform_auto(&s).unwrap() {
            OneHotOutput::Dense(m) => assert_eq!(m.ncols(), 2),
            OneHotOutput::Sparse(_) => panic!("expected dense"),
        }
    }

    #[test]
    fn transform_auto_sparse_when_flagged() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new().sparse_output(true);
        ohe.fit(&s).unwrap();
        match ohe.transform_auto(&s).unwrap() {
            OneHotOutput::Sparse(sp) => {
                assert_eq!(sp.ncols(), 2);
                assert_eq!(sp.nnz(), 2);
            }
            OneHotOutput::Dense(_) => panic!("expected sparse"),
        }
    }

    #[test]
    fn inverse_transform_single_column() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let encoded = ohe.fit_transform(&s).unwrap();
        let decoded = ohe.inverse_transform(&encoded).unwrap();
        for i in 0..s.nrows() {
            assert_eq!(decoded.get(i, 0), s.get(i, 0));
        }
    }

    #[test]
    fn inverse_transform_multiple_columns() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let encoded = ohe.fit_transform(&s).unwrap();
        let decoded = ohe.inverse_transform(&encoded).unwrap();
        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert_eq!(decoded.get(i, j), s.get(i, j));
            }
        }
    }

    #[test]
    fn inverse_transform_with_drop_first() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green"]).unwrap();
        let mut ohe = OneHotEncoder::new().drop(DropStrategy::First);
        let encoded = ohe.fit_transform(&s).unwrap();
        let decoded = ohe.inverse_transform(&encoded).unwrap();
        // Blue was dropped -> encoded as all-zeros -> decoded back to Blue
        for i in 0..s.nrows() {
            assert_eq!(decoded.get(i, 0), s.get(i, 0));
        }
    }

    #[test]
    fn inverse_transform_with_drop_multi_col() {
        let s =
            StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["c", "z"]]).unwrap();
        let mut ohe = OneHotEncoder::new().drop(DropStrategy::First);
        let encoded = ohe.fit_transform(&s).unwrap();
        let decoded = ohe.inverse_transform(&encoded).unwrap();
        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert_eq!(decoded.get(i, j), s.get(i, j));
            }
        }
    }

    #[test]
    fn inverse_transform_sparse_input() {
        let s = StrMatrix::from_column(["Red", "Blue", "Green", "Red"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        let sp = ohe.fit_transform_sparse(&s).unwrap();
        let decoded = ohe.inverse_transform_sparse(&sp).unwrap();
        for i in 0..s.nrows() {
            assert_eq!(decoded.get(i, 0), s.get(i, 0));
        }
    }

    #[test]
    fn inverse_transform_handle_unknown_ignore_zeros() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore);
        ohe.fit(&s).unwrap();
        let s2 = StrMatrix::from_column(["a", "z"]).unwrap();
        let encoded = ohe.transform(&s2).unwrap();
        // row 1 is all-zeros (unknown 'z')
        let decoded = ohe.inverse_transform(&encoded).unwrap();
        assert_eq!(decoded.get(0, 0), "a");
        // unknown -> empty string
        assert_eq!(decoded.get(1, 0), "");
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let ohe = OneHotEncoder::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(ohe.inverse_transform(&x).is_err());
    }

    #[test]
    fn inverse_transform_shape_mismatch() {
        let s = StrMatrix::from_column(["a", "b"]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();
        assert!(ohe.inverse_transform(&bad).is_err());
    }

    #[test]
    fn feature_names_short_input_pads_with_synthetic() {
        let s = StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"]]).unwrap();
        let mut ohe = OneHotEncoder::new();
        ohe.fit(&s).unwrap();
        // 2 columns but only 1 name provided
        let names = ohe.feature_names_out(Some(&["city".into()]));
        assert_eq!(names, vec!["city_a", "city_b", "x1_x", "x1_y"]);
    }
}
