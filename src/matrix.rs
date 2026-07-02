//! Dense and sparse matrix containers used throughout the crate.

use crate::error::{DatarustError, Result};

/// Row-major dense matrix of `f64` backed by `Vec<Vec<f64>>`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Matrix {
    pub(crate) data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a matrix from a nested vector, validating a rectangular shape.
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(DatarustError::EmptyInput("matrix has no rows".into()));
        }
        let cols = data[0].len();
        if cols == 0 {
            return Err(DatarustError::EmptyInput("matrix has no columns".into()));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(DatarustError::ShapeMismatch {
                    expected: format!("{} columns", cols),
                    actual: format!("{} columns at row {}", row.len(), i),
                });
            }
        }
        Ok(Self { data })
    }

    /// Creates a matrix from a nested vector of rows.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        Self::new(rows)
    }

    /// Creates a matrix from row-major flat data of the given shape.
    pub fn from_flat(rows: usize, cols: usize, flat: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        if flat.len() != rows * cols {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} elements", rows * cols),
                actual: format!("{} elements", flat.len()),
            });
        }
        let mut data = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * cols;
            data.push(flat[start..start + cols].to_vec());
        }
        Ok(Self { data })
    }

    /// Creates a matrix filled with zeros of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        Ok(Self {
            data: vec![vec![0.0; cols]; rows],
        })
    }

    /// Creates an `n` by `n` identity matrix.
    pub fn identity(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        let mut data = vec![vec![0.0; n]; n];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Ok(Self { data })
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.data[0].len()
    }

    /// Returns the element at row `i`, column `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    /// Sets the element at row `i`, column `j`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }

    /// Returns the row at index `i` as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    /// Returns column `j` as a new vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        self.data.iter().map(|r| r[j]).collect()
    }

    /// Iterates over the rows as slices.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[f64]> {
        self.data.iter().map(|v| v.as_slice())
    }

    /// Borrows the underlying vector of rows.
    pub fn rows_ref(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// Consumes the matrix and returns the underlying rows.
    pub fn into_rows(self) -> Vec<Vec<f64>> {
        self.data
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Matrix {
        let rows = self.nrows();
        let cols = self.ncols();
        let mut out = vec![vec![0.0; rows]; cols];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                out[j][i] = v;
            }
        }
        Matrix { data: out }
    }

    /// Multiplies two matrices and returns the product.
    #[allow(clippy::needless_range_loop)]
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix> {
        if self.ncols() != other.nrows() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("second operand with {} rows", self.ncols()),
                actual: format!("{} rows", other.nrows()),
            });
        }
        let m = self.nrows();
        let k = self.ncols();
        let n = other.ncols();
        let mut out = vec![vec![0.0; n]; m];
        // Indexed inner loops retained for cache-friendly row-major access.
        for (i, out_row) in out.iter_mut().enumerate() {
            let self_row = &self.data[i];
            for l in 0..k {
                let a = self_row[l];
                if a == 0.0 {
                    continue;
                }
                let other_row = &other.data[l];
                for j in 0..n {
                    out_row[j] += a * other_row[j];
                }
            }
        }
        Ok(Matrix { data: out })
    }

    /// Returns the mean of each column.
    pub fn column_mean(&self) -> Vec<f64> {
        crate::stats::column_mean(&self.data)
    }

    /// Creates a matrix from a vector of columns.
    pub fn from_columns(cols: Vec<Vec<f64>>) -> Result<Self> {
        if cols.is_empty() || cols[0].is_empty() {
            return Err(DatarustError::EmptyInput("no columns".into()));
        }
        let rows = cols[0].len();
        for c in &cols {
            if c.len() != rows {
                return Err(DatarustError::ShapeMismatch {
                    expected: format!("{} rows", rows),
                    actual: format!("{} rows", c.len()),
                });
            }
        }
        let mut data = vec![vec![0.0; cols.len()]; rows];
        for (j, col) in cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                data[i][j] = v;
            }
        }
        Ok(Self { data })
    }

    /// Select a subset of columns by index (0-based).
    ///
    /// ```rust
    /// use datarust::Matrix;
    ///
    /// let m = Matrix::new(vec![
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    /// ])?;
    /// let sub = m.select_columns(&[0, 2])?;
    /// assert_eq!(sub.ncols(), 2);
    /// assert_eq!(sub.get(0, 0), 1.0);
    /// assert_eq!(sub.get(0, 1), 3.0);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    pub fn select_columns(&self, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            return Err(DatarustError::EmptyInput("no columns selected".into()));
        }
        let ncols = self.ncols();
        for &c in indices {
            if c >= ncols {
                return Err(DatarustError::InvalidInput(format!(
                    "column index {} out of range (ncols {})",
                    c, ncols
                )));
            }
        }
        let out: Vec<Vec<f64>> = self
            .data
            .iter()
            .map(|row| indices.iter().map(|&c| row[c]).collect())
            .collect();
        Ok(Self { data: out })
    }

    /// Select a subset of rows by index (0-based).
    ///
    /// ```rust
    /// use datarust::Matrix;
    ///
    /// let m = Matrix::new(vec![
    ///     vec![10.0],
    ///     vec![20.0],
    ///     vec![30.0],
    /// ])?;
    /// let sub = m.select_rows(&[0, 2])?;
    /// assert_eq!(sub.nrows(), 2);
    /// assert_eq!(sub.get(0, 0), 10.0);
    /// assert_eq!(sub.get(1, 0), 30.0);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    pub fn select_rows(&self, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            return Err(DatarustError::EmptyInput("no rows selected".into()));
        }
        let nrows = self.nrows();
        for &r in indices {
            if r >= nrows {
                return Err(DatarustError::InvalidInput(format!(
                    "row index {} out of range (nrows {})",
                    r, nrows
                )));
            }
        }
        let out: Vec<Vec<f64>> = indices.iter().map(|&r| self.data[r].clone()).collect();
        Ok(Self { data: out })
    }
}

/// Row-major matrix of strings used by the categorical encoders.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrMatrix {
    pub(crate) data: Vec<Vec<String>>,
}

impl StrMatrix {
    /// Creates a string matrix from a nested vector, validating a rectangular shape.
    pub fn new(data: Vec<Vec<String>>) -> Result<Self> {
        if data.is_empty() {
            return Err(DatarustError::EmptyInput("matrix has no rows".into()));
        }
        let cols = data[0].len();
        if cols == 0 {
            return Err(DatarustError::EmptyInput("matrix has no columns".into()));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(DatarustError::ShapeMismatch {
                    expected: format!("{} columns", cols),
                    actual: format!("{} columns at row {}", row.len(), i),
                });
            }
        }
        Ok(Self { data })
    }

    /// Creates a single-column string matrix from an iterator of values.
    pub fn from_column<I, S>(col: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let data: Vec<Vec<String>> = col.into_iter().map(|s| vec![s.into()]).collect::<Vec<_>>();
        if data.is_empty() {
            return Err(DatarustError::EmptyInput("column has no rows".into()));
        }
        Self::new(data)
    }

    /// Creates a string matrix from an iterator of rows.
    pub fn from_strings<I, S>(rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<S>>,
        S: Into<String>,
    {
        let data: Vec<Vec<String>> = rows
            .into_iter()
            .map(|r| r.into_iter().map(|s| s.into()).collect())
            .collect();
        Self::new(data)
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.data[0].len()
    }

    /// Returns the string at row `i`, column `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &str {
        &self.data[i][j]
    }

    /// Returns column `j` as a new vector of strings.
    pub fn column(&self, j: usize) -> Vec<String> {
        self.data.iter().map(|r| r[j].clone()).collect()
    }

    /// Returns the row at index `i` as a slice.
    pub fn row(&self, i: usize) -> &[String] {
        &self.data[i]
    }
}

impl TryFrom<Vec<Vec<f64>>> for Matrix {
    type Error = DatarustError;

    /// Fallibly construct a [`Matrix`] from a nested vector.
    ///
    /// Returns an error if the rows are empty or jagged. This replaces the
    /// previous panicking `From` impl to keep validation consistent with
    /// [`Matrix::new`].
    fn try_from(data: Vec<Vec<f64>>) -> Result<Self> {
        Matrix::new(data)
    }
}

/// Compressed Sparse Row (CSR) matrix for memory-efficient storage of
/// mostly-zero 2-D data, mirroring scipy.sparse `csr_matrix`.
///
/// Three arrays define the non-zero entries:
/// - `indptr` (length `nrows + 1`): row `i` occupies
///   `indptr[i]..indptr[i+1]` in `indices`/`data`.
/// - `indices`: column index of each non-zero.
/// - `data`: value of each non-zero.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseMatrix {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl SparseMatrix {
    /// Build a CSR matrix from raw CSR arrays.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self> {
        if nrows == 0 || ncols == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        if indptr.len() != nrows + 1 {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} indptr entries", nrows + 1),
                actual: format!("{} indptr entries", indptr.len()),
            });
        }
        if indices.len() != data.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} indices", data.len()),
                actual: format!("{} indices", indices.len()),
            });
        }
        let nnz = data.len();
        if indptr[0] != 0 || indptr[nrows] != nnz {
            return Err(DatarustError::InvalidInput(
                "indptr must start at 0 and end at nnz".into(),
            ));
        }
        for &c in &indices {
            if c >= ncols {
                return Err(DatarustError::InvalidInput(format!(
                    "column index {} out of range (ncols {})",
                    c, ncols
                )));
            }
        }
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        })
    }

    /// Build from `(row, col, value)` triplets. Zero-valued triplets are
    /// dropped automatically. Within a row, entries are sorted by column.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self> {
        if nrows == 0 || ncols == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        let mut per_row: Vec<Vec<(usize, f64)>> = vec![vec![]; nrows];
        for &(r, c, v) in triplets {
            if r >= nrows {
                return Err(DatarustError::InvalidInput(format!(
                    "row {} out of range (nrows {})",
                    r, nrows
                )));
            }
            if c >= ncols {
                return Err(DatarustError::InvalidInput(format!(
                    "col {} out of range (ncols {})",
                    c, ncols
                )));
            }
            if v != 0.0 {
                per_row[r].push((c, v));
            }
        }
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for row_entries in &mut per_row {
            row_entries.sort_by_key(|(c, _)| *c);
            for &(c, v) in row_entries.iter() {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len());
        }
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        })
    }

    /// Create an all-zeros sparse matrix with the given shape.
    pub fn zeros(nrows: usize, ncols: usize) -> Result<Self> {
        if nrows == 0 || ncols == 0 {
            return Err(DatarustError::EmptyInput("zero dimension".into()));
        }
        Ok(Self {
            nrows,
            ncols,
            indptr: vec![0; nrows + 1],
            indices: vec![],
            data: vec![],
        })
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Density (fraction of non-zero entries).
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Get element at `(i, j)`. Returns 0.0 if not stored.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        // Binary search within the row's column indices.
        let slice = &self.indices[start..end];
        match slice.binary_search(&j) {
            Ok(local) => self.data[start + local],
            Err(_) => 0.0,
        }
    }

    /// Iterate over `(col, value)` non-zero entries in row `i`.
    pub fn row_nz(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        self.indices[start..end]
            .iter()
            .zip(self.data[start..end].iter())
            .map(|(&c, &v)| (c, v))
    }

    /// Convert to a dense [`Matrix`].
    pub fn to_dense(&self) -> Result<Matrix> {
        let mut rows = vec![vec![0.0; self.ncols]; self.nrows];
        for (i, row) in rows.iter_mut().enumerate() {
            for (c, v) in self.row_nz(i) {
                row[c] = v;
            }
        }
        Matrix::new(rows)
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn approx_eq_matrices(a: &Matrix, b: &Matrix, tol: f64) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (a.get(i, j) - b.get(i, j)).abs() > tol {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn approx_eq_vecs(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
}

#[cfg(test)]
#[macro_use]
mod assert_macros {
    #[allow(unused_macros)]
    macro_rules! assert_mat_eq {
        ($a:expr, $b:expr, $tol:expr) => {{
            assert!(
                $crate::matrix::approx_eq_matrices(&$a, &$b, $tol),
                "matrices not equal within tolerance {}\n left: {:?}\nright: {:?}",
                $tol,
                $a.rows_ref(),
                $b.rows_ref()
            );
        }};
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
    }

    #[test]
    fn new_jagged_rejected() {
        let err = Matrix::new(vec![vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn new_empty_rejected() {
        assert!(Matrix::new(vec![]).is_err());
        assert!(Matrix::new(vec![vec![]]).is_err());
    }

    #[test]
    fn from_flat() {
        let m = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn from_flat_bad_count() {
        let err = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn zeros_and_identity() {
        let z = Matrix::zeros(2, 3).unwrap();
        assert_eq!(z.get(1, 2), 0.0);
        let id = Matrix::identity(3).unwrap();
        assert_eq!(id.get(0, 0), 1.0);
        assert_eq!(id.get(0, 1), 0.0);
        assert_eq!(id.get(1, 1), 1.0);
    }

    #[test]
    fn transpose() {
        let m = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let t = m.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(1, 1), 5.0);
    }

    #[test]
    fn matmul() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        let c = a.matmul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn matmul_shape_mismatch() {
        let a = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        let b = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn from_columns() {
        let m = Matrix::from_columns(vec![vec![1.0, 2.0], vec![10.0, 20.0]]).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 20.0);
    }

    #[test]
    fn strmatrix_from_column() {
        let s = StrMatrix::from_column(["a", "b", "a"]).unwrap();
        assert_eq!(s.nrows(), 3);
        assert_eq!(s.ncols(), 1);
        assert_eq!(s.get(2, 0), "a");
    }

    #[test]
    fn strmatrix_from_strings() {
        let s = StrMatrix::from_strings(vec![vec!["x", "y"], vec!["x", "z"]]).unwrap();
        assert_eq!(s.ncols(), 2);
        assert_eq!(s.get(1, 1), "z");
    }

    #[test]
    fn sparse_from_triplets_basic() {
        let sp = SparseMatrix::from_triplets(
            3,
            4,
            &[(0, 0, 1.0), (1, 2, 3.0), (2, 3, 5.0), (0, 3, 7.0)],
        )
        .unwrap();
        assert_eq!(sp.nrows(), 3);
        assert_eq!(sp.ncols(), 4);
        assert_eq!(sp.nnz(), 4);
        assert_eq!(sp.get(0, 0), 1.0);
        assert_eq!(sp.get(1, 2), 3.0);
        assert_eq!(sp.get(0, 3), 7.0);
        assert_eq!(sp.get(1, 0), 0.0);
    }

    #[test]
    fn sparse_zero_triplets_dropped() {
        let sp =
            SparseMatrix::from_triplets(2, 2, &[(0, 0, 0.0), (0, 1, 5.0), (1, 0, 0.0)]).unwrap();
        assert_eq!(sp.nnz(), 1);
        assert_eq!(sp.get(0, 1), 5.0);
    }

    #[test]
    fn sparse_to_dense() {
        let sp = SparseMatrix::from_triplets(2, 3, &[(0, 1, 2.0), (1, 0, 4.0)]).unwrap();
        let dense = sp.to_dense().unwrap();
        assert_eq!(dense.row(0), [0.0, 2.0, 0.0]);
        assert_eq!(dense.row(1), [4.0, 0.0, 0.0]);
    }

    #[test]
    fn sparse_zeros() {
        let sp = SparseMatrix::zeros(2, 3).unwrap();
        assert_eq!(sp.nnz(), 0);
        assert_eq!(sp.density(), 0.0);
        assert_eq!(sp.get(0, 1), 0.0);
    }

    #[test]
    fn sparse_density() {
        let sp = SparseMatrix::from_triplets(2, 4, &[(0, 0, 1.0), (1, 3, 1.0)]).unwrap();
        assert!((sp.density() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn sparse_row_nz() {
        let sp =
            SparseMatrix::from_triplets(2, 3, &[(0, 1, 2.0), (0, 2, 9.0), (1, 0, 4.0)]).unwrap();
        let row0: Vec<(usize, f64)> = sp.row_nz(0).collect();
        assert_eq!(row0, vec![(1, 2.0), (2, 9.0)]);
        let row1: Vec<(usize, f64)> = sp.row_nz(1).collect();
        assert_eq!(row1, vec![(0, 4.0)]);
    }

    #[test]
    fn sparse_bad_indptr_rejected() {
        let err = SparseMatrix::new(2, 2, vec![0, 0, 5], vec![0], vec![1.0]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn sparse_col_out_of_range_rejected() {
        let err = SparseMatrix::from_triplets(2, 2, &[(0, 5, 1.0)]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn select_columns_basic() {
        let m = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let sub = m.select_columns(&[0, 2]).unwrap();
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.get(0, 0), 1.0);
        assert_eq!(sub.get(0, 1), 3.0);
        assert_eq!(sub.get(1, 0), 4.0);
        assert_eq!(sub.get(1, 1), 6.0);
    }

    #[test]
    fn select_columns_out_of_range() {
        let m = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
        assert!(m.select_columns(&[0, 5]).is_err());
    }

    #[test]
    fn select_columns_empty() {
        let m = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(m.select_columns(&[]).is_err());
    }

    #[test]
    fn select_rows_basic() {
        let m = Matrix::new(vec![vec![10.0], vec![20.0], vec![30.0]]).unwrap();
        let sub = m.select_rows(&[0, 2]).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.get(0, 0), 10.0);
        assert_eq!(sub.get(1, 0), 30.0);
    }

    #[test]
    fn select_rows_out_of_range() {
        let m = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(m.select_rows(&[0, 5]).is_err());
    }

    #[test]
    fn select_rows_empty() {
        let m = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(m.select_rows(&[]).is_err());
    }

    #[test]
    fn select_columns_reordered() {
        let m = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let sub = m.select_columns(&[2, 0]).unwrap();
        assert_eq!(sub.get(0, 0), 3.0);
        assert_eq!(sub.get(0, 1), 1.0);
    }

    #[test]
    fn select_rows_duplicates() {
        let m = Matrix::new(vec![vec![10.0], vec![20.0]]).unwrap();
        let sub = m.select_rows(&[0, 0, 1]).unwrap();
        assert_eq!(sub.nrows(), 3);
        assert_eq!(sub.get(0, 0), 10.0);
        assert_eq!(sub.get(1, 0), 10.0);
        assert_eq!(sub.get(2, 0), 20.0);
    }
}
