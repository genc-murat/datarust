//! Whole-dataset profile: shape, memory, duplicate rows, and per-column profiles.

use datarust::{stats, Matrix, StrMatrix};

use super::column::{ColumnProfile, FiveNumber, PrecomputedStats};
use crate::error::{ProfileError, Result};
use crate::infer;
use std::collections::HashSet;

/// Borrows column `j` of a [`StrMatrix`] as string slices, avoiding the
/// per-cell `String` clone of `StrMatrix::column`. Kept local so the published
/// package only relies on the stable `datarust` 0.6 API (`get`/`nrows`).
fn borrow_column(m: &StrMatrix, j: usize) -> Vec<&str> {
    (0..m.nrows()).map(|i| m.get(i, j)).collect()
}

/// Rough memory estimate for one column, in bytes, based on cell count and type.
fn column_bytes(column_type: crate::types::ColumnType, count: usize) -> usize {
    match column_type {
        // f64 per cell.
        crate::types::ColumnType::Numeric => count * 8,
        // 24-byte String header + payload estimate of ~10 bytes per cell.
        crate::types::ColumnType::Categorical => count * 24,
    }
}

use super::relationships::Relationships;

/// A complete profile of a dataset.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DatasetProfile {
    /// Number of rows in the source data.
    pub n_rows: usize,
    /// Number of columns profiled.
    pub n_columns: usize,
    /// Estimated in-memory footprint of the profiled cells, in bytes.
    pub memory_bytes: usize,
    /// Number of fully duplicated rows (exact equality across all columns).
    pub duplicate_rows: usize,
    /// Fraction of rows that are exact duplicates, in `[0.0, 1.0]`.
    pub duplicate_fraction: f64,
    /// Optional designated target column name for leakage hints.
    pub target_column: Option<String>,
    /// One [`ColumnProfile`] per column, in input order.
    pub columns: Vec<ColumnProfile>,
    /// Pairwise inter-column relationships (Pearson, Cramér's V, point-biserial).
    pub relationships: Option<Relationships>,
}

impl DatasetProfile {
    /// Designated target column setter for leakage detection.
    pub fn with_target(mut self, target: &str) -> Self {
        self.target_column = Some(target.to_string());
        self
    }

    /// Profiles a numeric [`Matrix`].
    ///
    /// Column names default to `x0..x{n-1}` when `names` is `None` or shorter
    /// than the column count.
    ///
    /// The mean, variance, and five-number summary of every column are computed
    /// in bulk over the flat row-major buffer (`Matrix::as_slice`) via
    /// `datarust::stats`'s fused Welford and quantile helpers — one fused pass
    /// for mean/variance and one sort per column for the quantiles, instead of
    /// the per-column `Vec` gathering the older path used. The remaining
    /// distributional statistics (skewness, kurtosis, histogram, outliers)
    /// still gather each column once, since `datarust` does not provide flat
    /// versions of those.
    pub fn from_matrix(m: &Matrix, names: Option<&[String]>) -> Result<Self> {
        let (rows, cols) = (m.nrows(), m.ncols());
        if rows == 0 || cols == 0 {
            return Err(ProfileError::EmptyInput(format!(
                "matrix is {}x{}",
                rows, cols
            )));
        }

        let default_names: Vec<String> = (0..cols).map(|j| format!("x{j}")).collect();
        let resolved: Vec<String> = match names {
            Some(ns) if ns.len() == cols => ns.to_vec(),
            _ => default_names,
        };

        // --- _flat fast path: mean/variance + quantiles in bulk -------------
        let data = m.as_slice();
        let (means, vars) = stats::column_mean_var_flat(data, rows, cols, 1);
        // Shape: [qs.len()][cols] → rows of [min, q1, median, q3, max] per col.
        let qs = &[0.0, 0.25, 0.5, 0.75, 1.0];
        let quantiles = stats::column_quantiles_many_flat(data, rows, cols, qs)?;

        let mut columns = Vec::with_capacity(cols);
        let mut memory_bytes = 0usize;
        let mut numeric_data: Vec<(String, Vec<f64>)> = Vec::with_capacity(cols);

        for (j, name) in resolved.iter().enumerate().take(cols) {
            let col = m.col(j);
            let col_has_nan = (0..rows).any(|i| !data[i * cols + j].is_finite());
            let precomputed = if col_has_nan {
                None
            } else {
                (|| {
                    let mean = *means.get(j)?;
                    let std = (*vars.get(j)?).sqrt();
                    let qrow = quantiles.get(0..qs.len())?;
                    let five = FiveNumber {
                        min: *qrow.first()?.get(j).unwrap_or(&f64::NAN),
                        q1: *qrow.get(1)?.get(j).unwrap_or(&f64::NAN),
                        median: *qrow.get(2)?.get(j).unwrap_or(&f64::NAN),
                        q3: *qrow.get(3)?.get(j).unwrap_or(&f64::NAN),
                        max: *qrow.get(4)?.get(j).unwrap_or(&f64::NAN),
                    };
                    Some(PrecomputedStats { mean, std, five })
                })()
            };
            columns.push(ColumnProfile::from_numeric_with_stats(
                name.clone(),
                &col,
                precomputed,
            ));
            memory_bytes += column_bytes(crate::types::ColumnType::Numeric, rows);
            numeric_data.push((name.clone(), col));
        }

        let num_refs: Vec<(&str, &[f64])> = numeric_data
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let empty_cat: Vec<(&str, &[&str])> = Vec::new();
        let relationships = Relationships::compute(&num_refs, &empty_cat);

        let duplicate_rows = count_duplicate_numeric(m);
        Ok(Self::finish(
            rows,
            cols,
            memory_bytes,
            duplicate_rows,
            columns,
            relationships,
        ))
    }

    /// Profiles a string [`StrMatrix`], inferring each column's type.
    pub fn from_str_matrix(m: &StrMatrix, names: Option<&[String]>) -> Result<Self> {
        let (rows, cols) = (m.nrows(), m.ncols());
        if rows == 0 || cols == 0 {
            return Err(ProfileError::EmptyInput(format!(
                "str-matrix is {}x{}",
                rows, cols
            )));
        }

        let default_names: Vec<String> = (0..cols).map(|j| format!("x{j}")).collect();
        let resolved: Vec<String> = match names {
            Some(ns) if ns.len() == cols => ns.to_vec(),
            _ => default_names,
        };

        let mut columns = Vec::with_capacity(cols);
        let mut memory_bytes = 0usize;
        let mut numeric_data: Vec<(String, Vec<f64>)> = Vec::new();
        let mut categorical_data: Vec<(String, Vec<&str>)> = Vec::new();

        for (j, name) in resolved.iter().enumerate().take(cols) {
            let cells = borrow_column(m, j);
            let profile = ColumnProfile::from_strings(name.clone(), &cells);
            memory_bytes += column_bytes(profile.column_type, rows);
            match profile.column_type {
                crate::types::ColumnType::Numeric => {
                    let parsed = infer::parse_numeric_column(&cells);
                    numeric_data.push((name.clone(), parsed));
                }
                crate::types::ColumnType::Categorical => {
                    categorical_data.push((name.clone(), cells));
                }
            }
            columns.push(profile);
        }

        let num_refs: Vec<(&str, &[f64])> = numeric_data
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let cat_refs: Vec<(&str, &[&str])> = categorical_data
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let relationships = Relationships::compute(&num_refs, &cat_refs);

        let duplicate_rows = count_duplicate_str(m);
        Ok(Self::finish(
            rows,
            cols,
            memory_bytes,
            duplicate_rows,
            columns,
            relationships,
        ))
    }

    /// Builds a profile over a mixed table: a numeric [`Matrix`] and a string
    /// [`StrMatrix`] side-by-side, sharing the same row count.
    ///
    /// `names` must contain one entry per column across both blocks (numerics
    /// first, then categoricals).
    pub fn from_table(
        numeric: Option<&Matrix>,
        categorical: Option<&StrMatrix>,
        names: &[String],
    ) -> Result<Self> {
        let n_numeric = numeric.map(|m| m.ncols()).unwrap_or(0);
        let n_categorical = categorical.map(|m| m.ncols()).unwrap_or(0);
        let cols = n_numeric + n_categorical;
        if cols == 0 {
            return Err(ProfileError::EmptyInput("table has no columns".into()));
        }
        if names.len() != cols {
            return Err(ProfileError::InvalidInput(format!(
                "expected {} column names, got {}",
                cols,
                names.len()
            )));
        }
        let rows = numeric
            .map(|m| m.nrows())
            .or_else(|| categorical.map(|m| m.nrows()));
        let rows = rows.ok_or_else(|| ProfileError::EmptyInput("table has no rows".into()))?;
        if let Some(nm) = numeric {
            if nm.nrows() != rows {
                return Err(ProfileError::InvalidInput(format!(
                    "numeric block has {} rows, expected {}",
                    nm.nrows(),
                    rows
                )));
            }
        }
        if let Some(cm) = categorical {
            if cm.nrows() != rows {
                return Err(ProfileError::InvalidInput(format!(
                    "categorical block has {} rows, expected {}",
                    cm.nrows(),
                    rows
                )));
            }
        }

        let mut columns = Vec::with_capacity(cols);
        let mut memory_bytes = 0usize;
        let mut numeric_data: Vec<(String, Vec<f64>)> = Vec::new();
        let mut categorical_data: Vec<(String, Vec<&str>)> = Vec::new();

        if let Some(nm) = numeric {
            for (j, name) in names.iter().enumerate().take(n_numeric) {
                let col = nm.col(j);
                columns.push(ColumnProfile::from_numeric(name.clone(), &col));
                memory_bytes += column_bytes(crate::types::ColumnType::Numeric, rows);
                numeric_data.push((name.clone(), col));
            }
        }
        if let Some(cm) = categorical {
            for (offset, j) in (n_numeric..cols).enumerate() {
                let cells = borrow_column(cm, offset);
                let profile = ColumnProfile::from_strings(names[j].clone(), &cells);
                memory_bytes += column_bytes(profile.column_type, rows);
                match profile.column_type {
                    crate::types::ColumnType::Numeric => {
                        let parsed = infer::parse_numeric_column(&cells);
                        numeric_data.push((names[j].clone(), parsed));
                    }
                    crate::types::ColumnType::Categorical => {
                        categorical_data.push((names[j].clone(), cells));
                    }
                }
                columns.push(profile);
            }
        }

        let num_refs: Vec<(&str, &[f64])> = numeric_data
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let cat_refs: Vec<(&str, &[&str])> = categorical_data
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let relationships = Relationships::compute(&num_refs, &cat_refs);

        let duplicate_rows = match (numeric, categorical) {
            (Some(nm), Some(cm)) => count_duplicate_table(nm, cm),
            (Some(nm), None) => count_duplicate_numeric(nm),
            (None, Some(cm)) => count_duplicate_str(cm),
            (None, None) => 0,
        };
        Ok(Self::finish(
            rows,
            cols,
            memory_bytes,
            duplicate_rows,
            columns,
            relationships,
        ))
    }

    fn finish(
        n_rows: usize,
        n_columns: usize,
        memory_bytes: usize,
        duplicate_rows: usize,
        columns: Vec<ColumnProfile>,
        relationships: Option<Relationships>,
    ) -> Self {
        let duplicate_fraction = if n_rows == 0 {
            0.0
        } else {
            duplicate_rows as f64 / n_rows as f64
        };
        DatasetProfile {
            n_rows,
            n_columns,
            memory_bytes,
            duplicate_rows,
            duplicate_fraction,
            target_column: None,
            columns,
            relationships,
        }
    }
}

fn count_duplicate_numeric(m: &Matrix) -> usize {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut seen: HashSet<Vec<u64>> = HashSet::with_capacity(rows.min(1 << 20));
    let mut dupes = 0usize;
    for i in 0..rows {
        // Rows containing NaN never equal anything under `f64`'s `PartialEq`
        // (NaN != NaN), so they are skipped entirely. `-0.0` and `0.0` are
        // equal under `PartialEq`, so both are canonicalized to the same bits.
        let mut row = Vec::with_capacity(cols);
        let mut has_nan = false;
        for j in 0..cols {
            let v = m.get(i, j);
            if v.is_nan() {
                has_nan = true;
                break;
            }
            row.push(if v == 0.0 { 0 } else { v.to_bits() });
        }
        if has_nan {
            continue;
        }
        if !seen.insert(row) {
            dupes += 1;
        }
    }
    dupes
}

fn count_duplicate_str(m: &StrMatrix) -> usize {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut seen: HashSet<Vec<&str>> = HashSet::with_capacity(rows.min(1 << 20));
    let mut dupes = 0usize;
    for i in 0..rows {
        let row: Vec<&str> = (0..cols).map(|j| m.get(i, j)).collect();
        if !seen.insert(row) {
            dupes += 1;
        }
    }
    dupes
}

fn count_duplicate_table(numeric: &Matrix, categorical: &StrMatrix) -> usize {
    let rows = numeric.nrows();
    let mut seen: HashSet<(Vec<u64>, Vec<&str>)> = HashSet::with_capacity(rows.min(1 << 20));
    let mut dupes = 0usize;
    for i in 0..rows {
        let mut nr: Vec<u64> = Vec::with_capacity(numeric.ncols());
        let mut has_nan = false;
        for j in 0..numeric.ncols() {
            let v = numeric.get(i, j);
            if v.is_nan() {
                has_nan = true;
                break;
            }
            nr.push(if v == 0.0 { 0 } else { v.to_bits() });
        }
        if has_nan {
            continue;
        }
        let cr: Vec<&str> = (0..categorical.ncols())
            .map(|j| categorical.get(i, j))
            .collect();
        if !seen.insert((nr, cr)) {
            dupes += 1;
        }
    }
    dupes
}

// Re-export the missing-marker check for callers building profiles by hand.
pub use infer::is_missing as _reexport_is_missing;
