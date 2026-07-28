//! Whole-dataset profile: shape, memory, duplicate rows, and per-column profiles.

use datarust::{stats, Matrix, StrMatrix};

use super::column::{ColumnProfile, FiveNumber, PrecomputedStats};
use crate::error::{ProfileError, Result};
use crate::infer;

/// Rough memory estimate for one column, in bytes, based on cell count and type.
fn column_bytes(column_type: crate::types::ColumnType, count: usize) -> usize {
    match column_type {
        // f64 per cell.
        crate::types::ColumnType::Numeric => count * 8,
        // 24-byte String header + payload estimate of ~10 bytes per cell.
        crate::types::ColumnType::Categorical => count * 24,
    }
}

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
    /// One [`ColumnProfile`] per column, in input order.
    pub columns: Vec<ColumnProfile>,
}

impl DatasetProfile {
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
        for (j, name) in resolved.iter().enumerate().take(cols) {
            // The flat helpers are not NaN-aware: if a column contains any
            // non-finite value, its bulk mean/variance/quantile are polluted
            // by NaN propagation. In that case fall back to the per-column
            // path, which filters missing values first. The fast path still
            // wins on the common (fully-finite) case.
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
            // Skew/kurtosis/histogram/outliers still need the raw column.
            let col = m.col(j);
            columns.push(ColumnProfile::from_numeric_with_stats(
                name.clone(),
                &col,
                precomputed,
            ));
            memory_bytes += column_bytes(crate::types::ColumnType::Numeric, rows);
        }

        let duplicate_rows = count_duplicate_numeric(m);
        Ok(Self::finish(
            rows,
            cols,
            memory_bytes,
            duplicate_rows,
            columns,
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
        for (j, name) in resolved.iter().enumerate().take(cols) {
            let cells = m.column(j);
            let profile = ColumnProfile::from_strings(name.clone(), &cells);
            memory_bytes += column_bytes(profile.column_type, rows);
            columns.push(profile);
        }

        let duplicate_rows = count_duplicate_str(m);
        Ok(Self::finish(
            rows,
            cols,
            memory_bytes,
            duplicate_rows,
            columns,
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
        if let Some(nm) = numeric {
            for (j, name) in names.iter().enumerate().take(n_numeric) {
                let col = nm.col(j);
                columns.push(ColumnProfile::from_numeric(name.clone(), &col));
                memory_bytes += column_bytes(crate::types::ColumnType::Numeric, rows);
            }
        }
        if let Some(cm) = categorical {
            for (offset, j) in (n_numeric..cols).enumerate() {
                let cells = cm.column(offset);
                let profile = ColumnProfile::from_strings(names[j].clone(), &cells);
                memory_bytes += column_bytes(profile.column_type, rows);
                columns.push(profile);
            }
        }

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
        ))
    }

    fn finish(
        n_rows: usize,
        n_columns: usize,
        memory_bytes: usize,
        duplicate_rows: usize,
        columns: Vec<ColumnProfile>,
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
            columns,
        }
    }
}

fn count_duplicate_numeric(m: &Matrix) -> usize {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut seen: Vec<Vec<f64>> = Vec::with_capacity(rows);
    let mut dupes = 0usize;
    for i in 0..rows {
        let row: Vec<f64> = (0..cols).map(|j| m.get(i, j)).collect();
        if seen.iter().any(|r| r == &row) {
            dupes += 1;
        } else {
            seen.push(row);
        }
    }
    // Dedup-by-difference: duplicate_rows counts rows beyond the first
    // occurrence. The loop above already does that correctly.
    let _ = rows; // suppress unused warning path
    dupes
}

fn count_duplicate_str(m: &StrMatrix) -> usize {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut seen: Vec<Vec<String>> = Vec::with_capacity(rows);
    let mut dupes = 0usize;
    for i in 0..rows {
        let row: Vec<String> = (0..cols).map(|j| m.get(i, j).to_string()).collect();
        if seen.iter().any(|r| r == &row) {
            dupes += 1;
        } else {
            seen.push(row);
        }
    }
    dupes
}

fn count_duplicate_table(numeric: &Matrix, categorical: &StrMatrix) -> usize {
    let rows = numeric.nrows();
    let mut seen: Vec<(Vec<f64>, Vec<String>)> = Vec::with_capacity(rows);
    let mut dupes = 0usize;
    for i in 0..rows {
        let nr: Vec<f64> = (0..numeric.ncols()).map(|j| numeric.get(i, j)).collect();
        let cr: Vec<String> = (0..categorical.ncols())
            .map(|j| categorical.get(i, j).to_string())
            .collect();
        let row = (nr, cr);
        if seen.iter().any(|r| r == &row) {
            dupes += 1;
        } else {
            seen.push(row);
        }
    }
    dupes
}

// Re-export the missing-marker check for callers building profiles by hand.
pub use infer::is_missing as _reexport_is_missing;
