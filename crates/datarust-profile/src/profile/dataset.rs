//! Whole-dataset profile: shape, memory, duplicate rows, and per-column profiles.

use datarust::{stats, Matrix, StrMatrix};

use super::column::{ColumnProfile, FiveNumber, PrecomputedStats};
use crate::error::{ProfileError, Result};
use crate::infer;
#[cfg(feature = "csv")]
use crate::types::ColumnType;
#[cfg(feature = "csv")]
use std::collections::HashMap;
use std::collections::HashSet;
#[cfg(feature = "csv")]
use std::path::Path;

/// Profiles a parsed CSV table, inferring each column's type.
///
/// Requires the `csv` feature. This is the composition of
/// [`crate::csv::read_csv_reader`] output and [`DatasetProfile::from_str_matrix`]
/// with the header names, so both produce bit-identical profiles.
#[cfg(feature = "csv")]
pub(crate) fn profile_from_csv_table(
    table: &crate::csv::CsvTable,
    names: Option<&[String]>,
) -> Result<DatasetProfile> {
    let fallback = table.column_names();
    let resolved: &[String] = match names {
        Some(ns) if ns.len() == table.ncols() => ns,
        _ => &fallback,
    };
    DatasetProfile::from_str_matrix(&table.rows, Some(resolved))
}

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
    /// Library-facing alias for [`profile_from_csv_table`], used by the
    /// `profile_csv*` free functions in the crate root.
    #[cfg(feature = "csv")]
    pub(crate) fn profile_from_csv_table_pub(
        table: &crate::csv::CsvTable,
        names: Option<&[String]>,
    ) -> Result<Self> {
        profile_from_csv_table(table, names)
    }

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

    /// Profiles a CSV file, inferring each column's type.
    ///
    /// Requires the `csv` feature. Header names become column names when the
    /// file has a header row (the default); `names` (when supplied with the
    /// matching length) overrides them.
    ///
    /// ```no_run
    /// use datarust_profile::DatasetProfile;
    /// use std::path::Path;
    ///
    /// let profile = DatasetProfile::from_csv(Path::new("data.csv")).unwrap();
    /// println!("{} rows", profile.n_rows);
    /// ```
    #[cfg(feature = "csv")]
    pub fn from_csv(path: &Path) -> Result<Self> {
        let table = crate::csv::read_csv(path)?;
        profile_from_csv_table(&table, None)
    }

    /// Profiles a CSV file with explicit [`crate::csv::CsvReadOptions`]
    /// (delimiter, header, trimming, ...). Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv_with(path: &Path, opts: &crate::csv::CsvReadOptions) -> Result<Self> {
        let table = crate::csv::read_csv_with(path, opts)?;
        profile_from_csv_table(&table, None)
    }

    /// Profiles a CSV file with per-column forced types.
    ///
    /// Requires the `csv` feature. Column `j` is profiled as `schema[j]`:
    /// `Numeric` columns are parsed with the crate's missing-marker semantics
    /// (non-parseable cells become missing) and grouped into a numeric block,
    /// `Categorical` columns stay strings. The schema must have one entry per
    /// CSV column. This mirrors the roadmap's `from_csv_with_types`.
    #[cfg(feature = "csv")]
    pub fn from_csv_with_types(path: &Path, schema: &[ColumnType]) -> Result<Self> {
        let table = crate::csv::read_csv(path)?;
        Self::from_csv_table_with_types(&table, schema)
    }

    /// Applies a forced-type schema to an already-parsed CSV table.
    #[cfg(feature = "csv")]
    pub(crate) fn from_csv_table_with_types(
        table: &crate::csv::CsvTable,
        schema: &[ColumnType],
    ) -> Result<Self> {
        if schema.len() != table.ncols() {
            return Err(ProfileError::InvalidInput(format!(
                "schema has {} entries, CSV has {} columns",
                schema.len(),
                table.ncols()
            )));
        }
        let names = table.column_names();
        // The schema path reorders columns back into CSV order by name, so
        // duplicate header names must be rejected instead of silently
        // collapsing the affected columns.
        {
            let mut seen = HashSet::new();
            if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
                return Err(ProfileError::InvalidInput(format!(
                    "duplicate column name '{dup}' in the CSV header; \
                     column names must be unique when a type schema is used"
                )));
            }
        }
        let rows = table.nrows();

        // Group columns by forced type, preserving input order within each group.
        let mut numeric_idx: Vec<usize> = Vec::new();
        let mut categorical_idx: Vec<usize> = Vec::new();
        for (j, kind) in schema.iter().enumerate() {
            match kind {
                ColumnType::Numeric => numeric_idx.push(j),
                ColumnType::Categorical => categorical_idx.push(j),
            }
        }

        // Transpose the required columns once, then split into blocks.
        let mut columns_of: Vec<Vec<String>> = Vec::with_capacity(table.ncols());
        for j in 0..table.ncols() {
            columns_of.push(
                (0..rows)
                    .map(|i| table.rows.get(i, j).to_string())
                    .collect(),
            );
        }

        let mut numeric_columns: Vec<Vec<f64>> = Vec::with_capacity(numeric_idx.len());
        for &j in &numeric_idx {
            numeric_columns.push(infer::parse_numeric_column(&columns_of[j]));
        }

        let numeric_matrix = if numeric_columns.is_empty() {
            None
        } else {
            Some(build_numeric_matrix(rows, &numeric_columns)?)
        };
        // Categorical columns are re-interleaved row-major for `StrMatrix`.
        let categorical_matrix = if categorical_idx.is_empty() {
            None
        } else {
            let cat_rows: Vec<Vec<String>> = (0..rows)
                .map(|i| {
                    categorical_idx
                        .iter()
                        .map(|&j| columns_of[j][i].clone())
                        .collect()
                })
                .collect();
            Some(StrMatrix::from_strings(cat_rows)?)
        };

        // Names in original column order: numeric names first, then categorical.
        let mut ordered_names: Vec<String> = Vec::with_capacity(table.ncols());
        for &j in &numeric_idx {
            ordered_names.push(names[j].clone());
        }
        for &j in &categorical_idx {
            ordered_names.push(names[j].clone());
        }

        let mut profile = Self::from_table(
            numeric_matrix.as_ref(),
            categorical_matrix.as_ref(),
            &ordered_names,
        )?;
        profile = Self::reorder_columns(profile, &names);
        Ok(profile)
    }

    /// Rebuilds a profile with `columns` permuted into the order given by
    /// `names` (matched by name). Relationships are left as computed.
    #[cfg(feature = "csv")]
    fn reorder_columns(mut profile: Self, names: &[String]) -> Self {
        let mut by_name: HashMap<String, ColumnProfile> = HashMap::new();
        for col in std::mem::take(&mut profile.columns) {
            by_name.insert(col.name.clone(), col);
        }
        profile.columns = names
            .iter()
            .map(|n| {
                by_name.remove(n).unwrap_or_else(|| {
                    // All names come from the same profile and duplicates
                    // are rejected before the reorder, so absence is a bug;
                    // degrade to an empty column instead of panicking.
                    ColumnProfile {
                        name: n.clone(),
                        column_type: crate::types::ColumnType::Categorical,
                        count: 0,
                        missing_count: 0,
                        missing_fraction: 0.0,
                        numeric: None,
                        categorical: None,
                    }
                })
            })
            .collect();
        profile
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

/// Builds a row-major `Matrix` from column-major `f64` vectors.
#[cfg(feature = "csv")]
fn build_numeric_matrix(rows: usize, columns: &[Vec<f64>]) -> Result<Matrix> {
    let cols = columns.len();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for col in columns {
            flat.push(col[i]);
        }
    }
    Matrix::from_flat(rows, cols, flat).map_err(ProfileError::from)
}
