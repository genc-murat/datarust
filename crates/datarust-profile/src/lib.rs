//! # datarust-profile
//!
//! One-call data profiling and data-quality reports for the
//! [datarust](https://crates.io/crates/datarust) ecosystem.
//!
//! `datarust-profile` takes a numeric [`datarust::Matrix`], a categorical
//! [`datarust::StrMatrix`], a mixed table of both — or, behind the `csv`
//! feature, a CSV file path — and produces a [`DatasetProfile`] describing
//! shape, memory footprint, duplicate rows, per-column descriptive statistics
//! (mean, std, five-number summary, cardinality, top value) and a list of
//! data-quality findings. Profiles can be rendered to JSON (via the `serde`
//! feature), to a self-contained HTML report, or to a terminal table (via
//! the `pretty` feature) — with no extra dependencies.
//!
//! ## Features
//!
//! - `csv` — pure-Rust RFC-4180 reader ([`csv`]) plus one-call entry points
//!   ([`profile_csv`], [`DatasetProfile::from_csv`],
//!   [`DatasetProfile::from_csv_with_types`]).
//! - `streaming` — single-pass bounded-memory CSV profiling
//!   ([`profile_csv_streaming`]). Implies `csv`.
//! - `pretty` — terminal-table renderer ([`report::to_terminal`]).
//! - `cli` — the `datarust-profile` binary (CSV → HTML/JSON/terminal).
//! - `serde` — JSON serialization of profiles and findings.
//!
//! ## Quick start
//!
//! ```no_run
//! use datarust::Matrix;
//! use datarust_profile::{profile_matrix, report};
//!
//! let m = Matrix::from_rows(vec![
//!     vec![1.0, 10.0],
//!     vec![2.0, 12.0],
//!     vec![3.0, f64::NAN],
//! ]).unwrap();
//!
//! let profile = profile_matrix(&m, None).unwrap();
//! println!("{}x{}, {} duplicates",
//!     profile.n_rows, profile.n_columns, profile.duplicate_rows);
//!
//! // HTML report (always available, no extra deps):
//! let html = report::to_html(&profile);
//! std::fs::write("profile.html", html).unwrap();
//! ```
//!
//! ## JSON output (feature `serde`)
//!
//! Requires the `serde` feature. The example below is marked `ignore` so that
//! it is not compiled by `cargo test --doc` when `serde` is disabled.
//!
//! ```rust,ignore
//! use datarust::Matrix;
//! use datarust_profile::{profile_matrix, report};
//!
//! let m = Matrix::from_rows(vec![vec![1.0]]).unwrap();
//! let profile = profile_matrix(&m, None).unwrap();
//! let json = report::to_json(
//!     &report::JsonReport::from_profile(&profile),
//! ).unwrap();
//! std::fs::write("profile.json", json).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "csv")]
pub mod csv;
pub mod error;
pub mod infer;
pub mod profile;
pub mod quality;
pub mod report;
pub mod types;

pub use error::{ProfileError, Result};
pub use profile::{
    CategoricalStats, ColumnProfile, CorrelationMatrix, DatasetProfile, FiveNumber, Histogram,
    NumericStats, PointBiserialEntry, Relationships,
};
pub use quality::{run_checks, QualityIssue, QualityKind, Thresholds};
pub use types::{ColumnType, Severity};

#[cfg(feature = "csv")]
pub use csv::{read_csv, read_csv_reader, read_csv_with, CsvReadOptions, CsvTable};
#[cfg(feature = "streaming")]
pub use streaming::{profile_csv_streaming, profile_csv_streaming_with, StreamingOptions};
#[cfg(feature = "streaming")]
pub mod streaming;

use datarust::{Matrix, StrMatrix};

/// Profiles a numeric [`Matrix`].
///
/// `names` optionally supplies column names; when `None` (or the wrong length)
/// columns are named `x0..x{n-1}`.
pub fn profile_matrix(m: &Matrix, names: Option<&[String]>) -> Result<DatasetProfile> {
    DatasetProfile::from_matrix(m, names)
}

/// Profiles a numeric [`Matrix`] and sets a designated target column for leakage detection.
pub fn profile_matrix_with_target(
    m: &Matrix,
    names: Option<&[String]>,
    target: &str,
) -> Result<DatasetProfile> {
    let profile = DatasetProfile::from_matrix(m, names)?;
    Ok(profile.with_target(target))
}

/// Profiles a string [`StrMatrix`], inferring each column's type.
pub fn profile_str_matrix(m: &StrMatrix, names: Option<&[String]>) -> Result<DatasetProfile> {
    DatasetProfile::from_str_matrix(m, names)
}

/// Profiles a mixed table: a numeric [`Matrix`] block followed by a
/// categorical [`StrMatrix`] block, both with the same row count.
///
/// `names` must have one entry per column across both blocks (numerics first).
pub fn profile_table(
    numeric: Option<&Matrix>,
    categorical: Option<&StrMatrix>,
    names: &[String],
) -> Result<DatasetProfile> {
    DatasetProfile::from_table(numeric, categorical, names)
}

/// Profiles a mixed table and sets a designated target column for leakage detection.
pub fn profile_table_with_target(
    numeric: Option<&Matrix>,
    categorical: Option<&StrMatrix>,
    names: &[String],
    target: &str,
) -> Result<DatasetProfile> {
    let profile = DatasetProfile::from_table(numeric, categorical, names)?;
    Ok(profile.with_target(target))
}

/// Profiles a CSV file, inferring each column's type.
///
/// Requires the `csv` feature. Header names become column names when the file
/// has a header row; `names` (when supplied with the matching length)
/// overrides them. Equivalent to [`DatasetProfile::from_csv`] modulo the
/// optional name override.
#[cfg(feature = "csv")]
pub fn profile_csv(path: &std::path::Path, names: Option<&[String]>) -> Result<DatasetProfile> {
    let table = csv::read_csv(path)?;
    DatasetProfile::profile_from_csv_table_pub(&table, names)
}

/// Profiles a CSV file with explicit [`csv::CsvReadOptions`]. Requires the
/// `csv` feature.
#[cfg(feature = "csv")]
pub fn profile_csv_with(
    path: &std::path::Path,
    opts: &csv::CsvReadOptions,
    names: Option<&[String]>,
) -> Result<DatasetProfile> {
    let table = csv::read_csv_with(path, opts)?;
    DatasetProfile::profile_from_csv_table_pub(&table, names)
}

/// Profiles a CSV file with per-column forced types (see
/// [`DatasetProfile::from_csv_with_types`]). Requires the `csv` feature.
#[cfg(feature = "csv")]
pub fn profile_csv_with_types(
    path: &std::path::Path,
    schema: &[ColumnType],
) -> Result<DatasetProfile> {
    DatasetProfile::from_csv_with_types(path, schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `profile_csv` and `profile_str_matrix` must produce bit-identical
    /// profiles on the same data: the CSV reader feeds the exact same
    /// `StrMatrix` path.
    #[cfg(feature = "csv")]
    #[test]
    fn profile_csv_matches_profile_str_matrix_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parity.csv");
        std::fs::write(
            &path,
            "age,income,city\n34,58000,Istanbul\n45,NA,Ankara\n29,45000,Izmir\n52,110000,Istanbul\n",
        )
        .unwrap();

        let via_csv = profile_csv(&path, None).unwrap();
        let table = csv::read_csv(&path).unwrap();
        let names = table.column_names();
        let via_matrix = profile_str_matrix(&table.rows, Some(&names)).unwrap();

        // `DatasetProfile` derives `PartialEq`, but the Pearson matrix may
        // contain NaN entries (zero-variance columns); compare columns
        // exactly and relationships structurally.
        assert_eq!(via_csv.columns, via_matrix.columns);
        assert_eq!(via_csv.n_rows, via_matrix.n_rows);
        assert_eq!(via_csv.n_columns, via_matrix.n_columns);
        assert_eq!(via_csv.duplicate_rows, via_matrix.duplicate_rows);
        assert_eq!(via_csv.memory_bytes, via_matrix.memory_bytes);
        match (&via_csv.relationships, &via_matrix.relationships) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.pearson.as_ref().map(|m| m.labels.clone()),
                    b.pearson.as_ref().map(|m| m.labels.clone())
                );
                if let (Some(pa), Some(pb)) = (&a.pearson, &b.pearson) {
                    for (ra, rb) in pa.values.iter().zip(pb.values.iter()) {
                        for (va, vb) in ra.iter().zip(rb.iter()) {
                            assert!(va.to_bits() == vb.to_bits() || (va.is_nan() && vb.is_nan()));
                        }
                    }
                }
                assert_eq!(
                    a.cramers_v.as_ref().map(|m| m.labels.clone()),
                    b.cramers_v.as_ref().map(|m| m.labels.clone())
                );
            }
            _ => panic!("relationships presence mismatch"),
        }
    }

    #[cfg(feature = "csv")]
    #[test]
    fn csv_with_types_rejects_duplicate_header_names() {
        // Regression: duplicate header names collapsed in the name-keyed
        // reorder, silently dropping the second column's stats.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup.csv");
        std::fs::write(&path, "a,a\n1,2\n3,4\n").unwrap();
        let schema = [ColumnType::Numeric, ColumnType::Numeric];
        let err = profile_csv_with_types(&path, &schema).unwrap_err();
        assert!(
            err.to_string().contains("duplicate column name 'a'"),
            "{err}"
        );
    }
}
