//! # datarust-profile
//!
//! One-call data profiling and data-quality reports for the
//! [datarust](https://crates.io/crates/datarust) ecosystem.
//!
//! `datarust-profile` takes a numeric [`datarust::Matrix`], a categorical
//! [`datarust::StrMatrix`], or a mixed table of both, and produces a
//! [`DatasetProfile`] describing shape, memory footprint, duplicate rows,
//! per-column descriptive statistics (mean, std, five-number summary,
//! cardinality, top value) and a list of data-quality findings. Profiles can
//! be rendered to JSON (via the `serde` feature) or to a self-contained HTML
//! report with no extra dependencies.
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

pub mod error;
pub mod infer;
pub mod profile;
pub mod quality;
pub mod report;
pub mod types;

pub use error::{ProfileError, Result};
pub use profile::{
    CategoricalStats, ColumnProfile, DatasetProfile, FiveNumber, Histogram, NumericStats,
};
pub use quality::{QualityIssue, QualityKind, Thresholds};
pub use types::{ColumnType, Severity};

use datarust::{Matrix, StrMatrix};

/// Profiles a numeric [`Matrix`].
///
/// `names` optionally supplies column names; when `None` (or the wrong length)
/// columns are named `x0..x{n-1}`.
pub fn profile_matrix(m: &Matrix, names: Option<&[String]>) -> Result<DatasetProfile> {
    DatasetProfile::from_matrix(m, names)
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
