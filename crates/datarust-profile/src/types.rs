//! Shared value types used across the profiling modules.

use std::fmt;

/// The inferred semantic type of a single column.
///
/// Type inference is heuristic: a column is `Numeric` when *all* non-empty
/// values parse as `f64`, and `Categorical` otherwise (or when the caller
/// declares it as such by profiling a [`datarust::StrMatrix`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ColumnType {
    /// Continuous-valued numeric column.
    Numeric,
    /// Discrete string/category column.
    Categorical,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnType::Numeric => f.write_str("numeric"),
            ColumnType::Categorical => f.write_str("categorical"),
        }
    }
}

/// A severity level attached to a [`crate::quality::QualityIssue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Severity {
    /// Informational; no action required.
    Info,
    /// Worth investigating; not necessarily wrong.
    Warning,
    /// Likely to bias downstream models or break preprocessing.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => f.write_str("info"),
            Severity::Warning => f.write_str("warning"),
            Severity::Critical => f.write_str("critical"),
        }
    }
}
