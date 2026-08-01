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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_numeric() {
        assert_eq!(ColumnType::Numeric.to_string(), "numeric");
    }

    #[test]
    fn display_categorical() {
        assert_eq!(ColumnType::Categorical.to_string(), "categorical");
    }

    #[test]
    fn severity_info() {
        assert_eq!(Severity::Info.to_string(), "info");
    }

    #[test]
    fn severity_warning() {
        assert_eq!(Severity::Warning.to_string(), "warning");
    }

    #[test]
    fn severity_critical() {
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn column_type_equality() {
        assert_eq!(ColumnType::Numeric, ColumnType::Numeric);
        assert_eq!(ColumnType::Categorical, ColumnType::Categorical);
        assert_ne!(ColumnType::Numeric, ColumnType::Categorical);
    }

    #[test]
    fn severity_equality() {
        assert_eq!(Severity::Info, Severity::Info);
        assert_ne!(Severity::Info, Severity::Critical);
    }

    #[test]
    fn column_type_clone() {
        let ct = ColumnType::Numeric;
        let ct2 = ct;
        assert_eq!(ct, ct2);
    }

    #[test]
    fn severity_clone() {
        let s = Severity::Warning;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn column_type_debug() {
        assert!(format!("{:?}", ColumnType::Numeric).contains("Numeric"));
    }

    #[test]
    fn severity_debug() {
        assert!(format!("{:?}", Severity::Critical).contains("Critical"));
    }
}
