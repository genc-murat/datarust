//! Column-type inference for string-typed data.
//!
//! A column is treated as [`ColumnType::Numeric`] when every non-empty cell
//! parses as `f64`; otherwise it is [`ColumnType::Categorical`]. This mirrors
//! the common pandas `convert_dtypes` heuristic and keeps inference free of
//! any external parser dependency.

use crate::types::ColumnType;

/// Returns `true` when the string represents a missing value placeholder.
///
/// Recognised empty markers: `""`, `"NA"`, `"N/A"`, `"null"`, `"NULL"`,
/// `"NaN"`, `"nan"`, `"None"`, `"none"`, `"-"`. The check is case-insensitive
/// for the alphabetic forms and trims surrounding whitespace.
pub fn is_missing(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    matches!(
        t.to_ascii_lowercase().as_str(),
        "na" | "n/a" | "null" | "nan" | "none" | "-" | "?"
    )
}

/// Infers the type of a single column from its raw string cells.
///
/// Empty/missing cells are skipped; the column is numeric only when **every**
/// non-missing cell parses as `f64`.
pub fn infer_column(cells: &[String]) -> ColumnType {
    let any_non_missing = cells.iter().any(|c| !is_missing(c));
    if !any_non_missing {
        // Treat an all-empty column as categorical (nothing numeric to summarise).
        return ColumnType::Categorical;
    }
    let all_numeric = cells
        .iter()
        .all(|c| is_missing(c) || c.trim().parse::<f64>().is_ok());
    if all_numeric {
        ColumnType::Numeric
    } else {
        ColumnType::Categorical
    }
}

/// Parses a column of strings into `f64`, replacing missing markers with
/// [`f64::NAN`]. Non-missing cells that fail to parse are also emitted as
/// `NaN`; callers that need strict parsing should validate with
/// [`infer_column`] first.
pub fn parse_numeric_column(cells: &[String]) -> Vec<f64> {
    cells
        .iter()
        .map(|c| {
            if is_missing(c) {
                f64::NAN
            } else {
                c.trim().parse::<f64>().unwrap_or(f64::NAN)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_missing_recognizes_all_markers() {
        assert!(is_missing(""));
        assert!(is_missing("NA"));
        assert!(is_missing("N/A"));
        assert!(is_missing("null"));
        assert!(is_missing("NULL"));
        assert!(is_missing("NaN"));
        assert!(is_missing("nan"));
        assert!(is_missing("None"));
        assert!(is_missing("none"));
        assert!(is_missing("-"));
        assert!(is_missing("?"));
        assert!(is_missing("  NA  "));
        assert!(is_missing("  nan  "));
        assert!(!is_missing("0"));
        assert!(!is_missing("1.5"));
        assert!(!is_missing("hello"));
    }

    #[test]
    fn infer_column_all_missing_is_categorical() {
        let cells = vec!["NA".to_string(), "null".to_string(), "".to_string()];
        assert_eq!(infer_column(&cells), ColumnType::Categorical);
    }

    #[test]
    fn infer_column_mixed_numeric_and_missing_is_numeric() {
        let cells = vec!["1.0".to_string(), "NA".to_string(), "2.0".to_string()];
        assert_eq!(infer_column(&cells), ColumnType::Numeric);
    }

    #[test]
    fn infer_column_one_non_numeric_is_categorical() {
        let cells = vec!["1.0".to_string(), "hello".to_string(), "2.0".to_string()];
        assert_eq!(infer_column(&cells), ColumnType::Categorical);
    }

    #[test]
    fn parse_numeric_column_handles_missing() {
        let cells = vec!["1.0".to_string(), "NA".to_string(), "2.0".to_string()];
        let parsed = parse_numeric_column(&cells);
        assert!((parsed[0] - 1.0).abs() < 1e-9);
        assert!(parsed[1].is_nan());
        assert!((parsed[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn parse_numeric_column_non_numeric_becomes_nan() {
        let cells = vec!["1.0".to_string(), "hello".to_string()];
        let parsed = parse_numeric_column(&cells);
        assert!((parsed[0] - 1.0).abs() < 1e-9);
        assert!(parsed[1].is_nan());
    }
}
