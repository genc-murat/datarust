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
