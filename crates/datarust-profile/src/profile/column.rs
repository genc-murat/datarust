//! Per-column profile records.

use std::collections::HashMap;

use crate::infer;
use crate::types::ColumnType;

/// Quantile stubs computed for numeric columns: min, Q1, median, Q3, max.
///
/// The five values correspond to the probabilities `[0.0, 0.25, 0.5, 0.75, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FiveNumber {
    /// Minimum (0% quantile).
    pub min: f64,
    /// First quartile (25% quantile).
    pub q1: f64,
    /// Median (50% quantile).
    pub median: f64,
    /// Third quartile (75% quantile).
    pub q3: f64,
    /// Maximum (100% quantile).
    pub max: f64,
}

/// The descriptive statistics computed for a numeric column.
///
/// Missing values (`NaN`) are excluded from every statistic and counted
/// separately in [`ColumnProfile::missing_count`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NumericStats {
    /// Arithmetic mean of the non-missing values (`NaN` if the column is empty).
    pub mean: f64,
    /// Sample standard deviation (ddof = 1) of the non-missing values.
    pub std: f64,
    /// Min/Q1/median/Q3/max of the non-missing values.
    pub five: FiveNumber,
}

/// The descriptive statistics computed for a categorical column.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CategoricalStats {
    /// Number of distinct non-missing values (cardinality).
    pub unique: usize,
    /// The most frequent non-missing value (ties broken by insertion order).
    pub top: String,
    /// Frequency of [`CategoricalStats::top`].
    pub freq: usize,
}

/// A full profile for a single column of the dataset.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ColumnProfile {
    /// Column name. Falls back to `x{j}` when no name was supplied.
    pub name: String,
    /// Inferred semantic type.
    pub column_type: ColumnType,
    /// Total number of rows (cells) in the column, including missing.
    pub count: usize,
    /// Number of missing/empty cells.
    pub missing_count: usize,
    /// Fraction of missing cells, in `[0.0, 1.0]`.
    pub missing_fraction: f64,
    /// Numeric descriptive stats. `None` for categorical columns.
    pub numeric: Option<NumericStats>,
    /// Categorical descriptive stats. `None` for numeric columns.
    pub categorical: Option<CategoricalStats>,
}

impl ColumnProfile {
    /// Builds a profile from a numeric slice, treating `NaN` as missing.
    pub(crate) fn from_numeric(name: String, values: &[f64]) -> Self {
        let count = values.len();
        let present: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let missing_count = count - present.len();
        let missing_fraction = if count == 0 {
            0.0
        } else {
            missing_count as f64 / count as f64
        };

        let numeric = if present.is_empty() {
            None
        } else {
            // Sort once; `stats::median_sorted` / `quantile` expect sorted input.
            let mut sorted = present.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));

            let mean = datarust::stats::mean(&present);
            let std = datarust::stats::std(&present, 1);
            let five = FiveNumber {
                min: datarust::stats::quantile(&sorted, 0.0).unwrap_or(f64::NAN),
                q1: datarust::stats::quantile(&sorted, 0.25).unwrap_or(f64::NAN),
                median: datarust::stats::median_sorted(&sorted).unwrap_or(f64::NAN),
                q3: datarust::stats::quantile(&sorted, 0.75).unwrap_or(f64::NAN),
                max: datarust::stats::quantile(&sorted, 1.0).unwrap_or(f64::NAN),
            };
            Some(NumericStats { mean, std, five })
        };

        ColumnProfile {
            name,
            column_type: ColumnType::Numeric,
            count,
            missing_count,
            missing_fraction,
            numeric,
            categorical: None,
        }
    }

    /// Builds a profile from raw string cells, inferring the column type.
    pub(crate) fn from_strings(name: String, cells: &[String]) -> Self {
        let count = cells.len();
        let missing_count = cells.iter().filter(|c| infer::is_missing(c)).count();
        let missing_fraction = if count == 0 {
            0.0
        } else {
            missing_count as f64 / count as f64
        };

        let column_type = infer::infer_column(cells);
        match column_type {
            ColumnType::Numeric => {
                let values = infer::parse_numeric_column(cells);
                let mut p = Self::from_numeric(name, &values);
                // Preserve the string-source count/fraction even when all cells
                // were missing (from_numeric would otherwise treat them as NaN).
                p.count = count;
                p.missing_count = missing_count;
                p.missing_fraction = missing_fraction;
                p
            }
            ColumnType::Categorical => {
                let categorical = compute_categorical(cells);
                ColumnProfile {
                    name,
                    column_type,
                    count,
                    missing_count,
                    missing_fraction,
                    numeric: None,
                    categorical,
                }
            }
        }
    }
}

/// Tallies the non-missing cells of a categorical column.
fn compute_categorical(cells: &[String]) -> Option<CategoricalStats> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for cell in cells {
        if infer::is_missing(cell) {
            continue;
        }
        let trimmed = cell.trim();
        match counts.get(trimmed) {
            None => {
                counts.insert(trimmed, 1);
                order.push(trimmed);
            }
            Some(c) => *counts.get_mut(trimmed).unwrap() = c + 1,
        }
    }
    if order.is_empty() {
        return None;
    }
    let (top, freq) = order
        .iter()
        .map(|k| (*k, counts[*k]))
        .max_by_key(|&(_, c)| c)
        .unwrap();
    Some(CategoricalStats {
        unique: order.len(),
        top: top.to_string(),
        freq,
    })
}
