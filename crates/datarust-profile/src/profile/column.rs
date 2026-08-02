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

/// An equal-width histogram over the non-missing values of a numeric column.
///
/// The bin count follows Sturges' rule (`ceil(log2(n) + 1)`, floored at 1).
/// `edges.len() == counts.len() + 1`; the final bin is inclusive of its upper
/// edge so the maximum value is never dropped.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Histogram {
    /// Bin edges, ascending. Length is `counts.len() + 1` (empty for an empty
    /// column).
    pub edges: Vec<f64>,
    /// Number of values falling into each bin. Length is one less than
    /// `edges`.
    pub counts: Vec<usize>,
}

impl Histogram {
    /// Returns the number of bins (== `counts.len()`).
    pub fn nbins(&self) -> usize {
        self.counts.len()
    }

    /// Returns the count in the tallest bin, or `0` for an empty histogram.
    pub fn max_count(&self) -> usize {
        self.counts.iter().copied().max().unwrap_or(0)
    }
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
    /// Fisher–Pearson sample skewness of the non-missing values (`0.0` when
    /// undefined, i.e. fewer than three values or zero variance).
    pub skewness: f64,
    /// Excess (Fisher) kurtosis of the non-missing values (`0.0` when
    /// undefined, i.e. fewer than four values or zero variance).
    pub kurtosis: f64,
    /// Equal-width histogram (Sturges bins) of the non-missing values.
    pub histogram: Histogram,
    /// Number of values outside the Tukey IQR fences (`Q1 − 1.5·IQR`,
    /// `Q3 + 1.5·IQR`).
    pub outlier_count: usize,
    /// Fraction of values outside the IQR fences, in `[0.0, 1.0]`.
    pub outlier_fraction: f64,
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
    /// `freq` divided by the number of non-missing cells, in `[0.0, 1.0]`.
    /// `1.0` means a single value dominates the whole column.
    pub imbalance_ratio: f64,
    /// The most frequent values and their counts, descending, capped at a
    /// small number for display. Useful for frequency bar charts.
    pub top_values: Vec<(String, usize)>,
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

/// Cap on the length of [`CategoricalStats::top_values`].
const TOP_VALUES_CAP: usize = 8;

/// Pre-computed mean/std/five-number summary, used by the `_flat` fast path in
/// [`crate::profile::DatasetProfile::from_matrix`] to avoid recomputing the
/// most expensive statistics per column.
///
/// When `Some`, [`ColumnProfile::from_numeric_with_stats`] trusts these values
/// instead of re-deriving them; the distributional stats (skewness, kurtosis,
/// histogram, outliers) are still computed from the raw values.
pub(crate) struct PrecomputedStats {
    pub(crate) mean: f64,
    pub(crate) std: f64,
    pub(crate) five: FiveNumber,
}

impl ColumnProfile {
    /// Builds a profile from a numeric slice, treating `NaN` as missing.
    pub(crate) fn from_numeric(name: String, values: &[f64]) -> Self {
        Self::from_numeric_with_stats(name, values, None)
    }

    /// Builds a profile from a numeric slice, optionally reusing pre-computed
    /// mean/std/five-number summary.
    ///
    /// When `precomputed` is `Some`, the caller (the `_flat` fast path)
    /// asserts responsibility for matching the values; when `None`, the stats
    /// are derived here exactly as in [`Self::from_numeric`].
    pub(crate) fn from_numeric_with_stats(
        name: String,
        values: &[f64],
        precomputed: Option<PrecomputedStats>,
    ) -> Self {
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

            let (mean, std, five) = match precomputed {
                Some(p) => (p.mean, p.std, p.five),
                None => {
                    let m = datarust::stats::mean(&present);
                    let s = datarust::stats::std(&present, 1);
                    let f = FiveNumber {
                        min: datarust::stats::quantile(&sorted, 0.0).unwrap_or(f64::NAN),
                        q1: datarust::stats::quantile(&sorted, 0.25).unwrap_or(f64::NAN),
                        median: datarust::stats::median_sorted(&sorted).unwrap_or(f64::NAN),
                        q3: datarust::stats::quantile(&sorted, 0.75).unwrap_or(f64::NAN),
                        max: datarust::stats::quantile(&sorted, 1.0).unwrap_or(f64::NAN),
                    };
                    (m, s, f)
                }
            };

            let skew = super::distribution::skewness(&present, mean, std);
            let kurt = super::distribution::kurtosis_excess(&present, mean, std);
            let histogram = super::distribution::histogram(&sorted, five.min, five.max);
            let (outlier_count, outlier_fraction) =
                super::distribution::outlier_count(&sorted, five.q1, five.q3);

            Some(NumericStats {
                mean,
                std,
                five,
                skewness: skew,
                kurtosis: kurt,
                histogram,
                outlier_count,
                outlier_fraction,
            })
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
    pub(crate) fn from_strings<T: AsRef<str>>(name: String, cells: &[T]) -> Self {
        let count = cells.len();
        let missing_count = cells
            .iter()
            .filter(|c| infer::is_missing(c.as_ref()))
            .count();
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
fn compute_categorical<T: AsRef<str>>(cells: &[T]) -> Option<CategoricalStats> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for cell in cells {
        let cell = cell.as_ref();
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
    let present_total: usize = order.iter().map(|k| counts[*k]).sum();

    // Top values: sort by count descending, tie-break by first-seen order.
    let mut entries: Vec<(&str, usize)> = order.iter().map(|k| (*k, counts[*k])).collect();
    entries.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    let top_values: Vec<(String, usize)> = entries
        .into_iter()
        .take(TOP_VALUES_CAP)
        .map(|(k, c)| (k.to_string(), c))
        .collect();

    let (top, freq) = {
        let first = top_values.first().expect("non-empty");
        (first.0.clone(), first.1)
    };
    let imbalance_ratio = if present_total == 0 {
        0.0
    } else {
        freq as f64 / present_total as f64
    };

    Some(CategoricalStats {
        unique: order.len(),
        top,
        freq,
        imbalance_ratio,
        top_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColumnType;

    #[test]
    fn from_numeric_handles_empty() {
        let p = ColumnProfile::from_numeric("x".to_string(), &[]);
        assert_eq!(p.name, "x");
        assert_eq!(p.column_type, ColumnType::Numeric);
        assert_eq!(p.count, 0);
        assert_eq!(p.missing_count, 0);
        assert_eq!(p.missing_fraction, 0.0);
        assert!(p.numeric.is_none());
    }

    #[test]
    fn from_numeric_all_nan() {
        let p = ColumnProfile::from_numeric("x".to_string(), &[f64::NAN, f64::NAN]);
        assert_eq!(p.count, 2);
        assert_eq!(p.missing_count, 2);
        assert_eq!(p.missing_fraction, 1.0);
        assert!(p.numeric.is_none());
    }

    #[test]
    fn from_numeric_computes_stats() {
        let p = ColumnProfile::from_numeric("x".to_string(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(p.count, 5);
        assert_eq!(p.missing_count, 0);
        let n = p.numeric.as_ref().unwrap();
        assert!((n.mean - 3.0).abs() < 1e-9);
        assert!((n.five.min - 1.0).abs() < 1e-9);
        assert!((n.five.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn from_numeric_with_nan() {
        let p = ColumnProfile::from_numeric("x".to_string(), &[1.0, f64::NAN, 3.0]);
        assert_eq!(p.count, 3);
        assert_eq!(p.missing_count, 1);
        assert!((p.missing_fraction - 1.0 / 3.0).abs() < 1e-9);
        let n = p.numeric.as_ref().unwrap();
        assert!((n.mean - 2.0).abs() < 1e-9);
    }

    #[test]
    fn from_strings_numeric() {
        let cells = vec!["1.0".to_string(), "2.0".to_string(), "3.0".to_string()];
        let p = ColumnProfile::from_strings("x".to_string(), &cells);
        assert_eq!(p.column_type, ColumnType::Numeric);
        assert!(p.numeric.is_some());
    }

    #[test]
    fn from_strings_categorical() {
        let cells = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let p = ColumnProfile::from_strings("x".to_string(), &cells);
        assert_eq!(p.column_type, ColumnType::Categorical);
        let c = p.categorical.as_ref().unwrap();
        assert_eq!(c.unique, 2);
        assert_eq!(c.top, "a");
        assert_eq!(c.freq, 2);
    }

    #[test]
    fn from_strings_with_missing() {
        let cells = vec!["1.0".to_string(), "NA".to_string(), "3.0".to_string()];
        let p = ColumnProfile::from_strings("x".to_string(), &cells);
        assert_eq!(p.column_type, ColumnType::Numeric);
        assert_eq!(p.missing_count, 1);
        assert!((p.missing_fraction - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn from_strings_all_missing_is_categorical() {
        let cells = vec!["NA".to_string(), "null".to_string(), "".to_string()];
        let p = ColumnProfile::from_strings("x".to_string(), &cells);
        assert_eq!(p.column_type, ColumnType::Categorical);
        assert_eq!(p.missing_count, 3);
        assert!(p.categorical.is_none());
    }

    #[test]
    fn histogram_nbins() {
        let h = crate::profile::distribution::histogram(&[1.0, 2.0, 3.0], 1.0, 3.0);
        assert_eq!(h.nbins(), h.counts.len());
        assert_eq!(h.max_count(), 1);
    }

    #[test]
    fn histogram_max_count() {
        let h = crate::profile::distribution::histogram(&[1.0, 1.0, 2.0, 3.0], 1.0, 3.0);
        assert_eq!(h.max_count(), 2);
    }

    #[test]
    fn histogram_empty_max_count() {
        let h = crate::profile::distribution::histogram(&[], 0.0, 0.0);
        assert_eq!(h.nbins(), 0);
        assert_eq!(h.max_count(), 0);
    }

    #[test]
    fn top_values_cap() {
        // Create many unique categorical values (non-numeric strings)
        let cells: Vec<String> = (0..20).map(|i| format!("val_{}", i)).collect();
        let p = ColumnProfile::from_strings("x".to_string(), &cells);
        let c = p.categorical.as_ref().unwrap();
        assert_eq!(c.unique, 20);
        assert!(c.top_values.len() <= 8); // TOP_VALUES_CAP
    }
}
