//! Pairwise relationship statistics: Pearson correlation, Cramér's V, and point-biserial correlation.

use crate::infer;
use std::collections::HashMap;

/// A square symmetric correlation or association matrix between named variables.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CorrelationMatrix {
    /// Variable names corresponding to rows and columns.
    pub labels: Vec<String>,
    /// Symmetric `p × p` matrix of values.
    pub values: Vec<Vec<f64>>,
}

/// A point-biserial correlation entry between a binary categorical column and a numeric column.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PointBiserialEntry {
    /// Name of the binary categorical column.
    pub categorical: String,
    /// Name of the continuous numeric column.
    pub numeric: String,
    /// Point-biserial correlation coefficient `r` in `[-1.0, 1.0]`.
    pub correlation: f64,
}

/// Container for all pairwise column relationships in a dataset.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Relationships {
    /// Pearson correlation matrix over numeric columns, if there are 2 or more numeric columns.
    pub pearson: Option<CorrelationMatrix>,
    /// Cramér's V matrix over categorical columns, if there are 2 or more categorical columns.
    pub cramers_v: Option<CorrelationMatrix>,
    /// Point-biserial correlations between binary categorical columns and numeric columns.
    pub point_biserial: Vec<PointBiserialEntry>,
}

impl Relationships {
    /// Computes relationships across numeric columns and string/categorical columns.
    pub fn compute<T: AsRef<str>>(
        numeric_cols: &[(&str, &[f64])],
        categorical_cols: &[(&str, &[T])],
    ) -> Option<Self> {
        let pearson = compute_pearson(numeric_cols);
        let cramers_v = compute_cramers_v(categorical_cols);
        let point_biserial = compute_point_biserial(numeric_cols, categorical_cols);

        if pearson.is_none() && cramers_v.is_none() && point_biserial.is_empty() {
            None
        } else {
            Some(Relationships {
                pearson,
                cramers_v,
                point_biserial,
            })
        }
    }
}

/// Computes Pearson correlation matrix for numeric columns using `datarust::stats::correlation_matrix`.
fn compute_pearson(cols: &[(&str, &[f64])]) -> Option<CorrelationMatrix> {
    if cols.len() < 2 {
        return None;
    }
    let n_rows = cols[0].1.len();
    if n_rows == 0 {
        return None;
    }

    // Build row-oriented data matrix for datarust::stats::correlation_matrix
    let mut rows_data: Vec<Vec<f64>> = vec![vec![0.0; cols.len()]; n_rows];
    for (j, (_, values)) in cols.iter().enumerate() {
        if values.len() != n_rows {
            return None;
        }
        for i in 0..n_rows {
            rows_data[i][j] = values[i];
        }
    }

    let labels: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
    let values = datarust::stats::correlation_matrix(&rows_data);

    Some(CorrelationMatrix { labels, values })
}

/// Encodes a categorical column into compact integer level codes so that
/// pairwise Cramér's V can be computed with `usize` keys instead of hashing
/// string tuples for every row of every column pair.
struct CodedColumn<'a> {
    /// Distinct non-missing trimmed levels, in first-seen order.
    levels: Vec<&'a str>,
    /// Per-row level code; `usize::MAX` marks a missing/empty cell.
    codes: Vec<usize>,
}

/// Builds a [`CodedColumn`] from raw string cells in one pass.
fn encode_column<'a, T: AsRef<str> + 'a>(cells: &'a [T]) -> CodedColumn<'a> {
    let mut map: HashMap<&'a str, usize> = HashMap::new();
    let mut levels: Vec<&'a str> = Vec::new();
    let mut codes = Vec::with_capacity(cells.len());
    for cell in cells {
        let cell = cell.as_ref();
        if infer::is_missing(cell) {
            codes.push(usize::MAX);
            continue;
        }
        let trimmed = cell.trim();
        let next = levels.len();
        let code = *map.entry(trimmed).or_insert_with(|| {
            levels.push(trimmed);
            next
        });
        codes.push(code);
    }
    CodedColumn { levels, codes }
}

/// Computes Cramér's V matrix for categorical columns.
fn compute_cramers_v<T: AsRef<str>>(cols: &[(&str, &[T])]) -> Option<CorrelationMatrix> {
    if cols.len() < 2 {
        return None;
    }
    let n_rows = cols[0].1.len();
    if n_rows == 0 {
        return None;
    }

    let encoded: Vec<CodedColumn> = cols.iter().map(|(_, values)| encode_column(values)).collect();

    let p = cols.len();
    let labels: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
    let mut values = vec![vec![0.0; p]; p];

    for i in 0..p {
        values[i][i] = 1.0;
        for j in (i + 1)..p {
            let v = cramers_v_from_codes(&encoded[i], &encoded[j]);
            values[i][j] = v;
            values[j][i] = v;
        }
    }

    Some(CorrelationMatrix { labels, values })
}

/// Computes Cramér's V between two pre-encoded columns.
///
/// Only rows where both columns are present contribute. The chi-squared
/// statistic is computed with the `Σ O²/E − N` identity, which sums over every
/// cell of the contingency table without materialising the empty ones. Small
/// tables (typical for categorical profiling) use a flat array; large tables
/// fall back to a hash map keyed by `usize`.
fn cramers_v_from_codes(a: &CodedColumn, b: &CodedColumn) -> f64 {
    let n = a.codes.len().min(b.codes.len());
    let a_r = a.levels.len();
    let a_c = b.levels.len();
    let mut total = 0usize;
    let mut row_totals = vec![0usize; a_r];
    let mut col_totals = vec![0usize; a_c];

    let flat_limit = 1 << 16;
    let mut flat: Option<Vec<usize>> =
        if a_r * a_c <= flat_limit { Some(vec![0; a_r * a_c]) } else { None };
    let mut counts: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let ri = a.codes[i];
        let ci = b.codes[i];
        if ri == usize::MAX || ci == usize::MAX {
            continue;
        }
        row_totals[ri] += 1;
        col_totals[ci] += 1;
        let key = ri * a_c + ci;
        match flat.as_mut() {
            Some(t) => t[key] += 1,
            None => *counts.entry(key).or_insert(0) += 1,
        }
        total += 1;
    }

    if total == 0 {
        return 0.0;
    }
    // Distinct levels restricted to rows where both columns are present.
    let r = row_totals.iter().filter(|&&t| t > 0).count();
    let c = col_totals.iter().filter(|&&t| t > 0).count();
    if r <= 1 || c <= 1 {
        return 0.0;
    }

    let n_f = total as f64;
    let mut chi2 = 0.0;
    match flat {
        Some(t) => {
            for (key, &obs) in t.iter().enumerate() {
                if obs == 0 {
                    continue;
                }
                let ri = key / a_c;
                let ci = key % a_c;
                let expected = row_totals[ri] as f64 * col_totals[ci] as f64 / n_f;
                if expected > 0.0 {
                    let obs_f = obs as f64;
                    chi2 += obs_f * obs_f / expected;
                }
            }
        }
        None => {
            for (&key, &obs) in &counts {
                let ri = key / a_c;
                let ci = key % a_c;
                let expected = row_totals[ri] as f64 * col_totals[ci] as f64 / n_f;
                if expected > 0.0 {
                    let obs_f = obs as f64;
                    chi2 += obs_f * obs_f / expected;
                }
            }
        }
    }
    chi2 -= n_f;

    let min_dim = (r - 1).min(c - 1) as f64;
    if min_dim == 0.0 {
        0.0
    } else {
        let v = (chi2 / (n_f * min_dim)).sqrt();
        if v.is_nan() {
            0.0
        } else {
            v.min(1.0)
        }
    }
}

/// Computes point-biserial correlation between binary categorical columns and numeric columns.
fn compute_point_biserial<T: AsRef<str>>(
    numeric_cols: &[(&str, &[f64])],
    categorical_cols: &[(&str, &[T])],
) -> Vec<PointBiserialEntry> {
    let mut entries = Vec::new();

    for (cat_name, cat_values) in categorical_cols {
        // Collect unique non-missing values in categorical column
        let mut unique_vals: Vec<&str> = Vec::new();
        for val in *cat_values {
            let val = val.as_ref();
            if infer::is_missing(val) {
                continue;
            }
            let trimmed = val.trim();
            if !unique_vals.contains(&trimmed) {
                unique_vals.push(trimmed);
            }
            if unique_vals.len() > 2 {
                break;
            }
        }

        // Must be exactly binary
        if unique_vals.len() != 2 {
            continue;
        }

        let val_0 = unique_vals[0];

        // Map cat values to 0.0 and 1.0 (or NaN if missing)
        let binary_encoded: Vec<f64> = cat_values
            .iter()
            .map(|cell| {
                let cell = cell.as_ref();
                if infer::is_missing(cell) {
                    f64::NAN
                } else if cell.trim() == val_0 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();

        for (num_name, num_values) in numeric_cols {
            if let Some(r) = calculate_point_biserial(&binary_encoded, num_values) {
                entries.push(PointBiserialEntry {
                    categorical: cat_name.to_string(),
                    numeric: num_name.to_string(),
                    correlation: r,
                });
            }
        }
    }

    entries
}

/// Calculates point-biserial correlation between a 0/1 indicator vector and a numeric vector.
fn calculate_point_biserial(binary: &[f64], numeric: &[f64]) -> Option<f64> {
    let n = binary.len().min(numeric.len());
    if n == 0 {
        return None;
    }

    // Filter out pairs where either value is NaN or missing
    let mut group_0 = Vec::new();
    let mut group_1 = Vec::new();

    for i in 0..n {
        let b = binary[i];
        let y = numeric[i];
        if b.is_finite() && y.is_finite() {
            if b == 0.0 {
                group_0.push(y);
            } else if b == 1.0 {
                group_1.push(y);
            }
        }
    }

    let n0 = group_0.len();
    let n1 = group_1.len();
    let total_n = n0 + n1;

    if n0 == 0 || n1 == 0 || total_n < 3 {
        return None;
    }

    let m0 = datarust::stats::mean(&group_0);
    let m1 = datarust::stats::mean(&group_1);

    let all_y: Vec<f64> = group_0.iter().chain(group_1.iter()).copied().collect();
    let s_y = datarust::stats::std(&all_y, 1);

    if s_y == 0.0 {
        return Some(0.0);
    }

    let r = ((m1 - m0) / s_y) * ((n0 * n1) as f64 / (total_n * (total_n - 1)) as f64).sqrt();

    if r.is_nan() {
        None
    } else {
        Some(r.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_pearson_returns_none_for_less_than_two_cols() {
        let cols = vec![("a", &[1.0, 2.0][..])];
        assert!(compute_pearson(&cols).is_none());
    }

    #[test]
    fn compute_pearson_returns_none_for_empty() {
        let cols = vec![("a", &[][..]), ("b", &[][..])];
        assert!(compute_pearson(&cols).is_none());
    }

    #[test]
    fn compute_pearson_perfect_correlation() {
        let cols = vec![("x", &[1.0, 2.0, 3.0][..]), ("y", &[2.0, 4.0, 6.0][..])];
        let mat = compute_pearson(&cols).unwrap();
        assert_eq!(mat.labels, vec!["x", "y"]);
        assert!((mat.values[0][1] - 1.0).abs() < 1e-6);
        assert!((mat.values[1][0] - 1.0).abs() < 1e-6);
        assert!((mat.values[0][0] - 1.0).abs() < 1e-6);
        assert!((mat.values[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_pearson_negative_correlation() {
        let cols = vec![("x", &[1.0, 2.0, 3.0][..]), ("y", &[3.0, 2.0, 1.0][..])];
        let mat = compute_pearson(&cols).unwrap();
        assert!((mat.values[0][1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_pearson_uncorrelated() {
        let cols = vec![("x", &[1.0, 2.0, 3.0][..]), ("y", &[2.0, 1.0, 3.0][..])];
        let mat = compute_pearson(&cols).unwrap();
        assert!(mat.values[0][1].abs() < 1.0);
    }

    #[test]
    fn compute_cramers_v_returns_none_for_less_than_two_cols() {
        let a = ["x".to_string()];
        let cols = vec![("a", &a[..])];
        assert!(compute_cramers_v(&cols).is_none());
    }

    #[test]
    fn compute_cramers_v_perfect_association() {
        let a = [
            "x".to_string(),
            "x".to_string(),
            "y".to_string(),
            "y".to_string(),
        ];
        let b = [
            "p".to_string(),
            "p".to_string(),
            "q".to_string(),
            "q".to_string(),
        ];
        let cols = vec![("a", &a[..]), ("b", &b[..])];
        let mat = compute_cramers_v(&cols).unwrap();
        assert_eq!(mat.labels, vec!["a", "b"]);
        assert!((mat.values[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_cramers_v_with_missing() {
        let a = ["x".to_string(), "NA".to_string(), "y".to_string()];
        let b = ["p".to_string(), "q".to_string(), "NA".to_string()];
        let cols = vec![("a", &a[..]), ("b", &b[..])];
        let mat = compute_cramers_v(&cols).unwrap();
        // Should still compute with valid pairs
        assert!(mat.values[0][1] >= 0.0);
    }

    #[test]
    fn compute_cramers_v_single_level() {
        let a = ["x".to_string(), "x".to_string(), "x".to_string()];
        let b = ["p".to_string(), "q".to_string(), "r".to_string()];
        let cols = vec![("a", &a[..]), ("b", &b[..])];
        let mat = compute_cramers_v(&cols).unwrap();
        // One column has only one level -> V = 0
        assert_eq!(mat.values[0][1], 0.0);
    }

    #[test]
    fn compute_point_biserial_binary_categorical() {
        let numeric = vec![("num", &[1.0, 1.0, 5.0, 5.0][..])];
        let cat = [
            "low".to_string(),
            "low".to_string(),
            "high".to_string(),
            "high".to_string(),
        ];
        let categorical = vec![("cat", &cat[..])];
        let entries = compute_point_biserial(&numeric, &categorical);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].categorical, "cat");
        assert_eq!(entries[0].numeric, "num");
        assert!((entries[0].correlation.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_point_biserial_non_binary_skipped() {
        let numeric = vec![("num", &[1.0, 2.0, 3.0][..])];
        let cat = ["a".to_string(), "b".to_string(), "c".to_string()];
        let categorical = vec![("cat", &cat[..])];
        let entries = compute_point_biserial(&numeric, &categorical);
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn compute_point_biserial_with_missing() {
        let numeric = vec![("num", &[1.0, f64::NAN, 5.0, 5.0][..])];
        let cat = [
            "low".to_string(),
            "low".to_string(),
            "high".to_string(),
            "high".to_string(),
        ];
        let categorical = vec![("cat", &cat[..])];
        let entries = compute_point_biserial(&numeric, &categorical);
        // Should still work, missing in numeric is filtered
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn relationships_compute_all() {
        let numeric = vec![
            ("x", &[1.0, 2.0, 3.0, 4.0][..]),
            ("y", &[2.0, 4.0, 6.0, 8.0][..]),
        ];
        let cat = [
            "p".to_string(),
            "p".to_string(),
            "q".to_string(),
            "q".to_string(),
        ];
        let categorical = vec![("a", &cat[..])];
        let rels = Relationships::compute(&numeric, &categorical).unwrap();
        assert!(rels.pearson.is_some());
        assert!(rels.cramers_v.is_none()); // only one categorical
        assert!(!rels.point_biserial.is_empty());
    }

    #[test]
    fn relationships_compute_only_categorical() {
        let numeric = vec![];
        let a = [
            "p".to_string(),
            "p".to_string(),
            "q".to_string(),
            "q".to_string(),
        ];
        let b = [
            "x".to_string(),
            "x".to_string(),
            "y".to_string(),
            "y".to_string(),
        ];
        let categorical = vec![("a", &a[..]), ("b", &b[..])];
        let rels = Relationships::compute(&numeric, &categorical).unwrap();
        assert!(rels.pearson.is_none());
        assert!(rels.cramers_v.is_some());
    }
}
