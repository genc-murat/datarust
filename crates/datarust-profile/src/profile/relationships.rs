//! Pairwise relationship statistics: Pearson correlation, Cramér's V, and point-biserial correlation.

use std::collections::{HashMap, HashSet};
use crate::infer;

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
    pub fn compute(
        numeric_cols: &[(&str, &[f64])],
        categorical_cols: &[(&str, &[String])],
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

/// Computes Cramér's V matrix for categorical columns.
fn compute_cramers_v(cols: &[(&str, &[String])]) -> Option<CorrelationMatrix> {
    if cols.len() < 2 {
        return None;
    }
    let n_rows = cols[0].1.len();
    if n_rows == 0 {
        return None;
    }

    let p = cols.len();
    let labels: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
    let mut values = vec![vec![0.0; p]; p];

    for i in 0..p {
        values[i][i] = 1.0;
        for j in (i + 1)..p {
            let v = calculate_cramers_v_pair(cols[i].1, cols[j].1);
            values[i][j] = v;
            values[j][i] = v;
        }
    }

    Some(CorrelationMatrix { labels, values })
}

/// Computes Cramér's V for two categorical columns.
fn calculate_cramers_v_pair(col_a: &[String], col_b: &[String]) -> f64 {
    let n_rows = col_a.len().min(col_b.len());
    if n_rows == 0 {
        return 0.0;
    }

    // Build contingency table ignoring missing values in either column
    let mut counts: HashMap<(&str, &str), usize> = HashMap::new();
    let mut row_levels: HashSet<&str> = HashSet::new();
    let mut col_levels: HashSet<&str> = HashSet::new();
    let mut total_valid = 0usize;

    for i in 0..n_rows {
        let cell_a = &col_a[i];
        let cell_b = &col_b[i];
        if infer::is_missing(cell_a) || infer::is_missing(cell_b) {
            continue;
        }
        let a_str = cell_a.trim();
        let b_str = cell_b.trim();
        row_levels.insert(a_str);
        col_levels.insert(b_str);
        *counts.entry((a_str, b_str)).or_insert(0) += 1;
        total_valid += 1;
    }

    let r = row_levels.len();
    let c = col_levels.len();

    if total_valid == 0 || r <= 1 || c <= 1 {
        return 0.0;
    }

    // Row totals and Col totals
    let mut row_totals: HashMap<&str, usize> = HashMap::new();
    let mut col_totals: HashMap<&str, usize> = HashMap::new();

    for (&(r_val, c_val), &cnt) in &counts {
        *row_totals.entry(r_val).or_insert(0) += cnt;
        *col_totals.entry(c_val).or_insert(0) += cnt;
    }

    // Chi-squared calculation
    let mut chi2 = 0.0;
    let n_f64 = total_valid as f64;

    for &r_val in &row_levels {
        let r_tot = *row_totals.get(r_val).unwrap_or(&0) as f64;
        for &c_val in &col_levels {
            let c_tot = *col_totals.get(c_val).unwrap_or(&0) as f64;
            let expected = (r_tot * c_tot) / n_f64;
            if expected > 0.0 {
                let observed = *counts.get(&(r_val, c_val)).unwrap_or(&0) as f64;
                let diff = observed - expected;
                chi2 += (diff * diff) / expected;
            }
        }
    }

    let min_dim = (r - 1).min(c - 1) as f64;
    if min_dim == 0.0 {
        0.0
    } else {
        let v = (chi2 / (n_f64 * min_dim)).sqrt();
        if v.is_nan() {
            0.0
        } else {
            v.min(1.0)
        }
    }
}

/// Computes point-biserial correlation between binary categorical columns and numeric columns.
fn compute_point_biserial(
    numeric_cols: &[(&str, &[f64])],
    categorical_cols: &[(&str, &[String])],
) -> Vec<PointBiserialEntry> {
    let mut entries = Vec::new();

    for (cat_name, cat_values) in categorical_cols {
        // Collect unique non-missing values in categorical column
        let mut unique_vals: Vec<&str> = Vec::new();
        for val in *cat_values {
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

    let r = ((m1 - m0) / s_y)
        * ((n0 * n1) as f64 / (total_n * (total_n - 1)) as f64).sqrt();

    if r.is_nan() {
        None
    } else {
        Some(r.clamp(-1.0, 1.0))
    }
}
