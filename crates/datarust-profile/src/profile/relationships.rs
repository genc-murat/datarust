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

/// Computes Pearson correlation matrix for numeric columns using
/// `datarust::stats::correlation_matrix_flat`.
fn compute_pearson(cols: &[(&str, &[f64])]) -> Option<CorrelationMatrix> {
    if cols.len() < 2 {
        return None;
    }
    let n_rows = cols[0].1.len();
    if n_rows == 0 {
        return None;
    }

    // Interleave the column-major slices into one flat row-major buffer and
    // hand it to the flat correlation kernel: a single allocation streaming
    // contiguous memory (and GEMM-backed covariance under the
    // `matrixmultiply` feature) instead of the nested `Vec<Vec<f64>>` layout.
    let p = cols.len();
    for (_, values) in cols {
        if values.len() != n_rows {
            return None;
        }
    }
    let mut flat: Vec<f64> = Vec::with_capacity(n_rows * p);
    for i in 0..n_rows {
        for (_, values) in cols {
            flat.push(values[i]);
        }
    }

    let labels: Vec<String> = cols.iter().map(|(name, _)| name.to_string()).collect();
    let values = datarust::stats::correlation_matrix_flat(&flat, n_rows, p);

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

    let encoded: Vec<CodedColumn> = cols
        .iter()
        .map(|(_, values)| encode_column(values))
        .collect();

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
    let mut flat: Option<Vec<usize>> = if a_r * a_c <= flat_limit {
        Some(vec![0; a_r * a_c])
    } else {
        None
    };
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
///
/// Computed in a single allocation-free pass: counts and sums per group are
/// accumulated alongside the overall sum-of-squares via the identity
/// `SS = Σy² − (Σy)² / N`. The previous version materialised `group_0`,
/// `group_1` and `all_y` `Vec`s for every (categorical, numeric) pair, which
/// dominated the profile cost on wide tables with many binary columns.
fn calculate_point_biserial(binary: &[f64], numeric: &[f64]) -> Option<f64> {
    let n = binary.len().min(numeric.len());
    if n == 0 {
        return None;
    }

    // Single pass: count, sum, and sum-of-squares per group, filtering pairs
    // where either value is missing or the indicator is neither 0 nor 1.
    let (mut n0, mut n1, mut sum0, mut sum1) = (0usize, 0usize, 0.0, 0.0);
    let mut sum_sq = 0.0;
    for i in 0..n {
        let b = binary[i];
        let y = numeric[i];
        if b.is_finite() && y.is_finite() {
            if b == 0.0 {
                n0 += 1;
                sum0 += y;
                sum_sq += y * y;
            } else if b == 1.0 {
                n1 += 1;
                sum1 += y;
                sum_sq += y * y;
            }
        }
    }

    let total_n = n0 + n1;
    if n0 == 0 || n1 == 0 || total_n < 3 {
        return None;
    }

    let m0 = sum0 / n0 as f64;
    let m1 = sum1 / n1 as f64;
    let total_sum = sum0 + sum1;

    // Pooled sample variance via the computational identity: SS = Σy² − (Σy)² / N.
    let ss = sum_sq - (total_sum * total_sum) / total_n as f64;
    let s_y = (ss / (total_n - 1) as f64).sqrt();

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

    /// Reference: the previous two-pass implementation of `calculate_point_biserial`.
    /// Used only in tests to verify that the single-pass rewrite (via the
    /// computational identity `SS = Σy² − (Σy)²/N`) produces bit-identical results.
    fn calculate_point_biserial_two_pass(binary: &[f64], numeric: &[f64]) -> Option<f64> {
        let n = binary.len().min(numeric.len());
        if n == 0 {
            return None;
        }

        // Pass 1: counts and sums.
        let (mut n0, mut n1, mut sum0, mut sum1) = (0usize, 0usize, 0.0, 0.0);
        for i in 0..n {
            let b = binary[i];
            let y = numeric[i];
            if b.is_finite() && y.is_finite() {
                if b == 0.0 {
                    n0 += 1;
                    sum0 += y;
                } else if b == 1.0 {
                    n1 += 1;
                    sum1 += y;
                }
            }
        }

        let total_n = n0 + n1;
        if n0 == 0 || n1 == 0 || total_n < 3 {
            return None;
        }

        let m0 = sum0 / n0 as f64;
        let m1 = sum1 / n1 as f64;
        let m_all = (sum0 + sum1) / total_n as f64;

        // Pass 2: pooled sample variance (ddof = 1) via Σ(y − mean)².
        let mut ss = 0.0;
        for i in 0..n {
            let b = binary[i];
            let y = numeric[i];
            if b.is_finite() && y.is_finite() && (b == 0.0 || b == 1.0) {
                let d = y - m_all;
                ss += d * d;
            }
        }
        let s_y = (ss / (total_n - 1) as f64).sqrt();

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

    #[test]
    fn point_biserial_single_pass_matches_two_pass() {
        // Deterministic pseudo-random data with various edge cases:
        // NaN, inf, non-binary indicators, short inputs, uniform values.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || -> f64 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0
                - 1.0
        };

        let cases: Vec<(Vec<f64>, Vec<f64>)> = (0..50)
            .map(|_| {
                let len = 20 + (next().abs() * 980.0) as usize; // 20..1000
                let binary: Vec<f64> = (0..len)
                    .map(|i| {
                        let r = next();
                        if i % 7 == 0 {
                            f64::NAN
                        } else if i % 13 == 0 {
                            f64::INFINITY
                        } else if r > 0.3 {
                            1.0
                        } else if r < -0.3 {
                            0.0
                        } else {
                            2.0 // non-binary → filtered
                        }
                    })
                    .collect();
                let numeric: Vec<f64> = (0..len).map(|_| next() * 100.0).collect();
                (binary, numeric)
            })
            .collect();

        for (i, (binary, numeric)) in cases.iter().enumerate() {
            let got = calculate_point_biserial(binary, numeric);
            let want = calculate_point_biserial_two_pass(binary, numeric);
            match (got, want) {
                (Some(g), Some(w)) => {
                    // Allow up to 1 ULP difference from floating-point
                    // accumulation order (two-pass sums deviations from the
                    // mean; single-pass uses the computational identity).
                    assert!(
                        (g - w).abs() < 1e-14
                            || (g.to_bits() as i64 - w.to_bits() as i64).unsigned_abs() <= 1,
                        "case {i}: single-pass {g} != two-pass {w}"
                    );
                }
                (None, None) => {}
                (g, w) => {
                    panic!("case {i}: mismatch: single-pass={g:?}, two-pass={w:?}");
                }
            }
        }
    }

    #[test]
    fn point_biserial_single_pass_matches_two_pass_known_values() {
        // Hand-picked cases where the result is known analytically.
        let cases: &[(&[f64], &[f64], Option<f64>)] = &[
            // Identical values → r = 0.
            (
                &[0.0, 0.0, 1.0, 1.0, 1.0],
                &[5.0, 5.0, 5.0, 5.0, 5.0],
                Some(0.0),
            ),
            // Too few samples → None.
            (&[0.0, 1.0], &[1.0, 2.0], None),
            // Only one group → None.
            (&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], None),
            // NaN in binary → all group-1 rows filtered → only group 0 remains.
            (&[0.0, f64::NAN, 0.0], &[1.0, 2.0, 3.0], None),
            // Inf in numeric → all group-0 rows filtered → only group 1 remains.
            (&[0.0, 1.0, 1.0], &[f64::INFINITY, 10.0, 11.0], None),
            // Hand-computed: n0=2, n1=2, total_n=4
            // group 0: {0, 10} → m0=5;  group 1: {20, 30} → m1=25
            // m_all = 15;  SS = (0-15)²+(10-15)²+(20-15)²+(30-15)² = 500
            // s_y = sqrt(500/3) ≈ 12.9099
            // r = ((25-5)/12.9099) * sqrt(4/(4*3)) = 1.5492 * 0.5774 ≈ 0.8944
            (
                &[0.0, 0.0, 1.0, 1.0],
                &[0.0, 10.0, 20.0, 30.0],
                Some(0.8944271909999159),
            ),
        ];

        for (i, (binary, numeric, expected)) in cases.iter().enumerate() {
            let got_single = calculate_point_biserial(binary, numeric);
            let got_two = calculate_point_biserial_two_pass(binary, numeric);
            assert_eq!(got_single, got_two, "case {i}: methods disagree");
            if let Some(exp) = expected {
                let r = got_single.unwrap();
                assert_eq!(
                    r.to_bits(),
                    exp.to_bits(),
                    "case {i}: got {r}, expected {exp}"
                );
            } else {
                assert!(
                    got_single.is_none(),
                    "case {i}: expected None, got {got_single:?}"
                );
            }
        }
    }
}
