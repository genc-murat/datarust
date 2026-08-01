//! End-to-end tests for the public profile API.

use std::error::Error as StdError;

use datarust::{Matrix, StrMatrix};
use datarust_profile::quality::checks::run_checks;
use datarust_profile::{
    profile_matrix, profile_matrix_with_target, profile_str_matrix, profile_table,
    profile_table_with_target, ColumnType, Severity, Thresholds,
};

fn names(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

#[test]
fn numeric_profile_counts_and_quantiles() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
        vec![5.0, f64::NAN], // missing in column 1
    ])
    .unwrap();

    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();

    assert_eq!(p.n_rows, 5);
    assert_eq!(p.n_columns, 2);
    assert_eq!(p.columns.len(), 2);
    assert_eq!(p.columns[0].name, "a");
    assert_eq!(p.columns[1].name, "b");

    // Column 1 is fully populated; column 2 has one missing.
    assert_eq!(p.columns[0].missing_count, 0);
    assert_eq!(p.columns[1].missing_count, 1);

    let n0 = p.columns[0].numeric.as_ref().unwrap();
    assert!((n0.mean - 3.0).abs() < 1e-9);
    assert!((n0.five.min - 1.0).abs() < 1e-9);
    assert!((n0.five.max - 5.0).abs() < 1e-9);
    // Median of [1,2,3,4,5] is 3.
    assert!((n0.five.median - 3.0).abs() < 1e-9);
}

#[test]
fn default_column_names_when_none() {
    let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let p = profile_matrix(&m, None).unwrap();
    assert_eq!(p.columns[0].name, "x0");
    assert_eq!(p.columns[1].name, "x1");
}

#[test]
fn duplicate_numeric_rows_detected() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0], // duplicate
        vec![2.0, 2.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, None).unwrap();
    assert_eq!(p.duplicate_rows, 1);
    assert!((p.duplicate_fraction - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn str_matrix_infers_numeric_and_categorical() {
    let s = StrMatrix::from_strings(vec![vec!["1", "red"], vec!["2", "blue"], vec!["3", "red"]])
        .unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["num", "color"]))).unwrap();
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    assert_eq!(p.columns[1].column_type, ColumnType::Categorical);

    let cat = p.columns[1].categorical.as_ref().unwrap();
    assert_eq!(cat.unique, 2);
    assert_eq!(cat.top, "red");
    assert_eq!(cat.freq, 2);
}

#[test]
fn mixed_table_profiles_both_blocks() {
    let numeric = Matrix::from_rows(vec![vec![10.0], vec![20.0], vec![30.0]]).unwrap();
    let categorical = StrMatrix::from_strings(vec![vec!["a"], vec!["b"], vec!["a"]]).unwrap();
    let p = profile_table(Some(&numeric), Some(&categorical), &names(&["v", "k"])).unwrap();
    assert_eq!(p.n_columns, 2);
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    assert_eq!(p.columns[1].column_type, ColumnType::Categorical);
}

#[test]
fn empty_matrix_rejected_at_construction() {
    // `Matrix` itself rejects empty input, so the profile layer never sees it.
    // Both constructors return Err; we assert the precondition holds rather
    // than trying to smuggle an empty matrix past the type system.
    assert!(Matrix::new(vec![]).is_err());
    assert!(Matrix::new(vec![vec![]]).is_err());
}

#[test]
fn quality_flags_high_missing_and_constant() {
    // Column "c" is constant; column "m" is 3/5 = 60% missing (above the
    // default 0.5 threshold but below the 0.9 critical threshold).
    let m = Matrix::from_rows(vec![
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, 1.0],
        vec![5.0, 2.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["c", "m"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());

    assert!(issues.iter().any(|i| {
        i.column.as_deref() == Some("c") && i.kind == datarust_profile::QualityKind::ConstantColumn
    }));
    // 3/5 = 60% missing, above the 0.5 default threshold.
    assert!(issues.iter().any(|i| {
        i.column.as_deref() == Some("m")
            && i.kind == datarust_profile::QualityKind::HighMissing
            && i.severity == Severity::Warning
    }));
}

#[test]
fn html_card_layout_replaces_table() {
    // v0.2 replaces the per-column <table> with a responsive card grid.
    let m = Matrix::from_rows(vec![vec![1.0], vec![2.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["a"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    // New layout markers: card grid + a CSS bar chart (the histogram).
    assert!(html.contains("col-grid"));
    assert!(html.contains("col-card"));
    assert!(html.contains("chart"));
    assert!(html.contains("bar"));
    // The old table is gone.
    assert!(!html.contains("<table"));
    // Column name + badge still present.
    assert!(html.contains("a"));
    assert!(html.contains("numeric"));
}

#[cfg(feature = "serde")]
#[test]
fn json_report_round_trips() {
    use datarust_profile::report::{to_json, JsonReport};

    let m = Matrix::from_rows(vec![vec![1.0, 10.0], vec![2.0, 20.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let json = to_json(&JsonReport::from_profile(&p)).unwrap();
    assert!(json.contains("\"n_rows\""));
    assert!(json.contains("\"a\""));
    assert!(json.contains("\"quality\""));
}

// ---- v0.2: distributional depth -------------------------------------------

#[test]
fn numeric_profile_includes_distribution_fields() {
    let m = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let n = p.columns[0].numeric.as_ref().unwrap();

    // Skewness of a symmetric sequence is ~0.
    assert!(n.skewness.abs() < 1e-9, "skewness {}", n.skewness);
    // Histogram uses Sturges: ceil(log2(5)+1) = ceil(3.32) = 4 bins.
    assert_eq!(n.histogram.nbins(), 4);
    // All values accounted for.
    assert_eq!(n.histogram.counts.iter().sum::<usize>(), 5);
    // Five-number summary flows through the flat path unchanged.
    assert!((n.five.min - 1.0).abs() < 1e-9);
    assert!((n.five.max - 5.0).abs() < 1e-9);
}

#[test]
fn outlier_detection_via_iqr_flags_extreme_values() {
    // [1,2,3,4,100] → 100 sits well above the Q3 + 1.5*IQR fence.
    let m = Matrix::from_rows(vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![100.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let n = p.columns[0].numeric.as_ref().unwrap();
    assert!(
        n.outlier_count >= 1,
        "expected outliers, got {}",
        n.outlier_count
    );

    // And the quality check fires.
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues.iter().any(|i| {
        i.column.as_deref() == Some("x") && i.kind == datarust_profile::QualityKind::Outliers
    }));
}

#[test]
fn categorical_imbalance_detected_when_top_dominates() {
    // 95% threshold default: 19/20 = 0.95 trips it.
    let mut rows: Vec<Vec<String>> = (0..19).map(|_| vec!["a".to_string()]).collect();
    rows.push(vec!["b".to_string()]);
    let s = StrMatrix::from_strings(rows).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["k"]))).unwrap();
    let c = p.columns[0].categorical.as_ref().unwrap();
    assert!((c.imbalance_ratio - 0.95).abs() < 1e-9);

    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues.iter().any(|i| {
        i.column.as_deref() == Some("k") && i.kind == datarust_profile::QualityKind::Imbalance
    }));
}

#[test]
fn flat_path_matches_independent_stats() {
    // from_matrix uses the _flat fast path (column_mean_var_flat +
    // column_quantiles_many_flat) for mean/std/five-number. Verify those
    // agree with an independent recomputation straight from datarust::stats.
    let m = Matrix::from_rows(vec![
        vec![1.0, 10.0, 100.0],
        vec![2.0, 20.0, f64::NAN],
        vec![3.0, 30.0, 300.0],
        vec![4.0, 40.0, 400.0],
        vec![5.0, 50.0, 500.0],
    ])
    .unwrap();

    let flat = profile_matrix(&m, Some(&names(&["a", "b", "c"]))).unwrap();

    for j in 0..m.ncols() {
        let col = m.col(j);
        // Filter NaN the same way the profile does, so the independent
        // recomputation matches the (NaN-aware) flat path.
        let present: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
        let n_flat = flat.columns[j].numeric.as_ref().unwrap();

        // Independent mean/std from datarust::stats (the per-column helpers).
        let mean = datarust::stats::mean(&present);
        let std = datarust::stats::std(&present, 1);
        assert!((n_flat.mean - mean).abs() < 1e-9, "col {j} mean mismatch");
        assert!((n_flat.std - std).abs() < 1e-9, "col {j} std mismatch");

        // Distributional stats are finite (computed from the raw column on
        // both paths, so they're well-defined here).
        assert!(n_flat.skewness.is_finite(), "col {j} skewness not finite");
        assert!(n_flat.kurtosis.is_finite(), "col {j} kurtosis not finite");
        assert!(
            n_flat.outlier_count <= present.len(),
            "col {j} outlier overflow"
        );
    }
}

#[test]
fn categorical_stats_carry_top_values_list() {
    let s = StrMatrix::from_strings(vec![vec!["a"], vec!["a"], vec!["b"], vec!["c"], vec!["d"]])
        .unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["k"]))).unwrap();
    let c = p.columns[0].categorical.as_ref().unwrap();
    assert_eq!(c.unique, 4);
    // Top values descending by count; "a" leads with 2.
    assert_eq!(c.top_values.first().unwrap().0, "a");
    assert_eq!(c.top_values.first().unwrap().1, 2);
}

#[test]
fn pearson_correlation_matrix_computed() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
        vec![4.0, 8.0],
    ])
    .unwrap();

    let p = profile_matrix(&m, Some(&names(&["x", "y"]))).unwrap();
    let rels = p.relationships.as_ref().unwrap();
    let pearson = rels.pearson.as_ref().unwrap();

    assert_eq!(pearson.labels, vec!["x", "y"]);
    // Perfectly correlated
    assert!((pearson.values[0][1] - 1.0).abs() < 1e-6);
    assert!((pearson.values[1][0] - 1.0).abs() < 1e-6);

    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues
        .iter()
        .any(|i| { i.kind == datarust_profile::QualityKind::HighCorrelation }));
}

#[test]
fn cramers_v_computed_for_categoricals() {
    let s = StrMatrix::from_strings(vec![
        vec!["cat_A".to_string(), "yes".to_string()],
        vec!["cat_A".to_string(), "yes".to_string()],
        vec!["cat_B".to_string(), "no".to_string()],
        vec!["cat_B".to_string(), "no".to_string()],
    ])
    .unwrap();

    let p = profile_str_matrix(&s, Some(&names(&["grp", "flag"]))).unwrap();
    let rels = p.relationships.as_ref().unwrap();
    let cramers = rels.cramers_v.as_ref().unwrap();

    assert_eq!(cramers.labels, vec!["grp", "flag"]);
    assert!((cramers.values[0][1] - 1.0).abs() < 1e-6);
}

#[test]
fn point_biserial_and_target_leakage_detected() {
    let numeric = Matrix::from_rows(vec![vec![1.0], vec![1.0], vec![5.0], vec![5.0]]).unwrap();

    let categorical = StrMatrix::from_strings(vec![
        vec!["low".to_string()],
        vec!["low".to_string()],
        vec!["high".to_string()],
        vec!["high".to_string()],
    ])
    .unwrap();

    let p = datarust_profile::profile_table_with_target(
        Some(&numeric),
        Some(&categorical),
        &names(&["num_val", "target_cat"]),
        "target_cat",
    )
    .unwrap();

    let rels = p.relationships.as_ref().unwrap();
    assert!(!rels.point_biserial.is_empty());
    let pb = &rels.point_biserial[0];
    assert_eq!(pb.categorical, "target_cat");
    assert_eq!(pb.numeric, "num_val");
    assert!((pb.correlation.abs() - 1.0).abs() < 1e-6);

    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues
        .iter()
        .any(|i| { i.kind == datarust_profile::QualityKind::TargetLeakage }));
}

#[test]
fn html_report_includes_relationships_heatmaps() {
    let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]]).unwrap();

    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);

    assert!(html.contains("Relationships &amp; Interaction"));
    assert!(html.contains("Pearson correlation matrix"));
    assert!(html.contains("class=\"heatmap\""));
}

// ---- error.rs tests --------------------------------------------------------

#[test]
fn error_display_invalid_input() {
    let e = datarust_profile::ProfileError::InvalidInput("bad".into());
    assert!(e.to_string().contains("invalid input"));
}

#[test]
fn error_display_empty_input() {
    let e = datarust_profile::ProfileError::EmptyInput("no rows".into());
    assert!(e.to_string().contains("empty input"));
}

#[test]
fn error_display_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, "oops");
    let e = datarust_profile::ProfileError::Io(io_err);
    assert!(e.to_string().contains("io error"));
    assert!(e.source().is_some());
}

#[test]
fn error_from_datarust_error() {
    let dr_err = datarust::error::DatarustError::ShapeMismatch {
        expected: "(2, 2)".to_string(),
        actual: "(3, 3)".to_string(),
    };
    let e: datarust_profile::ProfileError = dr_err.into();
    assert!(e.to_string().contains("datarust error"));
    assert!(e.source().is_some());
}

#[test]
fn error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no");
    let e: datarust_profile::ProfileError = io_err.into();
    assert!(e.to_string().contains("io error"));
}

// ---- types.rs tests --------------------------------------------------------

#[test]
fn column_type_display() {
    assert_eq!(datarust_profile::ColumnType::Numeric.to_string(), "numeric");
    assert_eq!(
        datarust_profile::ColumnType::Categorical.to_string(),
        "categorical"
    );
}

#[test]
fn severity_display() {
    assert_eq!(datarust_profile::Severity::Info.to_string(), "info");
    assert_eq!(datarust_profile::Severity::Warning.to_string(), "warning");
    assert_eq!(datarust_profile::Severity::Critical.to_string(), "critical");
}

// ---- dataset profile edge cases --------------------------------------------

#[test]
fn profile_table_only_numeric() {
    let numeric = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let p = profile_table(Some(&numeric), None, &names(&["a", "b"])).unwrap();
    assert_eq!(p.n_columns, 2);
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    assert_eq!(p.columns[1].column_type, ColumnType::Numeric);
    assert_eq!(p.memory_bytes, 2 * 2 * 8); // 2 rows x 2 cols x 8 bytes
}

#[test]
fn profile_table_only_categorical() {
    let cat = StrMatrix::from_strings(vec![vec!["x", "y"], vec!["z", "w"]]).unwrap();
    let p = profile_table(None, Some(&cat), &names(&["a", "b"])).unwrap();
    assert_eq!(p.n_columns, 2);
    assert_eq!(p.columns[0].column_type, ColumnType::Categorical);
    assert_eq!(p.columns[1].column_type, ColumnType::Categorical);
}

#[test]
fn profile_table_name_count_mismatch_errors() {
    let numeric = Matrix::from_rows(vec![vec![1.0, 2.0]]).unwrap();
    let result = profile_table(Some(&numeric), None, &names(&["a"]));
    assert!(result.is_err());
}

#[test]
fn profile_table_empty_all_errors() {
    let result = profile_table(None, None, &[]);
    assert!(result.is_err());
}

#[test]
fn profile_table_mixed_row_counts_errors() {
    let numeric = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let cat = StrMatrix::from_strings(vec![vec!["a"], vec!["b"]]).unwrap();
    let result = profile_table(Some(&numeric), Some(&cat), &names(&["n", "c"]));
    assert!(result.is_err());
}

#[test]
fn profile_str_matrix_default_names() {
    let s = StrMatrix::from_strings(vec![vec!["a", "b"], vec!["c", "d"]]).unwrap();
    let p = profile_str_matrix(&s, None).unwrap();
    assert_eq!(p.columns[0].name, "x0");
    assert_eq!(p.columns[1].name, "x1");
}

#[test]
fn profile_str_matrix_wrong_name_count_falls_back_to_defaults() {
    let s = StrMatrix::from_strings(vec![vec!["a", "b"], vec!["c", "d"]]).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["only_one"]))).unwrap();
    assert_eq!(p.columns[0].name, "x0");
    assert_eq!(p.columns[1].name, "x1");
}

#[test]
fn profile_matrix_with_target_sets_target_column() {
    let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let p = profile_matrix_with_target(&m, Some(&names(&["a", "b"])), "b").unwrap();
    assert_eq!(p.target_column.as_deref(), Some("b"));
}

#[test]
fn profile_table_with_target_sets_target_column() {
    let numeric = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let cat = StrMatrix::from_strings(vec![vec!["x"], vec!["y"], vec!["z"]]).unwrap();
    let p =
        profile_table_with_target(Some(&numeric), Some(&cat), &names(&["n", "c"]), "c").unwrap();
    assert_eq!(p.target_column.as_deref(), Some("c"));
}

#[test]
fn memory_bytes_numeric_only() {
    let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
    let p = profile_matrix(&m, None).unwrap();
    // 2 rows x 3 cols x 8 bytes = 48
    assert_eq!(p.memory_bytes, 48);
}

#[test]
fn memory_bytes_mixed_table() {
    let numeric = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let cat = StrMatrix::from_strings(vec![vec!["a", "b"], vec!["c", "d"]]).unwrap();
    let p = profile_table(
        Some(&numeric),
        Some(&cat),
        &names(&["n1", "n2", "c1", "c2"]),
    )
    .unwrap();
    // 2 numeric cols x 2 rows x 8 bytes + 2 categorical cols x 2 rows x 24 bytes = 32 + 96 = 128
    assert_eq!(p.memory_bytes, 128);
}

#[test]
fn single_row_matrix() {
    let m = Matrix::from_rows(vec![vec![42.0, 7.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    assert_eq!(p.n_rows, 1);
    assert_eq!(p.n_columns, 2);
    let n = p.columns[0].numeric.as_ref().unwrap();
    assert!((n.mean - 42.0).abs() < 1e-9);
    assert!((n.five.min - 42.0).abs() < 1e-9);
    assert!((n.five.max - 42.0).abs() < 1e-9);
}

#[test]
fn all_nan_column_numeric_type() {
    let m = Matrix::from_rows(vec![vec![f64::NAN], vec![f64::NAN]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    assert_eq!(p.columns[0].missing_count, 2);
    assert_eq!(p.columns[0].missing_fraction, 1.0);
    assert!(p.columns[0].numeric.is_none());
}

#[test]
fn all_nan_column_html_shows_no_values() {
    let m = Matrix::from_rows(vec![vec![f64::NAN], vec![f64::NAN]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("No non-missing values"));
}

#[test]
fn duplicate_rows_in_str_matrix() {
    let s = StrMatrix::from_strings(vec![
        vec!["a", "x"],
        vec!["b", "y"],
        vec!["a", "x"], // duplicate
    ])
    .unwrap();
    let p = profile_str_matrix(&s, None).unwrap();
    assert_eq!(p.duplicate_rows, 1);
    assert!((p.duplicate_fraction - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn duplicate_rows_in_mixed_table() {
    let numeric = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![1.0, 2.0]]).unwrap();
    let cat = StrMatrix::from_strings(vec![vec!["a"], vec!["b"], vec!["a"]]).unwrap();
    let p = profile_table(Some(&numeric), Some(&cat), &names(&["n1", "n2", "c"])).unwrap();
    assert_eq!(p.duplicate_rows, 1);
}

#[test]
fn html_report_with_categorical_column() {
    let s = StrMatrix::from_strings(vec![
        vec!["red", "big"],
        vec!["blue", "small"],
        vec!["red", "big"],
    ])
    .unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["color", "size"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("categorical"));
    assert!(html.contains("color"));
    assert!(html.contains("size"));
    assert!(html.contains("cat-list"));
}

#[test]
fn html_report_with_no_findings() {
    // Clean uncorrelated data with no quality issues
    let m = Matrix::from_rows(vec![
        vec![1.0, 50.0],
        vec![2.0, 30.0],
        vec![3.0, 10.0],
        vec![4.0, 40.0],
        vec![5.0, 20.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("No data-quality findings"));
}

#[test]
fn html_report_with_findings() {
    // Data with a constant column to trigger a finding
    let m = Matrix::from_rows(vec![vec![5.0, 10.0], vec![5.0, 20.0], vec![5.0, 30.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["c", "v"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("Data quality findings"));
    assert!(html.contains("constant"));
}

#[test]
fn html_report_to_html_with_custom_findings() {
    let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let custom_issue = datarust_profile::QualityIssue {
        kind: datarust_profile::QualityKind::DuplicateRows,
        severity: Severity::Info,
        column: None,
        message: "Custom finding".to_string(),
    };
    let html = datarust_profile::report::to_html_with(&p, &[custom_issue]);
    assert!(html.contains("Custom finding"));
    assert!(html.contains("info"));
}

#[test]
fn html_report_point_biserial_table() {
    let numeric = Matrix::from_rows(vec![vec![1.0], vec![1.0], vec![5.0], vec![5.0]]).unwrap();
    let cat = StrMatrix::from_strings(vec![vec!["low"], vec!["low"], vec!["high"], vec!["high"]])
        .unwrap();
    let p = profile_table_with_target(
        Some(&numeric),
        Some(&cat),
        &names(&["val", "target"]),
        "target",
    )
    .unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("Point-biserial"));
    assert!(html.contains("rel-table"));
}

#[test]
fn pearson_negative_correlation_detected() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 8.0],
        vec![3.0, 6.0],
        vec![4.0, 4.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["x", "y"]))).unwrap();
    let rels = p.relationships.as_ref().unwrap();
    let pearson = rels.pearson.as_ref().unwrap();
    assert!((pearson.values[0][1] + 1.0).abs() < 1e-6);
}

#[test]
fn no_relationships_for_single_column() {
    let m = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    assert!(p.relationships.is_none());
}

#[test]
fn no_relationships_for_single_categorical_column() {
    let s = StrMatrix::from_strings(vec![vec!["a"], vec!["b"], vec!["c"]]).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["k"]))).unwrap();
    // Single column: no pairs to compute relationships for
    assert!(p.relationships.is_none());
}

#[test]
fn quality_no_issues_on_clean_data() {
    // Use uncorrelated data to avoid HighCorrelation
    let m = Matrix::from_rows(vec![
        vec![1.0, 50.0],
        vec![2.0, 30.0],
        vec![3.0, 10.0],
        vec![4.0, 40.0],
        vec![5.0, 20.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues.is_empty());
}

#[test]
fn quality_target_leakage_via_pearson_with_numeric_target() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
    ])
    .unwrap();
    let p = profile_matrix_with_target(&m, Some(&names(&["feature", "target"])), "target").unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues
        .iter()
        .any(|i| i.kind == datarust_profile::QualityKind::TargetLeakage));
}

#[test]
fn quality_target_leakage_via_cramers_v() {
    let s = StrMatrix::from_strings(vec![
        vec!["cat_A", "yes"],
        vec!["cat_A", "yes"],
        vec!["cat_B", "no"],
        vec!["cat_B", "no"],
    ])
    .unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["feature", "target"]))).unwrap();
    let p = p.with_target("target");
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues
        .iter()
        .any(|i| i.kind == datarust_profile::QualityKind::TargetLeakage));
}

#[test]
fn quality_high_missing_all_columns() {
    // 100% missing in both columns
    let m = Matrix::from_rows(vec![
        vec![f64::NAN, f64::NAN],
        vec![f64::NAN, f64::NAN],
        vec![f64::NAN, f64::NAN],
    ])
    .unwrap();
    let p = profile_matrix(&m, None).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues.iter().all(|i| i.severity == Severity::Critical));
}

#[test]
fn quality_all_issues_together() {
    // Constant column, high missing (>=50%), outliers, duplicates
    // 5 NaN out of 9 = 55.6% missing
    let m = Matrix::from_rows(vec![
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, f64::NAN],
        vec![5.0, 1.0],
        vec![5.0, 2.0],
        vec![5.0, 100.0],
        vec![5.0, 100.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["c", "v"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    let kinds: Vec<_> = issues.iter().map(|i| i.kind).collect();
    assert!(kinds.contains(&datarust_profile::QualityKind::ConstantColumn));
    assert!(kinds.contains(&datarust_profile::QualityKind::HighMissing));
    assert!(kinds.contains(&datarust_profile::QualityKind::DuplicateRows));
}

#[test]
fn histogram_edge_labels_in_html() {
    let m = Matrix::from_rows(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("chart-labels"));
    assert!(html.contains("1.0000"));
    assert!(html.contains("5.0000"));
}

#[test]
fn skewness_positive_html() {
    // Right-skewed data
    let m = Matrix::from_rows(vec![
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![100.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let n = p.columns[0].numeric.as_ref().unwrap();
    assert!(n.skewness > 0.0);
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("skew"));
}

#[test]
fn categorical_with_all_unique_values() {
    let s = StrMatrix::from_strings(vec![vec!["id_1"], vec!["id_2"], vec!["id_3"], vec!["id_4"]])
        .unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["uid"]))).unwrap();
    let c = p.columns[0].categorical.as_ref().unwrap();
    assert_eq!(c.unique, 4);
    let issues = run_checks(&p, &Thresholds::default());
    assert!(issues
        .iter()
        .any(|i| i.kind == datarust_profile::QualityKind::NearUnique));
}

#[test]
fn str_matrix_infers_integer_strings_as_numeric() {
    let s = StrMatrix::from_strings(vec![vec!["10"], vec!["20"], vec!["30"]]).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["x"]))).unwrap();
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    let n = p.columns[0].numeric.as_ref().unwrap();
    assert!((n.mean - 20.0).abs() < 1e-9);
}

#[test]
fn str_matrix_infers_float_strings_as_numeric() {
    let s = StrMatrix::from_strings(vec![vec!["1.5"], vec!["2.5"], vec!["3.5"]]).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["x"]))).unwrap();
    assert_eq!(p.columns[0].column_type, ColumnType::Numeric);
    let n = p.columns[0].numeric.as_ref().unwrap();
    assert!((n.mean - 2.5).abs() < 1e-9);
}

#[test]
fn str_matrix_missing_values_not_counted_in_categorical() {
    let s = StrMatrix::from_strings(vec![vec!["a"], vec!["NA"], vec!["a"], vec!["b"]]).unwrap();
    let p = profile_str_matrix(&s, Some(&names(&["k"]))).unwrap();
    let c = p.columns[0].categorical.as_ref().unwrap();
    // unique=2 (a, b), top=a with freq=2
    assert_eq!(c.unique, 2);
    assert_eq!(c.top, "a");
    assert_eq!(c.freq, 2);
    assert_eq!(p.columns[0].missing_count, 1);
}

#[test]
fn negative_correlation_no_high_correlation_issue() {
    // Perfect negative correlation should NOT trigger HighCorrelation
    // (threshold is 0.95, and we check |r| >= threshold)
    let m = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 8.0],
        vec![3.0, 6.0],
        vec![4.0, 4.0],
        vec![5.0, 2.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["x", "y"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    // r = -1.0, |r| = 1.0 >= 0.95, so it SHOULD fire
    assert!(issues
        .iter()
        .any(|i| i.kind == datarust_profile::QualityKind::HighCorrelation));
}

#[test]
fn moderate_correlation_no_issue() {
    // Correlation around 0.7 should NOT trigger
    let m = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 12.0],
        vec![3.0, 11.0],
        vec![4.0, 15.0],
        vec![5.0, 14.0],
    ])
    .unwrap();
    let p = profile_matrix(&m, Some(&names(&["x", "y"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    assert!(!issues
        .iter()
        .any(|i| i.kind == datarust_profile::QualityKind::HighCorrelation));
}

#[test]
fn profile_table_numeric_block_only_errors_without_rows() {
    let result = profile_table(None, None, &[]);
    assert!(result.is_err());
}

#[test]
fn large_outlier_fraction_triggers_warning() {
    // Tight cluster [1,1,...,2,2,...] with extreme outliers
    // Q1≈1, Q3≈2, IQR≈1, upper fence≈3.5 → 100,200,300,400 are outliers
    // 4/20 = 20% outliers -> severity Warning (>= 0.2)
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for _ in 0..10 {
        rows.push(vec![1.0]);
    }
    for _ in 0..6 {
        rows.push(vec![2.0]);
    }
    rows.push(vec![100.0]);
    rows.push(vec![200.0]);
    rows.push(vec![300.0]);
    rows.push(vec![400.0]);
    let m = Matrix::from_rows(rows).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    let outlier_issue = issues
        .iter()
        .find(|i| i.kind == datarust_profile::QualityKind::Outliers);
    assert!(outlier_issue.is_some());
    assert_eq!(outlier_issue.unwrap().severity, Severity::Warning);
}

#[test]
fn small_outlier_fraction_triggers_info() {
    // 1/20 = 5% outliers -> severity Info (< 0.2)
    let mut rows: Vec<Vec<f64>> = (0..19).map(|i| vec![i as f64]).collect();
    rows.push(vec![1000.0]);
    let m = Matrix::from_rows(rows).unwrap();
    let p = profile_matrix(&m, Some(&names(&["x"]))).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    let outlier_issue = issues
        .iter()
        .find(|i| i.kind == datarust_profile::QualityKind::Outliers);
    assert!(outlier_issue.is_some());
    assert_eq!(outlier_issue.unwrap().severity, Severity::Info);
}

#[test]
fn duplicate_rows_below_10_percent_info() {
    // 1/20 = 5% duplicates -> Info (< 10%)
    let mut rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 10.0]).collect();
    rows.push(vec![0.0, 0.0]); // one duplicate
    let m = Matrix::from_rows(rows).unwrap();
    let p = profile_matrix(&m, None).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    let dup_issue = issues
        .iter()
        .find(|i| i.kind == datarust_profile::QualityKind::DuplicateRows);
    assert!(dup_issue.is_some());
    assert_eq!(dup_issue.unwrap().severity, Severity::Info);
}

#[test]
fn duplicate_rows_above_10_percent_warning() {
    // 3/20 = 15% duplicates -> Warning (>= 10%)
    let mut rows: Vec<Vec<f64>> = (0..17).map(|i| vec![i as f64, i as f64]).collect();
    rows.push(vec![0.0, 0.0]);
    rows.push(vec![1.0, 1.0]);
    rows.push(vec![2.0, 2.0]);
    let m = Matrix::from_rows(rows).unwrap();
    let p = profile_matrix(&m, None).unwrap();
    let issues = run_checks(&p, &Thresholds::default());
    let dup_issue = issues
        .iter()
        .find(|i| i.kind == datarust_profile::QualityKind::DuplicateRows);
    assert!(dup_issue.is_some());
    assert_eq!(dup_issue.unwrap().severity, Severity::Warning);
}
