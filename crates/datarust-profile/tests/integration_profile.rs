//! End-to-end tests for the public profile API.

use datarust::{Matrix, StrMatrix};
use datarust_profile::quality::checks::run_checks;
use datarust_profile::{
    profile_matrix, profile_str_matrix, profile_table, ColumnType, Severity, Thresholds,
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
    assert!(issues.iter().any(|i| {
        i.kind == datarust_profile::QualityKind::HighCorrelation
    }));
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
    let numeric = Matrix::from_rows(vec![
        vec![1.0],
        vec![1.0],
        vec![5.0],
        vec![5.0],
    ])
    .unwrap();

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
    assert!(issues.iter().any(|i| {
        i.kind == datarust_profile::QualityKind::TargetLeakage
    }));
}


#[test]
fn html_report_includes_relationships_heatmaps() {
    let m = Matrix::from_rows(vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
    ])
    .unwrap();

    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);

    assert!(html.contains("Relationships &amp; Interaction"));
    assert!(html.contains("Pearson correlation matrix"));
    assert!(html.contains("class=\"heatmap\""));
}

