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
fn html_report_contains_columns_and_findings() {
    let m = Matrix::from_rows(vec![vec![1.0, f64::NAN], vec![1.0, 2.0]]).unwrap();
    let p = profile_matrix(&m, Some(&names(&["a", "b"]))).unwrap();
    let html = datarust_profile::report::to_html(&p);
    assert!(html.contains("<table"));
    assert!(html.contains("a"));
    assert!(html.contains("numeric"));
    // NaN handling should not leak literal "NaN" into numeric cells.
    assert!(!html.contains("NaN</td>"));
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
