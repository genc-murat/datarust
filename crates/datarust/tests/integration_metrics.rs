//! Integration tests for `metrics::regression`.

use datarust::error::DatarustError;
use datarust::metrics::regression::{
    explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score,
};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn mse_rmse_toggle() {
    let y_true = vec![1.0, 2.0, 3.0];
    let y_pred = vec![1.0, 2.0, 4.0];
    let mse = mean_squared_error(&y_true, &y_pred, true).unwrap();
    let rmse = mean_squared_error(&y_true, &y_pred, false).unwrap();
    assert!(approx(mse, 1.0 / 3.0, 1e-12));
    assert!(approx(rmse, (1.0_f64 / 3.0).sqrt(), 1e-12));
}

#[test]
fn r2_mean_predictor_is_zero() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let mean = 2.5;
    let pred = vec![mean; 4];
    assert!(approx(r2_score(&y, &pred).unwrap(), 0.0, 1e-12));
}

#[test]
fn r2_worse_than_mean_is_negative() {
    let y = vec![1.0, 2.0, 3.0];
    // Predict anti-correlated → R² < 0.
    let pred = vec![3.0, 2.0, 1.0];
    let r2 = r2_score(&y, &pred).unwrap();
    assert!(r2 < 0.0, "expected negative r2, got {r2}");
}

#[test]
fn explained_variance_constant_y() {
    let y = vec![5.0, 5.0, 5.0];
    let pred = vec![5.0, 5.0, 5.0];
    assert!(approx(
        explained_variance_score(&y, &pred).unwrap(),
        1.0,
        1e-12
    ));
}

#[test]
fn max_error_basic() {
    let y_true = vec![1.0, 5.0, -3.0];
    let y_pred = vec![2.0, 3.0, 0.0];
    assert!(approx(max_error(&y_true, &y_pred).unwrap(), 3.0, 1e-12));
}

#[test]
fn length_mismatch_errors() {
    let err = mean_squared_error(&[1.0, 2.0], &[1.0], true).unwrap_err();
    assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    let err = r2_score(&[1.0, 2.0], &[1.0]).unwrap_err();
    assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
}

#[test]
fn empty_input_errors() {
    let err = mean_squared_error(&[], &[], true).unwrap_err();
    assert!(matches!(err, DatarustError::EmptyInput(_)));
    let err = r2_score(&[], &[]).unwrap_err();
    assert!(matches!(err, DatarustError::EmptyInput(_)));
}

#[test]
fn all_metrics_on_known_sample() {
    // sklearn reference values computed independently.
    let y_true = vec![3.0, -0.5, 2.0, 7.0];
    let y_pred = vec![2.5, 0.0, 2.0, 8.0];
    assert!(approx(
        mean_squared_error(&y_true, &y_pred, true).unwrap(),
        0.375,
        1e-12
    ));
    assert!(approx(
        mean_absolute_error(&y_true, &y_pred).unwrap(),
        0.5,
        1e-12
    ));
    assert!(approx(
        r2_score(&y_true, &y_pred).unwrap(),
        0.9486081370449679,
        1e-9
    ));
    assert!(approx(
        explained_variance_score(&y_true, &y_pred).unwrap(),
        0.9571734475374732,
        1e-9
    ));
    assert!(approx(max_error(&y_true, &y_pred).unwrap(), 1.0, 1e-12));
}
