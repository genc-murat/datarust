//! Regression metrics mirroring `sklearn.metrics`.
//!
//! Each function takes ground-truth `y_true` and predictions `y_pred` as flat
//! `&[f64]` slices and returns an `f64`. Lengths must match.

use crate::error::{DatarustError, Result};

fn check_lengths(y_true: &[f64], y_pred: &[f64]) -> Result<()> {
    if y_true.is_empty() {
        return Err(DatarustError::EmptyInput("y_true is empty".into()));
    }
    if y_true.len() != y_pred.len() {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} predictions", y_true.len()),
            actual: format!("{} predictions", y_pred.len()),
        });
    }
    Ok(())
}

/// Mean squared error (regression loss).
///
/// Mirrors `sklearn.metrics.mean_squared_error`. When `squared` is `false`,
/// returns the root mean squared error (RMSE) instead.
///
/// `MSE = (1/n) Σ (yᵢ − ŷᵢ)²`
///
/// ```rust
/// use datarust::metrics::regression::mean_squared_error;
///
/// let y_true = vec![3.0, -0.5, 2.0, 7.0];
/// let y_pred = vec![2.5, 0.0, 2.0, 8.0];
/// let mse = mean_squared_error(&y_true, &y_pred, true).unwrap();
/// assert!((mse - 0.375).abs() < 1e-12);
/// ```
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64], squared: bool) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let mut sum = 0.0;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        let d = t - p;
        sum += d * d;
    }
    let mse = sum / n;
    Ok(if squared { mse } else { mse.sqrt() })
}

/// Mean absolute error (regression loss).
///
/// Mirrors `sklearn.metrics.mean_absolute_error`.
///
/// `MAE = (1/n) Σ |yᵢ − ŷᵢ|`
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum();
    Ok(sum / n)
}

/// R² (coefficient of determination) regression score.
///
/// Mirrors `sklearn.metrics.r2_score`. Best possible score is 1.0; a model
/// that always predicts the mean of `y_true` scores 0.0; worse models score
/// negative.
///
/// `R² = 1 − SS_res / SS_tot` where `SS_res = Σ (yᵢ − ŷᵢ)²` and
/// `SS_tot = Σ (yᵢ − ȳ)²`.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        let d = t - p;
        ss_res += d * d;
        let dc = t - mean;
        ss_tot += dc * dc;
    }
    if ss_tot == 0.0 {
        // Constant y_true: sklearn returns 1.0 if predictions are also perfect,
        // else 0.0.
        return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Maximum residual error (regression metric).
///
/// Mirrors `sklearn.metrics.max_error`. Returns `max(|yᵢ − ŷᵢ|)`.
pub fn max_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let mut max = 0.0_f64;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        let d = (t - p).abs();
        if d > max {
            max = d;
        }
    }
    Ok(max)
}

/// Explained variance regression score.
///
/// Mirrors `sklearn.metrics.explained_variance_score`. Best possible score is
/// 1.0; lower values indicate worse models.
///
/// `explained_variance = 1 − Var(y − ŷ) / Var(y)`
pub fn explained_variance_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len() as f64;
    // Residual mean and variance.
    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| t - p)
        .collect();
    let res_mean = residuals.iter().sum::<f64>() / n;
    let res_var = residuals
        .iter()
        .map(|r| (r - res_mean).powi(2))
        .sum::<f64>()
        / n;

    let y_mean = y_true.iter().sum::<f64>() / n;
    let y_var = y_true.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n;

    if y_var == 0.0 {
        return Ok(if res_var == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - res_var / y_var)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sklearn reference values are computed against `sklearn.metrics` with the
    /// same inputs and hardcoded here as the expected ground truth.
    fn sample() -> (Vec<f64>, Vec<f64>) {
        // y_true and y_pred with known sklearn values.
        (vec![3.0, -0.5, 2.0, 7.0], vec![2.5, 0.0, 2.0, 8.0])
    }

    #[test]
    fn mse_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let mse = mean_squared_error(&y_true, &y_pred, true).unwrap();
        // sklearn: mean_squared_error -> 0.375
        assert!((mse - 0.375).abs() < 1e-12);
    }

    #[test]
    fn rmse_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let rmse = mean_squared_error(&y_true, &y_pred, false).unwrap();
        assert!((rmse - 0.375_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mae_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        // sklearn: mean_absolute_error -> 0.5
        assert!((mae - 0.5).abs() < 1e-12);
    }

    #[test]
    fn r2_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        // sklearn: r2_score -> 0.9486081370449679
        assert!((r2 - 0.9486081370449679).abs() < 1e-9);
    }

    #[test]
    fn max_error_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let me = max_error(&y_true, &y_pred).unwrap();
        // sklearn: max_error -> 1.0
        assert!((me - 1.0).abs() < 1e-12);
    }

    #[test]
    fn explained_variance_matches_sklearn() {
        let (y_true, y_pred) = sample();
        let ev = explained_variance_score(&y_true, &y_pred).unwrap();
        // Verified by hand: residuals=[0.5,-0.5,0,-1], res_var=0.3125,
        // y_var=7.296875 -> 1 - 0.3125/7.296875 ≈ 0.9571734475374732.
        assert!((ev - 0.9571734475374732).abs() < 1e-9);
    }

    #[test]
    fn perfect_predictions() {
        let y = vec![1.0, 2.0, 3.0];
        assert!((r2_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((explained_variance_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!(mean_squared_error(&y, &y, true).unwrap().abs() < 1e-12);
        assert!(mean_absolute_error(&y, &y).unwrap().abs() < 1e-12);
        assert!(max_error(&y, &y).unwrap().abs() < 1e-12);
    }

    #[test]
    fn mean_predictor_r2_is_zero() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mean = 2.5;
        let pred = vec![mean; 4];
        let r2 = r2_score(&y, &pred).unwrap();
        assert!(r2.abs() < 1e-12);
    }

    #[test]
    fn constant_y_true_perfect() {
        let y = vec![5.0, 5.0, 5.0];
        let pred = vec![5.0, 5.0, 5.0];
        assert!((r2_score(&y, &pred).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_y_true_imperfect() {
        let y = vec![5.0, 5.0, 5.0];
        let pred = vec![4.0, 5.0, 6.0];
        // sklearn returns 0.0 for non-perfect predictions on constant y_true.
        assert!((r2_score(&y, &pred).unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn length_mismatch_errors() {
        let err = mean_squared_error(&[1.0, 2.0], &[1.0], true).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn empty_errors() {
        let err = r2_score(&[], &[]).unwrap_err();
        assert!(matches!(err, DatarustError::EmptyInput(_)));
    }
}
