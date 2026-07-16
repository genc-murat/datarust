//! Binary classification metrics mirroring `sklearn.metrics`.
//!
//! Each function takes ground-truth `y_true` and predictions `y_pred` as flat
//! `&[f64]` slices. Classification labels are represented as `0.0` / `1.0`
//! floats, consistent with the [`Regressor`](crate::traits::Regressor) trait's
//! `Vec<f64>` output from [`LogisticRegression`](crate::linear_model::LogisticRegression).

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

/// 2×2 confusion matrix for binary labels `{0, 1}`.
///
/// Returns `[[tn, fp], [fn, tp]]` where:
/// - `tn` = true negatives (true 0, pred 0)
/// - `fp` = false positives (true 0, pred 1)
/// - `fn` = false negatives (true 1, pred 0)
/// - `tp` = true positives (true 1, pred 1)
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64]) -> Result<[[usize; 2]; 2]> {
    check_lengths(y_true, y_pred)?;
    let mut cm = [[0_usize; 2]; 2];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let ti = if t >= 0.5 { 1 } else { 0 };
        let pi = if p >= 0.5 { 1 } else { 0 };
        cm[ti][pi] += 1;
    }
    Ok(cm)
}

/// Fraction of correctly classified samples.
///
/// Mirrors `sklearn.metrics.accuracy_score`.
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len();
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| (**t >= 0.5) == (**p >= 0.5))
        .count();
    Ok(correct as f64 / n as f64)
}

/// Precision: `tp / (tp + fp)` — of all predicted positives, how many are real.
///
/// Mirrors `sklearn.metrics.precision_score`. Returns 0.0 when no positive
/// predictions are made (sklearn returns 0.0 with a warning by default).
pub fn precision_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let tp = cm[1][1];
    let fp = cm[0][1];
    if tp + fp == 0 {
        return Ok(0.0);
    }
    Ok(tp as f64 / (tp + fp) as f64)
}

/// Recall (sensitivity): `tp / (tp + fn)` — of all real positives, how many found.
///
/// Mirrors `sklearn.metrics.recall_score`. Returns 0.0 when there are no true
/// positives in the data.
pub fn recall_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let tp = cm[1][1];
    let fn_ = cm[1][0];
    if tp + fn_ == 0 {
        return Ok(0.0);
    }
    Ok(tp as f64 / (tp + fn_) as f64)
}

/// F1 score: harmonic mean of precision and recall.
///
/// `F1 = 2 · (precision · recall) / (precision + recall)`
///
/// Mirrors `sklearn.metrics.f1_score`. Returns 0.0 when both precision and
/// recall are zero.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let p = precision_score(y_true, y_pred)?;
    let r = recall_score(y_true, y_pred)?;
    if p + r == 0.0 {
        return Ok(0.0);
    }
    Ok(2.0 * p * r / (p + r))
}

/// Cross-entropy (log) loss for binary classification.
///
/// `log_loss = -(1/n) Σ [y_i log(p_i) + (1 − y_i) log(1 − p_i)]`
///
/// Mirrors `sklearn.metrics.log_loss`. `y_proba` holds predicted probabilities
/// of the positive class (values in `[0, 1]`). Probabilities are clipped to
/// `[eps, 1 − eps]` to avoid `log(0)`.
///
/// ```rust
/// use datarust::metrics::classification::log_loss;
///
/// let y_true = vec![0.0, 0.0, 1.0, 1.0];
/// let y_pred = vec![0.1, 0.2, 0.8, 0.9];
/// let ll = log_loss(&y_true, &y_pred, 1e-15).unwrap();
/// assert!(ll > 0.0);
/// ```
pub fn log_loss(y_true: &[f64], y_proba: &[f64], eps: f64) -> Result<f64> {
    check_lengths(y_true, y_proba)?;
    let n = y_true.len() as f64;
    let eps = eps.max(f64::MIN_POSITIVE);
    let mut sum = 0.0;
    for (&t, &p) in y_true.iter().zip(y_proba.iter()) {
        let pc = p.clamp(eps, 1.0 - eps);
        let ti = if t >= 0.5 { 1.0 } else { 0.0 };
        sum += ti * pc.ln() + (1.0 - ti) * (1.0 - pc).ln();
    }
    Ok(-sum / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample with a known confusion matrix: tn=2, fp=1, fn=1, tp=3.
    fn sample() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        )
    }

    #[test]
    fn confusion_matrix_known() {
        let (y_true, y_pred) = sample();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        // [[tn, fp], [fn, tp]] = [[2, 1], [1, 3]]
        assert_eq!(cm, [[2, 1], [1, 3]]);
    }

    #[test]
    fn accuracy_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let acc = accuracy_score(&y_true, &y_pred).unwrap();
        // (2 + 3) / 7
        assert!((acc - 5.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn precision_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let p = precision_score(&y_true, &y_pred).unwrap();
        // tp / (tp + fp) = 3 / 4
        assert!((p - 0.75).abs() < 1e-12);
    }

    #[test]
    fn recall_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let r = recall_score(&y_true, &y_pred).unwrap();
        // tp / (tp + fn) = 3 / 4
        assert!((r - 0.75).abs() < 1e-12);
    }

    #[test]
    fn f1_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        // 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        assert!((f1 - 0.75).abs() < 1e-12);
    }

    #[test]
    fn perfect_classifier() {
        let y = vec![0.0, 1.0, 0.0, 1.0];
        assert!((accuracy_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((precision_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((recall_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((f1_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_wrong_classifier() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![1.0, 0.0, 1.0, 0.0];
        assert!((accuracy_score(&y_true, &y_pred).unwrap() - 0.0).abs() < 1e-12);
        assert!((precision_score(&y_true, &y_pred).unwrap() - 0.0).abs() < 1e-12);
        assert!((recall_score(&y_true, &y_pred).unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn log_loss_perfect_confident() {
        let y = vec![0.0, 1.0];
        let p = vec![1e-15, 1.0 - 1e-15];
        let ll = log_loss(&y, &p, 1e-15).unwrap();
        assert!(ll < 1e-10, "log loss should be ~0, got {ll}");
    }

    #[test]
    fn log_loss_known_value() {
        // y_true = [0, 0, 1, 1], p = [0.1, 0.2, 0.8, 0.9]
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let p = vec![0.1, 0.2, 0.8, 0.9];
        let ll = log_loss(&y_true, &p, 1e-15).unwrap();
        // Computed by hand:
        // -(1/4) * [log(0.9) + log(0.8) + log(0.8) + log(0.9)]
        let expected = -(0.9_f64.ln() + 0.8_f64.ln() + 0.8_f64.ln() + 0.9_f64.ln()) / 4.0;
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn no_positive_predictions_precision_zero() {
        let y_true = vec![1.0, 1.0];
        let y_pred = vec![0.0, 0.0];
        assert!((precision_score(&y_true, &y_pred).unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn length_mismatch_errors() {
        let err = accuracy_score(&[1.0, 0.0], &[1.0]).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn empty_errors() {
        let err = accuracy_score(&[], &[]).unwrap_err();
        assert!(matches!(err, DatarustError::EmptyInput(_)));
    }
}
