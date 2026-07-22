//! Classification metrics mirroring `sklearn.metrics`.
//!
//! Each function takes ground-truth `y_true` and predictions `y_pred` as flat
//! `&[f64]` slices. Labels are represented as non-negative integer-valued floats
//! (`0.0`, `1.0`, `2.0`, …), consistent with the [`Predictor`](crate::traits::Predictor)
//! trait's hard-label `Vec<f64>` output. Binary labels (`{0.0, 1.0}`) and
//! multiclass labels (`{0.0, 1.0, 2.0, …}`) are both supported: the metric
//! functions auto-detect the number of classes and apply macro-averaging for
//! multiclass inputs.

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

/// Maps a label float to a `usize` class index. Labels are rounded to the
/// nearest integer and must be non-negative.
fn label_to_index(v: f64) -> Result<usize> {
    if v.is_nan() || v < 0.0 {
        return Err(DatarustError::InvalidInput(format!(
            "classification labels must be non-negative integers, found {v}"
        )));
    }
    Ok(v.round() as usize)
}

/// Detects the number of distinct classes present across both slices and
/// returns `(n_classes, label_indices_true, label_indices_pred)`.
fn encode_labels(y_true: &[f64], y_pred: &[f64]) -> Result<(usize, Vec<usize>, Vec<usize>)> {
    let mut idx_true = Vec::with_capacity(y_true.len());
    let mut idx_pred = Vec::with_capacity(y_pred.len());
    let mut max_label = 0usize;
    for &v in y_true {
        let i = label_to_index(v)?;
        max_label = max_label.max(i);
        idx_true.push(i);
    }
    for &v in y_pred {
        let i = label_to_index(v)?;
        max_label = max_label.max(i);
        idx_pred.push(i);
    }
    Ok((max_label + 1, idx_true, idx_pred))
}

/// Confusion matrix for arbitrary integer labels `{0, 1, 2, …}`.
///
/// Returns an `n_classes × n_classes` matrix where `cm[true_class][pred_class]`
/// is the count of samples with the given true/predicted label pair. The matrix
/// dimension is `max(label) + 1`, so all labels present in either input appear.
///
/// For binary `{0, 1}` input this reduces to the familiar `[[tn, fp], [fn, tp]]`
/// 2×2 layout.
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64]) -> Result<Vec<Vec<usize>>> {
    check_lengths(y_true, y_pred)?;
    let (n_classes, idx_true, idx_pred) = encode_labels(y_true, y_pred)?;
    let mut cm = vec![vec![0_usize; n_classes]; n_classes];
    for (&t, &p) in idx_true.iter().zip(idx_pred.iter()) {
        cm[t][p] += 1;
    }
    Ok(cm)
}

/// Fraction of correctly classified samples.
///
/// Mirrors `sklearn.metrics.accuracy_score`. Works for binary and multiclass
/// labels; two samples agree when their rounded integer labels are equal.
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len();
    let mut correct = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if label_to_index(t)? == label_to_index(p)? {
            correct += 1;
        }
    }
    Ok(correct as f64 / n as f64)
}

/// Per-class precision, recall, and F1 from a confusion matrix.
struct PerClassMetrics {
    precision: Vec<f64>,
    recall: Vec<f64>,
    f1: Vec<f64>,
}

/// Computes per-class precision/recall/F1 and returns them, handling
/// zero-denominators as 0.0 (sklearn's default behaviour).
fn per_class(cm: &[Vec<usize>]) -> PerClassMetrics {
    let k = cm.len();
    let mut precision = vec![0.0; k];
    let mut recall = vec![0.0; k];
    let mut f1 = vec![0.0; k];
    for c in 0..k {
        let tp = cm[c][c];
        let fp: usize = (0..k).filter(|&j| j != c).map(|j| cm[j][c]).sum();
        let fn_: usize = (0..k).filter(|&j| j != c).map(|j| cm[c][j]).sum();
        precision[c] = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        recall[c] = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        f1[c] = if precision[c] + recall[c] == 0.0 {
            0.0
        } else {
            2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
        };
    }
    PerClassMetrics {
        precision,
        recall,
        f1,
    }
}

/// Arithmetic mean of a slice (macro-average helper).
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Precision: of all predicted positives, how many are real.
///
/// For binary `{0, 1}` input this is the standard `tp / (tp + fp)` of the
/// positive class. For multiclass input it returns the **macro average** — the
/// mean of per-class precision, giving equal weight to each class.
///
/// Mirrors `sklearn.metrics.precision_score(average='macro')`.
pub fn precision_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let m = per_class(&cm);
    Ok(mean(&m.precision))
}

/// Recall (sensitivity): of all real positives, how many found.
///
/// For binary `{0, 1}` input this is the standard `tp / (tp + fn)` of the
/// positive class. For multiclass input it returns the **macro average** — the
/// mean of per-class recall.
///
/// Mirrors `sklearn.metrics.recall_score(average='macro')`.
pub fn recall_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let m = per_class(&cm);
    Ok(mean(&m.recall))
}

/// F1 score: harmonic mean of precision and recall.
///
/// For binary `{0, 1}` input this is the standard F1 of the positive class.
/// For multiclass input it returns the **macro average** — the mean of per-class
/// F1 scores.
///
/// Mirrors `sklearn.metrics.f1_score(average='macro')`.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let cm = confusion_matrix(y_true, y_pred)?;
    let m = per_class(&cm);
    Ok(mean(&m.f1))
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

/// Area under the ROC curve (binary classifier discrimination).
///
/// Computes the Receiver Operating Characteristic AUC by the rank-based
/// equivalence (Mann–Whitney U / Wilcoxon statistic): AUC =
/// `(Σ rank_positive − m(m+1)/2) / (m·n)` where `m` is the number of positives,
/// `n` the number of negatives, and ties are handled by averaging ranks.
///
/// Mirrors `sklearn.metrics.roc_auc_score` for binary `{0, 1}` targets. `y_score`
/// is the predicted probability (or any monotonic score) of the positive class.
///
/// Returns 0.5 for a random classifier, 1.0 for perfect separation. Errors if
/// fewer than two classes are present or inputs have mismatched length.
pub fn roc_auc_score(y_true: &[f64], y_score: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_score)?;
    // Collect (score, label) pairs; label = 1 for positive class.
    let mut pairs: Vec<(f64, f64)> = y_true
        .iter()
        .zip(y_score.iter())
        .map(|(&t, &s)| (s, if t >= 0.5 { 1.0 } else { 0.0 }))
        .collect();
    let m = pairs.iter().filter(|(_, l)| *l == 1.0).count();
    let n = pairs.len() - m;
    if m == 0 || n == 0 {
        return Err(DatarustError::InvalidInput(
            "roc_auc_score requires at least one sample of each class".into(),
        ));
    }
    // Sort by score ascending; ties get average rank.
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Assign average ranks for ties.
    let len = pairs.len();
    let mut ranks = vec![0.0_f64; len];
    let mut i = 0;
    while i < len {
        let mut j = i + 1;
        while j < len && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Average rank of positions i..j (1-based).
        let avg_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        for slot in &mut ranks[i..j] {
            *slot = avg_rank;
        }
        i = j;
    }
    let rank_sum_pos: f64 = pairs
        .iter()
        .zip(ranks.iter())
        .filter(|((_, l), _)| *l == 1.0)
        .map(|(_, r)| *r)
        .sum();
    let auc = (rank_sum_pos - m as f64 * (m as f64 + 1.0) / 2.0) / (m as f64 * n as f64);
    Ok(auc)
}

/// Average precision (area under the precision-recall curve) for binary
/// classification.
///
/// Computes a step-function approximation of the PR curve, mirroring
/// `sklearn.metrics.average_precision_score`. `y_score` is the predicted
/// probability (or any monotonic score) of the positive class.
///
/// Returns 1.0 for a perfect classifier, and the base rate (positive
/// prevalence) for a random one. Errors if fewer than two classes are present.
pub fn average_precision_score(y_true: &[f64], y_score: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_score)?;
    // Sort by descending score.
    let mut idx: Vec<usize> = (0..y_true.len()).collect();
    idx.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total_pos = y_true.iter().filter(|&&t| t >= 0.5).count();
    if total_pos == 0 || total_pos == y_true.len() {
        return Err(DatarustError::InvalidInput(
            "average_precision_score requires at least one sample of each class".into(),
        ));
    }
    // Walk down the ranked list accumulating TP; at each positive, precision
    // is tp/(tp+fp) and recall increases. AP = Σ (R_n − R_{n−1}) · P_n.
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0.0_f64;
    let mut prev_recall = 0.0_f64;
    for &i in &idx {
        if y_true[i] >= 0.5 {
            tp += 1;
        } else {
            fp += 1;
        }
        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / total_pos as f64;
        if recall > prev_recall {
            ap += (recall - prev_recall) * precision;
            prev_recall = recall;
        }
    }
    Ok(ap)
}

/// Cohen's kappa — agreement between `y_true` and `y_pred` corrected for
/// chance.
///
/// `κ = (p_o − p_e) / (1 − p_e)` where `p_o` is observed agreement and `p_e`
/// is expected (chance) agreement. Works for binary and multiclass integer
/// labels. Returns 1.0 for perfect agreement, 0.0 for chance-level agreement,
/// and negative values for worse-than-chance agreement.
///
/// Mirrors `sklearn.metrics.cohen_kappa_score`.
pub fn cohen_kappa_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len();
    let cm = confusion_matrix(y_true, y_pred)?;
    let k = cm.len();
    let n_f = n as f64;
    let p_o: f64 = (0..k).map(|c| cm[c][c] as f64).sum::<f64>() / n_f;
    // Marginal probabilities: true and predicted class counts.
    let row_sums: Vec<f64> = cm
        .iter()
        .map(|row| row.iter().sum::<usize>() as f64)
        .collect();
    let col_sums: Vec<f64> = (0..k)
        .map(|c| (0..k).map(|r| cm[r][c] as f64).sum::<f64>())
        .collect();
    let p_e: f64 = (0..k).map(|c| row_sums[c] * col_sums[c]).sum::<f64>() / (n_f * n_f);
    if (1.0 - p_e).abs() < 1e-15 {
        // Both annotators perfectly uniform on one class; agreement is undefined.
        return Ok(1.0);
    }
    Ok((p_o - p_e) / (1.0 - p_e))
}

/// Matthews correlation coefficient (MCC) for binary and multiclass
/// classification.
///
/// For binary `{0, 1}` labels this reduces to the familiar
/// `MCC = (tp·tn − fp·fn) / √((tp+fp)(tp+fn)(tn+fp)(tn+fn))`. For multiclass
/// labels it uses the general confusion-matrix formulation of Gorodkin (2004),
/// which sklearn also implements:
///
/// ```text
/// MCC = cov(x_t, x_p) / √(var(x_t) · var(x_p))
/// ```
///
/// where `x_t`, `x_p` are the one-hot encodings of the true/predicted labels.
///
/// Returns +1 for perfect prediction, 0 for random, −1 for inverse prediction.
/// When the denominator is zero, returns 0.0 (sklearn's default).
///
/// Mirrors `sklearn.metrics.matthews_corrcoef`.
pub fn matthews_corrcoef(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let cm = confusion_matrix(y_true, y_pred)?;
    let k = cm.len();
    let s: f64 = cm.iter().map(|r| r.iter().sum::<usize>() as f64).sum();
    // Marginal sums: t_k = Σ_l C_{k,l} (true class k count),
    //                 p_k = Σ_l C_{l,k} (predicted class k count).
    let mut t = vec![0.0_f64; k]; // row sums (true)
    let mut p = vec![0.0_f64; k]; // col sums (predicted)
    for i in 0..k {
        t[i] = cm[i].iter().sum::<usize>() as f64;
        p[i] = (0..k).map(|r| cm[r][i] as f64).sum();
    }
    // trace: total correct predictions.
    let trace: f64 = (0..k).map(|c| cm[c][c] as f64).sum();
    // Numerator: cov(x_t, x_p) · s.
    let cov_ytyp = trace * s - (0..k).map(|c| t[c] * p[c]).sum::<f64>();
    // Denominator: √(var(x_t) · var(x_p)).
    let cov_ytyt = s * s - (0..k).map(|c| t[c] * t[c]).sum::<f64>();
    let cov_ypyp = s * s - (0..k).map(|c| p[c] * p[c]).sum::<f64>();
    let denom = (cov_ytyt * cov_ypyp).sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(cov_ytyp / denom)
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
    fn confusion_matrix_binary_known() {
        let (y_true, y_pred) = sample();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm, vec![vec![2, 1], vec![1, 3]]);
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
        // Macro average: class 0 precision = 2/(2+1) = 2/3,
        // class 1 precision = 3/(3+1) = 3/4.  Mean = (2/3 + 3/4)/2 = 17/24.
        assert!((p - 17.0 / 24.0).abs() < 1e-12);
    }

    #[test]
    fn recall_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let r = recall_score(&y_true, &y_pred).unwrap();
        // Macro average: class 0 recall = 2/(2+1) = 2/3,
        // class 1 recall = 3/(3+1) = 3/4.  Mean = 17/24.
        assert!((r - 17.0 / 24.0).abs() < 1e-12);
    }

    #[test]
    fn f1_matches_hand_computation() {
        let (y_true, y_pred) = sample();
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        // Per-class F1 then macro-average.
        //   class 0: 2*(2/3*2/3)/(2/3+2/3) = 2/3
        //   class 1: 2*(3/4*3/4)/(3/4+3/4) = 3/4
        //   mean = (2/3 + 3/4)/2 = 17/24
        assert!((f1 - 17.0 / 24.0).abs() < 1e-12);
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

    // ── Multiclass tests ──────────────────────────────────────────────

    /// 3-class example. True labels 0/1/2, predictions with one error per class.
    /// Confusion matrix (true × pred):
    ///   class 0: 2 correct, 1 mispredicted as 2  → [2, 0, 1]
    ///   class 1: 2 correct, 1 mispredicted as 0  → [1, 2, 0]
    ///   class 2: 2 correct, 1 mispredicted as 1  → [0, 1, 2]
    fn multiclass_sample() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            vec![0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0],
        )
    }

    #[test]
    fn confusion_matrix_multiclass() {
        let (y_true, y_pred) = multiclass_sample();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm.len(), 3);
        assert_eq!(cm[0], vec![2, 0, 1]);
        assert_eq!(cm[1], vec![1, 2, 0]);
        assert_eq!(cm[2], vec![0, 1, 2]);
    }

    #[test]
    fn accuracy_multiclass() {
        let (y_true, y_pred) = multiclass_sample();
        let acc = accuracy_score(&y_true, &y_pred).unwrap();
        // 6 correct out of 9.
        assert!((acc - 6.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn macro_precision_recall_f1_multiclass() {
        let (y_true, y_pred) = multiclass_sample();
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        let m = per_class(&cm);
        // Per-class precision: tp/(tp+fp)
        //   class 0: 2/(2+1) = 2/3
        //   class 1: 2/(2+1) = 2/3
        //   class 2: 2/(2+1) = 2/3
        // Macro average = 2/3.
        for c in 0..3 {
            assert!(
                (m.precision[c] - 2.0 / 3.0).abs() < 1e-12,
                "precision[{c}]={}",
                m.precision[c]
            );
        }
        let p = precision_score(&y_true, &y_pred).unwrap();
        assert!((p - 2.0 / 3.0).abs() < 1e-12);
        // Per-class recall: tp/(tp+fn) — symmetric here.
        for c in 0..3 {
            assert!(
                (m.recall[c] - 2.0 / 3.0).abs() < 1e-12,
                "recall[{c}]={}",
                m.recall[c]
            );
        }
        let r = recall_score(&y_true, &y_pred).unwrap();
        assert!((r - 2.0 / 3.0).abs() < 1e-12);
        // F1: 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3.
        let f1 = f1_score(&y_true, &y_pred).unwrap();
        assert!((f1 - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn perfect_multiclass_classifier() {
        let y = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        assert!((accuracy_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((precision_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((recall_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((f1_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
        let cm = confusion_matrix(&y, &y).unwrap();
        assert_eq!(cm, vec![vec![2, 0, 0], vec![0, 2, 0], vec![0, 0, 2]]);
    }

    #[test]
    fn negative_label_rejected() {
        let err = confusion_matrix(&[-1.0, 0.0], &[0.0, 0.0]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    // ── ROC-AUC / PR-AUC tests ────────────────────────────────────────

    #[test]
    fn roc_auc_perfect_separation() {
        // All positives have higher scores than all negatives.
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let s = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        assert!((roc_auc_score(&y, &s).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn roc_auc_inverse() {
        // All negatives have higher scores → AUC = 0.
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let s = vec![0.9, 0.8, 0.2, 0.1];
        assert!((roc_auc_score(&y, &s).unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn roc_auc_ties_average() {
        // One tie between a positive and negative at score 0.5.
        // Negatives: [0.1, 0.5], Positives: [0.5, 0.9].
        // Ranks: 0.1→1, 0.5→2.5 (tie, avg of 2 & 3), 0.9→4.
        // Pos rank sum = 2.5 + 4 = 6.5; AUC = (6.5 − 3) / 4 = 0.875.
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let s = vec![0.1, 0.5, 0.5, 0.9];
        let auc = roc_auc_score(&y, &s).unwrap();
        assert!((auc - 0.875).abs() < 1e-12, "auc={auc}");
    }

    #[test]
    fn roc_auc_single_class_errors() {
        let err = roc_auc_score(&[1.0, 1.0], &[0.5, 0.6]).unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn average_precision_perfect() {
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let s = vec![0.1, 0.2, 0.8, 0.9];
        assert!((average_precision_score(&y, &s).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn average_precision_known_value() {
        // y = [0, 1, 0, 1], s = [0.3, 0.9, 0.4, 0.8].
        // Sort by score desc: 0.9(y=1), 0.8(y=1), 0.4(y=0), 0.3(y=0).
        // Both positives rank above both negatives → AP = 1.0.
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let s = vec![0.3, 0.9, 0.4, 0.8];
        let ap = average_precision_score(&y, &s).unwrap();
        assert!((ap - 1.0).abs() < 1e-12, "ap={ap}");
    }

    #[test]
    fn average_precision_imperfect() {
        // y = [1, 0, 1, 0], s = [0.9, 0.8, 0.4, 0.3].
        // Sort desc: 0.9(y=1), 0.8(y=0), 0.4(y=1), 0.3(y=0).
        //   i=0: tp=1, prec=1.0, recall=0.5, AP += 0.5
        //   i=1: tp=1, fp=1, prec=0.5
        //   i=2: tp=2, prec=0.667, recall=1.0, AP += 0.5*0.667
        // AP = 0.5 + 0.333... = 0.833...
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let s = vec![0.9, 0.8, 0.4, 0.3];
        let ap = average_precision_score(&y, &s).unwrap();
        assert!((ap - (0.5 + 0.5 * 2.0 / 3.0)).abs() < 1e-12, "ap={ap}");
    }

    // ── Cohen's kappa / Matthews correlation tests ─────────────────────

    #[test]
    fn cohen_kappa_perfect_agreement() {
        let y = vec![0.0, 1.0, 0.0, 1.0, 2.0, 2.0];
        assert!((cohen_kappa_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cohen_kappa_known_value() {
        // Binary: y_true = [0,0,1,1], y_pred = [0,1,0,1]
        // cm = [[1,1],[1,1]], p_o = 0.5, p_e = 0.5, kappa = 0.
        let t = vec![0.0, 0.0, 1.0, 1.0];
        let p = vec![0.0, 1.0, 0.0, 1.0];
        assert!((cohen_kappa_score(&t, &p).unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cohen_kappa_multiclass() {
        // 3-class with 6/9 correct; kappa should be between 0 and 1.
        let (t, p) = multiclass_sample();
        let k = cohen_kappa_score(&t, &p).unwrap();
        assert!(k > 0.0 && k < 1.0, "kappa={k}");
    }

    #[test]
    fn matthews_perfect() {
        let y = vec![0.0, 0.0, 1.0, 1.0];
        assert!((matthews_corrcoef(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matthews_inverse() {
        let t = vec![0.0, 0.0, 1.0, 1.0];
        let p = vec![1.0, 1.0, 0.0, 0.0];
        assert!((matthews_corrcoef(&t, &p).unwrap() - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn matthews_known_value() {
        // cm = [[2,1],[1,3]]: tp=3, tn=2, fp=1, fn=1
        // MCC = (3*2 - 1*1) / sqrt(4*4*3*3) = 5/12
        let (t, p) = sample();
        assert!((matthews_corrcoef(&t, &p).unwrap() - 5.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn matthews_multiclass_perfect() {
        let y = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        assert!((matthews_corrcoef(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matthews_multiclass_known_value() {
        // 3-class example with 6/9 correct.
        // cm = [[2,0,1],[1,2,0],[0,1,2]], trace=6, s=9.
        let (t, p) = multiclass_sample();
        let mcc = matthews_corrcoef(&t, &p).unwrap();
        // Hand-computed: t=[3,3,3], p=[3,3,3], cov_ytyp = 6·9 − 3·(3·3) = 54−27 = 27,
        // cov_ytyt = 81 − 3·9 = 54, cov_ypyp = 54, denom = 54.
        // MCC = 27/54 = 0.5.
        assert!((mcc - 0.5).abs() < 1e-12, "mcc={mcc}");
    }
}
