//! Classification metrics mirroring `sklearn.metrics`.
//!
//! Each function takes ground-truth `y_true` and predictions `y_pred` as flat
//! `&[f64]` slices. Labels are represented as non-negative integer-valued floats,
//! consistent with the [`Predictor`](crate::traits::Predictor) trait's hard-label
//! `Vec<f64>` output. Labels do not need to be contiguous: `{10.0, 20.0}` is a
//! valid two-class space and is compacted internally rather than allocating 21
//! confusion-matrix rows.

use crate::error::{DatarustError, Result};
use crate::label_space::canonical_label;
pub use crate::label_space::LabelSpace;

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

/// Confusion counts paired with the original label represented by each row and
/// column.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfusionMatrix {
    /// Sorted original labels. `labels[i]` names row and column `i`.
    pub labels: Vec<f64>,
    /// Counts in `[true_label_index][predicted_label_index]` order.
    pub counts: Vec<Vec<usize>>,
}

/// Confusion matrix for arbitrary non-negative integer labels.
///
/// Labels are compacted in sorted order. For example, `{10, 20}` produces a
/// 2×2 matrix rather than a 21×21 matrix. Use
/// [`confusion_matrix_labeled`] when the row/column label mapping is needed.
///
/// For binary `{0, 1}` input this reduces to the familiar `[[tn, fp], [fn, tp]]`
/// 2×2 layout.
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64]) -> Result<Vec<Vec<usize>>> {
    Ok(confusion_matrix_labeled(y_true, y_pred)?.counts)
}

/// Confusion matrix retaining the mapping from compact rows and columns to the
/// original class labels.
pub fn confusion_matrix_labeled(y_true: &[f64], y_pred: &[f64]) -> Result<ConfusionMatrix> {
    let space = LabelSpace::from_pair(y_true, y_pred)?;
    let n_classes = space.len();
    let mut counts = vec![vec![0_usize; n_classes]; n_classes];
    for (&truth, &predicted) in y_true.iter().zip(y_pred.iter()) {
        let truth_index = space.encode(truth)?;
        let predicted_index = space.encode(predicted)?;
        counts[truth_index][predicted_index] += 1;
    }
    Ok(ConfusionMatrix {
        labels: space.labels().to_vec(),
        counts,
    })
}

/// Fraction of correctly classified samples.
///
/// Mirrors `sklearn.metrics.accuracy_score`. Works for binary and multiclass
/// labels; two samples agree when their validated integer labels are equal.
pub fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    check_lengths(y_true, y_pred)?;
    let n = y_true.len();
    let mut correct = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if canonical_label(t)? == canonical_label(p)? {
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
    support: Vec<usize>,
    precision_defined: Vec<bool>,
    recall_defined: Vec<bool>,
    f1_defined: Vec<bool>,
}

/// Computes per-class precision/recall/F1 and returns them, handling
/// zero-denominators as 0.0 (sklearn's default behaviour).
fn per_class(cm: &[Vec<usize>]) -> PerClassMetrics {
    let k = cm.len();
    let mut precision = vec![0.0; k];
    let mut recall = vec![0.0; k];
    let mut f1 = vec![0.0; k];
    let mut support = vec![0; k];
    let mut precision_defined = vec![false; k];
    let mut recall_defined = vec![false; k];
    let mut f1_defined = vec![false; k];
    for c in 0..k {
        let tp = cm[c][c];
        let fp: usize = (0..k).filter(|&j| j != c).map(|j| cm[j][c]).sum();
        let fn_: usize = (0..k).filter(|&j| j != c).map(|j| cm[c][j]).sum();
        support[c] = tp + fn_;
        precision_defined[c] = tp + fp > 0;
        recall_defined[c] = tp + fn_ > 0;
        precision[c] = if !precision_defined[c] {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        recall[c] = if !recall_defined[c] {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        f1_defined[c] = precision[c] + recall[c] > 0.0;
        f1[c] = if !f1_defined[c] {
            0.0
        } else {
            2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
        };
    }
    PerClassMetrics {
        precision,
        recall,
        f1,
        support,
        precision_defined,
        recall_defined,
        f1_defined,
    }
}

/// Arithmetic mean of a slice (macro-average helper).
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Averaging strategy for multiclass precision, recall, and F1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Average {
    /// Score one selected positive label. The observed label space must contain
    /// no more than two classes.
    Binary {
        /// Original label to treat as the positive class.
        positive_label: f64,
    },
    /// Give every observed class equal weight.
    Macro,
    /// Weight each per-class score by its true-label support.
    Weighted,
    /// Pool all one-vs-rest decisions before computing the score. For
    /// single-label classification this equals accuracy.
    Micro,
}

/// Policy for a class whose requested metric has a zero denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroDivision {
    /// Return `0.0`, matching the compatibility helpers in this module.
    Zero,
    /// Return an [`InvalidInput`](DatarustError::InvalidInput) error.
    Error,
}

/// Per-class classification metrics in original-label order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassMetrics {
    /// Original class label.
    pub label: f64,
    /// One-vs-rest precision for this class.
    pub precision: f64,
    /// One-vs-rest recall for this class.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// Number of true rows carrying this label.
    pub support: usize,
}

#[derive(Debug, Clone, Copy)]
enum MetricKind {
    Precision,
    Recall,
    F1,
}

impl MetricKind {
    fn name(self) -> &'static str {
        match self {
            Self::Precision => "precision",
            Self::Recall => "recall",
            Self::F1 => "F1",
        }
    }
}

fn metric_is_defined(metrics: &PerClassMetrics, metric: MetricKind, index: usize) -> bool {
    match metric {
        MetricKind::Precision => metrics.precision_defined[index],
        MetricKind::Recall => metrics.recall_defined[index],
        MetricKind::F1 => metrics.f1_defined[index],
    }
}

fn undefined_metric_error(metric: MetricKind, label: f64) -> DatarustError {
    DatarustError::InvalidInput(format!(
        "{} is undefined for class {label} because its denominator is zero",
        metric.name()
    ))
}

fn averaged_score(
    y_true: &[f64],
    y_pred: &[f64],
    average: Average,
    metric: MetricKind,
    zero_division: ZeroDivision,
) -> Result<f64> {
    let labeled = confusion_matrix_labeled(y_true, y_pred)?;
    let per_class = per_class(&labeled.counts);
    let values = match metric {
        MetricKind::Precision => &per_class.precision,
        MetricKind::Recall => &per_class.recall,
        MetricKind::F1 => &per_class.f1,
    };

    match average {
        Average::Binary { positive_label } => {
            if labeled.labels.len() > 2 {
                return Err(DatarustError::InvalidInput(format!(
                    "binary averaging requires at most 2 observed classes, found {}",
                    labeled.labels.len()
                )));
            }
            let positive_label = canonical_label(positive_label)?;
            let index = labeled
                .labels
                .binary_search_by(|candidate| candidate.total_cmp(&positive_label))
                .ok();
            match index {
                Some(index)
                    if zero_division == ZeroDivision::Error
                        && !metric_is_defined(&per_class, metric, index) =>
                {
                    Err(undefined_metric_error(metric, positive_label))
                }
                Some(index) => Ok(values[index]),
                None if zero_division == ZeroDivision::Error => {
                    Err(undefined_metric_error(metric, positive_label))
                }
                None => Ok(0.0),
            }
        }
        Average::Macro => {
            if zero_division == ZeroDivision::Error {
                if let Some(index) =
                    (0..values.len()).find(|&index| !metric_is_defined(&per_class, metric, index))
                {
                    return Err(undefined_metric_error(metric, labeled.labels[index]));
                }
            }
            Ok(mean(values))
        }
        Average::Weighted => {
            let total_support: usize = per_class.support.iter().sum();
            if total_support == 0 {
                return Ok(0.0);
            }
            if zero_division == ZeroDivision::Error {
                if let Some(index) = (0..values.len()).find(|&index| {
                    per_class.support[index] > 0 && !metric_is_defined(&per_class, metric, index)
                }) {
                    return Err(undefined_metric_error(metric, labeled.labels[index]));
                }
            }
            Ok(values
                .iter()
                .zip(per_class.support.iter())
                .map(|(&value, &support)| value * support as f64)
                .sum::<f64>()
                / total_support as f64)
        }
        Average::Micro => {
            let correct: usize = labeled
                .counts
                .iter()
                .enumerate()
                .map(|(index, row)| row[index])
                .sum();
            let total: usize = labeled.counts.iter().flatten().sum();
            Ok(if total == 0 {
                0.0
            } else {
                correct as f64 / total as f64
            })
        }
    }
}

/// Return precision, recall, F1, and support for every observed class.
pub fn classification_report(y_true: &[f64], y_pred: &[f64]) -> Result<Vec<ClassMetrics>> {
    let labeled = confusion_matrix_labeled(y_true, y_pred)?;
    let metrics = per_class(&labeled.counts);
    Ok(labeled
        .labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| ClassMetrics {
            label,
            precision: metrics.precision[index],
            recall: metrics.recall[index],
            f1: metrics.f1[index],
            support: metrics.support[index],
        })
        .collect())
}

/// Precision with an explicit averaging strategy.
pub fn precision_score_with(y_true: &[f64], y_pred: &[f64], average: Average) -> Result<f64> {
    precision_score_with_options(y_true, y_pred, average, ZeroDivision::Zero)
}

/// Recall with an explicit averaging strategy.
pub fn recall_score_with(y_true: &[f64], y_pred: &[f64], average: Average) -> Result<f64> {
    recall_score_with_options(y_true, y_pred, average, ZeroDivision::Zero)
}

/// F1 with an explicit averaging strategy.
pub fn f1_score_with(y_true: &[f64], y_pred: &[f64], average: Average) -> Result<f64> {
    f1_score_with_options(y_true, y_pred, average, ZeroDivision::Zero)
}

/// Precision with explicit averaging and zero-division policies.
pub fn precision_score_with_options(
    y_true: &[f64],
    y_pred: &[f64],
    average: Average,
    zero_division: ZeroDivision,
) -> Result<f64> {
    averaged_score(
        y_true,
        y_pred,
        average,
        MetricKind::Precision,
        zero_division,
    )
}

/// Recall with explicit averaging and zero-division policies.
pub fn recall_score_with_options(
    y_true: &[f64],
    y_pred: &[f64],
    average: Average,
    zero_division: ZeroDivision,
) -> Result<f64> {
    averaged_score(y_true, y_pred, average, MetricKind::Recall, zero_division)
}

/// F1 with explicit averaging and zero-division policies.
pub fn f1_score_with_options(
    y_true: &[f64],
    y_pred: &[f64],
    average: Average,
    zero_division: ZeroDivision,
) -> Result<f64> {
    averaged_score(y_true, y_pred, average, MetricKind::F1, zero_division)
}

/// Precision macro-averaged over every observed class.
///
/// This compatibility helper is equivalent to
/// `precision_score_with(y_true, y_pred, Average::Macro)` for both binary and
/// multiclass inputs.
pub fn precision_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    precision_score_with(y_true, y_pred, Average::Macro)
}

/// Recall macro-averaged over every observed class.
///
/// This compatibility helper is equivalent to
/// `recall_score_with(y_true, y_pred, Average::Macro)`.
pub fn recall_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    recall_score_with(y_true, y_pred, Average::Macro)
}

/// F1 macro-averaged over every observed class.
///
/// This compatibility helper is equivalent to
/// `f1_score_with(y_true, y_pred, Average::Macro)`.
pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    f1_score_with(y_true, y_pred, Average::Macro)
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

    #[test]
    fn gapped_labels_are_compacted_and_retain_mapping() {
        let truth = vec![10.0, 10.0, 20.0, 20.0];
        let predicted = truth.clone();
        let labeled = confusion_matrix_labeled(&truth, &predicted).unwrap();

        assert_eq!(labeled.labels, vec![10.0, 20.0]);
        assert_eq!(labeled.counts, vec![vec![2, 0], vec![0, 2]]);
        assert_eq!(confusion_matrix(&truth, &predicted).unwrap().len(), 2);
        assert!((f1_score(&truth, &predicted).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn large_labels_do_not_control_matrix_allocation() {
        let truth = vec![10.0, 1_000_000_000_000.0];
        let labeled = confusion_matrix_labeled(&truth, &truth).unwrap();

        assert_eq!(labeled.labels, truth);
        assert_eq!(labeled.counts, vec![vec![1, 0], vec![0, 1]]);
    }

    #[test]
    fn fractional_and_non_finite_labels_are_rejected() {
        for invalid in [1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = confusion_matrix(&[0.0, invalid], &[0.0, 1.0]).unwrap_err();
            assert!(matches!(err, DatarustError::InvalidInput(_)));
        }
    }

    #[test]
    fn label_space_encodes_and_decodes_original_values() {
        let space = LabelSpace::fit(&[20.0, 10.0, 20.0, -0.0]).unwrap();
        assert_eq!(space.labels(), &[0.0, 10.0, 20.0]);
        assert_eq!(space.encode(10.0).unwrap(), 1);
        assert_eq!(space.decode(2).unwrap(), 20.0);
        assert!(space.encode(30.0).is_err());
        assert!(space.decode(3).is_err());
    }

    #[test]
    fn explicit_averaging_strategies_match_hand_computation() {
        // cm = [[2, 1], [0, 2]]
        let truth = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let predicted = vec![0.0, 0.0, 1.0, 1.0, 1.0];

        assert!(
            (precision_score_with(
                &truth,
                &predicted,
                Average::Binary {
                    positive_label: 1.0,
                },
            )
            .unwrap()
                - 2.0 / 3.0)
                .abs()
                < 1e-12
        );
        assert!(
            (recall_score_with(
                &truth,
                &predicted,
                Average::Binary {
                    positive_label: 1.0,
                },
            )
            .unwrap()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            (f1_score_with(
                &truth,
                &predicted,
                Average::Binary {
                    positive_label: 1.0,
                },
            )
            .unwrap()
                - 0.8)
                .abs()
                < 1e-12
        );

        assert!(
            (precision_score_with(&truth, &predicted, Average::Macro).unwrap() - 5.0 / 6.0).abs()
                < 1e-12
        );
        assert!(
            (precision_score_with(&truth, &predicted, Average::Weighted).unwrap() - 13.0 / 15.0)
                .abs()
                < 1e-12
        );
        assert!(
            (recall_score_with(&truth, &predicted, Average::Weighted).unwrap() - 0.8).abs() < 1e-12
        );
        assert!((f1_score_with(&truth, &predicted, Average::Micro).unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn classification_report_uses_original_labels() {
        let truth = vec![10.0, 10.0, 20.0];
        let predicted = vec![10.0, 20.0, 20.0];
        let report = classification_report(&truth, &predicted).unwrap();

        assert_eq!(report.len(), 2);
        assert_eq!(report[0].label, 10.0);
        assert_eq!(report[0].support, 2);
        assert_eq!(report[1].label, 20.0);
        assert_eq!(report[1].support, 1);
    }

    #[test]
    fn binary_average_rejects_multiclass_input() {
        let labels = vec![0.0, 1.0, 2.0];
        let err = f1_score_with(
            &labels,
            &labels,
            Average::Binary {
                positive_label: 1.0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, DatarustError::InvalidInput(_)));
    }

    #[test]
    fn zero_division_policy_can_return_zero_or_error() {
        let truth = vec![0.0, 1.0];
        let predicted = vec![0.0, 0.0];
        let average = Average::Binary {
            positive_label: 1.0,
        };

        assert_eq!(
            precision_score_with_options(&truth, &predicted, average, ZeroDivision::Zero).unwrap(),
            0.0
        );
        let err = precision_score_with_options(&truth, &predicted, average, ZeroDivision::Error)
            .unwrap_err();
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
