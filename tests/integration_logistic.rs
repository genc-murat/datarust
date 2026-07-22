//! Integration tests for `linear_model::LogisticRegression` and
//! `metrics::classification`.

use datarust::linear_model::LogisticRegression;
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
};
use datarust::traits::{Classifier, Predictor};
use datarust::Matrix;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Linearly separable: y = 1 when x0 > 0, no sample at the boundary.
fn separable() -> (Matrix, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (-5..=5)
        .map(|i| vec![i as f64 * 0.5])
        .filter(|r| r[0].abs() > 0.01)
        .collect();
    let x = Matrix::new(rows.clone()).unwrap();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 })
        .collect();
    (x, y)
}

#[test]
fn classifies_separable_data_perfectly() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new().with_max_iter(100);
    model.fit(&x, &y).unwrap();
    let classes = model.predict_class(&x).unwrap();
    assert_eq!(classes, y);
}

#[test]
fn score_is_one_on_separable_data() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    let acc = model.score(&x, &y).unwrap();
    assert!(approx(acc, 1.0, 1e-9));
}

#[test]
fn predict_returns_hard_labels() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    assert_eq!(model.predict(&x).unwrap(), y);
}

#[test]
fn predict_positive_proba_returns_probabilities() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    let probs = model.predict_positive_proba(&x).unwrap();
    for &p in &probs {
        assert!((0.0..=1.0).contains(&p));
    }
    // Negative samples should have p < 0.5, positive p > 0.5.
    for (&p, &t) in probs.iter().zip(y.iter()) {
        if t == 0.0 {
            assert!(p < 0.5);
        } else {
            assert!(p > 0.5);
        }
    }
}

#[test]
fn predict_class_thresholds_at_half() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    let probs = model.predict_positive_proba(&x).unwrap();
    let classes = model.predict_class(&x).unwrap();
    for (&p, &c) in probs.iter().zip(classes.iter()) {
        let expected = if p >= 0.5 { 1.0 } else { 0.0 };
        assert_eq!(c, expected);
    }
}

#[test]
fn n_iter_positive_and_bounded() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new().with_max_iter(50);
    model.fit(&x, &y).unwrap();
    assert!(model.n_iter() > 0);
    assert!(model.n_iter() <= 50);
}

#[test]
fn non_integer_labels_rejected() {
    // Multiclass integer labels {0, 1, 2} are now valid; only non-integer
    // values such as 2.5 are rejected.
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let mut model = LogisticRegression::new();
    let err = model.fit(&x, &[0.0, 1.0, 2.5]).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::InvalidInput(_)));
}

#[test]
fn predict_before_fit_errors() {
    let model = LogisticRegression::new();
    let x = Matrix::new(vec![vec![1.0]]).unwrap();
    assert!(matches!(
        model.predict(&x).unwrap_err(),
        datarust::DatarustError::NotFitted(_)
    ));
}

#[test]
fn predict_shape_mismatch_errors() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    let bad = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
    assert!(matches!(
        model.predict(&bad).unwrap_err(),
        datarust::DatarustError::ShapeMismatch { .. }
    ));
}

#[test]
fn is_fitted_flag() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    assert!(!model.is_fitted());
    model.fit(&x, &y).unwrap();
    assert!(model.is_fitted());
}

#[test]
fn fit_intercept_false_still_classifies() {
    let (x, y) = separable();
    let mut model = LogisticRegression::new()
        .with_fit_intercept(false)
        .with_max_iter(100);
    model.fit(&x, &y).unwrap();
    assert!(model.intercept()[0].abs() < 1e-12);
    let classes = model.predict_class(&x).unwrap();
    assert_eq!(classes, y);
}

#[test]
fn metrics_known_values() {
    let y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y_pred = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    // cm = [[2,1],[1,3]], accuracy=5/7.
    // Macro precision/recall/f1: per-class then averaged → 17/24.
    assert!((accuracy_score(&y_true, &y_pred).unwrap() - 5.0 / 7.0).abs() < 1e-12);
    assert!((precision_score(&y_true, &y_pred).unwrap() - 17.0 / 24.0).abs() < 1e-12);
    assert!((recall_score(&y_true, &y_pred).unwrap() - 17.0 / 24.0).abs() < 1e-12);
    assert!((f1_score(&y_true, &y_pred).unwrap() - 17.0 / 24.0).abs() < 1e-12);
    assert_eq!(
        confusion_matrix(&y_true, &y_pred).unwrap(),
        vec![vec![2, 1], vec![1, 3]]
    );
}

#[test]
fn log_loss_known_value() {
    let y_true = vec![0.0, 0.0, 1.0, 1.0];
    let p = vec![0.1, 0.2, 0.8, 0.9];
    let ll = log_loss(&y_true, &p, 1e-15).unwrap();
    let expected = -(0.9_f64.ln() + 0.8_f64.ln() + 0.8_f64.ln() + 0.9_f64.ln()) / 4.0;
    assert!((ll - expected).abs() < 1e-12);
}

#[test]
fn perfect_classifier_metrics() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    assert!((accuracy_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    assert!((precision_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    assert!((recall_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
    assert!((f1_score(&y, &y).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn metrics_length_mismatch_errors() {
    let err = accuracy_score(&[1.0, 0.0], &[1.0]).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::ShapeMismatch { .. }));
}

#[test]
fn metrics_empty_errors() {
    let err = accuracy_score(&[], &[]).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::EmptyInput(_)));
}

#[test]
fn logistic_implements_classifier_trait() {
    fn predict_via_trait<C: Classifier>(m: &C, x: &Matrix) -> usize {
        Predictor::predict(m, x).map(|v| v.len()).unwrap_or(0)
    }
    let (x, y) = separable();
    let mut model = LogisticRegression::new();
    model.fit(&x, &y).unwrap();
    assert_eq!(predict_via_trait(&model, &x), x.nrows());
}

// ── Multiclass integration tests ───────────────────────────────────────

/// Three linearly separable clusters in 2-D, placed at triangle vertices.
fn separable_3class_2d() -> (Matrix, Vec<f64>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut y = Vec::new();
    for _ in 0..15 {
        rows.push(vec![-5.0, -5.0]);
        y.push(0.0);
    }
    for _ in 0..15 {
        rows.push(vec![0.0, 5.0]);
        y.push(1.0);
    }
    for _ in 0..15 {
        rows.push(vec![5.0, -5.0]);
        y.push(2.0);
    }
    (Matrix::new(rows).unwrap(), y)
}

#[test]
fn multiclass_fits_and_predicts() {
    let (x, y) = separable_3class_2d();
    let mut model = LogisticRegression::new().with_max_iter(200);
    model.fit(&x, &y).unwrap();
    assert_eq!(model.classes(), [0.0, 1.0, 2.0]);
    let pred = model.predict(&x).unwrap();
    assert_eq!(pred.len(), y.len());
    let acc = model.score(&x, &y).unwrap();
    assert!((acc - 1.0).abs() < 1e-9, "multiclass accuracy={acc}");
}

#[test]
fn multiclass_predict_proba_columns_match_classes() {
    let (x, y) = separable_3class_2d();
    let mut model = LogisticRegression::new().with_max_iter(100);
    model.fit(&x, &y).unwrap();
    let probs = model.predict_proba(&x).unwrap();
    assert_eq!(probs.ncols(), 3);
    for i in 0..x.nrows() {
        let sum: f64 = (0..3).map(|c| probs.get(i, c)).sum();
        assert!((sum - 1.0).abs() < 1e-9, "row {i} sums to {sum}");
        for c in 0..3 {
            assert!((0.0..=1.0).contains(&probs.get(i, c)));
        }
    }
}

#[test]
fn multiclass_confusion_matrix_is_diagonal_on_separable() {
    let (x, y) = separable_3class_2d();
    let mut model = LogisticRegression::new().with_max_iter(200);
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    let cm = confusion_matrix(&y, &pred).unwrap();
    assert_eq!(cm.len(), 3);
    // Perfectly classified → off-diagonal entries are zero.
    for (i, row) in cm.iter().enumerate() {
        for (j, &count) in row.iter().enumerate() {
            if i == j {
                assert_eq!(count, 15);
            } else {
                assert_eq!(count, 0, "cm[{i}][{j}]={count}");
            }
        }
    }
}

#[test]
fn multiclass_macro_metrics_one_on_perfect() {
    let (x, y) = separable_3class_2d();
    let mut model = LogisticRegression::new().with_max_iter(200);
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    assert!((precision_score(&y, &pred).unwrap() - 1.0).abs() < 1e-9);
    assert!((recall_score(&y, &pred).unwrap() - 1.0).abs() < 1e-9);
    assert!((f1_score(&y, &pred).unwrap() - 1.0).abs() < 1e-9);
}
