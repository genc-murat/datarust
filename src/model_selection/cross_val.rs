//! Cross-validation scoring driver mirroring `sklearn.model_selection.cross_val_score`.

use crate::error::Result;
use crate::matrix::Matrix;
use crate::model_selection::kfold::KFold;
use crate::traits::Predictor;

/// Evaluate an estimator by K-fold cross-validation.
///
/// For each fold the estimator is cloned, fit on the training split, and scored
/// on the test split using the provided `scorer` function. Returns one score
/// per fold.
///
/// `scorer` is any closure `Fn(&[f64], &[f64]) -> Result<f64>` — typically
/// [`r2_score`](crate::metrics::regression::r2_score) for regression or
/// [`accuracy_score`](crate::metrics::classification::accuracy_score) for
/// classification. The estimator must implement [`Predictor`] and [`Clone`].
///
/// ```rust
/// use datarust::linear_model::LinearRegression;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{cross_val_score, KFold};
/// use datarust::Matrix;
///
/// let rows: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i as f64).sin()]).collect();
/// let x = Matrix::new(rows.clone())?;
/// let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
///
/// let model = LinearRegression::new();
/// let cv = KFold::new().with_n_splits(3);
/// let scores = cross_val_score(&model, &x, &y, &cv, r2_score)?;
/// assert_eq!(scores.len(), 3);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub fn cross_val_score<T, F>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    cv: &KFold,
    scorer: F,
) -> Result<Vec<f64>>
where
    T: Predictor + Clone,
    F: Fn(&[f64], &[f64]) -> Result<f64>,
{
    let n = x.nrows();
    let mut scores = Vec::new();
    for (train_idx, test_idx) in cv.split(n)? {
        let x_train = x.select_rows(&train_idx)?;
        let x_test = x.select_rows(&test_idx)?;
        let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

        let mut model = estimator.clone();
        model.fit(&x_train, &y_train)?;
        let pred = model.predict(&x_test)?;
        scores.push(scorer(&y_test, &pred)?);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::{LinearRegression, LogisticRegression};
    use crate::metrics::classification::accuracy_score;
    use crate::metrics::regression::r2_score;

    fn regression_data(n: usize) -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let i = i as f64;
                vec![i, i.sin()]
            })
            .collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn returns_one_score_per_fold() {
        let (x, y) = regression_data(30);
        let cv = KFold::new().with_n_splits(5);
        let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, r2_score).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn high_score_on_clean_linear_signal() {
        let (x, y) = regression_data(30);
        let cv = KFold::new().with_n_splits(3);
        let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, r2_score).unwrap();
        for s in &scores {
            assert!(*s > 0.99, "low R² score: {s}");
        }
    }

    #[test]
    fn classification_uses_accuracy_scorer() {
        // Linearly separable binary data.
        let rows: Vec<Vec<f64>> = (-10..=10)
            .filter(|&i| i != 0)
            .map(|i| vec![i as f64 * 0.5])
            .collect();
        let x = Matrix::new(rows.clone()).unwrap();
        let y: Vec<f64> = rows
            .iter()
            .map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let cv = KFold::new().with_n_splits(3);
        let scores =
            cross_val_score(&LogisticRegression::new(), &x, &y, &cv, accuracy_score).unwrap();
        assert_eq!(scores.len(), 3);
        // Separable data → perfect accuracy on every fold.
        for s in &scores {
            assert!((*s - 1.0).abs() < 1e-9, "low accuracy: {s}");
        }
    }

    #[test]
    fn works_with_custom_closure_scorer() {
        let (x, y) = regression_data(20);
        let cv = KFold::new().with_n_splits(2);
        let mse_scorer = |y_true: &[f64], y_pred: &[f64]| {
            let n = y_true.len() as f64;
            let s: f64 = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum();
            Ok(s / n)
        };
        let scores = cross_val_score(&LinearRegression::new(), &x, &y, &cv, mse_scorer).unwrap();
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(*s >= 0.0);
        }
    }
}
