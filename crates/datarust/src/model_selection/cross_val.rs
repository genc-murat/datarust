//! Cross-validation scoring driver mirroring `sklearn.model_selection.cross_val_score`.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::cross_validator::{CrossValidator, FoldIter};
use crate::traits::{PredictProba, Predictor};

/// Extract the second probability column for a binary classifier. Use this
/// response when the metric's positive label is the second fitted class.
pub fn binary_positive_probability<T: PredictProba>(estimator: &T, x: &Matrix) -> Result<Vec<f64>> {
    let probabilities = estimator.predict_proba(x)?;
    if probabilities.ncols() != 2 {
        return Err(DatarustError::InvalidInput(
            "binary probability scoring requires two fitted classes".into(),
        ));
    }
    Ok(probabilities.col(1))
}

pub(super) fn default_cv(classification: bool) -> Box<dyn CrossValidator> {
    if classification {
        Box::new(super::StratifiedKFold::new())
    } else {
        Box::new(super::KFold::new())
    }
}

/// Evaluate an estimator by cross-validation.
///
/// For each fold the estimator is cloned, fit on the training split, and scored
/// on the test split using the provided `scorer` function. Returns one score
/// per fold.
///
/// `scorer` is any closure `Fn(&[f64], &[f64]) -> Result<f64>` — typically
/// [`r2_score`](crate::metrics::regression::r2_score) for regression or
/// [`accuracy_score`](crate::metrics::classification::accuracy_score) for
/// classification. The estimator must implement [`Predictor`] and [`Clone`].
/// The cross-validator implements [`CrossValidator`] (see
/// [`KFold`](crate::model_selection::KFold) and
/// [`StratifiedKFold`](crate::model_selection::StratifiedKFold)).
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
pub fn cross_val_score<T, S, F>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    cv: &S,
    scorer: F,
) -> Result<Vec<f64>>
where
    T: Predictor + Clone,
    S: CrossValidator + ?Sized,
    F: Fn(&[f64], &[f64]) -> Result<f64>,
{
    cross_val_score_with_response(estimator, x, y, cv, scorer, T::predict)
}

/// Evaluate a metric using a chosen estimator response on each held-out fold.
/// Pass [`binary_positive_probability`] for binary probability metrics or a
/// function returning a decision score for each row.
pub fn cross_val_score_with_response<T, S, F, R>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    cv: &S,
    scorer: F,
    response: R,
) -> Result<Vec<f64>>
where
    T: Predictor + Clone,
    S: CrossValidator + ?Sized,
    F: Fn(&[f64], &[f64]) -> Result<f64>,
    R: Fn(&T, &Matrix) -> Result<Vec<f64>>,
{
    let mut scores = Vec::new();
    for (train, test) in checked_folds(cv, x, y)? {
        let train = Samples::select(x, y, &train)?;
        let test = Samples::select(x, y, &test)?;
        let mut model = estimator.clone();
        model.fit(&train.x, &train.y)?;
        scores.push(scorer(&test.y, &response(&model, &test.x)?)?);
    }
    Ok(scores)
}

pub(super) fn checked_folds<'a, S: CrossValidator + ?Sized>(
    cv: &'a S,
    x: &'a Matrix,
    y: &'a [f64],
) -> Result<std::iter::Peekable<FoldIter<'a>>> {
    let n = x.nrows();
    if y.len() != n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} targets", n),
            actual: format!("{} targets", y.len()),
        });
    }
    let mut folds = cv.split_iter(x, y)?.peekable();
    if folds.peek().is_none() {
        return Err(DatarustError::EmptyInput(
            "cross-validator produced no folds".into(),
        ));
    }
    Ok(folds)
}

/// Features and targets for one side of a fold, shared by candidates.
pub(super) struct Samples {
    pub x: Matrix,
    pub y: Vec<f64>,
}

impl Samples {
    pub fn select(x: &Matrix, y: &[f64], indices: &[usize]) -> Result<Self> {
        Ok(Self {
            x: x.select_rows(indices)?,
            y: indices.iter().map(|&i| y[i]).collect(),
        })
    }

    pub fn score<T, F>(&self, model: &T, scorer: F) -> Result<f64>
    where
        T: Predictor,
        F: Fn(&[f64], &[f64]) -> Result<f64>,
    {
        scorer(&self.y, &model.predict(&self.x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::{LinearRegression, LogisticRegression};
    use crate::metrics::classification::accuracy_score;
    use crate::metrics::regression::r2_score;
    use crate::model_selection::kfold::KFold;

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

    #[test]
    fn target_length_mismatch_is_rejected_before_indexing() {
        let (x, _) = regression_data(10);
        let cv = KFold::new().with_n_splits(2);
        let err = cross_val_score(&LinearRegression::new(), &x, &[1.0], &cv, r2_score).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }
}
