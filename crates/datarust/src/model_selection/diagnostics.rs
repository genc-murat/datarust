//! CV-driven diagnostics curves, mirroring
//! `sklearn.model_selection.validation_curve` and `learning_curve`.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::cross_validator::CrossValidator;
use crate::traits::{ParamValue, Params, Predictor};

/// Fit-and-score every CV fold, scoring both the training and test splits.
///
/// Returns `(train_scores, test_scores)` indexed `[fold]`. When `subsample`
/// is `Some(k)`, the training split is truncated to its first `k` samples
/// (used by [`learning_curve`]); scoring of the train split happens after
/// fitting on the (possibly truncated) subset.
fn cv_train_test_scores<T, S, F>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    cv: &S,
    scorer: F,
    subsample: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>)>
where
    T: Predictor + Clone,
    S: CrossValidator + ?Sized,
    F: Fn(&[f64], &[f64]) -> Result<f64>,
{
    let n = x.nrows();
    if y.len() != n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} rows in y", n),
            actual: format!("{} rows in y", y.len()),
        });
    }

    let mut train_scores = Vec::new();
    let mut test_scores = Vec::new();
    for (train_idx, test_idx) in cv.split(x, y)? {
        let train_idx = match subsample {
            Some(k) if train_idx.len() > k => train_idx[..k].to_vec(),
            _ => train_idx,
        };
        let x_train = x.select_rows(&train_idx)?;
        let x_test = x.select_rows(&test_idx)?;
        let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

        let mut model = estimator.clone();
        model.fit(&x_train, &y_train)?;
        train_scores.push(scorer(&y_train, &model.predict(&x_train)?)?);
        test_scores.push(scorer(&y_test, &model.predict(&x_test)?)?);
    }

    Ok((train_scores, test_scores))
}

/// Scores of a validation curve: `(train_scores, validation_scores)`, each
/// indexed `[param_value][fold]`.
pub type ValidationCurveScores = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// A learning curve: `(used_sizes, train_scores, validation_scores)` with
/// scores indexed `[size][fold]`.
pub type LearningCurveScores = (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Validation curve over a single hyperparameter.
///
/// For each value in `param_range` the estimator is parameterized via
/// [`Params::set_params`](crate::traits::Params::set_params) and evaluated by
/// cross-validation, scoring both the training and test folds. This shows
/// under- vs. over-fitting as the parameter moves.
///
/// Returns `(train_scores, validation_scores)`, each indexed
/// `[param_value][fold]`.
///
/// ```rust,ignore
/// use datarust::linear_model::Ridge;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{KFold, ParamValue, validation_curve};
///
/// let (train, test) = validation_curve(
///     &Ridge::new(),
///     &x, &y,
///     "alpha",
///     &[ParamValue::Float(0.1), ParamValue::Float(1.0), ParamValue::Float(10.0)],
///     &KFold::new().with_n_splits(3),
///     r2_score,
/// )?;
/// ```
pub fn validation_curve<T, S, F>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    param_name: &str,
    param_range: &[ParamValue],
    cv: &S,
    scorer: F,
) -> Result<ValidationCurveScores>
where
    T: Predictor + Params + Clone,
    S: CrossValidator + ?Sized,
    F: Fn(&[f64], &[f64]) -> Result<f64> + Copy,
{
    if param_range.is_empty() {
        return Err(DatarustError::EmptyInput("param_range is empty".into()));
    }

    let mut train_curve = Vec::with_capacity(param_range.len());
    let mut test_curve = Vec::with_capacity(param_range.len());
    for value in param_range {
        let mut model = estimator.clone();
        model.set_params(param_name, value.clone())?;
        let (train, test) = cv_train_test_scores(&model, x, y, cv, scorer, None)?;
        train_curve.push(train);
        test_curve.push(test);
    }

    Ok((train_curve, test_curve))
}

/// Learning curve over increasing training-set sizes.
///
/// `train_sizes` holds fractions in `(0, 1]` of each fold's training split.
/// For each size the training split is truncated to its first
/// `floor(fraction * n_train)` samples (sklearn's `shuffle=False` behavior),
/// fit, and scored on both the truncated train split and the untouched test
/// fold.
///
/// Returns `(used_sizes, train_scores, validation_scores)` with the absolute
/// sample counts actually used (deduplicated, ascending) and scores indexed
/// `[size][fold]`.
///
/// ```rust,ignore
/// use datarust::linear_model::LinearRegression;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{KFold, learning_curve};
///
/// let (sizes, train, test) = learning_curve(
///     &LinearRegression::new(),
///     &x, &y,
///     &[0.25, 0.5, 0.75, 1.0],
///     &KFold::new().with_n_splits(3),
///     r2_score,
/// )?;
/// ```
pub fn learning_curve<T, S, F>(
    estimator: &T,
    x: &Matrix,
    y: &[f64],
    train_sizes: &[f64],
    cv: &S,
    scorer: F,
) -> Result<LearningCurveScores>
where
    T: Predictor + Clone,
    S: CrossValidator + ?Sized,
    F: Fn(&[f64], &[f64]) -> Result<f64> + Copy,
{
    if train_sizes.is_empty() {
        return Err(DatarustError::EmptyInput("train_sizes is empty".into()));
    }
    if train_sizes
        .iter()
        .any(|&s| !(s.is_finite() && s > 0.0 && s <= 1.0))
    {
        return Err(DatarustError::InvalidInput(
            "train_sizes must contain fractions in (0, 1]".into(),
        ));
    }

    // First fold determines the size axis (all folds share the same
    // train-split size under KFold/StratifiedKFold).
    let folds = cv.split(x, y)?;
    let max_train = folds
        .iter()
        .map(|(train, _)| train.len())
        .min()
        .ok_or_else(|| DatarustError::EmptyInput("cross-validator produced no folds".into()))?;

    let mut sizes: Vec<usize> = train_sizes
        .iter()
        .map(|&s| ((s * max_train as f64).floor() as usize).max(1))
        .collect();
    sizes.sort_unstable();
    sizes.dedup();

    let mut train_curve = Vec::with_capacity(sizes.len());
    let mut test_curve = Vec::with_capacity(sizes.len());
    for &size in &sizes {
        let (train, test) = cv_train_test_scores(estimator, x, y, cv, scorer, Some(size))?;
        train_curve.push(train);
        test_curve.push(test);
    }

    Ok((sizes, train_curve, test_curve))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::{LinearRegression, Ridge};
    use crate::metrics::regression::r2_score;
    use crate::model_selection::KFold;

    fn make_data() -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let i = i as f64;
                vec![i, i.sin()]
            })
            .collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn validation_curve_shape_and_best_small_alpha() {
        let (x, y) = make_data();
        let cv = KFold::new().with_n_splits(3);
        let (train, test) = validation_curve(
            &Ridge::new(),
            &x,
            &y,
            "alpha",
            &[
                ParamValue::Float(0.01),
                ParamValue::Float(1.0),
                ParamValue::Float(100.0),
            ],
            &cv,
            r2_score,
        )
        .unwrap();

        assert_eq!(train.len(), 3);
        assert_eq!(test.len(), 3);
        assert!(train.iter().all(|f| f.len() == 3));
        assert!(test.iter().all(|f| f.len() == 3));
        // Least regularization fits this linear relationship best.
        assert!(test[0].iter().sum::<f64>() > test[2].iter().sum::<f64>());
        // Train scores degrade as alpha grows.
        assert!(train[0].iter().sum::<f64>() > train[2].iter().sum::<f64>());
    }

    #[test]
    fn validation_curve_unknown_param_errors() {
        let (x, y) = make_data();
        assert!(validation_curve(
            &Ridge::new(),
            &x,
            &y,
            "nope",
            &[ParamValue::Float(1.0)],
            &KFold::new(),
            r2_score,
        )
        .is_err());
    }

    #[test]
    fn learning_curve_sizes_and_shapes() {
        let (x, y) = make_data();
        let cv = KFold::new().with_n_splits(4);
        let (sizes, train, test) = learning_curve(
            &LinearRegression::new(),
            &x,
            &y,
            &[0.25, 0.5, 1.0],
            &cv,
            r2_score,
        )
        .unwrap();

        // 40 samples, 4 folds → train split is 30 rows.
        assert_eq!(sizes, vec![7, 15, 30]);
        assert_eq!(train.len(), 3);
        assert_eq!(test.len(), 3);
        assert!(train.iter().chain(&test).all(|f| f.len() == 4));
        // A linear model on linear data scores ~1 even with little data.
        assert!(test[0].iter().all(|s| *s > 0.9));
    }

    #[test]
    fn learning_curve_rejects_invalid_sizes() {
        let (x, y) = make_data();
        assert!(learning_curve(
            &LinearRegression::new(),
            &x,
            &y,
            &[],
            &KFold::new(),
            r2_score,
        )
        .is_err());
        assert!(learning_curve(
            &LinearRegression::new(),
            &x,
            &y,
            &[0.5, 1.5],
            &KFold::new(),
            r2_score,
        )
        .is_err());
    }
}
