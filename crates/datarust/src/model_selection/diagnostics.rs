//! CV-driven diagnostics curves, mirroring
//! `sklearn.model_selection.validation_curve` and `learning_curve`.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::cross_val::{checked_folds, Samples};
use crate::model_selection::cross_validator::CrossValidator;
use crate::traits::{ParamValue, Params, Predictor};

fn train_test_scores<T, F>(
    mut model: T,
    train: &Samples,
    test: &Samples,
    scorer: F,
) -> Result<(f64, f64)>
where
    T: Predictor,
    F: Fn(&[f64], &[f64]) -> Result<f64> + Copy,
{
    model.fit(&train.x, &train.y)?;
    Ok((train.score(&model, scorer)?, test.score(&model, scorer)?))
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
/// [`Params::set_params`] and evaluated by
/// cross-validation, scoring both the training and test folds. This shows
/// under- vs. over-fitting as the parameter moves.
///
/// Returns `(train_scores, validation_scores)`, each indexed
/// `[param_value][fold]`.
///
/// ```rust,ignore
/// use datarust::linear_model::Ridge;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{KFold, validation_curve};
/// use datarust::traits::ParamValue;
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

    let mut train_curve = vec![Vec::new(); param_range.len()];
    let mut test_curve = vec![Vec::new(); param_range.len()];
    for (train, test) in checked_folds(cv, x, y)? {
        let train = Samples::select(x, y, &train)?;
        let test = Samples::select(x, y, &test)?;
        for (point, value) in param_range.iter().enumerate() {
            let mut model = estimator.clone();
            model.set_params(param_name, value.clone())?;
            let (training, validation) = train_test_scores(model, &train, &test, scorer)?;
            train_curve[point].push(training);
            test_curve[point].push(validation);
        }
    }

    Ok((train_curve, test_curve))
}

/// Learning curve over increasing training-set sizes.
///
/// `train_sizes` holds fractions in `(0, 1]` of the first fold's training
/// size. Each fold uses the first requested number of training rows, or
/// all its training rows if it is smaller, matching sklearn's
/// `shuffle=False` behavior.
///
/// Returns `(used_sizes, train_scores, validation_scores)` with the absolute
/// requested sample counts (deduplicated, ascending) and scores indexed
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

    let mut folds = checked_folds(cv, x, y)?;
    let max_train = folds.peek().expect("checked_folds requires a fold").0.len();
    if max_train == 0 {
        return Err(DatarustError::EmptyInput(
            "first training fold is empty".into(),
        ));
    }

    let mut sizes: Vec<usize> = train_sizes
        .iter()
        .map(|&s| ((s * max_train as f64).floor() as usize).max(1))
        .collect();
    sizes.sort_unstable();
    sizes.dedup();

    let mut train_curve = vec![Vec::new(); sizes.len()];
    let mut test_curve = vec![Vec::new(); sizes.len()];
    for (train_indices, test) in folds {
        let test = Samples::select(x, y, &test)?;
        for (point, &size) in sizes.iter().enumerate() {
            let n_train = size.min(train_indices.len());
            let train = Samples::select(x, y, &train_indices[..n_train])?;
            let (training, validation) =
                train_test_scores(estimator.clone(), &train, &test, scorer)?;
            train_curve[point].push(training);
            test_curve[point].push(validation);
        }
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
