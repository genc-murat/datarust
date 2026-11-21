//! Exhaustive grid search over parameter combinations, mirroring
//! `sklearn.model_selection.GridSearchCV`.

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::cross_val::{checked_folds, Samples};
use crate::model_selection::cross_validator::CrossValidator;
use crate::traits::{Classifier, ParamValue, Params, PredictProba, Predictor, Regressor};

/// A single parameter grid point: parameter name and candidate values.
pub type ParamGrid = Vec<(&'static str, Vec<ParamValue>)>;

/// Cross-validation results for all candidates.
#[derive(Debug, Clone)]
pub struct CVResults {
    /// Mean test score per candidate.
    pub mean_test_score: Vec<f64>,
    /// Std dev of test scores per candidate.
    pub std_test_score: Vec<f64>,
    /// Raw fold scores: `[candidate][fold]`.
    pub split_scores: Vec<Vec<f64>>,
    /// Parameter combination for each candidate.
    pub params: Vec<Vec<(&'static str, ParamValue)>>,
    /// Fold index ranks per candidate.
    pub rank_test_score: Vec<usize>,
}

/// Exhaustive grid search over a parameter grid, using cross-validation.
///
/// Each combination in `param_grid` is evaluated on the same folds with the
/// provided scorer. Parameters are traversed in alphabetical order, with each
/// value list in its supplied order. When `refit` is `true`, the first
/// combination with the highest mean score is refit on the full dataset.
///
/// ```rust,ignore
/// use datarust::linear_model::Ridge;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{GridSearchCV, KFold};
/// use datarust::traits::ParamValue;
///
/// let mut gs = GridSearchCV::new(
///     Ridge::new(),
///     vec![("alpha", vec![ParamValue::Float(0.1), ParamValue::Float(1.0)])],
///     r2_score,
/// ).with_cv(KFold::new().with_n_splits(3));
/// gs.fit(&x, &y)?;
/// let best = gs.best_estimator().unwrap();
/// ```
#[derive(Debug)]
pub struct GridSearchCV<E: Predictor + Params + Clone> {
    estimator: E,
    param_grid: ParamGrid,
    cv: Box<dyn CrossValidator>,
    scorer: fn(&[f64], &[f64]) -> Result<f64>,
    response: fn(&E, &Matrix) -> Result<Vec<f64>>,
    refit: bool,
    // Fitted state
    cv_results_: Option<CVResults>,
    best_estimator_: Option<E>,
    best_params_: Option<Vec<(&'static str, ParamValue)>>,
    best_score_: Option<f64>,
    best_index_: Option<usize>,
    fitted: bool,
}

/// Cartesian product of lists of parameter values.
fn grid_product(grid: &ParamGrid) -> Vec<Vec<(&'static str, ParamValue)>> {
    if grid.is_empty() {
        return vec![vec![]];
    }
    let mut current: Vec<Vec<(&'static str, ParamValue)>> = vec![vec![]];
    for (name, values) in grid {
        let mut next = Vec::new();
        for prefix in &current {
            for value in values {
                let mut combo = prefix.clone();
                combo.push((*name, value.clone()));
                next.push(combo);
            }
        }
        current = next;
    }
    current
}

/// Apply a candidate parameter combination to a cloned estimator.
pub(crate) fn apply_candidate<E: Params + Clone>(
    estimator: &E,
    combo: &[(&'static str, ParamValue)],
) -> Result<E> {
    let mut model = estimator.clone();
    for (name, value) in combo {
        model.set_params(name, value.clone())?;
    }
    Ok(model)
}

/// Aggregate CV scores for a list of candidates: `(mean_scores, std_scores,
/// split_scores)`, one entry per candidate.
pub(crate) type CandidateScores = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

/// Score every candidate combination by cross-validation.
///
/// Returns one [`CandidateScores`] entry per candidate.
pub(crate) fn evaluate_candidates<E: Predictor + Params + Clone>(
    estimator: &E,
    combinations: &[Vec<(&'static str, ParamValue)>],
    x: &Matrix,
    y: &[f64],
    cv: &dyn CrossValidator,
    scorer: fn(&[f64], &[f64]) -> Result<f64>,
    response: fn(&E, &Matrix) -> Result<Vec<f64>>,
) -> Result<CandidateScores> {
    let mut all_split_scores = vec![Vec::new(); combinations.len()];
    for (train, test) in checked_folds(cv, x, y)? {
        let train = Samples::select(x, y, &train)?;
        let test = Samples::select(x, y, &test)?;
        for (combo, scores) in combinations.iter().zip(&mut all_split_scores) {
            let mut model = apply_candidate(estimator, combo)?;
            model.fit(&train.x, &train.y)?;
            scores.push(scorer(&test.y, &response(&model, &test.x)?)?);
        }
    }

    let mut mean_scores = Vec::with_capacity(combinations.len());
    let mut std_scores = Vec::with_capacity(combinations.len());
    for scores in &all_split_scores {
        let n_folds = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n_folds;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_folds;
        if !mean.is_finite() || !var.is_finite() {
            return Err(DatarustError::InvalidInput(
                "cross-validation scores must have a finite mean and variance".into(),
            ));
        }

        mean_scores.push(mean);
        std_scores.push(var.sqrt());
    }

    Ok((mean_scores, std_scores, all_split_scores))
}

/// Competition ranks for mean scores: highest mean gets rank 1, exact ties
/// share the same rank (1, 1, 3 — like sklearn's `rank_test_score`, which
/// compares scores for exact equality; identical candidates run identical
/// code paths, so their means are bit-identical here too).
pub(crate) fn rank_candidates(mean_scores: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = mean_scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0usize; mean_scores.len()];
    let mut current_rank = 1usize;
    for (idx, &(i, _)) in indexed.iter().enumerate() {
        if idx > 0 {
            let prev_score = indexed[idx - 1].1;
            if prev_score != mean_scores[i] {
                current_rank = idx + 1;
            }
        }
        ranks[i] = current_rank;
    }
    ranks
}

impl<E: Predictor + Params + Clone> GridSearchCV<E> {
    /// Create a grid search from a template estimator and parameter grid.
    pub fn new(
        estimator: E,
        mut param_grid: ParamGrid,
        scorer: fn(&[f64], &[f64]) -> Result<f64>,
    ) -> Self {
        param_grid.sort_by_key(|&(name, _)| name);
        let cv = super::cross_val::default_cv(estimator.is_classifier());
        Self {
            estimator,
            param_grid,
            cv,
            response: E::predict,
            scorer,
            refit: true,
            cv_results_: None,
            best_estimator_: None,
            best_params_: None,
            best_score_: None,
            best_index_: None,
            fitted: false,
        }
    }

    /// Choose the estimator output passed to the metric for CV and refit scoring.
    /// Defaults to [`Predictor::predict`]. For ROC-AUC with 0/1 labels use
    /// `.with_response(binary_positive_probability)`.
    pub fn with_response(mut self, response: fn(&E, &Matrix) -> Result<Vec<f64>>) -> Self {
        self.response = response;
        self
    }

    /// Override the default five-fold strategy (stratified for classifiers).
    pub fn with_cv(mut self, cv: impl CrossValidator + 'static) -> Self {
        self.cv = Box::new(cv);
        self
    }

    /// Whether to refit on the full dataset with the best parameters (default: true).
    pub fn with_refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// Run the grid search.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let combinations = grid_product(&self.param_grid);
        let n_candidates = combinations.len();
        if n_candidates == 0 {
            return Err(DatarustError::EmptyInput("param_grid is empty".into()));
        }

        let (mean_scores, std_scores, all_split_scores) = evaluate_candidates(
            &self.estimator,
            &combinations,
            x,
            y,
            self.cv.as_ref(),
            self.scorer,
            self.response,
        )?;
        let ranks = rank_candidates(&mean_scores);

        // Keep the first candidate at the highest mean.
        let mut best_index = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &mean) in mean_scores.iter().enumerate() {
            if mean > best_score {
                best_score = mean;
                best_index = i;
            }
        }

        let best_estimator = if self.refit {
            let mut best = apply_candidate(&self.estimator, &combinations[best_index])?;
            best.fit(x, y)?;
            Some(best)
        } else {
            None
        };
        self.best_params_ = Some(combinations[best_index].clone());
        self.cv_results_ = Some(CVResults {
            mean_test_score: mean_scores,
            std_test_score: std_scores,
            split_scores: all_split_scores,
            params: combinations,
            rank_test_score: ranks,
        });

        self.best_score_ = Some(best_score);
        self.best_index_ = Some(best_index);
        self.best_estimator_ = best_estimator;
        self.fitted = true;
        Ok(())
    }

    /// Cross-validation results (available after `fit`).
    pub fn cv_results(&self) -> Option<&CVResults> {
        self.cv_results_.as_ref()
    }

    /// Refitted estimator with the best parameters (available after `fit` with
    /// `refit=true`).
    pub fn best_estimator(&self) -> Option<&E> {
        self.best_estimator_.as_ref()
    }

    /// Best parameter combination.
    pub fn best_params(&self) -> Option<&[(&'static str, ParamValue)]> {
        self.best_params_.as_deref()
    }

    /// Mean cross-validation score of the best candidate.
    pub fn best_score(&self) -> Option<f64> {
        self.best_score_
    }

    /// Score the refitted estimator on the supplied data with the search scorer.
    pub fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let estimator = self
            .best_estimator_
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("GridSearchCV".into()))?;
        (self.scorer)(y, &(self.response)(estimator, x)?)
    }

    /// Index of the best candidate in `cv_results().params`.
    pub fn best_index(&self) -> Option<usize> {
        self.best_index_
    }
}

impl<E: Predictor + Params + Clone> crate::traits::Estimator for GridSearchCV<E> {}

impl<E: Predictor + Params + Clone> crate::traits::Predictor for GridSearchCV<E> {
    fn is_classifier(&self) -> bool {
        self.estimator.is_classifier()
    }

    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        GridSearchCV::fit(self, x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.best_estimator_
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("GridSearchCV".into()))?
            .predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl<E: Predictor + PredictProba + Params + Clone> crate::traits::PredictProba for GridSearchCV<E> {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.best_estimator_
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("GridSearchCV".into()))?
            .predict_proba(x)
    }
}

impl<E: Predictor + Params + Clone + Classifier> crate::traits::Classifier for GridSearchCV<E> {
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        GridSearchCV::score(self, x, y)
    }
}

impl<E: Predictor + Params + Clone + Regressor> crate::traits::Regressor for GridSearchCV<E> {
    fn name(&self) -> &'static str {
        "GridSearchCV"
    }

    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        GridSearchCV::score(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::Ridge;
    use crate::metrics::regression::r2_score;
    use crate::model_selection::KFold;

    fn make_data() -> (Matrix, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let i = i as f64;
                vec![i, i.sin()]
            })
            .collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
        (Matrix::new(rows).unwrap(), y)
    }

    #[test]
    fn grid_search_finds_best_alpha() {
        let (x, y) = make_data();
        let mut gs = GridSearchCV::new(
            Ridge::new(),
            vec![(
                "alpha",
                vec![
                    ParamValue::Float(10.0),
                    ParamValue::Float(1.0),
                    ParamValue::Float(0.1),
                ],
            )],
            r2_score,
        )
        .with_cv(KFold::new().with_n_splits(3));

        gs.fit(&x, &y).unwrap();
        let results = gs.cv_results().unwrap();
        assert_eq!(results.mean_test_score.len(), 3);

        // Alpha=0.1 should be best (least regularization for this linear data)
        let best = gs.best_params().unwrap();
        let alpha = best.iter().find(|(n, _)| *n == "alpha").map(|(_, v)| v);
        assert!(matches!(alpha, Some(ParamValue::Float(a)) if *a == 0.1));

        assert!(gs.best_score().unwrap() > 0.99);
        assert!(gs.best_estimator().is_some());
    }

    #[test]
    fn grid_search_is_deterministic() {
        let (x, y) = make_data();
        let run = || {
            let mut gs = GridSearchCV::new(
                Ridge::new(),
                vec![(
                    "alpha",
                    vec![ParamValue::Float(0.1), ParamValue::Float(10.0)],
                )],
                r2_score,
            )
            .with_cv(KFold::new().with_n_splits(2));
            gs.fit(&x, &y).unwrap();
            gs.best_score().unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn grid_search_tie_break_first_candidate_wins() {
        // All candidates identical → first index wins
        let (x, y) = make_data();
        let mut gs = GridSearchCV::new(
            Ridge::new(),
            vec![(
                "alpha",
                vec![ParamValue::Float(1.0), ParamValue::Float(1.0)],
            )],
            r2_score,
        )
        .with_cv(KFold::new().with_n_splits(2));

        gs.fit(&x, &y).unwrap();
        assert_eq!(gs.best_index(), Some(0));
    }

    #[test]
    fn grid_search_refit_false_has_no_estimator() {
        let (x, y) = make_data();
        let mut gs = GridSearchCV::new(
            Ridge::new(),
            vec![("alpha", vec![ParamValue::Float(1.0)])],
            r2_score,
        )
        .with_refit(false);
        gs.fit(&x, &y).unwrap();
        assert!(gs.best_estimator().is_none());
        assert!(gs.is_fitted());
    }
}
