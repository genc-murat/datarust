//! Randomized hyperparameter search over parameter distributions, mirroring
//! `sklearn.model_selection.RandomizedSearchCV`.

use std::collections::{HashMap, HashSet};

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::model_selection::cross_validator::CrossValidator;
use crate::model_selection::grid_search::{
    apply_candidate, evaluate_candidates, rank_candidates, CVResults,
};
use crate::model_selection::rng::Rng;
use crate::traits::{ParamValue, Params, PredictProba, Predictor};

/// A parameter distribution to sample from, mirroring sklearn's
/// distributions-vs-lists distinction in `ParameterSampler`.
#[derive(Debug, Clone)]
pub enum ParamDistribution {
    /// Discrete choices; one value is drawn uniformly per candidate.
    Choices(Vec<ParamValue>),
    /// Uniform `f64` in `[lo, hi)`.
    Uniform(f64, f64),
    /// Uniform `usize` in `[lo, hi]` (inclusive, like `scipy.stats.randint`).
    RandInt(usize, usize),
}

/// A randomized parameter space: parameter name and its distribution.
pub type ParamSpace = Vec<(&'static str, ParamDistribution)>;

/// Draw `n_iter` candidate combinations from `space` using `rng`.
///
/// Sampling contract (deliberately NOT bit-exact with sklearn's
/// `ParameterSampler`; datarust uses its own xorshift64 stream):
///
/// - If every entry is [`ParamDistribution::Choices`], candidates are drawn
///   *without replacement* from the exhaustive cartesian product (sklearn
///   behavior for list-only spaces). At most `product size` distinct
///   candidates are returned.
/// - Otherwise each candidate draws every parameter independently *with
///   replacement*: `Choices` picks a random element, `Uniform` draws from
///   `[lo, hi)`, `RandInt` draws from the inclusive integer range.
fn sample_combinations(
    space: &ParamSpace,
    n_iter: usize,
    rng: &mut Rng,
) -> Result<Vec<Vec<(&'static str, ParamValue)>>> {
    validate_space(space)?;

    let all_choices = space
        .iter()
        .all(|(_, d)| matches!(d, ParamDistribution::Choices(_)));

    if all_choices {
        let choices: Vec<_> = space
            .iter()
            .map(|(name, d)| match d {
                ParamDistribution::Choices(values) => (*name, values.as_slice()),
                _ => unreachable!("checked above"),
            })
            .collect();
        Ok(sample_choices(&choices, n_iter, rng))
    } else {
        let mut combos = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let mut combo = Vec::with_capacity(space.len());
            for (name, dist) in space {
                let value = match dist {
                    ParamDistribution::Choices(values) => {
                        values[draw_index(rng, values.len())].clone()
                    }
                    ParamDistribution::Uniform(lo, hi) => {
                        ParamValue::Float(rng.next_range(*lo, *hi))
                    }
                    ParamDistribution::RandInt(lo, hi) => {
                        // `hi - lo + 1` overflows only for the full usize span
                        // (lo == 0, hi == usize::MAX); sample that degenerate
                        // case directly from the u64 stream instead.
                        let index = if *lo == 0 && *hi == usize::MAX {
                            rng.next_u64() as usize
                        } else {
                            draw_index(rng, *hi - *lo + 1)
                        };
                        ParamValue::Int(lo + index)
                    }
                };
                combo.push((*name, value));
            }
            combos.push(combo);
        }
        Ok(combos)
    }
}

/// Draw a uniform integer without the precision loss of a float conversion.
fn draw_index(rng: &mut Rng, bound: usize) -> usize {
    let bound = bound as u64;
    let cutoff = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= cutoff {
            return (value % bound) as usize;
        }
    }
}

fn sample_choices(
    choices: &[(&'static str, &[ParamValue])],
    n_iter: usize,
    rng: &mut Rng,
) -> Vec<Vec<(&'static str, ParamValue)>> {
    let product = choices
        .iter()
        .try_fold(1usize, |size, (_, values)| size.checked_mul(values.len()));
    let Some(product) = product else {
        // The product is too wide for an index. Draw one index per dimension
        // and keep unique tuples. The space is larger than usize::MAX.
        let mut seen = HashSet::new();
        let mut combinations = Vec::with_capacity(n_iter);
        while combinations.len() < n_iter {
            let indices: Vec<_> = choices
                .iter()
                .map(|(_, values)| draw_index(rng, values.len()))
                .collect();
            if seen.insert(indices.clone()) {
                combinations.push(
                    choices
                        .iter()
                        .zip(indices)
                        .map(|(&(name, values), index)| (name, values[index].clone()))
                        .collect(),
                );
            }
        }
        return combinations;
    };

    // Draw from a virtual Fisher–Yates shuffle, storing only moved indices.
    let count = n_iter.min(product);
    let mut moved = HashMap::with_capacity(count);
    let mut combinations = Vec::with_capacity(count);
    for remaining in (product - count..product).rev().map(|index| index + 1) {
        let position = draw_index(rng, remaining);
        let last = remaining - 1;
        let mut index = moved.remove(&position).unwrap_or(position);
        let replacement = moved.remove(&last).unwrap_or(last);
        if position != last {
            moved.insert(position, replacement);
        }
        let mut combination = Vec::with_capacity(choices.len());
        for &(name, values) in choices.iter().rev() {
            combination.push((name, values[index % values.len()].clone()));
            index /= values.len();
        }
        combination.reverse();
        combinations.push(combination);
    }
    combinations
}

fn validate_space(space: &ParamSpace) -> Result<()> {
    if space.is_empty() {
        return Err(DatarustError::EmptyInput("param_space is empty".into()));
    }
    for (name, dist) in space {
        match dist {
            ParamDistribution::Choices(values) if values.is_empty() => {
                return Err(DatarustError::InvalidInput(format!(
                    "parameter '{name}' has an empty choices list"
                )));
            }
            ParamDistribution::Uniform(lo, hi)
                if !lo.is_finite() || !hi.is_finite() || lo >= hi =>
            {
                return Err(DatarustError::InvalidInput(format!(
                    "parameter '{name}' requires finite lo < hi for Uniform"
                )));
            }
            ParamDistribution::RandInt(lo, hi) if lo > hi => {
                return Err(DatarustError::InvalidInput(format!(
                    "parameter '{name}' requires lo <= hi for RandInt"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Randomized search over a parameter space, using cross-validation.
///
/// Draws `n_iter` candidates from the [`ParamSpace`] with datarust's seeded
/// [`Rng`] (deterministic for a fixed `random_state`), evaluates
/// each by cross-validation, and keeps the best. See
/// [`GridSearchCV`](crate::model_selection::GridSearchCV) for the exhaustive
/// counterpart; result fields mirror it.
///
/// ```rust,ignore
/// use datarust::linear_model::Lasso;
/// use datarust::metrics::regression::r2_score;
/// use datarust::model_selection::{ParamDistribution, RandomizedSearchCV};
///
/// let mut rs = RandomizedSearchCV::new(
///     Lasso::new(),
///     vec![("alpha", ParamDistribution::Uniform(1e-4, 10.0))],
///     r2_score,
/// )
/// .with_n_iter(20)
/// .with_random_state(42);
/// rs.fit(&x, &y)?;
/// ```
#[derive(Debug)]
pub struct RandomizedSearchCV<E: Predictor + Params + Clone> {
    estimator: E,
    param_space: ParamSpace,
    cv: Box<dyn CrossValidator>,
    scorer: fn(&[f64], &[f64]) -> Result<f64>,
    n_iter: usize,
    response: fn(&E, &Matrix) -> Result<Vec<f64>>,
    refit: bool,
    random_state: u64,
    // Fitted state
    cv_results_: Option<CVResults>,
    best_estimator_: Option<E>,
    best_params_: Option<Vec<(&'static str, ParamValue)>>,
    best_score_: Option<f64>,
    best_index_: Option<usize>,
    fitted: bool,
}

impl<E: Predictor + Params + Clone> RandomizedSearchCV<E> {
    /// Create a randomized search from a template estimator and parameter
    /// space.
    pub fn new(
        estimator: E,
        param_space: ParamSpace,
        scorer: fn(&[f64], &[f64]) -> Result<f64>,
    ) -> Self {
        let cv = super::cross_val::default_cv(estimator.is_classifier());
        Self {
            estimator,
            param_space,
            cv,
            response: E::predict,
            scorer,
            n_iter: 10,
            refit: true,
            random_state: 0,
            cv_results_: None,
            best_estimator_: None,
            best_params_: None,
            best_score_: None,
            best_index_: None,
            fitted: false,
        }
    }

    /// Number of sampled candidates (default: 10).
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter.max(1);
        self
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

    /// Seed for the sampling stream (default: 0 — deterministic by default,
    /// unlike sklearn's `random_state=None`).
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    /// Run the randomized search.
    pub fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let mut rng = Rng::new(self.random_state);
        let combinations = sample_combinations(&self.param_space, self.n_iter, &mut rng)?;

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

        // Best candidate: highest mean, ties broken by draw order.
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
            .ok_or_else(|| DatarustError::NotFitted("RandomizedSearchCV".into()))?;
        (self.scorer)(y, &(self.response)(estimator, x)?)
    }

    /// Index of the best candidate in `cv_results().params`.
    pub fn best_index(&self) -> Option<usize> {
        self.best_index_
    }
}

impl<E: Predictor + Params + Clone> crate::traits::Estimator for RandomizedSearchCV<E> {}

impl<E: Predictor + Params + Clone> crate::traits::Predictor for RandomizedSearchCV<E> {
    fn is_classifier(&self) -> bool {
        self.estimator.is_classifier()
    }

    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        RandomizedSearchCV::fit(self, x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.best_estimator_
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("RandomizedSearchCV".into()))?
            .predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl<E: Predictor + PredictProba + Params + Clone> crate::traits::PredictProba
    for RandomizedSearchCV<E>
{
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.best_estimator_
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("RandomizedSearchCV".into()))?
            .predict_proba(x)
    }
}

impl<E: Predictor + Params + Clone + crate::traits::Classifier> crate::traits::Classifier
    for RandomizedSearchCV<E>
{
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        RandomizedSearchCV::score(self, x, y)
    }
}

impl<E: Predictor + Params + Clone + crate::traits::Regressor> crate::traits::Regressor
    for RandomizedSearchCV<E>
{
    fn name(&self) -> &'static str {
        "RandomizedSearchCV"
    }

    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        RandomizedSearchCV::score(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::Lasso;
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
    fn randomized_search_is_deterministic_per_seed() {
        let (x, y) = make_data();
        let run = |seed: u64| {
            let mut rs = RandomizedSearchCV::new(
                Lasso::new(),
                vec![("alpha", ParamDistribution::Uniform(1e-3, 10.0))],
                r2_score,
            )
            .with_n_iter(8)
            .with_random_state(seed)
            .with_cv(KFold::new().with_n_splits(3));
            rs.fit(&x, &y).unwrap();
            (rs.best_score().unwrap(), rs.best_params().unwrap().to_vec())
        };
        assert_eq!(run(42), run(42));
        // Different seeds explore different candidates for a continuous space.
        let a = run(1);
        let b = run(2);
        assert!((a.0 - b.0).abs() > 0.0 || a.1 != b.1);
    }

    #[test]
    fn choices_can_sample_a_product_larger_than_usize() {
        let choices = ParamDistribution::Choices((0..10).map(ParamValue::Int).collect());
        let space = (0..24).map(|_| ("param", choices.clone())).collect();
        let combinations = sample_combinations(&space, 10, &mut Rng::new(42)).unwrap();
        assert_eq!(combinations.len(), 10);
        for (index, candidate) in combinations.iter().enumerate() {
            assert_eq!(candidate.len(), 24);
            assert!(!combinations[..index].contains(candidate));
        }
    }

    #[test]
    fn continuous_uniform_stays_in_bounds() {
        let (x, y) = make_data();
        let mut rs = RandomizedSearchCV::new(
            Lasso::new(),
            vec![("alpha", ParamDistribution::Uniform(0.5, 2.0))],
            r2_score,
        )
        .with_n_iter(6)
        .with_random_state(7)
        .with_cv(KFold::new().with_n_splits(3));
        rs.fit(&x, &y).unwrap();
        let results = rs.cv_results().unwrap();
        assert_eq!(results.params.len(), 6);
        for combo in &results.params {
            match combo[0].1 {
                ParamValue::Float(a) => assert!((0.5..2.0).contains(&a)),
                ref other => panic!("expected Float, got {other:?}"),
            }
        }
    }

    #[test]
    fn rand_int_respects_inclusive_bounds() {
        let (x, y) = make_data();
        let mut rs = RandomizedSearchCV::new(
            Lasso::new(),
            vec![
                ("max_iter", ParamDistribution::RandInt(5, 50)),
                ("alpha", ParamDistribution::Uniform(0.1, 1.0)),
            ],
            r2_score,
        )
        .with_n_iter(5)
        .with_random_state(3)
        .with_cv(KFold::new().with_n_splits(2));
        rs.fit(&x, &y).unwrap();
        for combo in rs.cv_results().unwrap().params.iter() {
            for (name, value) in combo {
                if *name == "max_iter" {
                    assert!(matches!(value, ParamValue::Int(n) if (5..=50).contains(n)));
                }
            }
        }
    }

    #[test]
    fn choices_only_space_is_sampled_without_replacement() {
        let (x, y) = make_data();
        let mut rs = RandomizedSearchCV::new(
            Lasso::new(),
            vec![
                (
                    "alpha",
                    ParamDistribution::Choices(vec![
                        ParamValue::Float(0.1),
                        ParamValue::Float(1.0),
                        ParamValue::Float(10.0),
                    ]),
                ),
                (
                    "fit_intercept",
                    ParamDistribution::Choices(vec![
                        ParamValue::Bool(true),
                        ParamValue::Bool(false),
                    ]),
                ),
            ],
            r2_score,
        )
        .with_n_iter(6) // > product size 3*2 = 6 → exactly the full product
        .with_random_state(1)
        .with_cv(KFold::new().with_n_splits(2));
        rs.fit(&x, &y).unwrap();
        let params = &rs.cv_results().unwrap().params;
        assert_eq!(params.len(), 6);
        let mut keys: Vec<String> = params.iter().map(|c| format!("{c:?}")).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 6, "no duplicate candidates");
    }

    #[test]
    fn empty_space_errors() {
        let (x, y) = make_data();
        let mut rs = RandomizedSearchCV::new(Lasso::new(), vec![], r2_score);
        assert!(matches!(rs.fit(&x, &y), Err(DatarustError::EmptyInput(_))));
    }

    #[test]
    fn rand_int_full_usize_span_does_not_overflow() {
        // `hi - lo + 1` overflows usize when lo == 0 && hi == usize::MAX;
        // the sampler must fall back to the u64 stream instead of panicking
        // (debug) or wrapping (release).
        let (x, y) = make_data();
        let mut rs = RandomizedSearchCV::new(
            Lasso::new(),
            vec![
                ("max_iter", ParamDistribution::RandInt(0, usize::MAX)),
                ("alpha", ParamDistribution::Uniform(0.1, 1.0)),
            ],
            r2_score,
        )
        .with_n_iter(4)
        .with_random_state(5)
        .with_cv(KFold::new().with_n_splits(2));
        rs.fit(&x, &y).unwrap();
        for combo in rs.cv_results().unwrap().params.iter() {
            for (name, value) in combo {
                if *name == "max_iter" {
                    assert!(matches!(value, ParamValue::Int(_)));
                }
            }
        }
    }
}
