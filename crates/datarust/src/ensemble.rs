//! Random-forest and extremely-randomized-tree ensembles built from the
//! CART tree kernel.
//!
//! Each forest fits independent [`DecisionTreeClassifier`] or
//! [`DecisionTreeRegressor`] estimators on bootstrap samples; the
//! [`ExtraTreesClassifier`] / [`ExtraTreesRegressor`] variants fit on the
//! full sample by default and draw one random threshold per candidate
//! feature instead of scanning midpoints. Feature subsampling is delegated
//! to the tree splitter, so `max_features` is applied independently at
//! every split, matching scikit-learn's forest semantics. The forest itself
//! aggregates class probabilities by averaging trees and regression
//! predictions by their arithmetic mean.
//!
//! The implementation is dense-input and single-output, like the v0.7 tree
//! module. Missing values, sample weights, out-of-bag scoring, and parallel
//! fitting are intentionally outside this first forest release.

use crate::error::{DatarustError, Result};
use crate::label_space::LabelSpace;
use crate::matrix::Matrix;
use crate::traits::{
    Classifier, Estimator, ParamValue, Params, PredictProba, Predictor, Regressor,
};
use crate::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};

const SKLEARN_MAX_INT: u32 = 2_147_483_647;

/// Minimal legacy MT19937 used by scikit-learn's `RandomState` forest seed
/// and bootstrap-index generation. Keeping this local preserves the tree
/// kernel's existing exact seed contract without changing the public RNG API.
struct LegacyRandomState {
    mt: [u32; 624],
    index: usize,
}

impl LegacyRandomState {
    fn new(seed: u64) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed as u32;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    self.mt[i] ^= 0x9908_b0df;
                }
            }
            self.index = 0;
        }
        let mut value = self.mt[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    /// Legacy masked-rejection integer draw in `[0, high)`.
    fn randint(&mut self, high: usize) -> usize {
        if high <= 1 {
            return 0;
        }
        // Smallest 2^k−1 ≥ high−1, mirroring numpy's bit-smeared
        // `_gen_mask`; `next_power_of_two` would give high−2 when high−1
        // is already a power of two.
        let bound = (high - 1) as u32;
        let mask = u32::MAX >> bound.leading_zeros();
        loop {
            let value = (self.next_u32() & mask) as usize;
            if value < high {
                return value;
            }
        }
    }

    fn seed31(&mut self) -> u64 {
        self.randint(SKLEARN_MAX_INT as usize) as u64
    }
}

/// Shared random-forest configuration.
#[derive(Debug, Clone, PartialEq)]
struct ForestOptions {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_impurity_decrease: f64,
    max_features: MaxFeatures,
    bootstrap: bool,
    random_state: u64,
    random_split: bool,
}

impl ForestOptions {
    fn validate(&self) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(DatarustError::InvalidConfig(
                "n_estimators must be at least 1".into(),
            ));
        }
        if let Some(depth) = self.max_depth {
            if depth == 0 {
                return Err(DatarustError::InvalidConfig(
                    "max_depth must be at least 1".into(),
                ));
            }
        }
        if self.min_samples_split < 2 {
            return Err(DatarustError::InvalidConfig(
                "min_samples_split must be at least 2".into(),
            ));
        }
        if self.min_samples_leaf == 0 {
            return Err(DatarustError::InvalidConfig(
                "min_samples_leaf must be at least 1".into(),
            ));
        }
        if self.min_impurity_decrease.is_nan() || self.min_impurity_decrease < 0.0 {
            return Err(DatarustError::InvalidConfig(
                "min_impurity_decrease must be non-negative".into(),
            ));
        }
        if self.random_state > u32::MAX as u64 {
            return Err(DatarustError::InvalidConfig(
                "random_state must fit in the tree seed range".into(),
            ));
        }
        if matches!(self.max_features, MaxFeatures::Count(0)) {
            return Err(DatarustError::InvalidConfig(
                "max_features count must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// Row sample for one tree, derived deterministically from the tree's
    /// seed: bootstrap draws from a per-tree stream (keeping estimator
    /// addition deterministic and isolating each tree's draws), else the
    /// original row order. Storing only the seeds lets
    /// `estimators_samples` regenerate indices on demand instead of the
    /// model retaining `n_estimators × n_samples` indices for its lifetime.
    fn sample_for(&self, tree_seed: u64, n_samples: usize) -> Vec<usize> {
        if !self.bootstrap {
            return (0..n_samples).collect();
        }
        let mut sampler = LegacyRandomState::new(tree_seed);
        (0..n_samples).map(|_| sampler.randint(n_samples)).collect()
    }

    fn tree_classifier(&self, criterion: Criterion, seed: u64) -> DecisionTreeClassifier {
        let mut tree = DecisionTreeClassifier::new()
            .with_criterion(criterion)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_impurity_decrease(self.min_impurity_decrease)
            .with_max_features(self.max_features)
            .with_random_state(seed)
            .with_random_split(self.random_split);
        tree = tree.with_max_depth(self.max_depth);
        tree
    }

    fn tree_regressor(&self, seed: u64) -> DecisionTreeRegressor {
        let mut tree = DecisionTreeRegressor::new()
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_impurity_decrease(self.min_impurity_decrease)
            .with_max_features(self.max_features)
            .with_random_state(seed)
            .with_random_split(self.random_split);
        tree = tree.with_max_depth(self.max_depth);
        tree
    }

    fn set_shared_param(&mut self, name: &str, value: ParamValue) -> Result<bool> {
        match (name, value) {
            ("n_estimators", ParamValue::Int(0)) => Err(DatarustError::InvalidConfig(
                "n_estimators must be at least 1".into(),
            )),
            ("n_estimators", ParamValue::Int(value)) => {
                self.n_estimators = value;
                Ok(true)
            }
            ("max_depth", ParamValue::Int(0)) => {
                self.max_depth = None;
                Ok(true)
            }
            ("max_depth", ParamValue::Int(value)) => {
                self.max_depth = Some(value);
                Ok(true)
            }
            ("min_samples_split", ParamValue::Int(value)) if value < 2 => Err(
                DatarustError::InvalidConfig("min_samples_split must be at least 2".into()),
            ),
            ("min_samples_split", ParamValue::Int(value)) => {
                self.min_samples_split = value;
                Ok(true)
            }
            ("min_samples_leaf", ParamValue::Int(0)) => Err(DatarustError::InvalidConfig(
                "min_samples_leaf must be at least 1".into(),
            )),
            ("min_samples_leaf", ParamValue::Int(value)) => {
                self.min_samples_leaf = value;
                Ok(true)
            }
            ("min_impurity_decrease", ParamValue::Float(value))
                if value.is_nan() || value < 0.0 =>
            {
                Err(DatarustError::InvalidConfig(
                    "min_impurity_decrease must be non-negative".into(),
                ))
            }
            ("min_impurity_decrease", ParamValue::Float(value)) => {
                self.min_impurity_decrease = value;
                Ok(true)
            }
            ("bootstrap", ParamValue::Bool(value)) => {
                self.bootstrap = value;
                Ok(true)
            }
            ("random_state", ParamValue::Int(value)) if value > u32::MAX as usize => Err(
                DatarustError::InvalidConfig("random_state must fit in the tree seed range".into()),
            ),
            ("random_state", ParamValue::Int(value)) => {
                self.random_state = value as u64;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn params(&self) -> Vec<(&'static str, ParamValue)> {
        vec![
            ("n_estimators", ParamValue::Int(self.n_estimators)),
            ("max_depth", ParamValue::Int(self.max_depth.unwrap_or(0))),
            ("min_samples_split", ParamValue::Int(self.min_samples_split)),
            ("min_samples_leaf", ParamValue::Int(self.min_samples_leaf)),
            (
                "min_impurity_decrease",
                ParamValue::Float(self.min_impurity_decrease),
            ),
            (
                "max_features",
                ParamValue::Int(max_features_code(self.max_features)),
            ),
            ("bootstrap", ParamValue::Bool(self.bootstrap)),
            ("random_state", ParamValue::Int(self.random_state as usize)),
        ]
    }
}

impl Default for ForestOptions {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            max_features: MaxFeatures::All,
            bootstrap: true,
            random_state: 0,
            random_split: false,
        }
    }
}

fn max_features_code(value: MaxFeatures) -> usize {
    match value {
        MaxFeatures::All => 0,
        MaxFeatures::Sqrt => 1,
        MaxFeatures::Log2 => 2,
        MaxFeatures::Count(count) => count.saturating_add(2),
    }
}

fn max_features_from_code(value: usize) -> Result<MaxFeatures> {
    match value {
        0 => Ok(MaxFeatures::All),
        1 => Ok(MaxFeatures::Sqrt),
        2 => Ok(MaxFeatures::Log2),
        value => value
            .checked_sub(2)
            .filter(|&count| count > 0)
            .map(MaxFeatures::Count)
            .ok_or_else(|| {
                DatarustError::InvalidConfig("max_features count must be at least 1".into())
            }),
    }
}

fn sample_weights(n_samples: usize, indices: &[usize]) -> Vec<f64> {
    let mut weights = vec![0.0; n_samples];
    for &index in indices {
        weights[index] += 1.0;
    }
    weights
}

fn validate_forest_input(x: &Matrix, y: &[f64], options: &ForestOptions) -> Result<()> {
    options.validate()?;
    if x.nrows() != y.len() {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} rows in y", x.nrows()),
            actual: format!("{} rows in y", y.len()),
        });
    }
    if x.nrows() == 0 {
        return Err(DatarustError::EmptyInput("X is empty".into()));
    }
    x.validate_finite()?;
    Ok(())
}

fn validate_prediction_shape(x: &Matrix, n_features: usize) -> Result<()> {
    if x.ncols() != n_features {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} features", n_features),
            actual: format!("{} features", x.ncols()),
        });
    }
    x.validate_finite()?;
    if x.nrows() == 0 {
        return Err(DatarustError::EmptyInput("X has no rows".into()));
    }
    Ok(())
}

/// Random-forest classifier using bootstrap aggregation of CART classifiers.
///
/// The default is 100 bootstrap trees with `max_features = sqrt(n_features)`
/// at each split, matching scikit-learn's `RandomForestClassifier` defaults.
/// `random_state` defaults to zero so fits are reproducible; call
/// [`with_random_state`](Self::with_random_state) to choose another seed.
///
/// ```rust
/// use datarust::ensemble::RandomForestClassifier;
/// use datarust::traits::{PredictProba, Predictor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0], vec![0.0, 1.0],
///     vec![1.0, 0.0], vec![1.0, 1.0],
/// ])?;
/// let y = vec![0.0, 1.0, 1.0, 0.0];
/// let mut forest = RandomForestClassifier::new()
///     .with_n_estimators(25)
///     .with_random_state(7);
/// forest.fit(&x, &y)?;
/// let predictions = forest.predict(&x)?;
/// let probabilities = forest.predict_proba(&x)?;
/// assert_eq!(predictions.len(), 4);
/// assert_eq!(probabilities.ncols(), 2);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    options: ForestOptions,
    criterion: Criterion,
    n_features: usize,
    /// Rows seen at fit time; drives on-demand sample regeneration.
    n_samples: usize,
    classes: Option<LabelSpace>,
    trees: Vec<DecisionTreeClassifier>,
    tree_seeds: Vec<u64>,
    /// Estimator label used in error messages; ExtraTrees delegates here.
    name: &'static str,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestClassifier {
    /// Creates a classifier with scikit-learn-style defaults.
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self {
            options: ForestOptions {
                max_features: MaxFeatures::Sqrt,
                ..ForestOptions::default()
            },
            criterion: Criterion::Gini,
            n_features: 0,
            n_samples: 0,
            classes: None,
            trees: Vec::new(),
            tree_seeds: Vec::new(),
            name: "RandomForestClassifier",
        }
    }

    /// Sets the number of trees.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.options.n_estimators = n;
        self
    }

    /// Sets the tree split criterion (`Gini` or `Entropy`).
    /// `Criterion::SquaredError` is regression-only and is rejected by
    /// `fit`.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Sets the maximum depth of each tree (`None` means unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.options.max_depth = depth;
        self
    }

    /// Sets the minimum samples required to split a node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.options.min_samples_split = n;
        self
    }

    /// Sets the minimum samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.options.min_samples_leaf = n;
        self
    }

    /// Sets the minimum impurity decrease required for a split.
    pub fn with_min_impurity_decrease(mut self, value: f64) -> Self {
        self.options.min_impurity_decrease = value;
        self
    }

    /// Sets the number of features considered at each split.
    pub fn with_max_features(mut self, value: MaxFeatures) -> Self {
        self.options.max_features = value;
        self
    }

    /// Enables or disables bootstrap sampling.
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.options.bootstrap = bootstrap;
        self
    }

    /// Sets the deterministic forest seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.options.random_state = seed;
        self
    }

    /// Number of fitted trees.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Number of features seen during fitting.
    pub fn n_features_in(&self) -> usize {
        self.n_features
    }

    /// Borrow the fitted decision trees.
    pub fn trees(&self) -> &[DecisionTreeClassifier] {
        &self.trees
    }

    /// Sorted original class labels.
    pub fn classes(&self) -> Option<&[f64]> {
        self.classes.as_ref().map(LabelSpace::labels)
    }

    /// Regenerates the row indices used to fit each tree (scikit-learn's
    /// `estimators_samples_` semantics). With bootstrap enabled, each inner
    /// vector contains `n_samples` draws with replacement; with bootstrap
    /// disabled it is the original row order. Indices are derived from the
    /// stored per-tree seeds on demand, so the fitted model does not retain
    /// `n_estimators × n_samples` indices.
    pub fn estimators_samples(&self) -> Vec<Vec<usize>> {
        self.tree_seeds
            .iter()
            .map(|&seed| self.options.sample_for(seed, self.n_samples))
            .collect()
    }

    /// Mean normalized impurity reduction across the non-trivial fitted
    /// trees, matching sklearn's `feature_importances_` behavior.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.trees.is_empty() {
            return Err(DatarustError::NotFitted(self.name.into()));
        }
        let mut all_importances = Vec::new();
        for tree in &self.trees {
            if tree.tree().is_some_and(|fitted| fitted.n_nodes() > 1) {
                all_importances.push(tree.feature_importances()?);
            }
        }
        if all_importances.is_empty() {
            return Ok(vec![0.0; self.n_features]);
        }
        let mut out = vec![0.0; self.n_features];
        for importances in &all_importances {
            for (dst, value) in out.iter_mut().zip(importances) {
                *dst += *value;
            }
        }
        let count = all_importances.len() as f64;
        for value in &mut out {
            *value /= count;
        }
        normalize_importances(&mut out);
        Ok(out)
    }

    fn fit_impl(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        validate_forest_input(x, y, &self.options)?;
        if matches!(self.criterion, Criterion::SquaredError) {
            return Err(DatarustError::InvalidInput(format!(
                "{} requires Criterion::Gini or Criterion::Entropy; \
                 Criterion::SquaredError is regression-only",
                self.name
            )));
        }
        let classes = LabelSpace::fit(y)?;
        let mut stream = LegacyRandomState::new(self.options.random_state);
        let mut trees = Vec::with_capacity(self.options.n_estimators);
        let mut tree_seeds = Vec::with_capacity(self.options.n_estimators);
        for _ in 0..self.options.n_estimators {
            let tree_seed = stream.seed31();
            let sample_indices = self.options.sample_for(tree_seed, x.nrows());
            let weights = sample_weights(x.nrows(), &sample_indices);
            let mut tree = self.options.tree_classifier(self.criterion, tree_seed);
            tree.fit_weighted_impl(x, y, Some(weights), Some(classes.clone()))?;
            trees.push(tree);
            tree_seeds.push(tree_seed);
        }
        self.n_features = x.ncols();
        self.n_samples = x.nrows();
        self.classes = Some(classes);
        self.trees = trees;
        self.tree_seeds = tree_seeds;
        Ok(())
    }

    fn probabilities(&self, x: &Matrix) -> Result<Matrix> {
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted(self.name.into()))?;
        validate_prediction_shape(x, self.n_features)?;
        let n = x.nrows();
        let k = classes.len();
        let mut data = vec![0.0; n * k];
        for tree in &self.trees {
            let local = tree.predict_proba(x)?;
            let local_classes =
                tree.tree()
                    .and_then(|fitted| fitted.classes())
                    .ok_or_else(|| {
                        DatarustError::InvalidInput("classifier tree has no classes".into())
                    })?;
            // Each tree's class list is fixed at fit; map local column
            // indices to forest indices once instead of binary-searching
            // per (row, class).
            let to_global = local_classes
                .iter()
                .map(|&label| classes.encode(label))
                .collect::<Result<Vec<usize>>>()?;
            for i in 0..n {
                for (local_index, &global_index) in to_global.iter().enumerate() {
                    data[i * k + global_index] += local.get(i, local_index);
                }
            }
        }
        let count = self.trees.len() as f64;
        for value in &mut data {
            *value /= count;
        }
        Matrix::from_flat(n, k, data)
    }
}

impl Estimator for RandomForestClassifier {}

impl Predictor for RandomForestClassifier {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_impl(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let probabilities = self.probabilities(x)?;
        let classes = self.classes.as_ref().expect("validated above");
        let mut out = Vec::with_capacity(x.nrows());
        for row in probabilities.iter_rows() {
            let mut best = None;
            for (index, &probability) in row.iter().enumerate() {
                if best.map_or(true, |(_, current)| probability > current) {
                    best = Some((index, probability));
                }
            }
            let best = best
                .map(|(index, _)| index)
                .ok_or_else(|| DatarustError::InvalidInput("empty probability row".into()))?;
            out.push(classes.decode(best)?);
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        !self.trees.is_empty() && self.classes.is_some()
    }
}

impl Classifier for RandomForestClassifier {}

impl PredictProba for RandomForestClassifier {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.probabilities(x)
    }
}

impl Params for RandomForestClassifier {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = self.options.params();
        params.push((
            "criterion",
            ParamValue::Int(match self.criterion {
                Criterion::Gini => 0,
                Criterion::Entropy => 1,
                Criterion::SquaredError => 2,
            }),
        ));
        params
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        if self.options.set_shared_param(name, value.clone())? {
            return Ok(());
        }
        match (name, value) {
            ("criterion", ParamValue::Int(0)) => self.criterion = Criterion::Gini,
            ("criterion", ParamValue::Int(1)) => self.criterion = Criterion::Entropy,
            // Storable for `get_params` round-trips; rejected by `fit`.
            ("criterion", ParamValue::Int(2)) => self.criterion = Criterion::SquaredError,
            ("criterion", _) => {
                return Err(DatarustError::InvalidInput(
                    "criterion expects Int(0)=gini, Int(1)=entropy or Int(2)=squared_error".into(),
                ));
            }
            ("max_features", ParamValue::Int(value)) => {
                self.options.max_features = max_features_from_code(value)?;
            }
            _ => {
                return Err(DatarustError::InvalidInput(format!(
                    "unknown parameter '{name}' for {}",
                    self.name
                )));
            }
        }
        Ok(())
    }
}

/// Random-forest regressor using bootstrap aggregation of CART regressors.
///
/// The default is 100 bootstrap trees and all features considered at each
/// split, matching scikit-learn's `RandomForestRegressor` default.
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    options: ForestOptions,
    n_features: usize,
    /// Rows seen at fit time; drives on-demand sample regeneration.
    n_samples: usize,
    trees: Vec<DecisionTreeRegressor>,
    tree_seeds: Vec<u64>,
    /// Estimator label used in error messages; ExtraTrees delegates here.
    name: &'static str,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestRegressor {
    /// Creates a regressor with scikit-learn-style defaults.
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self {
            options: ForestOptions::default(),
            n_features: 0,
            n_samples: 0,
            trees: Vec::new(),
            tree_seeds: Vec::new(),
            name: "RandomForestRegressor",
        }
    }

    /// Sets the number of trees.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.options.n_estimators = n;
        self
    }

    /// Sets the maximum depth of each tree (`None` means unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.options.max_depth = depth;
        self
    }

    /// Sets the minimum samples required to split a node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.options.min_samples_split = n;
        self
    }

    /// Sets the minimum samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.options.min_samples_leaf = n;
        self
    }

    /// Sets the minimum impurity decrease required for a split.
    pub fn with_min_impurity_decrease(mut self, value: f64) -> Self {
        self.options.min_impurity_decrease = value;
        self
    }

    /// Sets the number of features considered at each split.
    pub fn with_max_features(mut self, value: MaxFeatures) -> Self {
        self.options.max_features = value;
        self
    }

    /// Enables or disables bootstrap sampling.
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.options.bootstrap = bootstrap;
        self
    }

    /// Sets the deterministic forest seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.options.random_state = seed;
        self
    }

    /// Number of fitted trees.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Number of features seen at fit time.
    pub fn n_features_in(&self) -> usize {
        self.n_features
    }

    /// Borrow the fitted decision trees.
    pub fn trees(&self) -> &[DecisionTreeRegressor] {
        &self.trees
    }

    /// Regenerates the row indices used to fit each tree (scikit-learn's
    /// `estimators_samples_` semantics). With bootstrap enabled, each inner
    /// vector contains `n_samples` draws with replacement; with bootstrap
    /// disabled it is the original row order. Indices are derived from the
    /// stored per-tree seeds on demand, so the fitted model does not retain
    /// `n_estimators × n_samples` indices.
    pub fn estimators_samples(&self) -> Vec<Vec<usize>> {
        self.tree_seeds
            .iter()
            .map(|&seed| self.options.sample_for(seed, self.n_samples))
            .collect()
    }

    /// Mean normalized impurity reduction across the non-trivial fitted
    /// trees, matching sklearn's `feature_importances_` behavior.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.trees.is_empty() {
            return Err(DatarustError::NotFitted(self.name.into()));
        }
        let mut all_importances = Vec::new();
        for tree in &self.trees {
            if tree.tree().is_some_and(|fitted| fitted.n_nodes() > 1) {
                all_importances.push(tree.feature_importances()?);
            }
        }
        if all_importances.is_empty() {
            return Ok(vec![0.0; self.n_features]);
        }
        let mut out = vec![0.0; self.n_features];
        for importances in &all_importances {
            for (dst, value) in out.iter_mut().zip(importances) {
                *dst += *value;
            }
        }
        let count = all_importances.len() as f64;
        for value in &mut out {
            *value /= count;
        }
        normalize_importances(&mut out);
        Ok(out)
    }

    fn fit_impl(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        validate_forest_input(x, y, &self.options)?;
        for (index, &value) in y.iter().enumerate() {
            if !value.is_finite() {
                return Err(DatarustError::InvalidInput(format!(
                    "y contains a non-finite value at index {index}"
                )));
            }
        }
        let mut stream = LegacyRandomState::new(self.options.random_state);
        let mut trees = Vec::with_capacity(self.options.n_estimators);
        let mut tree_seeds = Vec::with_capacity(self.options.n_estimators);
        for _ in 0..self.options.n_estimators {
            let tree_seed = stream.seed31();
            let sample_indices = self.options.sample_for(tree_seed, x.nrows());
            let weights = sample_weights(x.nrows(), &sample_indices);
            let mut tree = self.options.tree_regressor(tree_seed);
            tree.fit_weighted_impl(x, y, Some(weights))?;
            trees.push(tree);
            tree_seeds.push(tree_seed);
        }
        self.n_features = x.ncols();
        self.n_samples = x.nrows();
        self.trees = trees;
        self.tree_seeds = tree_seeds;
        Ok(())
    }
}

impl Estimator for RandomForestRegressor {}

impl Predictor for RandomForestRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_impl(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        if self.trees.is_empty() {
            return Err(DatarustError::NotFitted(self.name.into()));
        }
        validate_prediction_shape(x, self.n_features)?;
        let mut out = vec![0.0; x.nrows()];
        for tree in &self.trees {
            for (value, prediction) in out.iter_mut().zip(tree.predict(x)?) {
                *value += prediction;
            }
        }
        let count = self.trees.len() as f64;
        for value in &mut out {
            *value /= count;
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        !self.trees.is_empty()
    }
}

impl Regressor for RandomForestRegressor {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl Params for RandomForestRegressor {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        self.options.params()
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        if self.options.set_shared_param(name, value.clone())? {
            return Ok(());
        }
        if name == "max_features" {
            if let ParamValue::Int(value) = value {
                self.options.max_features = max_features_from_code(value)?;
                return Ok(());
            }
        }
        Err(DatarustError::InvalidInput(format!(
            "unknown parameter '{name}' for {}",
            self.name
        )))
    }
}

fn normalize_importances(values: &mut [f64]) {
    let total = crate::tree::numpy_pairwise_sum(values);
    if total > 0.0 {
        for value in values {
            *value /= total;
        }
    }
}

/// Extremely-randomized-tree classifier (scikit-learn's
/// `ExtraTreesClassifier`).
///
/// Like [`RandomForestClassifier`], but each split draws a single random
/// threshold between the feature's min and max instead of scanning all
/// midpoints, and trees are fitted on the full sample by default
/// (`bootstrap = false`). Defaults: 100 trees,
/// `max_features = sqrt(n_features)`, Gini criterion.
///
/// ```rust
/// use datarust::ensemble::ExtraTreesClassifier;
/// use datarust::traits::{PredictProba, Predictor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0], vec![0.0, 1.0],
///     vec![1.0, 0.0], vec![1.0, 1.0],
/// ])?;
/// let y = vec![0.0, 1.0, 1.0, 0.0];
/// let mut forest = ExtraTreesClassifier::new()
///     .with_n_estimators(25)
///     .with_random_state(7);
/// forest.fit(&x, &y)?;
/// assert_eq!(forest.predict(&x)?.len(), 4);
/// assert_eq!(forest.predict_proba(&x)?.ncols(), 2);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ExtraTreesClassifier {
    inner: RandomForestClassifier,
}

impl Default for ExtraTreesClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtraTreesClassifier {
    /// Creates a classifier with scikit-learn-style ExtraTrees defaults
    /// (100 non-bootstrap trees, sqrt feature subsampling, Gini).
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        let mut inner = RandomForestClassifier::new().with_bootstrap(false);
        inner.name = "ExtraTreesClassifier";
        inner.options.random_split = true;
        Self { inner }
    }

    /// Sets the number of trees.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.inner = self.inner.with_n_estimators(n);
        self
    }

    /// Sets the tree split criterion (`Gini` or `Entropy`).
    /// `Criterion::SquaredError` is regression-only and is rejected by
    /// `fit`.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.inner = self.inner.with_criterion(criterion);
        self
    }

    /// Sets the maximum depth of each tree (`None` means unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.inner = self.inner.with_max_depth(depth);
        self
    }

    /// Sets the minimum samples required to split a node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.inner = self.inner.with_min_samples_split(n);
        self
    }

    /// Sets the minimum samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.inner = self.inner.with_min_samples_leaf(n);
        self
    }

    /// Sets the minimum impurity decrease required for a split.
    pub fn with_min_impurity_decrease(mut self, value: f64) -> Self {
        self.inner = self.inner.with_min_impurity_decrease(value);
        self
    }

    /// Sets the number of features considered at each split.
    pub fn with_max_features(mut self, value: MaxFeatures) -> Self {
        self.inner = self.inner.with_max_features(value);
        self
    }

    /// Enables or disables bootstrap sampling (disabled by default).
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.inner = self.inner.with_bootstrap(bootstrap);
        self
    }

    /// Sets the deterministic forest seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.inner = self.inner.with_random_state(seed);
        self
    }

    /// Number of fitted trees.
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Number of features seen during fitting.
    pub fn n_features_in(&self) -> usize {
        self.inner.n_features_in()
    }

    /// Borrow the fitted decision trees.
    pub fn trees(&self) -> &[DecisionTreeClassifier] {
        self.inner.trees()
    }

    /// Sorted original class labels.
    pub fn classes(&self) -> Option<&[f64]> {
        self.inner.classes()
    }

    /// Row indices used to fit each tree (identity order unless bootstrap
    /// is enabled); see [`RandomForestClassifier::estimators_samples`].
    pub fn estimators_samples(&self) -> Vec<Vec<usize>> {
        self.inner.estimators_samples()
    }

    /// Mean normalized impurity reduction across the non-trivial fitted
    /// trees, matching sklearn's `feature_importances_` behavior.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        self.inner.feature_importances()
    }
}

impl Estimator for ExtraTreesClassifier {}

impl Predictor for ExtraTreesClassifier {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.inner.fit(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.inner.predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }
}

impl Classifier for ExtraTreesClassifier {}

impl PredictProba for ExtraTreesClassifier {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.inner.predict_proba(x)
    }
}

impl Params for ExtraTreesClassifier {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        self.inner.get_params()
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.inner.set_params(name, value)
    }
}

/// Extremely-randomized-tree regressor (scikit-learn's
/// `ExtraTreesRegressor`).
///
/// Like [`RandomForestRegressor`], but each split draws a single random
/// threshold between the feature's min and max, and trees are fitted on the
/// full sample by default (`bootstrap = false`). Defaults: 100 trees, all
/// features considered at each split, squared-error criterion.
///
/// ```rust
/// use datarust::ensemble::ExtraTreesRegressor;
/// use datarust::traits::Predictor;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![0.0, 0.0], vec![0.0, 1.0],
///     vec![1.0, 0.0], vec![1.0, 1.0],
/// ])?;
/// let y = vec![0.0, 1.0, 1.0, 2.0];
/// let mut forest = ExtraTreesRegressor::new()
///     .with_n_estimators(25)
///     .with_random_state(7);
/// forest.fit(&x, &y)?;
/// assert_eq!(forest.predict(&x)?.len(), 4);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ExtraTreesRegressor {
    inner: RandomForestRegressor,
}

impl Default for ExtraTreesRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtraTreesRegressor {
    /// Creates a regressor with scikit-learn-style ExtraTrees defaults
    /// (100 non-bootstrap trees, all features, squared error).
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        let mut inner = RandomForestRegressor::new().with_bootstrap(false);
        inner.name = "ExtraTreesRegressor";
        inner.options.random_split = true;
        Self { inner }
    }

    /// Sets the number of trees.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.inner = self.inner.with_n_estimators(n);
        self
    }

    /// Sets the maximum depth of each tree (`None` means unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.inner = self.inner.with_max_depth(depth);
        self
    }

    /// Sets the minimum samples required to split a node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.inner = self.inner.with_min_samples_split(n);
        self
    }

    /// Sets the minimum samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.inner = self.inner.with_min_samples_leaf(n);
        self
    }

    /// Sets the minimum impurity decrease required for a split.
    pub fn with_min_impurity_decrease(mut self, value: f64) -> Self {
        self.inner = self.inner.with_min_impurity_decrease(value);
        self
    }

    /// Sets the number of features considered at each split.
    pub fn with_max_features(mut self, value: MaxFeatures) -> Self {
        self.inner = self.inner.with_max_features(value);
        self
    }

    /// Enables or disables bootstrap sampling (disabled by default).
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.inner = self.inner.with_bootstrap(bootstrap);
        self
    }

    /// Sets the deterministic forest seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.inner = self.inner.with_random_state(seed);
        self
    }

    /// Number of fitted trees.
    pub fn n_estimators(&self) -> usize {
        self.inner.n_estimators()
    }

    /// Number of features seen during fitting.
    pub fn n_features_in(&self) -> usize {
        self.inner.n_features_in()
    }

    /// Borrow the fitted decision trees.
    pub fn trees(&self) -> &[DecisionTreeRegressor] {
        self.inner.trees()
    }

    /// Row indices used to fit each tree (identity order unless bootstrap
    /// is enabled); see [`RandomForestRegressor::estimators_samples`].
    pub fn estimators_samples(&self) -> Vec<Vec<usize>> {
        self.inner.estimators_samples()
    }

    /// Mean normalized impurity reduction across the non-trivial fitted
    /// trees, matching sklearn's `feature_importances_` behavior.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        self.inner.feature_importances()
    }
}

impl Estimator for ExtraTreesRegressor {}

impl Predictor for ExtraTreesRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.inner.fit(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.inner.predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }
}

impl Regressor for ExtraTreesRegressor {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

impl Params for ExtraTreesRegressor {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        self.inner.get_params()
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        self.inner.set_params(name, value)
    }
}

// ---------------------------------------------------------------------------
// Sampling utilities for Bagging
// ---------------------------------------------------------------------------

/// Fisher-Yates shuffle of `arange(n)`, matching numpy's legacy
/// `RandomState.permutation`.
fn shuffled_arange(state: &mut LegacyRandomState, n: usize) -> Vec<usize> {
    let mut arr: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = state.randint(i + 1);
        arr.swap(i, j);
    }
    arr
}

/// Tracking-selection SWOR: draws uniformly until the desired count of
/// distinct values is reached. Order is insertion order (draw order),
/// matching sklearn's set-based implementation.
fn tracking_selection(
    state: &mut LegacyRandomState,
    n_population: usize,
    n_samples: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(n_samples);
    let mut selected = vec![false; n_population];
    while out.len() < n_samples {
        let j = state.randint(n_population);
        if !selected[j] {
            selected[j] = true;
            out.push(j);
        }
    }
    out
}

/// Reservoir-sampling SWOR (sklearn's `_sample_without_replacement_with_reservoir_sampling`).
fn reservoir_swor(
    state: &mut LegacyRandomState,
    n_population: usize,
    n_samples: usize,
) -> Vec<usize> {
    let mut out: Vec<usize> = (0..n_samples).collect();
    for i in n_samples..n_population {
        let j = state.randint(i + 1);
        if j < n_samples {
            out[j] = i;
        }
    }
    out
}

/// Auto-dispatched SWOR matching sklearn's `sample_without_replacement`.
fn sample_without_replacement(
    state: &mut LegacyRandomState,
    n_population: usize,
    n_samples: usize,
) -> Vec<usize> {
    if n_samples == 0 {
        return vec![];
    }
    if n_samples >= n_population {
        // `>=` so that ratio 1.0 returns the identity (reservoir
        // short-circuit) without consuming random state.
        return (0..n_population).collect();
    }
    let ratio = n_samples as f64 / n_population as f64;
    if ratio > 0.01 && ratio < 0.99 {
        let perm = shuffled_arange(state, n_population);
        perm[..n_samples].to_vec()
    } else if ratio < 0.2 {
        tracking_selection(state, n_population, n_samples)
    } else {
        reservoir_swor(state, n_population, n_samples)
    }
}

/// Bincount: for each `indices[i]`, increment the count at slot `indices[i]`.
fn bincount(indices: &[usize], n: usize) -> Vec<f64> {
    let mut counts = vec![0.0; n];
    for &i in indices {
        counts[i] += 1.0;
    }
    counts
}

// ---------------------------------------------------------------------------
// BaggingClassifier / BaggingRegressor
// ---------------------------------------------------------------------------

/// Bagging meta-estimator for classification (sklearn's
/// `BaggingClassifier`), built on [`DecisionTreeClassifier`] as the base
/// estimator.
///
/// Each tree is fitted on a bootstrap (or random subset) of the training
/// data, with an optional random feature subset per estimator.
/// Predictions average the class probabilities across all estimators.
#[derive(Debug, Clone)]
pub struct BaggingClassifier {
    base: DecisionTreeClassifier,
    n_estimators: usize,
    max_samples: BaggingMaxSamples,
    max_features: BaggingMaxFeatures,
    bootstrap: bool,
    bootstrap_features: bool,
    random_state: u64,

    // Fitted state
    estimators_: Vec<DecisionTreeClassifier>,
    estimators_samples_: Vec<Vec<usize>>,
    estimators_features_: Vec<Vec<usize>>,
    classes_: LabelSpace,
    n_features_: usize,
}

impl Default for BaggingClassifier {
    fn default() -> Self {
        Self {
            base: DecisionTreeClassifier::default(),
            n_estimators: 10,
            max_samples: BaggingMaxSamples::Fraction(1.0),
            max_features: BaggingMaxFeatures::Fraction(1.0),
            bootstrap: true,
            bootstrap_features: false,
            random_state: 0,
            estimators_: Vec::new(),
            estimators_samples_: Vec::new(),
            estimators_features_: Vec::new(),
            classes_: LabelSpace::fit(&[0.0]).unwrap(),
            n_features_: 0,
        }
    }
}

impl BaggingClassifier {
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of base estimators.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Fraction (`0 < f ≤ 1`) or absolute count of samples per estimator.
    pub fn with_max_samples(mut self, value: BaggingMaxSamples) -> Self {
        self.max_samples = value;
        self
    }

    /// Fraction or absolute count of features drawn for each base estimator.
    pub fn with_max_features(mut self, value: BaggingMaxFeatures) -> Self {
        self.max_features = value;
        self
    }

    /// Whether to draw samples with replacement (bootstrap).
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Whether features are drawn with replacement.
    pub fn with_bootstrap_features(mut self, v: bool) -> Self {
        self.bootstrap_features = v;
        self
    }

    /// Seed for the random state (deterministic sampling).
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    /// Maximum depth of each base tree (`None` for unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.base = self.base.with_max_depth(depth);
        self
    }

    /// Fitted base estimators, one per ensemble member.
    pub fn estimators(&self) -> &[DecisionTreeClassifier] {
        &self.estimators_
    }

    /// Bootstrap sample indices per estimator (usize indices into the training set).
    pub fn estimators_samples(&self) -> &[Vec<usize>] {
        &self.estimators_samples_
    }

    /// Feature indices per estimator (column positions in the training matrix).
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        &self.estimators_features_
    }

    /// Number of classes in the training labels.
    pub fn n_classes(&self) -> usize {
        self.classes_.len()
    }
}

impl Estimator for BaggingClassifier {}

impl Predictor for BaggingClassifier {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features_ = n_features;

        self.classes_ = LabelSpace::fit(y)?;

        let n_draw_samples = self.max_samples.resolve(n_samples)?;
        let n_draw_features = self.max_features.resolve(n_features)?;

        let mut rng = LegacyRandomState::new(self.random_state);

        for _ in 0..self.n_estimators {
            let forest_seed = rng.seed31();
            // sklearn's `_set_random_states` derives a fresh seed for the
            // base estimator from the forest-level seed, so that setting the
            // same seed at the forest level changes every tree (just like
            // `RandomForest`). Feature and sample draws, however, consume
            // the forest-level seed directly (same `RandomState`).
            let tree_seed = LegacyRandomState::new(forest_seed).seed31();
            let mut rs = LegacyRandomState::new(forest_seed);

            // Features first, then samples (matching sklearn's order, both
            // consuming from the same `RandomState`).
            let features = if self.bootstrap_features || n_draw_features != n_features {
                if self.bootstrap_features {
                    (0..n_draw_features)
                        .map(|_| rs.randint(n_features))
                        .collect()
                } else {
                    sample_without_replacement(&mut rs, n_features, n_draw_features)
                }
            } else {
                (0..n_features).collect::<Vec<_>>()
            };

            let sample_indices = if self.bootstrap {
                (0..n_draw_samples).map(|_| rs.randint(n_samples)).collect()
            } else {
                sample_without_replacement(&mut rs, n_samples, n_draw_samples)
            };

            let x_sub =
                if features.len() == n_features && (0..n_features).eq(features.iter().copied()) {
                    x.clone()
                } else {
                    x.select_columns(&features)?
                };

            // Weight path: bincount for bootstrap, zero mask for non-bootstrap.
            let sample_weight = if self.bootstrap {
                bincount(&sample_indices, n_samples)
            } else {
                let mut w = vec![1.0; n_samples];
                let mut selected = vec![false; n_samples];
                for &idx in &sample_indices {
                    selected[idx] = true;
                }
                for (i, w_i) in w.iter_mut().enumerate() {
                    if !selected[i] {
                        *w_i = 0.0;
                    }
                }
                w
            };

            let mut tree = self.base.clone();
            tree.set_params("random_state", ParamValue::Int(tree_seed as usize))?;
            tree.fit_weighted_impl(&x_sub, y, Some(sample_weight), Some(self.classes_.clone()))?;

            self.estimators_.push(tree);
            self.estimators_samples_.push(sample_indices);
            self.estimators_features_.push(features);
        }

        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let pred = self.predict_proba(x)?;
        let n_classes = self.classes_.len();
        let n_samples = pred.nrows();
        let proba = pred.as_slice();
        let mut preds = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let base = i * n_classes;
            let mut best_c = 0usize;
            let mut best_p = proba[base];
            for c in 1..n_classes {
                let p = proba[base + c];
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
            }
            preds.push(self.classes_.labels()[best_c]);
        }
        Ok(preds)
    }

    fn is_fitted(&self) -> bool {
        !self.estimators_.is_empty()
    }
}

impl PredictProba for BaggingClassifier {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        let n_classes = self.classes_.len();
        let n_samples = x.nrows();
        let mut proba = vec![0.0; n_samples * n_classes];

        let n = self.estimators_.len() as f64;
        for (est, features) in self.estimators_.iter().zip(&self.estimators_features_) {
            let x_sub = if features.len() == self.n_features_
                && (0..self.n_features_).eq(features.iter().copied())
            {
                x.clone()
            } else {
                x.select_columns(features)?
            };
            let est_proba = est.predict_proba(&x_sub)?;
            let est_proba_slice = est_proba.as_slice();
            let est_classes = est_proba.ncols();
            for i in 0..n_samples {
                let base = i * n_classes;
                let est_base = i * est_classes;
                for c in 0..est_classes {
                    proba[base + c] += est_proba_slice[est_base + c];
                }
            }
        }

        for p in &mut proba {
            *p /= n;
        }
        Matrix::from_flat(n_samples, n_classes, proba)
    }
}

impl Params for BaggingClassifier {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = self.base.get_params();
        params.push(("n_estimators", ParamValue::Int(self.n_estimators)));
        params.push(("max_samples", self.max_samples.to_param_value()));
        params.push(("max_features", self.max_features.to_param_value()));
        params.push(("bootstrap", ParamValue::Bool(self.bootstrap)));
        params.push((
            "bootstrap_features",
            ParamValue::Bool(self.bootstrap_features),
        ));
        params.push(("random_state", ParamValue::Int(self.random_state as usize)));
        params
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => {
                self.n_estimators = match value {
                    ParamValue::Int(v) => v,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "n_estimators expects Int".into(),
                        ))
                    }
                };
            }
            "max_samples" => self.max_samples = BaggingMaxSamples::from_param_value(value)?,
            "max_features" => self.max_features = BaggingMaxFeatures::from_param_value(value)?,
            "bootstrap" => {
                self.bootstrap = match value {
                    ParamValue::Bool(b) => b,
                    _ => return Err(DatarustError::InvalidInput("bootstrap expects Bool".into())),
                };
            }
            "bootstrap_features" => {
                self.bootstrap_features = match value {
                    ParamValue::Bool(b) => b,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "bootstrap_features expects Bool".into(),
                        ))
                    }
                };
            }
            "random_state" => {
                self.random_state = match value {
                    ParamValue::Int(v) => v as u64,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "random_state expects Int".into(),
                        ))
                    }
                };
            }
            _ => {
                self.base.set_params(name, value)?;
            }
        }
        Ok(())
    }
}

impl Classifier for BaggingClassifier {}

// ---------------------------------------------------------------------------
// BaggingRegressor
// ---------------------------------------------------------------------------

/// Bagging meta-estimator for regression (sklearn's `BaggingRegressor`),
/// built on [`DecisionTreeRegressor`] as the base estimator.
#[derive(Debug, Clone)]
pub struct BaggingRegressor {
    base: DecisionTreeRegressor,
    n_estimators: usize,
    max_samples: BaggingMaxSamples,
    max_features: BaggingMaxFeatures,
    bootstrap: bool,
    bootstrap_features: bool,
    random_state: u64,

    // Fitted state
    estimators_: Vec<DecisionTreeRegressor>,
    estimators_samples_: Vec<Vec<usize>>,
    estimators_features_: Vec<Vec<usize>>,
    n_features_: usize,
}

impl Default for BaggingRegressor {
    fn default() -> Self {
        Self {
            base: DecisionTreeRegressor::default(),
            n_estimators: 10,
            max_samples: BaggingMaxSamples::Fraction(1.0),
            max_features: BaggingMaxFeatures::Fraction(1.0),
            bootstrap: true,
            bootstrap_features: false,
            random_state: 0,
            estimators_: Vec::new(),
            estimators_samples_: Vec::new(),
            estimators_features_: Vec::new(),
            n_features_: 0,
        }
    }
}

impl BaggingRegressor {
    /// New Bagging classifier with sensible defaults: 10 estimators, bootstrap
    /// on the full sample, all features.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of base estimators.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Fraction (`0 < f ≤ 1`) or absolute count of samples per estimator.
    pub fn with_max_samples(mut self, value: BaggingMaxSamples) -> Self {
        self.max_samples = value;
        self
    }

    /// Fraction or absolute count of features drawn for each base estimator.
    pub fn with_max_features(mut self, value: BaggingMaxFeatures) -> Self {
        self.max_features = value;
        self
    }

    /// Whether to draw samples with replacement (bootstrap).
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Whether features are drawn with replacement.
    pub fn with_bootstrap_features(mut self, v: bool) -> Self {
        self.bootstrap_features = v;
        self
    }

    /// Seed for the random state (deterministic sampling).
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = seed;
        self
    }

    /// Maximum depth of each base tree (`None` for unlimited).
    pub fn with_max_depth(mut self, depth: Option<usize>) -> Self {
        self.base = self.base.with_max_depth(depth);
        self
    }

    /// Fitted base estimators, one per ensemble member.
    pub fn estimators(&self) -> &[DecisionTreeRegressor] {
        &self.estimators_
    }

    /// Bootstrap sample indices per estimator (usize indices into the training set).
    pub fn estimators_samples(&self) -> &[Vec<usize>] {
        &self.estimators_samples_
    }

    /// Feature indices per estimator (column positions in the training matrix).
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        &self.estimators_features_
    }
}

impl Estimator for BaggingRegressor {}

impl Predictor for BaggingRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features_ = n_features;

        let n_draw_samples = self.max_samples.resolve(n_samples)?;
        let n_draw_features = self.max_features.resolve(n_features)?;

        let mut rng = LegacyRandomState::new(self.random_state);

        for _ in 0..self.n_estimators {
            let forest_seed = rng.seed31();
            // sklearn's `_set_random_states` derives a fresh seed for the
            // base estimator from the forest-level seed, so that setting the
            // same seed at the forest level changes every tree (just like
            // `RandomForest`). Feature and sample draws, however, consume
            // the forest-level seed directly (same `RandomState`).
            let tree_seed = LegacyRandomState::new(forest_seed).seed31();
            let mut rs = LegacyRandomState::new(forest_seed);

            let features = if self.bootstrap_features || n_draw_features != n_features {
                if self.bootstrap_features {
                    (0..n_draw_features)
                        .map(|_| rs.randint(n_features))
                        .collect()
                } else {
                    sample_without_replacement(&mut rs, n_features, n_draw_features)
                }
            } else {
                (0..n_features).collect::<Vec<_>>()
            };

            let sample_indices = if self.bootstrap {
                (0..n_draw_samples).map(|_| rs.randint(n_samples)).collect()
            } else {
                sample_without_replacement(&mut rs, n_samples, n_draw_samples)
            };

            let x_sub =
                if features.len() == n_features && (0..n_features).eq(features.iter().copied()) {
                    x.clone()
                } else {
                    x.select_columns(&features)?
                };

            let sample_weight = if self.bootstrap {
                bincount(&sample_indices, n_samples)
            } else {
                let mut w = vec![1.0; n_samples];
                let mut selected = vec![false; n_samples];
                for &idx in &sample_indices {
                    selected[idx] = true;
                }
                for (i, w_i) in w.iter_mut().enumerate() {
                    if !selected[i] {
                        *w_i = 0.0;
                    }
                }
                w
            };

            let mut tree = self.base.clone();
            tree.set_params("random_state", ParamValue::Int(tree_seed as usize))?;
            tree.fit_weighted_impl(&x_sub, y, Some(sample_weight))?;

            self.estimators_.push(tree);
            self.estimators_samples_.push(sample_indices);
            self.estimators_features_.push(features);
        }

        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let n_samples = x.nrows();
        if self.estimators_.is_empty() {
            return Err(DatarustError::NotFitted(
                "BaggingRegressor not fitted".into(),
            ));
        }
        let mut sum = vec![0.0; n_samples];
        let n = self.estimators_.len() as f64;
        for (est, features) in self.estimators_.iter().zip(&self.estimators_features_) {
            let x_sub = if features.len() == self.n_features_
                && (0..self.n_features_).eq(features.iter().copied())
            {
                x.clone()
            } else {
                x.select_columns(features)?
            };
            let preds = est.predict(&x_sub)?;
            for (s, p) in sum.iter_mut().zip(preds) {
                *s += p;
            }
        }
        for v in &mut sum {
            *v /= n;
        }
        Ok(sum)
    }

    fn is_fitted(&self) -> bool {
        !self.estimators_.is_empty()
    }
}

impl Regressor for BaggingRegressor {
    fn name(&self) -> &'static str {
        "BaggingRegressor"
    }
}

impl Params for BaggingRegressor {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = self.base.get_params();
        params.push(("n_estimators", ParamValue::Int(self.n_estimators)));
        params.push(("max_samples", self.max_samples.to_param_value()));
        params.push(("max_features", self.max_features.to_param_value()));
        params.push(("bootstrap", ParamValue::Bool(self.bootstrap)));
        params.push((
            "bootstrap_features",
            ParamValue::Bool(self.bootstrap_features),
        ));
        params.push(("random_state", ParamValue::Int(self.random_state as usize)));
        params
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "n_estimators" => {
                self.n_estimators = match value {
                    ParamValue::Int(v) => v,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "n_estimators expects Int".into(),
                        ))
                    }
                };
            }
            "max_samples" => self.max_samples = BaggingMaxSamples::from_param_value(value)?,
            "max_features" => self.max_features = BaggingMaxFeatures::from_param_value(value)?,
            "bootstrap" => {
                self.bootstrap = match value {
                    ParamValue::Bool(b) => b,
                    _ => return Err(DatarustError::InvalidInput("bootstrap expects Bool".into())),
                };
            }
            "bootstrap_features" => {
                self.bootstrap_features = match value {
                    ParamValue::Bool(b) => b,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "bootstrap_features expects Bool".into(),
                        ))
                    }
                };
            }
            "random_state" => {
                self.random_state = match value {
                    ParamValue::Int(v) => v as u64,
                    _ => {
                        return Err(DatarustError::InvalidInput(
                            "random_state expects Int".into(),
                        ))
                    }
                };
            }
            _ => {
                self.base.set_params(name, value)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bagging parameter enums
// ---------------------------------------------------------------------------

/// Value for `max_samples` in Bagging: a float fraction (0 < f ≤ 1) or an
/// absolute integer count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaggingMaxSamples {
    /// Fraction of the total sample count (`0 < f ≤ 1`).
    Fraction(f64),
    /// Absolute number of samples per estimator.
    Count(usize),
}

impl BaggingMaxSamples {
    /// Converts a fraction or count into an absolute sample size,
    /// truncating fractions (matching sklearn's `int(frac * n)`).
    pub fn resolve(self, n: usize) -> Result<usize> {
        match self {
            Self::Fraction(f) => {
                let count = (f * n as f64).trunc() as usize;
                Ok(count.max(1))
            }
            Self::Count(c) => {
                if c == 0 || c > n {
                    Err(DatarustError::InvalidInput(format!(
                        "max_samples count {c} must be in [1, {n}]"
                    )))
                } else {
                    Ok(c)
                }
            }
        }
    }

    fn to_param_value(self) -> ParamValue {
        match self {
            Self::Fraction(f) => ParamValue::Float(f),
            Self::Count(c) => ParamValue::Int(c),
        }
    }

    fn from_param_value(value: ParamValue) -> Result<Self> {
        match value {
            ParamValue::Float(f) => Ok(Self::Fraction(f)),
            ParamValue::Int(c) => Ok(Self::Count(c)),
            other => Err(DatarustError::InvalidInput(format!(
                "max_samples must be Float or Int, got {:?}",
                other
            ))),
        }
    }
}

/// Value for `max_features` in Bagging: a float fraction or an absolute
/// integer count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaggingMaxFeatures {
    /// Fraction of the total sample count (`0 < f ≤ 1`).
    Fraction(f64),
    /// Absolute number of samples per estimator.
    Count(usize),
}

impl BaggingMaxFeatures {
    /// Converts a fraction or count into an absolute feature count,
    /// truncating fractions. A count of zero means "all features".
    pub fn resolve(self, n: usize) -> Result<usize> {
        match self {
            Self::Fraction(f) => {
                let count = (f * n as f64).trunc() as usize;
                Ok(count.max(1))
            }
            Self::Count(c) => {
                if c == 0 {
                    Ok(n)
                } else if c > n {
                    Err(DatarustError::InvalidInput(format!(
                        "max_features {c} exceeds n_features {n}"
                    )))
                } else {
                    Ok(c)
                }
            }
        }
    }

    fn to_param_value(self) -> ParamValue {
        match self {
            Self::Fraction(f) => ParamValue::Float(f),
            Self::Count(c) => ParamValue::Int(c),
        }
    }

    fn from_param_value(value: ParamValue) -> Result<Self> {
        match value {
            ParamValue::Float(f) => Ok(Self::Fraction(f)),
            ParamValue::Int(c) => Ok(Self::Count(c)),
            other => Err(DatarustError::InvalidInput(format!(
                "max_features must be Float or Int, got {:?}",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{PredictProba, Predictor};

    fn simple_data() -> (Matrix, Vec<f64>) {
        let x = Matrix::new(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 1.0],
            vec![3.0, 0.0],
            vec![3.0, 1.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn classifier_fits_and_aggregates_probabilities() {
        let (x, y) = simple_data();
        let mut forest = RandomForestClassifier::new()
            .with_n_estimators(11)
            .with_max_depth(Some(3))
            .with_random_state(42);
        forest.fit(&x, &y).unwrap();
        assert_eq!(forest.n_estimators(), 11);
        assert_eq!(forest.classes(), Some(&[0.0, 1.0][..]));
        let proba = forest.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for i in 0..x.nrows() {
            assert!((proba.get(i, 0) + proba.get(i, 1) - 1.0).abs() < 1e-12);
        }
        assert_eq!(forest.predict(&x).unwrap().len(), x.nrows());
    }

    #[test]
    fn regressor_averages_tree_predictions() {
        let (x, y) = simple_data();
        let mut forest = RandomForestRegressor::new()
            .with_n_estimators(7)
            .with_max_depth(Some(2))
            .with_random_state(9);
        forest.fit(&x, &y).unwrap();
        let actual = forest.predict(&x).unwrap();
        for (row, &value) in actual.iter().enumerate() {
            let expected: f64 = forest
                .trees()
                .iter()
                .map(|tree| tree.predict(&x).unwrap()[row])
                .sum::<f64>()
                / forest.n_estimators() as f64;
            assert_eq!(value, expected);
        }
    }

    #[test]
    fn bootstrap_and_feature_subsampling_are_deterministic() {
        let (x, y) = simple_data();
        let build = || {
            let mut forest = RandomForestClassifier::new()
                .with_n_estimators(9)
                .with_max_features(MaxFeatures::Count(1))
                .with_random_state(17);
            forest.fit(&x, &y).unwrap();
            forest
        };
        let a = build();
        let b = build();
        assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
        assert_eq!(
            a.feature_importances().unwrap(),
            b.feature_importances().unwrap()
        );
    }

    #[test]
    fn no_bootstrap_uses_full_sample_count() {
        let (x, y) = simple_data();
        let mut forest = RandomForestRegressor::new()
            .with_n_estimators(3)
            .with_bootstrap(false)
            .with_max_depth(Some(1));
        forest.fit(&x, &y).unwrap();
        for tree in forest.trees() {
            assert_eq!(tree.tree().unwrap().node_n_samples(0), Some(x.nrows()));
        }
    }

    #[test]
    fn randint_mask_covers_every_index() {
        // `high − 1` a power of two (2, 3, 5, 9, 17, 33) is where a wrong
        // mask silently drops the top index.
        for high in [2usize, 3, 5, 9, 17, 33, 100] {
            let mut state = LegacyRandomState::new(0);
            let mut seen = vec![false; high];
            for _ in 0..10_000 {
                seen[state.randint(high)] = true;
            }
            let missing: Vec<usize> = seen
                .iter()
                .enumerate()
                .filter(|(_, &s)| !s)
                .map(|(i, _)| i)
                .collect();
            assert!(missing.is_empty(), "high = {high} never drew {missing:?}");
        }
    }

    #[test]
    fn bootstrap_can_draw_last_row_when_n_is_power_of_two_plus_one() {
        let x = Matrix::new((0..9).map(|i| vec![i as f64]).collect()).unwrap();
        let y: Vec<f64> = (0..9).map(|i| (i % 2) as f64).collect();
        let mut forest = RandomForestClassifier::new()
            .with_n_estimators(50)
            .with_max_depth(Some(2))
            .with_random_state(3);
        forest.fit(&x, &y).unwrap();
        assert!(
            forest
                .estimators_samples()
                .iter()
                .any(|indices| indices.contains(&8)),
            "row 8 never entered any bootstrap sample"
        );
    }
}
