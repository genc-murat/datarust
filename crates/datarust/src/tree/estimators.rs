//! `DecisionTreeClassifier` and `DecisionTreeRegressor`.

use super::{Criterion, MaxFeatures, TreeBuilder, TreeOptions, TreeStructure};
use crate::error::{DatarustError, Result};
use crate::label_space::LabelSpace;
use crate::matrix::Matrix;
use crate::traits::{
    Classifier, Estimator, ParamValue, Params, PredictProba, Predictor, Regressor,
};

/// CART classification tree (Gini / entropy criteria).
///
/// # Example
///
/// ```rust
/// use datarust::tree::DecisionTreeClassifier;
/// use datarust::traits::{Classifier, Predictor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![1.0, 1.0],
///     vec![2.0, 1.0],
///     vec![5.0, 4.0],
///     vec![6.0, 5.0],
/// ])?;
/// let y = vec![0.0, 0.0, 1.0, 1.0];
///
/// let mut clf = DecisionTreeClassifier::new();
/// clf.fit(&x, &y)?;
/// assert_eq!(clf.predict(&x)?, vec![0.0, 0.0, 1.0, 1.0]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    options: TreeOptions,
    criterion: Criterion,
    n_features: usize,
    tree: Option<TreeStructure>,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeClassifier {
    /// Creates a classifier with sklearn-default parameters.
    pub fn new() -> Self {
        DecisionTreeClassifier {
            options: TreeOptions::default(),
            criterion: Criterion::Gini,
            n_features: 0,
            tree: None,
        }
    }

    /// Sets the split criterion (default `Gini`). `Criterion::SquaredError`
    /// is regression-only and is rejected by `fit`.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Sets the maximum tree depth (`None` = unlimited, the default).
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.options.max_depth = max_depth;
        self
    }

    /// Sets the minimum samples required to split a node (default 2).
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.options.min_samples_split = n;
        self
    }

    /// Sets the minimum samples required in each leaf (default 1).
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.options.min_samples_leaf = n;
        self
    }

    /// Sets the minimum impurity decrease required to split (default 0).
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self {
        self.options.min_impurity_decrease = v;
        self
    }

    /// Sets the per-split feature subset strategy (default all features).
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.options.max_features = max_features;
        self
    }

    /// Sets the seed for deterministic per-split feature subsampling.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.options.random_state = seed;
        self
    }

    /// The fitted tree structure, or `None` before `fit`.
    pub fn tree(&self) -> Option<&TreeStructure> {
        self.tree.as_ref()
    }

    /// Number of features seen at fit time.
    pub fn n_features_in(&self) -> usize {
        self.n_features
    }

    /// Per-feature normalized impurity reduction (sklearn's
    /// `feature_importances_`). Requires `fit`.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("DecisionTreeClassifier".into()))?;
        Ok(tree.feature_importances(self.n_features))
    }

    fn fit_impl(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_weighted_impl(x, y, None, None)
    }

    pub(crate) fn fit_weighted_impl(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weights: Option<Vec<f64>>,
        known_classes: Option<LabelSpace>,
    ) -> Result<()> {
        self.options.validate()?;
        if matches!(self.criterion, Criterion::SquaredError) {
            return Err(DatarustError::InvalidInput(
                "DecisionTreeClassifier requires Criterion::Gini or Criterion::Entropy; \
                 Criterion::SquaredError is regression-only"
                    .into(),
            ));
        }
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
        let classes = known_classes.unwrap_or(LabelSpace::fit(y)?);
        let n_classes = classes.len();
        let codes: Vec<usize> = y
            .iter()
            .map(|&v| classes.encode(v))
            .collect::<Result<Vec<_>>>()?;
        let mut builder = TreeBuilder::new_with_weights(
            x,
            y,
            Some(&codes),
            n_classes,
            self.criterion,
            self.options.clone(),
            sample_weights,
        );
        builder.build();
        self.n_features = x.ncols();
        self.tree = Some(TreeStructure::new(builder.nodes, Some(classes)));
        Ok(())
    }
}

impl Estimator for DecisionTreeClassifier {}

impl Predictor for DecisionTreeClassifier {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_impl(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("DecisionTreeClassifier".into()))?;
        let classes = tree.classes.as_ref().expect("classification tree");
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let mut out = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let leaf = tree.leaf_for(x.row(i))?;
            let class_idx = tree.nodes[leaf].value.argmax_class()?;
            out.push(classes.decode(class_idx)?);
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }
}

impl Classifier for DecisionTreeClassifier {}

impl PredictProba for DecisionTreeClassifier {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("DecisionTreeClassifier".into()))?;
        let classes = tree.classes.as_ref().expect("classification tree");
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let n_classes = classes.len();
        let mut flat: Vec<f64> = Vec::with_capacity(x.nrows() * n_classes);
        for i in 0..x.nrows() {
            let leaf = tree.leaf_for(x.row(i))?;
            flat.extend(tree.nodes[leaf].value.class_probabilities()?);
        }
        Matrix::from_flat(x.nrows(), n_classes, flat)
    }
}

impl Params for DecisionTreeClassifier {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = self.options.shared_params();
        params.push(("criterion", ParamValue::Int(self.criterion_code())));
        params.push((
            "max_depth",
            ParamValue::Int(self.options.max_depth.unwrap_or(0)),
        ));
        params.push(("max_features", ParamValue::Int(self.max_features_code())));
        params
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        if self.options.set_shared_param(name, value.clone())? {
            return Ok(());
        }
        match (name, value) {
            ("criterion", ParamValue::Int(0)) => {
                self.criterion = Criterion::Gini;
                Ok(())
            }
            ("criterion", ParamValue::Int(1)) => {
                self.criterion = Criterion::Entropy;
                Ok(())
            }
            // Storable for `get_params` round-trips; rejected by `fit`.
            ("criterion", ParamValue::Int(2)) => {
                self.criterion = Criterion::SquaredError;
                Ok(())
            }
            ("criterion", _) => Err(DatarustError::InvalidInput(
                "criterion expects Int(0)=gini, Int(1)=entropy or Int(2)=squared_error".into(),
            )),
            ("max_depth", ParamValue::Int(0)) => {
                self.options.max_depth = None;
                Ok(())
            }
            ("max_depth", ParamValue::Int(v)) => {
                self.options.max_depth = Some(v);
                Ok(())
            }
            ("max_features", ParamValue::Int(v)) => {
                self.options.max_features = match v {
                    0 => MaxFeatures::All,
                    1 => MaxFeatures::Sqrt,
                    2 => MaxFeatures::Log2,
                    k => MaxFeatures::Count(k - 2),
                };
                Ok(())
            }
            _ => Err(DatarustError::InvalidInput(format!(
                "unknown parameter '{name}' for DecisionTreeClassifier"
            ))),
        }
    }
}

impl DecisionTreeClassifier {
    /// Stable integer code for the criterion (0 gini, 1 entropy; 2 marks
    /// the regression-only squared error, which `fit` rejects).
    fn criterion_code(&self) -> usize {
        match self.criterion {
            Criterion::Gini => 0,
            Criterion::Entropy => 1,
            Criterion::SquaredError => 2,
        }
    }

    /// Stable integer code for `max_features` (0 all, 1 sqrt, 2 log2, k+2
    /// an explicit count).
    fn max_features_code(&self) -> usize {
        match self.options.max_features {
            MaxFeatures::All => 0,
            MaxFeatures::Sqrt => 1,
            MaxFeatures::Log2 => 2,
            MaxFeatures::Count(k) => k.saturating_add(2),
        }
    }
}

/// CART regression tree (squared-error criterion).
///
/// # Example
///
/// ```rust
/// use datarust::tree::DecisionTreeRegressor;
/// use datarust::traits::{Predictor, Regressor};
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![9.0], vec![10.0]])?;
/// let y = vec![1.0, 1.5, 9.0, 9.5];
///
/// let mut reg = DecisionTreeRegressor::new();
/// reg.fit(&x, &y)?;
/// // With default settings the tree grows to pure leaves, so training
/// // predictions reproduce the targets exactly (sklearn parity).
/// let pred = reg.predict(&x)?;
/// assert_eq!(pred, vec![1.0, 1.5, 9.0, 9.5]);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    options: TreeOptions,
    n_features: usize,
    tree: Option<TreeStructure>,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeRegressor {
    /// Creates a regressor with sklearn-default parameters.
    pub fn new() -> Self {
        DecisionTreeRegressor {
            options: TreeOptions::default(),
            n_features: 0,
            tree: None,
        }
    }

    /// Sets the maximum tree depth (`None` = unlimited, the default).
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.options.max_depth = max_depth;
        self
    }

    /// Sets the minimum samples required to split a node (default 2).
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.options.min_samples_split = n;
        self
    }

    /// Sets the minimum samples required in each leaf (default 1).
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.options.min_samples_leaf = n;
        self
    }

    /// Sets the minimum impurity decrease required to split (default 0).
    pub fn with_min_impurity_decrease(mut self, v: f64) -> Self {
        self.options.min_impurity_decrease = v;
        self
    }

    /// Sets the per-split feature subset strategy (default all features).
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.options.max_features = max_features;
        self
    }

    /// Sets the seed for deterministic per-split feature subsampling.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.options.random_state = seed;
        self
    }

    /// The fitted tree structure, or `None` before `fit`.
    pub fn tree(&self) -> Option<&TreeStructure> {
        self.tree.as_ref()
    }

    /// Number of features seen at fit time.
    pub fn n_features_in(&self) -> usize {
        self.n_features
    }

    /// Per-feature normalized impurity reduction (sklearn's
    /// `feature_importances_`). Requires `fit`.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("DecisionTreeRegressor".into()))?;
        Ok(tree.feature_importances(self.n_features))
    }

    fn fit_impl(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_weighted_impl(x, y, None)
    }

    pub(crate) fn fit_weighted_impl(
        &mut self,
        x: &Matrix,
        y: &[f64],
        sample_weights: Option<Vec<f64>>,
    ) -> Result<()> {
        self.options.validate()?;
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
        for (i, &v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(DatarustError::InvalidInput(format!(
                    "y contains a non-finite value at index {i}"
                )));
            }
        }
        let mut builder = TreeBuilder::new_with_weights(
            x,
            y,
            None,
            0,
            Criterion::SquaredError,
            self.options.clone(),
            sample_weights,
        );
        builder.build();
        self.n_features = x.ncols();
        self.tree = Some(TreeStructure::new(builder.nodes, None));
        Ok(())
    }
}

impl Estimator for DecisionTreeRegressor {}

impl Predictor for DecisionTreeRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_impl(x, y)
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| DatarustError::NotFitted("DecisionTreeRegressor".into()))?;
        if x.ncols() != self.n_features {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} features", x.ncols()),
            });
        }
        x.validate_finite()?;
        let mut out = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            let leaf = tree.leaf_for(x.row(i))?;
            out.push(tree.nodes[leaf].value.regression_mean()?);
        }
        Ok(out)
    }

    fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }
}

impl Regressor for DecisionTreeRegressor {
    fn name(&self) -> &'static str {
        "DecisionTreeRegressor"
    }
}

impl Params for DecisionTreeRegressor {
    fn get_params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = self.options.shared_params();
        params.push((
            "max_depth",
            ParamValue::Int(self.options.max_depth.unwrap_or(0)),
        ));
        params.push(("max_features", ParamValue::Int(self.max_features_code())));
        params
    }

    fn set_params(&mut self, name: &str, value: ParamValue) -> Result<()> {
        if self.options.set_shared_param(name, value.clone())? {
            return Ok(());
        }
        match (name, value) {
            ("max_depth", ParamValue::Int(0)) => {
                self.options.max_depth = None;
                Ok(())
            }
            ("max_depth", ParamValue::Int(v)) => {
                self.options.max_depth = Some(v);
                Ok(())
            }
            ("max_features", ParamValue::Int(v)) => {
                self.options.max_features = match v {
                    0 => MaxFeatures::All,
                    1 => MaxFeatures::Sqrt,
                    2 => MaxFeatures::Log2,
                    k => MaxFeatures::Count(k - 2),
                };
                Ok(())
            }
            _ => Err(DatarustError::InvalidInput(format!(
                "unknown parameter '{name}' for DecisionTreeRegressor"
            ))),
        }
    }
}

impl DecisionTreeRegressor {
    /// Stable integer code for `max_features` (0 all, 1 sqrt, 2 log2, k+2
    /// an explicit count).
    fn max_features_code(&self) -> usize {
        match self.options.max_features {
            MaxFeatures::All => 0,
            MaxFeatures::Sqrt => 1,
            MaxFeatures::Log2 => 2,
            MaxFeatures::Count(k) => k.saturating_add(2),
        }
    }
}
