use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{Classifier, Estimator, FeatureNames, PredictProba, Predictor, Regressor};
use crate::transformer_kind::TransformerKind;
use crate::Transformer;

/// Sequential pipeline of numeric transformers, mirroring
/// `sklearn.pipeline.Pipeline` (numeric subset).
///
/// Each step is a [`TransformerKind`] variant.  Data flows through each step
/// in order; `fit_transform` fits each step on the output of the previous one.
///
/// ```rust,ignore
/// use datarust::scaler::{StandardScaler, MinMaxScaler};
/// use datarust::transformer_kind::TransformerKind;
///
/// let mut p = Pipeline::new()
///     .push("std", TransformerKind::StandardScaler(StandardScaler::new()))
///     .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
/// let out = p.fit_transform(&x)?;
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pipeline {
    steps: Vec<(String, TransformerKind)>,
}

impl Pipeline {
    /// Creates a new empty pipeline.
    pub fn new() -> Self {
        Self { steps: vec![] }
    }

    /// Appends a named transformer step to the end of the pipeline.
    pub fn push<S: Into<String>>(mut self, name: S, t: TransformerKind) -> Self {
        self.steps.push((name.into(), t));
        self
    }

    /// Returns the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether the pipeline contains no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the names of the steps in order.
    pub fn names(&self) -> Vec<&str> {
        self.steps.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Returns the ordered list of (name, transformer) steps.
    pub fn steps(&self) -> &[(String, TransformerKind)] {
        &self.steps
    }

    /// Returns mutable access to the ordered list of steps.
    pub fn steps_mut(&mut self) -> &mut Vec<(String, TransformerKind)> {
        &mut self.steps
    }

    /// Access a step by name.
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::StandardScaler;
    /// use datarust::transformer_kind::TransformerKind;
    ///
    /// let p = Pipeline::new()
    ///     .push("scale", TransformerKind::StandardScaler(StandardScaler::new()));
    /// assert!(p.get_step("scale").is_some());
    /// assert!(p.get_step("nope").is_none());
    /// ```
    pub fn get_step(&self, name: &str) -> Option<&TransformerKind> {
        self.steps.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Mutable access to a step by name.
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::{StandardScaler, MaxAbsScaler};
    /// use datarust::transformer_kind::TransformerKind;
    /// use datarust::traits::Transformer;
    ///
    /// let mut p = Pipeline::new()
    ///     .push("a", TransformerKind::StandardScaler(StandardScaler::new()));
    /// let step = p.get_step_mut("a").unwrap();
    /// *step = TransformerKind::MaxAbsScaler(MaxAbsScaler::new());
    /// assert_eq!(p.get_step("a").unwrap().name(), "MaxAbsScaler");
    /// ```
    pub fn get_step_mut(&mut self, name: &str) -> Option<&mut TransformerKind> {
        self.steps
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    /// Access a step and its name by index (0-based).
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::StandardScaler;
    /// use datarust::transformer_kind::TransformerKind;
    /// use datarust::traits::Transformer;
    ///
    /// let p = Pipeline::new()
    ///     .push("scale", TransformerKind::StandardScaler(StandardScaler::new()));
    /// let (name, step) = p.step(0).unwrap();
    /// assert_eq!(name, "scale");
    /// assert_eq!(step.name(), "StandardScaler");
    /// ```
    pub fn step(&self, index: usize) -> Option<(&String, &TransformerKind)> {
        self.steps.get(index).map(|(n, t)| (n, t))
    }

    /// Mutable access to a step and its name by index.
    pub fn step_mut(&mut self, index: usize) -> Option<(&mut String, &mut TransformerKind)> {
        self.steps.get_mut(index).map(|(n, t)| (n, t))
    }

    /// Remove a step by index, returning its name and transformer.
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::StandardScaler;
    /// use datarust::transformer_kind::TransformerKind;
    ///
    /// let mut p = Pipeline::new()
    ///     .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
    ///     .push("b", TransformerKind::StandardScaler(StandardScaler::new()));
    /// let (name, _) = p.remove_step(0).unwrap();
    /// assert_eq!(name, "a");
    /// assert_eq!(p.len(), 1);
    /// ```
    pub fn remove_step(&mut self, index: usize) -> Option<(String, TransformerKind)> {
        if index < self.steps.len() {
            Some(self.steps.remove(index))
        } else {
            None
        }
    }

    /// Insert a step at a given index (0-based), shifting later steps right.
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::{StandardScaler, MaxAbsScaler};
    /// use datarust::transformer_kind::TransformerKind;
    ///
    /// let p = Pipeline::new()
    ///     .push("b", TransformerKind::StandardScaler(StandardScaler::new()))
    ///     .insert_step(0, "a", TransformerKind::MaxAbsScaler(MaxAbsScaler::new()));
    /// assert_eq!(p.names(), vec!["a", "b"]);
    /// ```
    pub fn insert_step(mut self, index: usize, name: &str, t: TransformerKind) -> Self {
        self.steps.insert(index, (name.to_string(), t));
        self
    }

    /// Replace a step by name, returning the previous step if found.
    ///
    /// ```rust
    /// use datarust::pipeline::Pipeline;
    /// use datarust::scaler::{StandardScaler, RobustScaler};
    /// use datarust::transformer_kind::TransformerKind;
    /// use datarust::traits::Transformer;
    ///
    /// let mut p = Pipeline::new()
    ///     .push("s", TransformerKind::StandardScaler(StandardScaler::new()));
    /// let old = p.set_step("s", TransformerKind::RobustScaler(RobustScaler::new()));
    /// assert!(old.is_some());
    /// assert_eq!(p.get_step("s").unwrap().name(), "RobustScaler");
    /// ```
    pub fn set_step(&mut self, name: &str, t: TransformerKind) -> Option<TransformerKind> {
        self.steps
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, old)| std::mem::replace(old, t))
    }

    /// Attach a final supervised estimator to this preprocessing pipeline.
    ///
    /// The returned [`SupervisedPipeline`] fits every transformer on the
    /// training fold before fitting the final estimator, preventing feature
    /// selection and scaling leakage during cross-validation.
    pub fn with_estimator<E>(self, estimator: E) -> SupervisedPipeline<E> {
        SupervisedPipeline::from_pipeline(self, estimator)
    }

    /// Fit all preprocessing steps with target values available to supervised
    /// transformers such as [`SelectKBest`](crate::selection::SelectKBest).
    pub fn fit_with_target(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fit_transform_with_target(x, y).map(|_| ())
    }

    /// Fit all preprocessing steps with target values and return transformed
    /// training features.
    pub fn fit_transform_with_target(&mut self, x: &Matrix, y: &[f64]) -> Result<Matrix> {
        if self.steps.is_empty() {
            return Err(DatarustError::InvalidInput("empty pipeline".into()));
        }
        if y.len() != x.nrows() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} targets", x.nrows()),
                actual: format!("{} targets", y.len()),
            });
        }
        let mut current = x.clone();
        for (_name, step) in self.steps.iter_mut() {
            step.fit_with_target(&current, y)?;
            current = step.transform(&current)?;
        }
        Ok(current)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for Pipeline {
    fn name(&self) -> &'static str {
        "Pipeline"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        if self.steps.is_empty() {
            return Err(DatarustError::InvalidInput("empty pipeline".into()));
        }
        let mut current = x.clone();
        for (_name, step) in self.steps.iter_mut() {
            step.fit(&current)?;
            current = step.transform(&current)?;
        }
        Ok(())
    }

    fn fit_with_target(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        Pipeline::fit_with_target(self, x, y)
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if self.steps.is_empty() {
            return Err(DatarustError::InvalidInput("empty pipeline".into()));
        }
        let mut current = x.clone();
        for (_name, step) in self.steps.iter() {
            current = step.transform(&current)?;
        }
        Ok(current)
    }

    fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix> {
        if self.steps.is_empty() {
            return Err(DatarustError::InvalidInput("empty pipeline".into()));
        }
        let mut current = x.clone();
        for (_name, step) in self.steps.iter_mut() {
            current = step.fit_transform(&current)?;
        }
        Ok(current)
    }

    fn is_fitted(&self) -> bool {
        self.steps.iter().all(|(_, t)| Transformer::is_fitted(t))
    }
}

impl FeatureNames for Pipeline {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let mut current: Option<Vec<String>> = input_features.map(|f| f.to_vec());
        for (_name, step) in &self.steps {
            let out = step.feature_names_out(current.as_deref());
            current = Some(out);
        }
        current.unwrap_or_default()
    }
}

/// A sklearn-style supervised pipeline: zero or more preprocessing steps plus
/// a final estimator.
///
/// Unlike [`Pipeline`], this type accepts target values during `fit` and
/// exposes the final estimator's `predict`, `score`, and (when supported)
/// `predict_proba` operations. It is generic over the estimator, retaining
/// static dispatch and type safety while remaining serde-serializable when the
/// selected estimator is serializable.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SupervisedPipeline<E> {
    transformers: Pipeline,
    estimator: E,
    fitted: bool,
}

impl<E> SupervisedPipeline<E> {
    /// Create a supervised pipeline containing only a final estimator.
    pub fn new(estimator: E) -> Self {
        Self {
            transformers: Pipeline::new(),
            estimator,
            fitted: false,
        }
    }

    /// Create a supervised pipeline from an existing preprocessing pipeline.
    pub fn from_pipeline(transformers: Pipeline, estimator: E) -> Self {
        Self {
            transformers,
            estimator,
            fitted: false,
        }
    }

    /// Add a preprocessing step before the final estimator.
    pub fn push<S: Into<String>>(mut self, name: S, transformer: TransformerKind) -> Self {
        self.transformers = self.transformers.push(name, transformer);
        self.fitted = false;
        self
    }

    /// Return the fitted preprocessing pipeline.
    pub fn transformers(&self) -> &Pipeline {
        &self.transformers
    }

    /// Return the final estimator.
    pub fn estimator(&self) -> &E {
        &self.estimator
    }

    /// Return mutable access to the final estimator configuration.
    ///
    /// Mutable access may change learned model state, so the pipeline must be
    /// fitted again before prediction.
    pub fn estimator_mut(&mut self) -> &mut E {
        self.fitted = false;
        &mut self.estimator
    }

    /// Transform features through the fitted preprocessing steps. An
    /// estimator-only pipeline returns a clone of its input.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("SupervisedPipeline".into()));
        }
        if self.transformers.is_empty() {
            Ok(x.clone())
        } else {
            self.transformers.transform(x)
        }
    }

    fn fit_transform_features(&mut self, x: &Matrix, y: &[f64]) -> Result<Matrix> {
        if self.transformers.is_empty() {
            if y.len() != x.nrows() {
                return Err(DatarustError::ShapeMismatch {
                    expected: format!("{} targets", x.nrows()),
                    actual: format!("{} targets", y.len()),
                });
            }
            Ok(x.clone())
        } else {
            self.transformers.fit_transform_with_target(x, y)
        }
    }
}

impl<E> Estimator for SupervisedPipeline<E> {}

impl<E: Predictor> Predictor for SupervisedPipeline<E> {
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()> {
        self.fitted = false;
        let transformed = self.fit_transform_features(x, y)?;
        self.estimator.fit(&transformed, y)?;
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        let transformed = self.transform(x)?;
        self.estimator.predict(&transformed)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl<E: Regressor> Regressor for SupervisedPipeline<E> {
    fn name(&self) -> &'static str {
        "SupervisedPipeline"
    }
}

impl<E: Classifier> Classifier for SupervisedPipeline<E> {}

impl<E: PredictProba> PredictProba for SupervisedPipeline<E> {
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        let transformed = self.transform(x)?;
        self.estimator.predict_proba(&transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::LinearRegression;
    use crate::scaler::{MinMaxScaler, StandardScaler};
    use crate::selection::{ScoreFunc, SelectKBest};
    use crate::traits::{Predictor, Transformer};
    use crate::transformer_kind::TransformerKind;

    fn m1() -> Matrix {
        Matrix::new(vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ])
        .unwrap()
    }

    #[test]
    fn chain_scalers() {
        let mut p = Pipeline::new()
            .push(
                "std",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        let out = p.fit_transform(&m1()).unwrap();
        assert!((out.get(0, 0) - 0.0).abs() < 1e-9);
        assert!((out.get(3, 0) - 1.0).abs() < 1e-9);
        assert!((out.get(0, 1) - 0.0).abs() < 1e-9);
        assert!((out.get(3, 1) - 1.0).abs() < 1e-9);
        assert!(p.is_fitted());
    }

    #[test]
    fn fit_then_transform_consistent() {
        let mut p = Pipeline::new().push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        p.fit(&m1()).unwrap();
        let t1 = p.transform(&m1()).unwrap();
        let mut p2 = Pipeline::new().push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        let t2 = p2.fit_transform(&m1()).unwrap();
        for i in 0..t1.nrows() {
            for j in 0..t1.ncols() {
                assert!((t1.get(i, j) - t2.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let p = Pipeline::new().push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(matches!(
            p.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn empty_pipeline_errors() {
        let mut p = Pipeline::new();
        assert!(matches!(
            p.fit_transform(&m1()),
            Err(DatarustError::InvalidInput(_))
        ));
    }

    #[test]
    fn step_error_propagates() {
        let mut p = Pipeline::new()
            .push(
                "std",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        p.fit(&m1()).unwrap();
        let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert!(p.transform(&bad).is_err());
    }

    #[test]
    fn target_fit_dispatches_through_transformer_trait() {
        let x = Matrix::new(vec![
            vec![-2.0, 0.2],
            vec![-1.0, 0.8],
            vec![1.0, -0.4],
            vec![2.0, 0.1],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let selector = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        let mut pipeline = Pipeline::new().push("select", TransformerKind::SelectKBest(selector));

        let transformer: &mut dyn Transformer = &mut pipeline;
        transformer.fit_with_target(&x, &y).unwrap();

        assert!(pipeline.is_fitted());
        assert_eq!(pipeline.transform(&x).unwrap().ncols(), 1);
    }

    #[test]
    fn supervised_pipeline_invalidates_fitted_state_after_mutation_or_failed_fit() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let mut pipeline = SupervisedPipeline::new(LinearRegression::new());
        pipeline.fit(&x, &y).unwrap();
        assert!(pipeline.is_fitted());

        assert!(pipeline.fit(&x, &[3.0, 5.0]).is_err());
        assert!(!pipeline.is_fitted());
        assert!(matches!(
            pipeline.predict(&x),
            Err(DatarustError::NotFitted(_))
        ));

        pipeline.fit(&x, &y).unwrap();
        let _ = pipeline.estimator_mut();
        assert!(!pipeline.is_fitted());

        pipeline.fit(&x, &y).unwrap();
        pipeline = pipeline.push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(!pipeline.is_fitted());
        assert!(matches!(
            pipeline.predict(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn names_preserved() {
        let p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .push("b", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        assert_eq!(p.names(), vec!["a", "b"]);
    }

    #[test]
    fn feature_names_passthrough_scalers() {
        let p = Pipeline::new()
            .push(
                "std",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        let names = p.feature_names_out(Some(&["age".into(), "salary".into()]));
        assert_eq!(names, vec!["age", "salary"]);
    }

    #[test]
    fn feature_names_chained_into_pca() {
        use crate::decomposition::{PCAComponents, PCA};
        let mut p = Pipeline::new()
            .push(
                "std",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .push(
                "pca",
                TransformerKind::PCA(PCA::new(PCAComponents::Count(2))),
            );
        let x = Matrix::new(vec![
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
        ])
        .unwrap();
        p.fit(&x).unwrap();
        let names = p.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["pca0", "pca1"]);
    }

    #[test]
    fn feature_names_default_input() {
        use crate::decomposition::TruncatedSVD;
        let p = Pipeline::new().push(
            "svd",
            TransformerKind::TruncatedSVD(TruncatedSVD::new(2).unwrap()),
        );
        let names = p.feature_names_out(None);
        assert_eq!(names, vec!["svd0", "svd1"]);
    }

    #[test]
    fn get_step_by_name() {
        let p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .push("b", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        assert!(p.get_step("a").is_some());
        assert!(p.get_step("b").is_some());
        assert!(p.get_step("c").is_none());
    }

    #[test]
    fn get_step_by_name_mut() {
        let mut p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .push("b", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        let step = p.get_step_mut("a").unwrap();
        assert_eq!(step.name(), "StandardScaler");
        // mutate: replace with RobustScaler
        *step = TransformerKind::MaxAbsScaler(crate::scaler::MaxAbsScaler::new());
        assert_eq!(p.get_step("a").unwrap().name(), "MaxAbsScaler");
    }

    #[test]
    fn step_by_index() {
        let p = Pipeline::new().push("a", TransformerKind::StandardScaler(StandardScaler::new()));
        let (name, step) = p.step(0).unwrap();
        assert_eq!(name, "a");
        assert_eq!(step.name(), "StandardScaler");
        assert!(p.step(5).is_none());
    }

    #[test]
    fn remove_step_removes_and_returns() {
        let mut p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .push("b", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        let (name, _) = p.remove_step(0).unwrap();
        assert_eq!(name, "a");
        assert_eq!(p.len(), 1);
        assert!(p.remove_step(5).is_none());
    }

    #[test]
    fn insert_step_adds_at_position() {
        let p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .insert_step(
                0,
                "z",
                TransformerKind::MaxAbsScaler(crate::scaler::MaxAbsScaler::new()),
            );
        assert_eq!(p.len(), 2);
        assert_eq!(p.step(0).unwrap().0, "z");
        assert_eq!(p.step(1).unwrap().0, "a");
    }

    #[test]
    fn set_step_replaces_by_name() {
        let mut p = Pipeline::new()
            .push("a", TransformerKind::StandardScaler(StandardScaler::new()))
            .push("b", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
        let old = p.set_step(
            "a",
            TransformerKind::MaxAbsScaler(crate::scaler::MaxAbsScaler::new()),
        );
        assert!(old.is_some());
        assert_eq!(p.get_step("a").unwrap().name(), "MaxAbsScaler");
        assert!(p
            .set_step(
                "nonexistent",
                TransformerKind::StandardScaler(StandardScaler::new())
            )
            .is_none());
    }

    #[test]
    fn pipeline_with_function_transformer() {
        use crate::function_transformer::FunctionTransformer;
        fn times_two(x: &Matrix) -> Result<Matrix> {
            let out: Vec<Vec<f64>> = x
                .rows_ref()
                .iter()
                .map(|row| row.iter().map(|&v| v * 2.0).collect())
                .collect();
            Matrix::new(out)
        }
        let mut p = Pipeline::new()
            .push(
                "std",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .push(
                "times2",
                TransformerKind::FunctionTransformer(FunctionTransformer::new(times_two)),
            );
        let x = Matrix::new(vec![vec![0.0, 10.0], vec![1.0, 100.0]]).unwrap();
        let out = p.fit_transform(&x).unwrap();
        // std: col0 mean=0.5, std=0.5; col1 mean=55, std=45
        // row0 col0: (0-0.5)/0.5 = -1, then *2 = -2
        assert!((out.get(0, 0) - (-2.0)).abs() < 1e-9);
        // row0 col1: (10-55)/45 = -1, then *2 = -2
        assert!((out.get(0, 1) - (-2.0)).abs() < 1e-9);
        assert!(p.is_fitted());
        // Verify we can access the function transformer step by name
        let step = p.get_step("times2").unwrap();
        assert_eq!(step.name(), "FunctionTransformer");
    }
}
