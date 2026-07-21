//! Core traits shared by the numeric and categorical transformers.

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};

/// Shared base contract for every estimator in datarust.
///
/// `Estimator` is the common base for transformers and supervised predictors.
/// It deliberately does not prescribe input or output types; those belong to
/// the more specific [`Transformer`], [`Regressor`], and [`Classifier`] traits.
pub trait Estimator {}

/// Trait for numeric transformers operating on `Matrix -> Matrix`.
///
/// All scalers, decompositions, and other numeric transformers implement this
/// trait.  Call [`fit`](Transformer::fit) to learn parameters from training
/// data, then [`transform`](Transformer::transform) to apply the learned
/// transformation to new data.
///
/// ```rust
/// use datarust::scaler::StandardScaler;
/// use datarust::traits::Transformer;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![1.0, 10.0],
///     vec![2.0, 20.0],
///     vec![3.0, 30.0],
/// ])?;
/// let mut s = StandardScaler::new();
/// let out = s.fit_transform(&x)?;
/// assert_eq!(out.ncols(), 2);
/// // Inverse supported on StandardScaler
/// let back = s.inverse_transform(&out)?;
/// for i in 0..3 {
///     for j in 0..2 {
///         assert!((back.get(i, j) - x.get(i, j)).abs() < 1e-9);
///     }
/// }
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait Transformer: Estimator {
    /// Name of the transformer, used for diagnostics.
    fn name(&self) -> &'static str;

    /// Fit the transformer on training data.
    fn fit(&mut self, x: &Matrix) -> Result<()>;

    /// Transform data using fitted parameters.
    fn transform(&self, x: &Matrix) -> Result<Matrix>;

    /// Convenience: fit then transform.
    fn fit_transform(&mut self, x: &Matrix) -> Result<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Fit the transformer with supervised target values when they are
    /// available. Unsupervised transformers ignore `y` by default. Supervised
    /// feature selectors override this method, allowing them to live safely in
    /// a [`SupervisedPipeline`](crate::pipeline::SupervisedPipeline).
    fn fit_with_target(&mut self, x: &Matrix, _y: &[f64]) -> Result<()> {
        self.fit(x)
    }

    /// Reverse the transformation, recovering an approximation of the
    /// original input.  Not all transformers support this; the default
    /// implementation returns an error.
    fn inverse_transform(&self, _x: &Matrix) -> Result<Matrix> {
        Err(DatarustError::InvalidInput(format!(
            "{} does not support inverse_transform",
            Transformer::name(self)
        )))
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Trait for transformers that can report their output feature names.
///
/// `input_features` are the names of the columns fed to `fit`. When `None`,
/// implementations generate synthetic names (e.g. `x0`, `x1`).
pub trait FeatureNames {
    /// Returns the output feature names given the optional input feature names.
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String>;
}

/// Helper: generate default input names `x0..x{n-1}`.
pub fn default_input_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("x{}", i)).collect()
}

/// Trait for categorical (string) transformers operating on `StrMatrix -> Matrix`.
///
/// All categorical encoders that accept a [`StrMatrix`] and
/// produce a [`Matrix`] implement this trait.
pub trait CategoricalTransformer: Estimator {
    /// Human-readable name of the transformer.
    fn name(&self) -> &'static str;

    /// Fit the transformer on categorical training data.
    fn fit(&mut self, x: &StrMatrix) -> Result<()>;

    /// Transform categorical data using fitted parameters, returning a numeric
    /// [`Matrix`].
    fn transform(&self, x: &StrMatrix) -> Result<Matrix>;

    /// Convenience: fit then transform.
    fn fit_transform(&mut self, x: &StrMatrix) -> Result<Matrix> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Reverse the transformation, recovering category strings from numeric
    /// codes.  Not all encoders support this; the default implementation
    /// returns an error.
    fn inverse_transform(&self, _y: &Matrix) -> Result<StrMatrix> {
        Err(DatarustError::InvalidInput(format!(
            "{} does not support inverse_transform",
            CategoricalTransformer::name(self)
        )))
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Trait for supervised encoders that require target values during `fit`.
///
/// Input is a [`StrMatrix`] of categorical features and a slice of target
/// values `&[f64]`.  Output is a numeric [`Matrix`].
pub trait TargetTransformer: Estimator {
    /// Human-readable name of the transformer.
    fn name(&self) -> &'static str;

    /// Fit the transformer on categorical features and target values.
    fn fit(&mut self, x: &StrMatrix, y: &[f64]) -> Result<()>;

    /// Transform categorical data using fitted parameters.
    fn transform(&self, x: &StrMatrix) -> Result<Matrix>;

    /// Convenience: fit then transform.
    fn fit_transform(&mut self, x: &StrMatrix, y: &[f64]) -> Result<Matrix> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Reverse the transformation.  Not all supervised encoders support this;
    /// the default implementation returns an error.
    fn inverse_transform(&self, _y: &Matrix) -> Result<StrMatrix> {
        Err(DatarustError::InvalidInput(format!(
            "{} does not support inverse_transform",
            TargetTransformer::name(self)
        )))
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Shared fitting and prediction contract for supervised estimators operating
/// on numeric features and targets.
pub trait Predictor: Estimator {
    /// Fit the estimator on training features and target values.
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()>;

    /// Predict one numeric value per input row. For classifiers this is the
    /// predicted class label; probability estimates are exposed separately by
    /// [`PredictProba`].
    fn predict(&self, x: &Matrix) -> Result<Vec<f64>>;

    /// Convenience: fit then predict on the same data.
    fn fit_predict(&mut self, x: &Matrix, y: &[f64]) -> Result<Vec<f64>> {
        self.fit(x, y)?;
        self.predict(x)
    }

    /// Whether the estimator has learned fitted state.
    fn is_fitted(&self) -> bool;
}

/// Trait for regression estimators operating on `Matrix` features and
/// `&[f64]` targets.
///
/// Regression models (e.g. [`LinearRegression`]) implement this trait.
/// Call [`fit`](Predictor::fit) to learn coefficients from training data and
/// targets, then [`predict`](Predictor::predict) to generate predictions for
/// new data.
///
/// [`LinearRegression`]: crate::linear_model::LinearRegression
///
/// ```rust
/// use datarust::linear_model::LinearRegression;
/// use datarust::traits::Predictor;
/// use datarust::Matrix;
///
/// let x = Matrix::new(vec![
///     vec![1.0],
///     vec![2.0],
///     vec![3.0],
///     vec![4.0],
/// ])?;
/// let y = vec![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
/// let pred = model.predict(&x)?;
/// assert!((pred[0] - 3.0).abs() < 1e-9);
/// assert!((pred[3] - 9.0).abs() < 1e-9);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait Regressor: Predictor {
    /// Name of the estimator, used for diagnostics.
    ///
    /// Kept on `Regressor` for backwards compatibility with code that uses
    /// regression estimators through this trait.
    fn name(&self) -> &'static str;

    /// R² (coefficient of determination) of the prediction.
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let prediction = Predictor::predict(self, x)?;
        crate::metrics::regression::r2_score(y, &prediction)
    }
}

/// Trait for classifiers that return one class label per input row.
pub trait Classifier: Predictor {
    /// Mean classification accuracy, mirroring sklearn's default classifier
    /// score.
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64> {
        let prediction = self.predict(x)?;
        crate::metrics::classification::accuracy_score(y, &prediction)
    }
}

/// Trait for classifiers that expose a probability for every class.
///
/// The returned matrix is shaped `(n_samples, n_classes)` and follows the
/// estimator's class order. Binary [`LogisticRegression`](crate::LogisticRegression)
/// returns columns `[P(class=0), P(class=1)]`.
pub trait PredictProba: Classifier {
    /// Return per-class probability estimates.
    fn predict_proba(&self, x: &Matrix) -> Result<Matrix>;
}

/// Trait for 1-D label encoders that map `&[String]` to `Vec<usize>`.
///
/// Used to encode target labels for supervised learning.
pub trait LabelTransformer: Estimator {
    /// Human-readable name of the transformer.
    fn name(&self) -> &'static str;

    /// Fit on label data, learning the unique sorted classes.
    fn fit(&mut self, x: &[String]) -> Result<()>;

    /// Transform labels to integer indices.
    fn transform(&self, x: &[String]) -> Result<Vec<usize>>;

    /// Reverses the transformation: indices back to strings.
    fn inverse_transform(&self, x: &[usize]) -> Result<Vec<String>>;

    /// Convenience: fit then transform.
    fn fit_transform(&mut self, x: &[String]) -> Result<Vec<usize>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

macro_rules! impl_estimator_from_transformer {
    ($($ty:path),+ $(,)?) => {
        $(
            impl Estimator for $ty {}
        )+
    };
}

impl_estimator_from_transformer!(
    crate::scaler::StandardScaler,
    crate::scaler::MinMaxScaler,
    crate::scaler::MaxAbsScaler,
    crate::scaler::RobustScaler,
    crate::scaler::Normalizer,
    crate::scaler::Binarizer,
    crate::scaler::KBinsDiscretizer,
    crate::scaler::QuantileTransformer,
    crate::scaler::PowerTransformer,
    crate::polynomial::PolynomialFeatures,
    crate::selection::VarianceThreshold,
    crate::selection::SelectKBest,
    crate::decomposition::PCA,
    crate::decomposition::TruncatedSVD,
    crate::imputer::SimpleImputer,
    crate::imputer::KnnImputer,
    crate::function_transformer::FunctionTransformer,
    crate::transformer_kind::TransformerKind,
    crate::pipeline::Pipeline,
);

macro_rules! impl_estimator_from_categorical_transformer {
    ($($ty:path),+ $(,)?) => {
        $(
            impl Estimator for $ty {}
        )+
    };
}

impl_estimator_from_categorical_transformer!(
    crate::encoder::OneHotEncoder,
    crate::encoder::OrdinalEncoder,
    crate::encoder::FrequencyEncoder,
    crate::categorical_kind::CategoricalTransformerKind,
);

macro_rules! impl_estimator_from_target_transformer {
    ($($ty:path),+ $(,)?) => {
        $(
            impl Estimator for $ty {}
        )+
    };
}

impl_estimator_from_target_transformer!(
    crate::encoder::TargetEncoder,
    crate::target_kind::TargetTransformerKind,
);

impl Estimator for crate::encoder::LabelEncoder {}
