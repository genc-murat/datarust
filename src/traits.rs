//! Core traits shared by the numeric and categorical transformers.

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};

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
pub trait Transformer {
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

    /// Reverse the transformation, recovering an approximation of the
    /// original input.  Not all transformers support this; the default
    /// implementation returns an error.
    fn inverse_transform(&self, _x: &Matrix) -> Result<Matrix> {
        Err(DatarustError::InvalidInput(format!(
            "{} does not support inverse_transform",
            self.name()
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
pub trait CategoricalTransformer {
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
            self.name()
        )))
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Trait for supervised encoders that require target values during `fit`.
///
/// Input is a [`StrMatrix`] of categorical features and a slice of target
/// values `&[f64]`.  Output is a numeric [`Matrix`].
pub trait TargetTransformer {
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
            self.name()
        )))
    }

    /// Whether the transformer has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Trait for regression estimators operating on `Matrix` features and
/// `&[f64]` targets.
///
/// Regression models (e.g. [`LinearRegression`]) implement this trait.
/// Call [`fit`](Regressor::fit) to learn coefficients from training data and
/// targets, then [`predict`](Regressor::predict) to generate predictions for
/// new data.
///
/// [`LinearRegression`]: crate::linear_model::LinearRegression
///
/// ```rust
/// use datarust::linear_model::LinearRegression;
/// use datarust::traits::Regressor;
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
pub trait Regressor {
    /// Name of the estimator, used for diagnostics.
    fn name(&self) -> &'static str;

    /// Fit the estimator on training features and target values.
    fn fit(&mut self, x: &Matrix, y: &[f64]) -> Result<()>;

    /// Predict target values for the given features.
    fn predict(&self, x: &Matrix) -> Result<Vec<f64>>;

    /// Convenience: fit then predict on the same data.
    fn fit_predict(&mut self, x: &Matrix, y: &[f64]) -> Result<Vec<f64>> {
        self.fit(x, y)?;
        self.predict(x)
    }

    /// Whether the estimator has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Trait for 1-D label encoders that map `&[String]` to `Vec<usize>`.
///
/// Used to encode target labels for supervised learning.
pub trait LabelTransformer {
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
