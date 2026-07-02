use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;

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
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String>;
}

/// Helper: generate default input names `x0..x{n-1}`.
pub fn default_input_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("x{}", i)).collect()
}
