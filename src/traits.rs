use crate::error::Result;
use crate::matrix::Matrix;

/// Trait for numeric transformers operating on `Matrix -> Matrix`.
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
