use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Apply a user-defined function as a transformer, mirroring
/// `sklearn.preprocessing.FunctionTransformer`.
///
/// The function is stored as a function pointer.
///
/// # Serialization
///
/// The function pointers are **not** serialized.  After deserialization call
/// [`set_func`](FunctionTransformer::set_func) to restore the function before
/// calling `transform` or `inverse_transform`.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionTransformer {
    #[cfg_attr(feature = "serde", serde(skip))]
    func: Option<fn(&Matrix) -> Result<Matrix>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    inv_func: Option<fn(&Matrix) -> Result<Matrix>>,
    n_features_in_: usize,
    fitted: bool,
}

impl FunctionTransformer {
    pub fn new(func: fn(&Matrix) -> Result<Matrix>) -> Self {
        Self {
            func: Some(func),
            inv_func: None,
            n_features_in_: 0,
            fitted: false,
        }
    }

    pub fn with_inverse(mut self, inv_func: fn(&Matrix) -> Result<Matrix>) -> Self {
        self.inv_func = Some(inv_func);
        self
    }

    pub fn set_func(&mut self, func: fn(&Matrix) -> Result<Matrix>) {
        self.func = Some(func);
    }

    pub fn set_inv_func(&mut self, inv_func: fn(&Matrix) -> Result<Matrix>) {
        self.inv_func = Some(inv_func);
    }
}

impl std::fmt::Debug for FunctionTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTransformer")
            .field("n_features_in_", &self.n_features_in_)
            .field("fitted", &self.fitted)
            .field("has_func", &self.func.is_some())
            .field("has_inv_func", &self.inv_func.is_some())
            .finish()
    }
}

impl Transformer for FunctionTransformer {
    fn name(&self) -> &'static str {
        "FunctionTransformer"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        self.n_features_in_ = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("FunctionTransformer".into()));
        }
        let func = self.func.ok_or_else(|| {
            DatarustError::InvalidInput(
                "FunctionTransformer: func not set (call set_func after deserialization)".into(),
            )
        })?;
        func(x)
    }

    fn inverse_transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("FunctionTransformer".into()));
        }
        let inv = self.inv_func.ok_or_else(|| {
            DatarustError::InvalidInput("FunctionTransformer: inverse_func not set".into())
        })?;
        inv(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for FunctionTransformer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.n_features_in_),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: &Matrix) -> Result<Matrix> {
        let out: Vec<Vec<f64>> = x
            .rows_ref()
            .iter()
            .map(|row| row.iter().map(|&v| v * v).collect())
            .collect();
        Matrix::new(out)
    }

    fn sqrt_transform(x: &Matrix) -> Result<Matrix> {
        let out: Vec<Vec<f64>> = x
            .rows_ref()
            .iter()
            .map(|row| row.iter().map(|&v| v.sqrt()).collect())
            .collect();
        Matrix::new(out)
    }

    fn m1() -> Matrix {
        Matrix::new(vec![vec![1.0, 4.0], vec![9.0, 16.0], vec![25.0, 36.0]]).unwrap()
    }

    #[test]
    fn apply_square_function() {
        let mut ft = FunctionTransformer::new(square);
        let x = m1();
        let out = ft.fit_transform(&x).unwrap();
        assert!((out.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 16.0).abs() < 1e-12);
        assert!((out.get(1, 0) - 81.0).abs() < 1e-12);
        assert!((out.get(2, 1) - 1296.0).abs() < 1e-12);
        assert!(ft.is_fitted());
    }

    #[test]
    fn inverse_transform_round_trip() {
        let mut ft = FunctionTransformer::new(square).with_inverse(sqrt_transform);
        let x = Matrix::new(vec![vec![1.0, 4.0], vec![9.0, 16.0]]).unwrap();
        let out = ft.fit_transform(&x).unwrap();
        let back = ft.inverse_transform(&out).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((back.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn transform_before_fit_errors() {
        let ft = FunctionTransformer::new(square);
        assert!(matches!(
            ft.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn inverse_transform_without_inv_func_errors() {
        let mut ft = FunctionTransformer::new(square);
        ft.fit(&m1()).unwrap();
        assert!(ft.inverse_transform(&m1()).is_err());
    }

    #[test]
    fn inverse_transform_before_fit_errors() {
        let ft = FunctionTransformer::new(square).with_inverse(sqrt_transform);
        assert!(matches!(
            ft.inverse_transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_without_func_errors() {
        // Simulate deserialized state (func = None)
        let ft = FunctionTransformer {
            func: None,
            inv_func: None,
            n_features_in_: 2,
            fitted: true,
        };
        assert!(ft.transform(&m1()).is_err());
    }

    #[test]
    fn feature_names_passthrough() {
        let mut ft = FunctionTransformer::new(square);
        ft.fit(&m1()).unwrap();
        let names = ft.feature_names_out(Some(&["a".into(), "b".into()]));
        assert_eq!(names, vec!["a", "b"]);
        let default = ft.feature_names_out(None);
        assert_eq!(default, vec!["x0", "x1"]);
    }

    #[test]
    fn func_can_be_set_after_construction() {
        let mut ft = FunctionTransformer {
            func: None,
            inv_func: None,
            n_features_in_: 2,
            fitted: true,
        };
        ft.set_func(square);
        let x = Matrix::new(vec![vec![2.0, 3.0]]).unwrap();
        let out = ft.transform(&x).unwrap();
        assert!((out.get(0, 0) - 4.0).abs() < 1e-12);
        assert!((out.get(0, 1) - 9.0).abs() < 1e-12);
    }
}
