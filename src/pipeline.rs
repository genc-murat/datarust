use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::FeatureNames;
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pipeline {
    steps: Vec<(String, TransformerKind)>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { steps: vec![] }
    }

    pub fn push<S: Into<String>>(mut self, name: S, t: TransformerKind) -> Self {
        self.steps.push((name.into(), t));
        self
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        self.steps.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn steps(&self) -> &[(String, TransformerKind)] {
        &self.steps
    }

    pub fn steps_mut(&mut self) -> &mut Vec<(String, TransformerKind)> {
        &mut self.steps
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
        self.steps.iter().all(|(_, t)| t.is_fitted())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaler::{MinMaxScaler, StandardScaler};
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
}
