//! Integration tests for end-to-end preprocessing pipelines.

use datarust::encoder::OneHotEncoder;
use datarust::imputer::{ImputeStrategy, SimpleImputer};
use datarust::pipeline::Pipeline;
use datarust::scaler::{MinMaxScaler, Norm, Normalizer, RobustScaler, StandardScaler};
use datarust::transformer_kind::TransformerKind;
use datarust::Transformer;
use datarust::{CategoricalTransformerKind, PredictProba, Predictor};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn end_to_end_numeric_pipeline() {
    let raw = datarust::Matrix::new(vec![
        vec![1.0, 100.0, f64::NAN],
        vec![2.0, 200.0, 3.0],
        vec![3.0, 300.0, 6.0],
        vec![4.0, 400.0, 9.0],
    ])
    .unwrap();

    let mut pipe = Pipeline::new()
        .push(
            "impute",
            TransformerKind::SimpleImputer(SimpleImputer::new(ImputeStrategy::Mean)),
        )
        .push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .push(
            "norm",
            TransformerKind::Normalizer(Normalizer::new(Norm::L2)),
        );

    let out = pipe.fit_transform(&raw).unwrap();
    assert!(pipe.is_fitted());

    for i in 0..out.nrows() {
        let n: f64 = out.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(approx(n, 1.0, 1e-9), "row {} norm {}", i, n);
    }
}

#[test]
fn pipeline_minmax_then_robust_in_range() {
    let raw =
        datarust::Matrix::new(vec![vec![10.0, -5.0], vec![20.0, 5.0], vec![30.0, 15.0]]).unwrap();

    let mut pipe = Pipeline::new()
        .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()))
        .push("robust", TransformerKind::RobustScaler(RobustScaler::new()));

    let out = pipe.fit_transform(&raw).unwrap();
    assert_eq!(out.nrows(), 3);
    assert_eq!(out.ncols(), 2);
    for i in 0..3 {
        for j in 0..2 {
            assert!(out.get(i, j).is_finite());
        }
    }
}

#[test]
fn train_then_inference_preserves_fitted_params() {
    let train = datarust::Matrix::new(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
    ])
    .unwrap();

    let mut pipe = Pipeline::new().push(
        "std",
        TransformerKind::StandardScaler(StandardScaler::new()),
    );
    let train_out = pipe.fit_transform(&train).unwrap();

    let test = datarust::Matrix::new(vec![vec![1.0, 10.0], vec![4.0, 40.0]]).unwrap();
    let out = pipe.transform(&test).unwrap();
    for (j, train_row) in [0usize, 3].iter().enumerate() {
        for c in 0..2 {
            assert!(approx(out.get(j, c), train_out.get(*train_row, c), 1e-9));
        }
    }
}

#[test]
fn pipeline_empty_step_errors() {
    let raw = datarust::Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
    let mut pipe = Pipeline::new();
    assert!(pipe.fit_transform(&raw).is_err());
}

#[test]
fn variance_threshold_inside_pipeline() {
    use datarust::selection::VarianceThreshold;
    let raw = datarust::Matrix::new(vec![
        vec![5.0, 1.0],
        vec![5.0, 2.0],
        vec![5.0, 3.0],
        vec![5.0, 4.0],
    ])
    .unwrap();
    let mut pipe = Pipeline::new()
        .push(
            "vt",
            TransformerKind::VarianceThreshold(VarianceThreshold::default()),
        )
        .push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
    let out = pipe.fit_transform(&raw).unwrap();
    assert_eq!(out.ncols(), 1);
    assert_eq!(out.nrows(), 4);
}

#[test]
fn onehot_then_scaler_via_column_transformer() {
    use datarust::compose::{ColumnTransformer, Remainder, Table};
    let numeric =
        datarust::Matrix::new(vec![vec![1.0, 100.0], vec![2.0, 200.0], vec![3.0, 300.0]]).unwrap();
    let categorical =
        datarust::StrMatrix::from_strings(vec![vec!["a"], vec!["b"], vec!["a"]]).unwrap();
    let table = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Passthrough)
        .add_numeric(
            "num",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_categorical(
            "cat",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        );
    let out = ct.fit_transform(&table).unwrap();
    assert_eq!(out.ncols(), 4);
    assert_eq!(out.nrows(), 3);
}

#[test]
fn supervised_pipeline_fits_selector_and_final_classifier() {
    use datarust::linear_model::{LogisticRegression, LogisticSolver};
    use datarust::selection::{ScoreFunc, SelectKBest};

    // The first feature alone separates the classes. The selector must see y
    // while the final estimator must only see the selected feature.
    let x = datarust::Matrix::new(vec![
        vec![-4.0, 0.1],
        vec![-3.0, 0.7],
        vec![-2.0, -0.2],
        vec![-1.0, 0.4],
        vec![1.0, -0.5],
        vec![2.0, 0.3],
        vec![3.0, -0.1],
        vec![4.0, 0.6],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let selector = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
    let mut pipe = Pipeline::new()
        .push("select", TransformerKind::SelectKBest(selector))
        .with_estimator(LogisticRegression::new().with_solver(LogisticSolver::Svd));

    assert!(!pipe.is_fitted());
    pipe.fit(&x, &y).unwrap();
    assert!(pipe.is_fitted());
    assert_eq!(pipe.predict(&x).unwrap(), y);

    let probabilities = pipe.predict_proba(&x).unwrap();
    assert_eq!(
        (probabilities.nrows(), probabilities.ncols()),
        (x.nrows(), 2)
    );
    for i in 0..x.nrows() {
        assert!(approx(
            probabilities.get(i, 0) + probabilities.get(i, 1),
            1.0,
            1e-12
        ));
    }
}

#[test]
fn target_aware_pipeline_rejects_mismatched_targets() {
    use datarust::selection::{ScoreFunc, SelectKBest};

    let x = datarust::Matrix::new(vec![vec![0.0], vec![1.0]]).unwrap();
    let selector = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
    let mut pipe = Pipeline::new().push("select", TransformerKind::SelectKBest(selector));
    assert!(matches!(
        pipe.fit_transform_with_target(&x, &[0.0]),
        Err(datarust::DatarustError::ShapeMismatch { .. })
    ));
}
