//! Hyperparameter search integration: nested `step__param` addressing through
//! pipelines, plus grid / randomized search and CV curves on composed models.

use datarust::compose::{ColumnTransformer, Remainder};
use datarust::linear_model::Ridge;
use datarust::matrix::Matrix;
use datarust::metrics::regression::r2_score;
use datarust::model_selection::{
    validation_curve, GridSearchCV, KFold, ParamDistribution, RandomizedSearchCV,
};
use datarust::scaler::StandardScaler;
use datarust::traits::{ParamValue as PV, Params, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::SupervisedPipeline;

/// Deterministic two-feature regression data: `y = 2*x0 + x1`.
fn make_data() -> (Matrix, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..40)
        .map(|i| {
            let i = i as f64;
            vec![i, 10.0 - i / 4.0]
        })
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + r[1]).collect();
    (Matrix::new(rows).unwrap(), y)
}

fn make_pipeline() -> SupervisedPipeline<Ridge> {
    SupervisedPipeline::new(Ridge::new()).push(
        "scaler",
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
}

#[test]
fn supervised_pipeline_exposes_nested_params() {
    let pipeline = make_pipeline();

    let params = pipeline.get_params();
    assert!(params
        .iter()
        .any(|(n, v)| *n == "estimator__alpha" && *v == PV::Float(1.0)));
    assert!(params
        .iter()
        .any(|(n, v)| *n == "scaler__with_mean" && *v == PV::Bool(true)));

    let mut pipeline = pipeline;
    pipeline
        .set_params("estimator__alpha", PV::Float(0.5))
        .unwrap();
    pipeline
        .set_params("scaler__with_mean", PV::Bool(false))
        .unwrap();
    let params = pipeline.get_params();
    assert!(params
        .iter()
        .any(|(n, v)| *n == "estimator__alpha" && *v == PV::Float(0.5)));
    assert!(params
        .iter()
        .any(|(n, v)| *n == "scaler__with_mean" && *v == PV::Bool(false)));

    assert!(pipeline
        .set_params("estimator__nope", PV::Float(1.0))
        .is_err());
    assert!(pipeline.set_params("scaler__nope", PV::Bool(true)).is_err());
}

#[test]
fn grid_search_over_pipeline_grid() {
    let (x, y) = make_data();
    let mut gs = GridSearchCV::new(
        make_pipeline(),
        vec![
            (
                "estimator__alpha",
                vec![PV::Float(10.0), PV::Float(1.0), PV::Float(0.1)],
            ),
            ("scaler__with_mean", vec![PV::Bool(true), PV::Bool(false)]),
        ],
        r2_score,
    )
    .with_cv(KFold::new().with_n_splits(3));

    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();
    assert_eq!(results.params.len(), 6);
    assert_eq!(results.mean_test_score.len(), 6);
    assert_eq!(results.rank_test_score.len(), 6);
    // 3 folds per candidate.
    assert!(results.split_scores.iter().all(|f| f.len() == 3));

    // Least regularization wins on this noiseless linear target.
    let best = gs.best_params().unwrap();
    assert!(matches!(
        best.iter().find(|(n, _)| *n == "estimator__alpha"),
        Some((_, PV::Float(a))) if *a == 0.1
    ));
    assert!(gs.best_score().unwrap() > 0.99);

    // The refitted best pipeline predicts on new data (small bias remains
    // because alpha=0.1 shrinks the coefficients off the exact OLS solution).
    let best_pipeline = gs.best_estimator().unwrap();
    let x_new = Matrix::new(vec![vec![41.0, 10.0 - 41.0 / 4.0]]).unwrap();
    let pred = best_pipeline.predict(&x_new).unwrap();
    let truth = 2.0 * 41.0 + (10.0 - 41.0 / 4.0);
    assert!(
        (pred[0] - truth).abs() / truth.abs() < 0.05,
        "pred {pred:?} vs truth {truth}"
    );
}

#[test]
fn randomized_search_over_pipeline_choices() {
    let (x, y) = make_data();
    // Choices-only space: product = 3 alphas x 2 with_mean = 6, sampled
    // without replacement, so n_iter=6 covers every grid point.
    let mut rs = RandomizedSearchCV::new(
        make_pipeline(),
        vec![
            (
                "estimator__alpha",
                ParamDistribution::Choices(vec![PV::Float(0.1), PV::Float(1.0), PV::Float(10.0)]),
            ),
            (
                "scaler__with_mean",
                ParamDistribution::Choices(vec![PV::Bool(true), PV::Bool(false)]),
            ),
        ],
        r2_score,
    )
    .with_n_iter(6)
    .with_random_state(9)
    .with_cv(KFold::new().with_n_splits(3));

    rs.fit(&x, &y).unwrap();

    let params = rs.cv_results().unwrap().params.clone();
    assert_eq!(params.len(), 6);
    let mut keys: Vec<String> = params.iter().map(|c| format!("{c:?}")).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 6, "choices-only space has no duplicates");
    assert!(rs.best_score().unwrap() > 0.99);
}

#[test]
fn validation_curve_over_nested_estimator_param() {
    let (x, y) = make_data();
    let (train, test) = validation_curve(
        &make_pipeline(),
        &x,
        &y,
        "estimator__alpha",
        &[PV::Float(0.01), PV::Float(1.0), PV::Float(100.0)],
        &KFold::new().with_n_splits(3),
        r2_score,
    )
    .unwrap();

    assert_eq!(train.len(), 3);
    assert_eq!(test.len(), 3);
    assert!(train.iter().chain(&test).all(|f| f.len() == 3));
    // Under-regularized fits win on a noiseless linear target.
    assert!(test[0].iter().sum::<f64>() > test[2].iter().sum::<f64>());
}

#[test]
fn column_transformer_params_round_trip() {
    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Passthrough)
        .add_numeric(
            "num",
            vec![0, 1],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );

    let params = ct.get_params();
    assert!(params
        .iter()
        .any(|(n, v)| *n == "num__with_mean" && *v == PV::Bool(true)));

    ct.set_params("num__with_mean", PV::Bool(false)).unwrap();
    ct.set_params("remainder", PV::Int(0)).unwrap();
    let params = ct.get_params();
    assert!(params
        .iter()
        .any(|(n, v)| *n == "num__with_mean" && *v == PV::Bool(false)));
    assert!(params
        .iter()
        .any(|(n, v)| *n == "remainder" && *v == PV::Int(0)));

    assert!(ct.set_params("nope__with_mean", PV::Bool(true)).is_err());
    assert!(ct.set_params("remainder", PV::Int(2)).is_err());
}
