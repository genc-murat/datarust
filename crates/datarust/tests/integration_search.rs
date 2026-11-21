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

#[test]
fn grid_search_tunes_knn_neighbors_and_weights() {
    use datarust::metrics::classification::accuracy_score;
    use datarust::neighbors::KNeighborsClassifier;
    use datarust::traits::{Classifier, Params};

    // Two separable clusters with a noisy borderline pair; larger k wins on
    // this noiseless data.
    let rows: Vec<Vec<f64>> = (0..24)
        .map(|i| {
            let i = i as f64;
            vec![i, (i / 3.0).sin()]
        })
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| (r[0] > 11.0) as i32 as f64).collect();
    let x = Matrix::new(rows).unwrap();

    let mut gs = GridSearchCV::new(
        KNeighborsClassifier::new(),
        vec![
            ("n_neighbors", vec![PV::Int(1), PV::Int(3), PV::Int(5)]),
            ("weights", vec![PV::Int(0), PV::Int(1)]), // Uniform / Distance
        ],
        accuracy_score,
    )
    .with_cv(KFold::new().with_n_splits(3));
    gs.fit(&x, &y).unwrap();

    assert_eq!(gs.cv_results().unwrap().params.len(), 6);
    assert!(gs.best_score().unwrap() > 0.9);

    // Round-trip through set_params mirrors the search's own addressing.
    let mut knn = KNeighborsClassifier::new();
    knn.set_params("n_neighbors", PV::Int(3)).unwrap();
    knn.set_params("weights", PV::Int(1)).unwrap();
    let weights = knn
        .get_params()
        .into_iter()
        .find(|(n, _)| *n == "weights")
        .map(|(_, v)| v);
    assert_eq!(weights, Some(PV::Int(1)));
    knn.fit(&x, &y).unwrap();
    assert!(knn.score(&x, &y).unwrap() > 0.9);
}

type TestFold = (Vec<usize>, Vec<usize>);

#[test]
fn cv_candidates_and_curves_reuse_the_same_split() {
    use datarust::model_selection::{learning_curve, CrossValidator};
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug)]
    struct RotatingSplit(Rc<Cell<usize>>);

    impl CrossValidator for RotatingSplit {
        fn n_splits(&self) -> usize {
            1
        }

        fn split(&self, x: &Matrix, _: &[f64]) -> datarust::Result<Vec<TestFold>> {
            let test = self.0.get();
            self.0.set(test + 1);
            Ok(vec![(
                (0..x.nrows()).filter(|&row| row != test).collect(),
                vec![test],
            )])
        }
    }

    let (x, y) = make_data();
    let scores = |actual: &[f64], _: &[f64]| Ok(actual[0]);
    let calls = Rc::new(Cell::new(0));
    let mut search = GridSearchCV::new(
        Ridge::new(),
        vec![("alpha", vec![PV::Float(0.5), PV::Float(1.0)])],
        scores,
    )
    .with_cv(RotatingSplit(calls.clone()));
    search.fit(&x, &y).unwrap();
    assert_eq!(search.cv_results().unwrap().mean_test_score, vec![y[0]; 2]);
    assert_eq!(calls.replace(0), 1);

    let cv = RotatingSplit(calls.clone());
    let (_, validation) = validation_curve(
        &Ridge::new(),
        &x,
        &y,
        "alpha",
        &[PV::Float(0.5), PV::Float(1.0)],
        &cv,
        scores,
    )
    .unwrap();
    assert_eq!(validation, vec![vec![y[0]]; 2]);
    assert_eq!(calls.replace(0), 1);

    let (_, _, validation) =
        learning_curve(&Ridge::new(), &x, &y, &[0.5, 1.0], &cv, scores).unwrap();
    assert_eq!(validation, vec![vec![y[0]]; 2]);
    assert_eq!(calls.get(), 1);
}

#[test]
fn learning_curve_uses_first_training_fold_for_unequal_sizes() {
    use datarust::model_selection::{learning_curve, CrossValidator};

    #[derive(Debug)]
    struct UnequalFolds;
    impl CrossValidator for UnequalFolds {
        fn n_splits(&self) -> usize {
            2
        }

        fn split(&self, _: &Matrix, _: &[f64]) -> datarust::Result<Vec<TestFold>> {
            Ok(vec![
                ((0..30).collect(), (30..40).collect()),
                ((20..40).collect(), (0..20).collect()),
            ])
        }
    }

    let (x, y) = make_data();
    let (sizes, train, _) =
        learning_curve(&Ridge::new(), &x, &y, &[1.0], &UnequalFolds, |actual, _| {
            Ok(actual.len() as f64)
        })
        .unwrap();
    assert_eq!(sizes, [30]);
    assert_eq!(train, vec![vec![30.0, 20.0]]);
}

#[test]
fn search_drops_previous_model_when_refit_is_disabled() {
    let (x, y) = make_data();
    let cv = KFold::new().with_n_splits(2);
    let mut grid = GridSearchCV::new(
        Ridge::new(),
        vec![("alpha", vec![PV::Float(1.0)])],
        r2_score,
    )
    .with_cv(cv.clone());
    let mut random = RandomizedSearchCV::new(
        Ridge::new(),
        vec![("alpha", ParamDistribution::Choices(vec![PV::Float(1.0)]))],
        r2_score,
    )
    .with_cv(cv);
    grid.fit(&x, &y).unwrap();
    random.fit(&x, &y).unwrap();
    assert!(grid.best_estimator().is_some());
    assert!(random.best_estimator().is_some());

    let mut grid = grid.with_refit(false);
    let mut random = random.with_refit(false);
    grid.fit(&x, &y).unwrap();
    random.fit(&x, &y).unwrap();
    assert!(grid.best_estimator().is_none());
    assert!(random.best_estimator().is_none());
    assert!(grid.predict(&x).is_err());
    assert!(random.predict(&x).is_err());
    assert!(grid.score(&x, &y).is_err());
    assert!(random.score(&x, &y).is_err());
}

#[test]
fn search_scoring_uses_the_configured_metric_through_estimator_traits() {
    use datarust::neighbors::KNeighborsClassifier;
    use datarust::traits::{Classifier, Regressor};

    fn searches<E: Predictor + Params + Clone>(
        model: E,
        name: &'static str,
        value: PV,
        x: &Matrix,
        y: &[f64],
    ) -> (GridSearchCV<E>, RandomizedSearchCV<E>) {
        let metric = |actual: &[f64], predicted: &[f64]| Ok(actual[0] + 10.0 * predicted[0]);
        let mut grid = GridSearchCV::new(model.clone(), vec![(name, vec![value.clone()])], metric);
        let mut random = RandomizedSearchCV::new(
            model,
            vec![(name, ParamDistribution::Choices(vec![value]))],
            metric,
        );
        grid.fit(x, y).unwrap();
        random.fit(x, y).unwrap();
        (grid, random)
    }

    let (x, y) = make_data();
    let query = x.select_rows(&[2]).unwrap();
    let (grid, random) = searches(Ridge::new(), "alpha", PV::Float(1.0), &x, &y);
    let expected = y[2] + 10.0 * grid.best_estimator().unwrap().predict(&query).unwrap()[0];
    assert_eq!(Regressor::score(&grid, &query, &y[2..3]).unwrap(), expected);
    assert_eq!(
        Regressor::score(&random, &query, &y[2..3]).unwrap(),
        expected
    );

    let labels: Vec<f64> = y.iter().map(|v| f64::from(*v < 30.0)).collect();
    let (grid, random) = searches(
        KNeighborsClassifier::new(),
        "n_neighbors",
        PV::Int(1),
        &x,
        &labels,
    );
    let expected = labels[2] + 10.0 * grid.best_estimator().unwrap().predict(&query).unwrap()[0];
    assert_eq!(
        Classifier::score(&grid, &query, &labels[2..3]).unwrap(),
        expected
    );
    assert_eq!(
        Classifier::score(&random, &query, &labels[2..3]).unwrap(),
        expected
    );
}

#[test]
fn grid_uses_sorted_parameter_names_and_preserves_each_value_list_order() {
    let (x, y) = make_data();
    let mut search = GridSearchCV::new(
        Ridge::new(),
        vec![
            ("fit_intercept", vec![PV::Bool(true), PV::Bool(false)]),
            ("alpha", vec![PV::Float(1.0), PV::Float(0.5)]),
        ],
        |_, _| Ok(1.0),
    );
    search.fit(&x, &y).unwrap();
    let expected = vec![
        vec![("alpha", PV::Float(1.0)), ("fit_intercept", PV::Bool(true))],
        vec![
            ("alpha", PV::Float(1.0)),
            ("fit_intercept", PV::Bool(false)),
        ],
        vec![("alpha", PV::Float(0.5)), ("fit_intercept", PV::Bool(true))],
        vec![
            ("alpha", PV::Float(0.5)),
            ("fit_intercept", PV::Bool(false)),
        ],
    ];
    assert_eq!(search.cv_results().unwrap().params, expected);
    assert_eq!(search.best_params(), Some(expected[0].as_slice()));
}

#[test]
fn cv_consumers_evaluate_each_fold_before_requesting_the_next() {
    use datarust::model_selection::cross_validator::FoldIter;
    use datarust::model_selection::{cross_val_score, learning_curve, CrossValidator};
    use datarust::DatarustError;

    #[derive(Debug)]
    struct StreamingFolds;
    impl CrossValidator for StreamingFolds {
        fn n_splits(&self) -> usize {
            2
        }

        fn split(&self, _: &Matrix, _: &[f64]) -> datarust::Result<Vec<TestFold>> {
            panic!("evaluation should use the streaming interface")
        }

        fn split_iter<'a>(&'a self, _: &'a Matrix, _: &'a [f64]) -> datarust::Result<FoldIter<'a>> {
            let first = std::iter::once((vec![0, 1, 2], vec![3, 4]));
            let second = std::iter::once_with(|| panic!("scoring should have stopped on fold one"));
            Ok(Box::new(first.chain(second)))
        }
    }

    fn stop(_: &[f64], _: &[f64]) -> datarust::Result<f64> {
        Err(DatarustError::InvalidInput("stop on fold one".into()))
    }

    fn stopped_by_scorer<T>(result: datarust::Result<T>) {
        assert!(
            matches!(result, Err(DatarustError::InvalidInput(message)) if message == "stop on fold one")
        );
    }

    let (x, y) = make_data();
    let model = Ridge::new();
    let cv = StreamingFolds;
    let values = [PV::Float(0.1), PV::Float(1.0)];
    stopped_by_scorer(cross_val_score(&model, &x, &y, &cv, stop));
    stopped_by_scorer(validation_curve(
        &model, &x, &y, "alpha", &values, &cv, stop,
    ));
    stopped_by_scorer(learning_curve(&model, &x, &y, &[0.5, 1.0], &cv, stop));
    stopped_by_scorer(
        GridSearchCV::new(model, vec![("alpha", values.to_vec())], stop)
            .with_cv(cv)
            .fit(&x, &y),
    );
}

#[test]
fn changing_remainder_requires_fitting_the_categorical_encoders() {
    use datarust::compose::Table;
    use datarust::{DatarustError, StrMatrix};

    let numeric = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let categorical = StrMatrix::new(vec![vec!["one".into()], vec!["two".into()]]).unwrap();
    let table = Table::new(numeric, categorical).unwrap();
    let mut transformer = ColumnTransformer::new().add_numeric(
        "scaled",
        vec![0],
        TransformerKind::StandardScaler(StandardScaler::new()),
    );
    transformer.fit(&table).unwrap();
    assert_eq!(transformer.transform(&table).unwrap().ncols(), 1);
    transformer.set_params("remainder", PV::Int(1)).unwrap();
    assert!(matches!(
        transformer.transform(&table),
        Err(DatarustError::NotFitted(_))
    ));
    transformer.fit(&table).unwrap();
    assert_eq!(transformer.transform(&table).unwrap().ncols(), 3);
}

#[test]
fn failed_full_data_refit_does_not_publish_new_cv_results() {
    use datarust::model_selection::CrossValidator;

    #[derive(Debug)]
    struct LeaveLastRowOut;

    impl CrossValidator for LeaveLastRowOut {
        fn n_splits(&self) -> usize {
            1
        }

        fn split(&self, x: &Matrix, _: &[f64]) -> datarust::Result<Vec<TestFold>> {
            let last = x.nrows() - 1;
            Ok(vec![((0..last - 1).collect(), vec![last - 1])])
        }
    }

    let (x, y) = make_data();
    let scorer = |actual: &[f64], _: &[f64]| Ok(actual[0]);
    let mut grid = GridSearchCV::new(Ridge::new(), vec![("alpha", vec![PV::Float(1.0)])], scorer)
        .with_cv(LeaveLastRowOut);
    let mut random = RandomizedSearchCV::new(
        Ridge::new(),
        vec![("alpha", ParamDistribution::Choices(vec![PV::Float(1.0)]))],
        scorer,
    )
    .with_cv(LeaveLastRowOut);
    grid.fit(&x, &y).unwrap();
    random.fit(&x, &y).unwrap();
    let score = grid.best_score();
    let original = grid.predict(&x).unwrap();

    let mut new_targets: Vec<f64> = y.iter().map(|v| v + 100.0).collect();
    *new_targets.last_mut().unwrap() = f64::NAN;
    assert!(grid.fit(&x, &new_targets).is_err());
    assert!(random.fit(&x, &new_targets).is_err());
    assert_eq!(grid.best_score(), score);
    assert_eq!(random.best_score(), score);
    assert_eq!(grid.predict(&x).unwrap(), original);
    assert_eq!(random.predict(&x).unwrap(), original);
}

#[test]
fn probability_search_uses_stratified_folds_and_same_response_after_refit() {
    use datarust::metrics::classification::roc_auc_score;
    use datarust::model_selection::{
        binary_positive_probability, cross_val_score_with_response, StratifiedKFold,
    };
    use datarust::KNeighborsClassifier;
    let y: Vec<_> = (0..100).map(|i| f64::from(i >= 80)).collect();
    let x = Matrix::new((0..100).map(|i| vec![(i % 7) as f64 + y[i] * 2.]).collect()).unwrap();
    let model = SupervisedPipeline::new(KNeighborsClassifier::new().with_n_neighbors(9));
    let expected = cross_val_score_with_response(
        &model,
        &x,
        &y,
        &StratifiedKFold::new(),
        roc_auc_score,
        binary_positive_probability,
    )
    .unwrap();
    let mut grid = GridSearchCV::new(model.clone(), vec![], roc_auc_score)
        .with_response(binary_positive_probability);
    let mut random = RandomizedSearchCV::new(
        model,
        vec![(
            "estimator__n_neighbors",
            ParamDistribution::Choices(vec![PV::Int(9)]),
        )],
        roc_auc_score,
    )
    .with_response(binary_positive_probability);
    grid.fit(&x, &y).unwrap();
    random.fit(&x, &y).unwrap();
    assert_eq!(grid.cv_results().unwrap().split_scores[0], expected);
    assert_eq!(random.cv_results().unwrap().split_scores[0], expected);
    let full = roc_auc_score(
        &y,
        &binary_positive_probability(grid.best_estimator().unwrap(), &x).unwrap(),
    )
    .unwrap();
    assert_ne!(full, roc_auc_score(&y, &grid.predict(&x).unwrap()).unwrap());
    assert_eq!(grid.score(&x, &y).unwrap(), full);
    assert_eq!(random.score(&x, &y).unwrap(), full);
}

#[test]
fn singleton_r2_cannot_rank_candidates() {
    let x = Matrix::from_flat(5, 1, vec![0., 1., 2., 3., 4.]).unwrap();
    let mut grid = GridSearchCV::new(Ridge::new(), vec![], r2_score);
    assert!(grid.fit(&x, &[2., 5., 8., 11., 14.]).is_err());
    assert!(grid.best_index().is_none());
    assert!(r2_score(&[1.], &[1.]).is_err());
}
