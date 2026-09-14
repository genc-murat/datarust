use datarust::ensemble::{RandomForestClassifier, RandomForestRegressor};
use datarust::traits::{PredictProba, Predictor};
use datarust::{Matrix, MaxFeatures, ParamValue, Params};

fn fixture() -> Matrix {
    Matrix::new(vec![
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 0.0, 1.0],
        vec![2.0, 1.0, 0.0],
        vec![3.0, 0.0, 0.0],
        vec![3.0, 1.0, 1.0],
    ])
    .unwrap()
}

#[test]
fn classifier_matches_sklearn_reference_predictions_probabilities_and_importances() {
    let x = fixture();
    let y = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let mut model = RandomForestClassifier::new()
        .with_n_estimators(5)
        .with_max_depth(Some(3))
        .with_max_features(MaxFeatures::Count(2))
        .with_random_state(13);
    model.fit(&x, &y).unwrap();

    assert_eq!(model.predict(&x).unwrap(), y);
    assert_eq!(
        model.predict_proba(&x).unwrap().as_slice(),
        &[0.8, 0.2, 0.2, 0.8, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.2, 0.8, 0.4, 0.6, 1.0, 0.0,]
    );
    let expected_importances: [f64; 3] =
        [0.13142857142857145, 0.04571428571428573, 0.8228571428571428];
    for (actual, expected) in model
        .feature_importances()
        .unwrap()
        .iter()
        .zip(expected_importances)
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn regressor_matches_sklearn_reference_predictions_importances_and_bootstrap_indices() {
    let x = fixture();
    let y = vec![0.5, 2.0, 1.0, 2.5, 1.5, 3.0, 2.0, 3.5];
    let mut model = RandomForestRegressor::new()
        .with_n_estimators(5)
        .with_max_depth(Some(3))
        .with_max_features(MaxFeatures::Count(2))
        .with_random_state(13);
    model.fit(&x, &y).unwrap();

    let expected_predictions: [f64; 8] = [0.9, 2.2, 1.0, 2.4, 1.4, 2.8, 1.5, 3.5];
    for (actual, expected) in model.predict(&x).unwrap().iter().zip(expected_predictions) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let expected_importances: [f64; 3] =
        [0.24508660036871355, 0.6780638747828597, 0.07684952484842672];
    for (actual, expected) in model
        .feature_importances()
        .unwrap()
        .iter()
        .zip(expected_importances)
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(
        model.estimators_samples(),
        vec![
            vec![7, 4, 4, 3, 1, 2, 7, 6],
            vec![3, 6, 3, 2, 1, 7, 5, 1],
            vec![3, 5, 7, 2, 2, 4, 5, 0],
            vec![0, 1, 7, 5, 3, 0, 4, 2],
            vec![1, 7, 2, 4, 5, 4, 5, 1],
        ]
    );
}

#[test]
fn bootstrap_false_uses_each_training_row_once_and_params_round_trip() {
    let x = fixture();
    let y = vec![0.5, 2.0, 1.0, 2.5, 1.5, 3.0, 2.0, 3.5];
    let mut model = RandomForestRegressor::new()
        .with_n_estimators(3)
        .with_bootstrap(false)
        .with_max_features(MaxFeatures::Count(2));
    model.fit(&x, &y).unwrap();
    assert!(model
        .estimators_samples()
        .iter()
        .all(|samples| samples == &(0..x.nrows()).collect::<Vec<_>>()));
    for params in model.get_params() {
        model.set_params(params.0, params.1).unwrap();
    }
    // Params encode max_features as 0=all, 1=sqrt, 2=log2, k+2=Count(k);
    // Int(4) therefore round-trips Count(2) set above.
    assert!(model
        .get_params()
        .contains(&("max_features", ParamValue::Int(4))));
}

#[test]
fn classifier_accepts_one_class_and_returns_one_probability_column() {
    let x = fixture();
    let y = vec![7.0; 8];
    let mut model = RandomForestClassifier::new()
        .with_n_estimators(3)
        .with_random_state(5);
    model.fit(&x, &y).unwrap();
    assert_eq!(model.classes(), Some(&[7.0][..]));
    let proba = model.predict_proba(&x).unwrap();
    assert_eq!(proba.ncols(), 1);
    assert!(proba.as_slice().iter().all(|&value| value == 1.0));
    assert_eq!(model.predict(&x).unwrap(), y);
}

#[test]
fn classifier_is_deterministic_and_preserves_external_class_labels() {
    let x = fixture();
    let y = vec![10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0];
    let build = || {
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(7)
            .with_max_features(MaxFeatures::Count(1))
            .with_random_state(99);
        model.fit(&x, &y).unwrap();
        model
    };
    let a = build();
    let b = build();
    assert_eq!(a.classes(), Some(&[10.0, 20.0][..]));
    assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
    assert_eq!(a.predict_proba(&x).unwrap(), b.predict_proba(&x).unwrap());
}

#[test]
fn classifier_rejects_regression_criterion_at_fit() {
    use datarust::tree::Criterion;
    use datarust::DatarustError;

    // Regression: the forest used to fit "successfully" and only fail on
    // the first predict. `fit` must reject the regression-only criterion
    // before any tree is built.
    let x = fixture();
    let y = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let mut model = RandomForestClassifier::new()
        .with_n_estimators(3)
        .with_criterion(Criterion::SquaredError);
    let err = model.fit(&x, &y).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)), "got {err:?}");
    assert_eq!(model.n_estimators(), 0);
}
