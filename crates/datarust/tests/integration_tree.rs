//! Integration tests for `tree::{DecisionTreeClassifier, DecisionTreeRegressor}`
//! and the shared `rng` module.

use datarust::model_selection::rng::Rng;
use datarust::traits::{Params, PredictProba, Predictor};
use datarust::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};
use datarust::{DatarustError, Matrix};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// XOR data: XOR with one feature is a classic tree problem where linear
/// models fail; an unrestricted tree must classify it perfectly.
fn xor_data() -> (Matrix, Vec<f64>) {
    let x = Matrix::new(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ])
    .unwrap();
    let y = vec![0.0, 1.0, 1.0, 0.0];
    (x, y)
}

#[test]
fn classifier_fits_xor_perfectly() {
    let (x, y) = xor_data();
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    assert!(clf.is_fitted());
    let pred = clf.predict(&x).unwrap();
    assert_eq!(pred, y, "unrestricted tree must fit XOR exactly");
}

#[test]
fn max_depth_one_fails_xor() {
    let (x, y) = xor_data();
    let mut clf = DecisionTreeClassifier::new().with_max_depth(Some(1));
    clf.fit(&x, &y).unwrap();
    let pred = clf.predict(&x).unwrap();
    let n_correct = pred.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    assert!(
        n_correct < y.len(),
        "depth-1 tree cannot solve XOR (got {n_correct}/4)"
    );
}

#[test]
fn regressor_recovers_group_means() {
    // Two well-separated groups; a depth-1 tree splits between them and each
    // leaf predicts the group mean.
    let x = Matrix::new(vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![10.0],
        vec![11.0],
        vec![12.0],
    ])
    .unwrap();
    let y = vec![1.0, 2.0, 3.0, 20.0, 21.0, 22.0];

    let mut reg = DecisionTreeRegressor::new().with_max_depth(Some(1));
    reg.fit(&x, &y).unwrap();
    let pred = reg.predict(&x).unwrap();

    for p in pred.iter().take(3) {
        assert!(approx(*p, 2.0, 1e-12), "left leaf mean, got {p}");
    }
    for p in pred.iter().skip(3) {
        assert!(approx(*p, 21.0, 1e-12), "right leaf mean, got {p}");
    }
}

#[test]
fn regressor_fit_predict_agrees_with_predict() {
    // The tree interpolates a single-feature step function exactly (pure
    // leaves), and the general trait contract holds: predict returns one
    // value per row and `fit_predict` agrees with `predict`.
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![9.0], vec![10.0]]).unwrap();
    let y = vec![1.0, 1.0, 10.0, 10.0];

    let mut tree_reg = DecisionTreeRegressor::new();
    tree_reg.fit(&x, &y).unwrap();
    let tree_pred = tree_reg.predict(&x).unwrap();
    assert_eq!(tree_pred, y, "pure leaves reproduce training targets");

    assert_eq!(tree_pred.len(), x.nrows());
    let fp = tree_reg.fit_predict(&x, &y).unwrap();
    assert_eq!(fp, tree_pred);
}

#[test]
fn predict_proba_rows_sum_to_one() {
    let x = Matrix::new(vec![
        vec![1.0, 1.0],
        vec![1.2, 1.1],
        vec![5.0, 5.0],
        vec![5.2, 4.9],
        vec![9.0, 9.0],
        vec![8.8, 9.1],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    let proba = clf.predict_proba(&x).unwrap();

    assert_eq!(proba.nrows(), 6);
    assert_eq!(proba.ncols(), 3);
    for i in 0..proba.nrows() {
        let row_sum: f64 = (0..proba.ncols()).map(|k| proba.get(i, k)).sum();
        assert!(approx(row_sum, 1.0, 1e-12), "row {i} sums to {row_sum}");
        let argmax = (0..proba.ncols())
            .max_by(|a, b| proba.get(i, *a).total_cmp(&proba.get(i, *b)))
            .unwrap();
        assert_eq!(
            proba.get(i, argmax),
            1.0,
            "pure leaves give argmax 1.0 (row {i})"
        );
    }
}

#[test]
fn proba_aligns_with_predict() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
    let y = vec![1.0, 1.0, 1.0, 0.0];
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    let pred = clf.predict(&x).unwrap();
    let proba = clf.predict_proba(&x).unwrap();
    let classes = clf.tree().unwrap().classes().unwrap().to_vec();
    for (i, p) in pred.iter().enumerate() {
        let argmax = (0..classes.len())
            .max_by(|a, b| proba.get(i, *a).total_cmp(&proba.get(i, *b)))
            .unwrap();
        assert_eq!(classes[argmax], *p);
    }
}

#[test]
fn feature_importances_sum_to_one_when_splittable() {
    let x = Matrix::new(vec![
        vec![1.0, 5.0],
        vec![2.0, 5.0],
        vec![3.0, 5.0],
        vec![10.0, 5.0],
        vec![11.0, 5.0],
        vec![12.0, 5.0],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    let imp = clf.feature_importances().unwrap();

    let total: f64 = imp.iter().sum();
    assert!(approx(total, 1.0, 1e-9), "importances sum to {total}");
    assert_eq!(imp.len(), x.ncols());
    assert!(approx(imp[0], 1.0, 1e-9), "only feature 0 is informative");
    assert_eq!(imp[1], 0.0, "feature 1 never used");
}

#[test]
fn min_samples_leaf_enforced() {
    // 12 samples; without constraints the tree grows deep. With
    // min_samples_leaf=4 every leaf must hold ≥ 4 samples — verified through
    // the public `leaf_for` routing of every training row.
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..12 {
        rows.push(vec![i as f64]);
        y.push(if i < 6 { 0.0 } else { 1.0 });
    }
    let x = Matrix::new(rows).unwrap();

    let mut clf = DecisionTreeClassifier::new().with_min_samples_leaf(4);
    clf.fit(&x, &y).unwrap();
    let tree = clf.tree().unwrap();

    let mut per_leaf: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..x.nrows() {
        *per_leaf
            .entry(tree.leaf_for(x.row(i)).unwrap())
            .or_insert(0) += 1;
    }
    assert!(!per_leaf.is_empty());
    for (&leaf, &n) in &per_leaf {
        assert!(
            n >= 4,
            "leaf {leaf} has only {n} samples (< min_samples_leaf)"
        );
    }
    assert!(
        tree.n_leaves() <= 3,
        "12 samples with min_samples_leaf=4 allow at most 3 leaves"
    );
}

#[test]
fn min_samples_leaf_can_block_all_splits() {
    // 12 samples with min_samples_leaf=7: any split would leave one side
    // with < 7 samples, so the tree must be a single leaf.
    let rows: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
    let x = Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..12).map(|i| (i % 2) as f64).collect();

    let mut clf = DecisionTreeClassifier::new().with_min_samples_leaf(7);
    clf.fit(&x, &y).unwrap();
    let tree = clf.tree().unwrap();
    assert_eq!(tree.n_nodes(), 1, "no valid split → single root leaf");
    assert_eq!(tree.n_leaves(), 1);
    assert_eq!(tree.get_depth(), 0);

    // Regression behaves identically.
    let mut reg = DecisionTreeRegressor::new().with_min_samples_leaf(7);
    reg.fit(&x, &y).unwrap();
    assert_eq!(reg.tree().unwrap().n_nodes(), 1);
}

#[test]
fn sqrt_feature_subsampling_still_fits_noise_heavy_data() {
    // 10 features but only feature 9 separates the classes. With sqrt(10) = 3
    // features considered per split and a fixed seed, the tree may not find
    // the split on the first try — this is a fit-quality check on a
    // deterministic configuration, not a proof of the subset itself.
    let mut rng = Rng::new(42);
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..60 {
        let label = (i % 2) as f64;
        let mut row: Vec<f64> = (0..9).map(|_| rng.next_unit() * 100.0).collect();
        row.push(label * 10.0);
        rows.push(row);
        y.push(label);
    }
    let x = Matrix::new(rows).unwrap();

    let mut clf = DecisionTreeClassifier::new()
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(7);
    clf.fit(&x, &y).unwrap();
    assert!(clf.is_fitted());
    assert_eq!(clf.feature_importances().unwrap().len(), 10);
    let pred = clf.predict(&x).unwrap();
    let n_correct = pred.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    assert!(n_correct >= 54, "expected good fit, got {n_correct}/60");
}

#[test]
fn constant_feature_column_never_splits_and_tree_still_fits() {
    let x = Matrix::new(vec![
        vec![5.0, 0.0],
        vec![5.0, 1.0],
        vec![5.0, 2.0],
        vec![5.0, 3.0],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    let tree = clf.tree().unwrap();
    for n in 0..tree.n_nodes() {
        assert_ne!(tree.node_feature(n), Some(0), "split on constant column");
    }
    assert_eq!(clf.predict(&x).unwrap(), y);
}

#[test]
fn single_sample_fits_to_a_single_leaf() {
    let x = Matrix::new(vec![vec![1.0, 2.0]]).unwrap();
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &[3.0]).unwrap();
    assert_eq!(clf.tree().unwrap().n_nodes(), 1);
    assert_eq!(clf.predict(&x).unwrap(), vec![3.0]);
    assert_eq!(clf.predict_proba(&x).unwrap().as_slice(), &[1.0]);

    let mut reg = DecisionTreeRegressor::new();
    reg.fit(&x, &[7.5]).unwrap();
    assert_eq!(reg.tree().unwrap().n_nodes(), 1);
    assert_eq!(reg.predict(&x).unwrap(), vec![7.5]);
}

#[test]
fn all_same_labels_fit_to_a_single_leaf() {
    let x = Matrix::new(vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]]).unwrap();
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &[4.0, 4.0, 4.0]).unwrap();
    assert_eq!(clf.tree().unwrap().n_nodes(), 1);
    assert_eq!(clf.predict(&x).unwrap(), vec![4.0, 4.0, 4.0]);
}

#[test]
fn min_impurity_decrease_can_suppress_all_splits() {
    let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let mut clf = DecisionTreeClassifier::new().with_min_impurity_decrease(2.0);
    clf.fit(&x, &y).unwrap();
    assert_eq!(
        clf.tree().unwrap().n_nodes(),
        1,
        "no split improves gini by 2.0"
    );
    assert_eq!(clf.predict(&x).unwrap(), vec![0.0; 4]);
}

#[test]
fn log2_feature_strategy_fits() {
    let x = Matrix::new(vec![
        vec![0.0, 0.0, 3.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 3.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 0.0, 3.0],
        vec![2.0, 1.0, 1.0],
        vec![3.0, 0.0, 3.0],
        vec![3.0, 1.0, 1.0],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let mut clf = DecisionTreeClassifier::new()
        .with_max_features(MaxFeatures::Log2)
        .with_random_state(3);
    clf.fit(&x, &y).unwrap();
    assert_eq!(clf.predict(&x).unwrap(), y);
}

#[test]
fn non_finite_y_is_rejected() {
    let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0]]).unwrap();

    let mut clf = DecisionTreeClassifier::new();
    let err = clf.fit(&x, &[0.0, f64::NAN, 1.0]).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)), "got {err:?}");

    let mut reg = DecisionTreeRegressor::new();
    let err = reg.fit(&x, &[0.0, f64::INFINITY, 1.0]).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)), "got {err:?}");
}

#[test]
fn reproducible_with_random_state() {
    let mut rng = Rng::new(1234);
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..80 {
        let label = (i % 2) as f64;
        let mut row: Vec<f64> = (0..6).map(|_| rng.next_unit()).collect();
        row.push(label);
        rows.push(row);
        y.push(label);
    }
    let x = Matrix::new(rows).unwrap();

    let build = || {
        DecisionTreeClassifier::new()
            .with_max_features(MaxFeatures::Sqrt)
            .with_random_state(99)
    };

    let mut a = build();
    a.fit(&x, &y).unwrap();
    let mut b = build();
    b.fit(&x, &y).unwrap();

    let ta = a.tree().unwrap();
    let tb = b.tree().unwrap();
    assert_eq!(ta.n_nodes(), tb.n_nodes());
    assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
}

#[test]
fn params_round_trip_and_unknown_param() {
    let clf = DecisionTreeClassifier::new().with_max_depth(Some(3));
    let params = clf.get_params();
    assert!(params.iter().any(|(k, _)| *k == "max_depth"));
    assert!(params.iter().any(|(k, _)| *k == "criterion"));
    assert!(params.iter().any(|(k, _)| *k == "max_features"));

    // Round-trip through set_params.
    let mut clone = DecisionTreeClassifier::new();
    for (k, v) in params {
        clone.set_params(k, v).unwrap();
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..8 {
        x.push(vec![i as f64]);
        y.push((i % 2) as f64);
    }
    let x = Matrix::new(x).unwrap();
    let mut a = clf;
    a.fit(&x, &y).unwrap();
    let mut b = clone;
    b.fit(&x, &y).unwrap();
    assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());

    // Unknown parameter names are rejected.
    let mut c = DecisionTreeClassifier::new();
    let err = c
        .set_params("bogus", datarust::traits::ParamValue::Int(1))
        .unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)));

    let mut r = DecisionTreeRegressor::new();
    let err = r
        .set_params("bogus", datarust::traits::ParamValue::Int(1))
        .unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)));
}

#[test]
fn regressor_params_preserve_max_features() {
    let mut original = DecisionTreeRegressor::new()
        .with_max_features(MaxFeatures::Count(1))
        .with_random_state(17);
    let params = original.get_params();

    let mut restored = DecisionTreeRegressor::new();
    for (name, value) in params {
        restored.set_params(name, value).unwrap();
    }

    let x = Matrix::new(vec![
        vec![0.0, 10.0],
        vec![1.0, 11.0],
        vec![2.0, 12.0],
        vec![9.0, 3.0],
        vec![10.0, 4.0],
        vec![11.0, 5.0],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    original.fit(&x, &y).unwrap();
    restored.fit(&x, &y).unwrap();
    assert_eq!(original.predict(&x).unwrap(), restored.predict(&x).unwrap());
}

#[test]
fn unfitted_predict_errors() {
    let x = Matrix::new(vec![vec![1.0]]).unwrap();
    let clf = DecisionTreeClassifier::new();
    let err = clf.predict(&x).unwrap_err();
    assert!(matches!(err, DatarustError::NotFitted(_)));

    let reg = DecisionTreeRegressor::new();
    let err = reg.predict(&x).unwrap_err();
    assert!(matches!(err, DatarustError::NotFitted(_)));

    let err = clf.feature_importances().unwrap_err();
    assert!(matches!(err, DatarustError::NotFitted(_)));
}

#[test]
fn classifier_rejects_regression_criterion_at_fit() {
    // Regression: `with_criterion(Criterion::SquaredError)` used to fit a
    // regression tree that only failed later, per row, at predict time.
    let x = Matrix::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let y = vec![0.0, 0.0, 1.0, 1.0];
    let mut clf = DecisionTreeClassifier::new().with_criterion(Criterion::SquaredError);
    let err = clf.fit(&x, &y).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)), "got {err:?}");
    assert!(!clf.is_fitted());

    // The invalid criterion round-trips through the params API.
    let params = clf.get_params();
    let mut restored = DecisionTreeClassifier::new();
    for (name, value) in params {
        restored.set_params(name, value).unwrap();
    }
    let err = restored.fit(&x, &y).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)), "got {err:?}");
}

#[test]
fn fit_rejects_mismatched_lengths() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let mut clf = DecisionTreeClassifier::new();
    let err = clf.fit(&x, &[0.0, 1.0]).unwrap_err();
    assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
}

#[test]
fn fit_rejects_non_finite_input() {
    let x = Matrix::new(vec![vec![1.0], vec![f64::NAN]]).unwrap();
    let mut clf = DecisionTreeClassifier::new();
    let err = clf.fit(&x, &[0.0, 1.0]).unwrap_err();
    assert!(matches!(err, DatarustError::InvalidInput(_)));
}

#[test]
fn rng_promotion_is_shared_and_deterministic() {
    // The promoted `datarust::rng::Rng` and the legacy
    // `model_selection::rng::Rng` are the same type and produce identical
    // sequences for a given seed.
    let mut a = datarust::rng::Rng::new(0xDEADBEEF);
    let mut b = Rng::new(0xDEADBEEF);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.next_unit(), b.next_unit());
    }
    // And it's stateful: consecutive draws differ.
    let mut c = Rng::new(0xDEADBEEF);
    let first = c.next_u64();
    let second = c.next_u64();
    assert_ne!(first, second);
}

#[test]
fn supports_more_than_255_classes_without_label_collisions() {
    let rows: Vec<Vec<f64>> = (0..300).map(|i| vec![i as f64]).collect();
    let x = Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..300).map(|i| i as f64).collect();

    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();
    assert_eq!(clf.tree().unwrap().classes().unwrap().len(), 300);
    assert_eq!(clf.predict(&x).unwrap(), y);

    let proba = clf.predict_proba(&x).unwrap();
    assert_eq!(proba.ncols(), 300);
    for row in 0..x.nrows() {
        assert_eq!(proba.get(row, row), 1.0);
    }
}

#[test]
fn predict_rejects_wrong_shape_and_non_finite_values() {
    let x = Matrix::new(vec![vec![0.0], vec![1.0]]).unwrap();
    let y = vec![0.0, 1.0];
    let mut clf = DecisionTreeClassifier::new();
    clf.fit(&x, &y).unwrap();

    let wrong_shape = Matrix::new(vec![vec![0.0, 1.0]]).unwrap();
    assert!(matches!(
        clf.predict(&wrong_shape),
        Err(DatarustError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        clf.predict_proba(&wrong_shape),
        Err(DatarustError::ShapeMismatch { .. })
    ));

    let nan = Matrix::new(vec![vec![f64::NAN]]).unwrap();
    let infinity = Matrix::new(vec![vec![f64::INFINITY]]).unwrap();
    assert!(matches!(
        clf.predict(&nan),
        Err(DatarustError::InvalidInput(_))
    ));
    assert!(matches!(
        clf.predict_proba(&infinity),
        Err(DatarustError::InvalidInput(_))
    ));

    let mut reg = DecisionTreeRegressor::new();
    reg.fit(&x, &y).unwrap();
    assert!(matches!(
        reg.predict(&wrong_shape),
        Err(DatarustError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        reg.predict(&nan),
        Err(DatarustError::InvalidInput(_))
    ));
}
