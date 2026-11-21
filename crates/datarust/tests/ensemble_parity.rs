//! Bit-exact parity tests for the ensemble module against scikit-learn
//! 1.6.1 / numpy 2.0.2 fixtures (see `ensemble_fixtures.rs`).

#[path = "ensemble_fixtures.rs"]
mod ensemble_fixtures;

use datarust::ensemble::{ExtraTreesClassifier, ExtraTreesRegressor};
use datarust::traits::{Params, PredictProba, Predictor};
use datarust::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};
use datarust::Matrix;

use ensemble_fixtures::{EtClfFixture, EtTreeFixture, ET_CLF_FIXTURES, ET_REG_FIXTURES};

fn assert_bits(actual: f64, expected: f64, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: got {actual}, want {expected}"
    );
}

fn check_slice(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_bits(a, e, &format!("{ctx}[{i}]"));
    }
}

fn parse_criterion(name: &str) -> Criterion {
    match name {
        "gini" => Criterion::Gini,
        "entropy" => Criterion::Entropy,
        other => panic!("unknown criterion {other}"),
    }
}

/// Fixture `max_features` strings: `"sqrt"`, `"all"`, an integer count
/// (`"2"`), or a float fraction (`"1.0"`, `"0.7"` → `int(frac * n)`).
fn parse_max_features(value: &str, n_features: usize) -> MaxFeatures {
    match value {
        "sqrt" => MaxFeatures::Sqrt,
        "all" => MaxFeatures::All,
        other => {
            let parsed: f64 = other.parse().expect("numeric max_features");
            if other.contains('.') {
                MaxFeatures::Count(((parsed * n_features as f64) as usize).max(1))
            } else {
                MaxFeatures::Count(parsed as usize)
            }
        }
    }
}

fn check_tree_structure(fitted: &datarust::tree::TreeStructure, t: &EtTreeFixture, ctx: &str) {
    assert_eq!(fitted.n_nodes(), t.features.len(), "{ctx}: node count");
    for i in 0..t.features.len() {
        let node_ctx = format!("{ctx} node {i}");
        if t.features[i] == -2 {
            assert!(
                fitted.node_feature(i).is_none(),
                "{node_ctx}: expected leaf"
            );
            assert!(
                fitted.node_threshold(i).is_none(),
                "{node_ctx}: leaf threshold"
            );
        } else {
            assert_eq!(
                fitted.node_feature(i),
                Some(t.features[i] as usize),
                "{node_ctx}: feature"
            );
            assert_bits(
                fitted.node_threshold(i).expect("internal threshold"),
                t.thresholds[i],
                &format!("{node_ctx} threshold"),
            );
        }
        assert_bits(
            fitted.node_impurity(i).expect("impurity"),
            t.impurities[i],
            &format!("{node_ctx} impurity"),
        );
        assert_eq!(
            fitted.node_n_samples(i).expect("n_node"),
            t.n_node[i],
            "{node_ctx}: n_node_samples"
        );
    }
}

fn check_clf_tree(tree: &DecisionTreeClassifier, t: &EtTreeFixture, n_classes: usize, ctx: &str) {
    let fitted = tree.tree().expect("fitted tree");
    check_tree_structure(fitted, t, ctx);
    for i in 0..t.features.len() {
        let probs = fitted
            .node_class_probabilities(i)
            .expect("classification node value");
        assert_eq!(probs.len(), n_classes, "{ctx} node {i}: value width");
        for (c, &p) in probs.iter().enumerate() {
            assert_bits(
                p,
                t.values[i * n_classes + c],
                &format!("{ctx} node {i} class {c}"),
            );
        }
    }
}

fn check_reg_tree(tree: &DecisionTreeRegressor, t: &EtTreeFixture, ctx: &str) {
    let fitted = tree.tree().expect("fitted tree");
    check_tree_structure(fitted, t, ctx);
    for (i, &expected) in t.values.iter().enumerate().take(t.features.len()) {
        assert_bits(
            fitted
                .node_regression_value(i)
                .expect("regression node value"),
            expected,
            &format!("{ctx} node {i} value"),
        );
    }
}

fn check_samples(actual: &[Vec<usize>], expected: &[&[usize]], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: estimator count");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.as_slice(), *e, "{ctx}: est_samples[{i}]");
    }
}

fn fit_et_clf(f: &EtClfFixture) -> (ExtraTreesClassifier, Matrix) {
    let x = Matrix::from_flat(f.rows, f.cols, f.x.to_vec()).expect("fixture X");
    let mut model = ExtraTreesClassifier::new()
        .with_n_estimators(f.n_estimators)
        .with_criterion(parse_criterion(f.criterion))
        .with_max_depth((f.max_depth > 0).then_some(f.max_depth))
        .with_max_features(parse_max_features(f.max_features, f.cols))
        .with_bootstrap(f.bootstrap)
        .with_min_samples_leaf(f.min_samples_leaf)
        .with_random_state(f.seed);
    model.fit(&x, f.y).expect("fit");
    (model, x)
}

#[test]
fn extra_trees_classifier_parity() {
    for f in ET_CLF_FIXTURES {
        let (model, x) = fit_et_clf(f);
        assert_eq!(
            model.n_estimators(),
            f.n_estimators,
            "{}: n_estimators",
            f.name
        );
        for (i, (tree, t)) in model.trees().iter().zip(f.trees).enumerate() {
            check_clf_tree(tree, t, f.n_classes, &format!("{} tree {i}", f.name));
        }
        if !f.est_samples.is_empty() {
            check_samples(&model.estimators_samples(), f.est_samples, f.name);
        }
        check_slice(
            &model.predict(&x).expect("predict"),
            f.preds,
            &format!("{} preds", f.name),
        );
        let proba = model.predict_proba(&x).expect("predict_proba");
        assert_eq!((proba.nrows(), proba.ncols()), (f.rows, f.n_classes));
        check_slice(proba.as_slice(), f.proba, &format!("{} proba", f.name));
        check_slice(
            &model.feature_importances().expect("importances"),
            f.importances,
            &format!("{} importances", f.name),
        );
    }
}

#[test]
fn extra_trees_regressor_parity() {
    for f in ET_REG_FIXTURES {
        let x = Matrix::from_flat(f.rows, f.cols, f.x.to_vec()).expect("fixture X");
        let mut model = ExtraTreesRegressor::new()
            .with_n_estimators(f.n_estimators)
            .with_max_depth((f.max_depth > 0).then_some(f.max_depth))
            .with_max_features(parse_max_features(f.max_features, f.cols))
            .with_bootstrap(f.bootstrap)
            .with_min_samples_leaf(f.min_samples_leaf)
            .with_random_state(f.seed);
        model.fit(&x, f.y).expect("fit");
        assert_eq!(
            model.n_estimators(),
            f.n_estimators,
            "{}: n_estimators",
            f.name
        );
        for (i, (tree, t)) in model.trees().iter().zip(f.trees).enumerate() {
            check_reg_tree(tree, t, &format!("{} tree {i}", f.name));
        }
        if !f.est_samples.is_empty() {
            check_samples(&model.estimators_samples(), f.est_samples, f.name);
        }
        check_slice(
            &model.predict(&x).expect("predict"),
            f.preds,
            &format!("{} preds", f.name),
        );
        check_slice(
            &model.feature_importances().expect("importances"),
            f.importances,
            &format!("{} importances", f.name),
        );
    }
}

/// Refitting with the same seed reproduces predictions bit-for-bit, and
/// ExtraTrees defaults differ from RandomForest (no bootstrap).
#[test]
fn extra_trees_determinism_and_defaults() {
    let f = &ET_CLF_FIXTURES[0];
    let (a, x) = fit_et_clf(f);
    let (b, _) = fit_et_clf(f);
    check_slice(
        &a.predict(&x).unwrap(),
        &b.predict(&x).unwrap(),
        "refit preds",
    );
    let defaults = ExtraTreesClassifier::new();
    let params = defaults.get_params();
    let bootstrap = params
        .iter()
        .find(|(name, _)| *name == "bootstrap")
        .map(|(_, value)| value.clone());
    assert_eq!(bootstrap, Some(datarust::traits::ParamValue::Bool(false)));
    assert!(!defaults.is_fitted());
}
