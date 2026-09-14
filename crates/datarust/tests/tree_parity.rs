//! Bit-exact parity tests against scikit-learn 1.6.1 fixtures.
//!
//! Fixtures (`tree_fixtures.rs`) were generated with seeded sklearn runs
//! (`random_state` fixed, numpy 2.0.2 legacy `RandomState`), so every value
//! — split features, thresholds, impurities, node values, predictions,
//! probabilities, and feature importances — is deterministic and must match
//! **bit-for-bit** (`to_bits()` comparison), since our splitter is an exact
//! transliteration of sklearn's.

#[path = "tree_fixtures.rs"]
mod tree_fixtures;

use tree_fixtures::{CLF_FIXTURES, REG_FIXTURES};

use datarust::matrix::Matrix;
use datarust::traits::{PredictProba, Predictor};
use datarust::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};

fn build_classifier(f: &tree_fixtures::ClfFixture) -> DecisionTreeClassifier {
    let max_features = match f.max_features {
        "all" => MaxFeatures::All,
        "sqrt" => MaxFeatures::Sqrt,
        other => panic!("unknown max_features {other:?} in fixture {}", f.name),
    };
    let mut clf = DecisionTreeClassifier::new()
        .with_criterion(f.criterion.parse::<Criterion>().unwrap())
        .with_max_features(max_features)
        .with_random_state(f.seed);
    if f.max_depth > 0 {
        clf = clf.with_max_depth(Some(f.max_depth));
    }
    clf
}

fn build_regressor(f: &tree_fixtures::RegFixture) -> DecisionTreeRegressor {
    let mut reg = DecisionTreeRegressor::new().with_random_state(f.seed);
    if f.max_depth > 0 {
        reg = reg.with_max_depth(Some(f.max_depth));
    }
    reg
}

#[test]
fn classifier_parity_on_reference_fixtures() {
    for f in CLF_FIXTURES {
        let x = Matrix::from_flat(f.rows, f.cols, f.x.to_vec()).unwrap();
        let mut clf = build_classifier(f);
        clf.fit(&x, f.y).unwrap();
        let tree = clf.tree().unwrap();

        assert_eq!(tree.n_nodes(), f.features.len(), "[{}] node count", f.name);

        for (n, feature) in f.features.iter().enumerate() {
            if *feature == -2 {
                assert!(
                    tree.node_feature(n).is_none(),
                    "[{}] node {n} should be a leaf",
                    f.name
                );
                assert!(
                    tree.node_threshold(n).is_none(),
                    "[{}] leaf node {n} must not carry a threshold",
                    f.name
                );
            } else {
                assert_eq!(
                    tree.node_feature(n),
                    Some(*feature as usize),
                    "[{}] node {n} feature",
                    f.name
                );
                assert_eq!(
                    &tree.node_threshold(n).unwrap().to_bits(),
                    &f.thresholds[n].to_bits(),
                    "[{}] node {n} threshold",
                    f.name
                );
            }
            assert_eq!(
                tree.node_n_samples(n),
                Some(f.n_node[n]),
                "[{}] node {n} sample count",
                f.name
            );
            assert_eq!(
                &tree.node_impurity(n).unwrap().to_bits(),
                &f.impurities[n].to_bits(),
                "[{}] node {n} impurity",
                f.name
            );
        }

        // Class-count values, bit-exact. sklearn ≥ 1.3 stores *normalized*
        // proportions in `tree_.value`; ours store counts, so compare the
        // proportion `count / n_node`.
        let root_value = tree.root_class_counts().unwrap();
        let root_n = f.n_node[0] as f64;
        for (k, v) in root_value.iter().enumerate() {
            assert_eq!(
                (v / root_n).to_bits(),
                f.values[k].to_bits(),
                "[{}] root class proportion {k}",
                f.name
            );
        }

        // Predictions, probabilities, and importances: bit-exact.
        let preds = clf.predict(&x).unwrap();
        assert_eq!(preds.len(), f.preds.len(), "[{}] prediction count", f.name);
        for (i, (p, e)) in preds.iter().zip(f.preds).enumerate() {
            assert_eq!(p.to_bits(), e.to_bits(), "[{}] prediction {i}", f.name);
        }

        let proba = PredictProba::predict_proba(&clf, &x).unwrap();
        assert_eq!(
            proba.nrows() * proba.ncols(),
            f.proba.len(),
            "[{}] proba shape",
            f.name
        );
        for (i, e) in f.proba.iter().enumerate() {
            let (r, c) = (i / f.n_classes, i % f.n_classes);
            assert_eq!(
                proba.get(r, c).to_bits(),
                e.to_bits(),
                "[{}] proba row {r} col {c}",
                f.name
            );
        }

        let importances = clf.feature_importances().unwrap();
        for (k, e) in f.importances.iter().enumerate() {
            assert_eq!(
                importances[k].to_bits(),
                e.to_bits(),
                "[{}] importance of feature {k}",
                f.name
            );
        }
    }
}

#[test]
fn regressor_parity_on_reference_fixtures() {
    for f in REG_FIXTURES {
        let x = Matrix::from_flat(f.rows, f.cols, f.x.to_vec()).unwrap();
        let mut reg = build_regressor(f);
        reg.fit(&x, f.y).unwrap();
        let tree = reg.tree().unwrap();

        assert_eq!(tree.n_nodes(), f.features.len(), "[{}] node count", f.name);

        for (n, feature) in f.features.iter().enumerate() {
            if *feature == -2 {
                assert!(
                    tree.node_feature(n).is_none(),
                    "[{}] node {n} should be a leaf",
                    f.name
                );
            } else {
                assert_eq!(
                    tree.node_feature(n),
                    Some(*feature as usize),
                    "[{}] node {n} feature",
                    f.name
                );
                assert_eq!(
                    &tree.node_threshold(n).unwrap().to_bits(),
                    &f.thresholds[n].to_bits(),
                    "[{}] node {n} threshold",
                    f.name
                );
            }
            assert_eq!(
                tree.node_n_samples(n),
                Some(f.n_node[n]),
                "[{}] node {n} sample count",
                f.name
            );
            assert_eq!(
                &tree.node_impurity(n).unwrap().to_bits(),
                &f.impurities[n].to_bits(),
                "[{}] node {n} impurity",
                f.name
            );
            // Regression node value (target mean), bit-exact.
            assert_eq!(
                &tree.node_regression_value(n).unwrap().to_bits(),
                &f.values[n].to_bits(),
                "[{}] node {n} value",
                f.name
            );
        }

        let preds = reg.predict(&x).unwrap();
        assert_eq!(preds.len(), f.preds.len(), "[{}] prediction count", f.name);
        for (i, (p, e)) in preds.iter().zip(f.preds).enumerate() {
            assert_eq!(p.to_bits(), e.to_bits(), "[{}] prediction {i}", f.name);
        }

        let importances = reg.feature_importances().unwrap();
        for (k, e) in f.importances.iter().enumerate() {
            assert_eq!(
                importances[k].to_bits(),
                e.to_bits(),
                "[{}] importance of feature {k}",
                f.name
            );
        }
    }
}

#[test]
fn entropy_criterion_parity_on_iris_maxdepth2() {
    // Balanced 30-row iris subsample (rows 0–9, 50–59, 100–109, taken from
    // the full-iris fixture), entropy criterion, depth 2, seeded — measured
    // node-for-node against sklearn 1.6.1.
    let full = &CLF_FIXTURES[0];
    assert_eq!(full.name, "iris_gini_rs7_full");
    let keep: Vec<usize> = (0..10).chain(50..60).chain(100..110).collect();
    let x30: Vec<f64> = keep
        .iter()
        .flat_map(|&r| full.x[r * 4..r * 4 + 4].iter().copied())
        .collect();
    let y: Vec<f64> = (0..30).map(|i| (i / 10) as f64).collect();

    let x = Matrix::from_flat(30, 4, x30).unwrap();
    let mut ours = DecisionTreeClassifier::new()
        .with_criterion(Criterion::Entropy)
        .with_max_depth(Some(2))
        .with_random_state(7);
    ours.fit(&x, &y).unwrap();
    let tree = ours.tree().unwrap();

    assert_eq!(tree.n_nodes(), 5);
    assert_eq!(tree.node_feature(0), Some(2));
    assert_eq!(tree.node_threshold(0).unwrap().to_bits(), 2.5f64.to_bits());
    assert_eq!(tree.node_feature(2), Some(3));
    assert_eq!(
        tree.node_threshold(2).unwrap().to_bits(),
        1.6500000357627869f64.to_bits()
    );
    assert_eq!(
        tree.node_impurity(0).unwrap().to_bits(),
        1.5849625007211563f64.to_bits()
    );
    assert_eq!(tree.node_n_samples(0), Some(30));
    assert_eq!(tree.node_n_samples(1), Some(10));
    assert_eq!(tree.node_n_samples(2), Some(20));
    assert_eq!(tree.get_depth(), 2);
    assert_eq!(tree.n_leaves(), 3);

    let preds = ours.predict(&x).unwrap();
    for (p, t) in preds.iter().zip(&y) {
        assert_eq!(p, t);
    }
}
