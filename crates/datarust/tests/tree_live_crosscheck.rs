//! Live scikit-learn cross-check on configs **not** covered by the
//! `tree_fixtures.rs` parities: a fresh dataset (wine), fresh seeds, and
//! parameters the fixture list never exercises (`min_samples_leaf`,
//! `max_features="sqrt"` under entropy).
//!
//! The expected values were captured from a live scikit-learn 1.6.1 run as
//! shortest-round-trip decimal literals (Rust's f64 parser is correctly
//! rounded, so parsing yields bit-identical values).
//!
//! Note: diabetes is intentionally absent — datarust's embedded diabetes
//! dataset stores the *unnormalized* measurements, so `load_diabetes`-derived
//! expectations would not apply; its parity is covered by the `tree_fixtures`
//! tests, which carry their own X arrays.
#![cfg(feature = "datasets")]

use datarust::datasets::{iris, wine};
use datarust::traits::Predictor;
use datarust::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};

/// Asserts our fitted tree matches the captured sklearn tree bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn assert_tree_matches(
    name: &str,
    features: &[i32],
    thresholds: &[f64],
    impurities: &[f64],
    importances: &[f64],
    ours: &datarust::tree::TreeStructure,
    ours_importances: &[f64],
    n_features: usize,
) {
    assert_eq!(ours.n_nodes(), features.len(), "[{name}] node count");
    for (n, &f) in features.iter().enumerate() {
        if f == -2 {
            assert!(ours.node_feature(n).is_none(), "[{name}] node {n} leaf");
        } else {
            assert_eq!(
                ours.node_feature(n),
                Some(f as usize),
                "[{name}] node {n} feature"
            );
            assert_eq!(
                ours.node_threshold(n).unwrap().to_bits(),
                thresholds[n].to_bits(),
                "[{name}] node {n} threshold"
            );
        }
        assert_eq!(
            ours.node_impurity(n).unwrap().to_bits(),
            impurities[n].to_bits(),
            "[{name}] node {n} impurity"
        );
    }
    assert_eq!(
        ours_importances.len(),
        n_features,
        "[{name}] importance length"
    );
    for (k, &e) in importances.iter().enumerate() {
        assert_eq!(
            ours_importances[k].to_bits(),
            e.to_bits(),
            "[{name}] importance {k}"
        );
    }
}

#[test]
fn live_sklearn_crosscheck_outside_fixtures() {
    // wine: wine_gini_rs11_depth4
    {
        let d = wine::load();
        let x = d.features();
        let y = d.targets();
        let mut m = DecisionTreeClassifier::new()
            .with_max_depth(Some(4))
            .with_random_state(11);
        m.fit(&x, y).unwrap();
        assert_tree_matches(
            "wine_gini_rs11_depth4",
            &[
                12, 11, 10, 6, -2, -2, 2, -2, -2, 6, -2, 0, -2, -2, 6, 9, -2, -2, 4, -2, -2,
            ],
            &[
                755.0,
                2.1149998903274536,
                0.9350000023841858,
                1.5800000429153442,
                -2.0,
                -2.0,
                2.4499999284744263,
                -2.0,
                -2.0,
                0.7950000166893005,
                -2.0,
                13.174999713897705,
                -2.0,
                -2.0,
                2.165000081062317,
                3.6050000190734863,
                -2.0,
                -2.0,
                135.5,
                -2.0,
                -2.0,
            ],
            &[
                0.6583133442747128,
                0.49216784351919485,
                0.22684310018903586,
                0.04874999999999996,
                0.0,
                0.0,
                0.2777777777777778,
                0.0,
                0.0,
                0.11739644970414198,
                0.0,
                0.06147644242882333,
                0.0,
                0.48,
                0.26464691468032975,
                0.375,
                0.0,
                0.0,
                0.06549841999425454,
                0.0,
                0.0,
            ],
            &[
                0.01283340962648672,
                0.0,
                0.014520560461003322,
                0.0,
                0.033668011373987373,
                0.0,
                0.14440427175529344,
                0.0,
                0.0,
                0.02613700882980598,
                0.059401718859817494,
                0.3185672624090835,
                0.3904677566845222,
            ],
            m.tree().unwrap(),
            &m.feature_importances().unwrap(),
            x.ncols(),
        );
    }
    // iris: iris_entropy_rs999_sqrt
    {
        let d = iris::load();
        let x = d.features();
        let y = d.targets();
        let mut m = DecisionTreeClassifier::new()
            .with_criterion(Criterion::Entropy)
            .with_max_features(MaxFeatures::Sqrt)
            .with_random_state(999);
        m.fit(&x, y).unwrap();
        assert_tree_matches(
            "iris_entropy_rs999_sqrt",
            &[
                2, -2, 2, 3, -2, -2, 3, 0, 2, -2, 1, -2, 3, -2, -2, -2, 2, 0, -2, -2, -2,
            ],
            &[
                2.449999988079071,
                -2.0,
                4.75,
                1.6500000357627869,
                -2.0,
                -2.0,
                1.75,
                7.049999952316284,
                4.950000047683716,
                -2.0,
                2.649999976158142,
                -2.0,
                1.550000011920929,
                -2.0,
                -2.0,
                -2.0,
                4.8500001430511475,
                5.950000047683716,
                -2.0,
                -2.0,
                -2.0,
            ],
            &[
                1.5849625007211563,
                0.0,
                1.0,
                0.15374218032876188,
                0.0,
                0.0,
                0.49716776141607527,
                0.9910760598382221,
                0.9544340029249649,
                0.0,
                0.9709505944546686,
                0.0,
                0.9182958340544896,
                0.0,
                0.0,
                0.0,
                0.15109697051711368,
                0.9182958340544896,
                0.0,
                0.0,
                0.0,
            ],
            &[
                0.01698925585713631,
                0.008832450690597573,
                0.8852282718044832,
                0.08895002164778301,
            ],
            m.tree().unwrap(),
            &m.feature_importances().unwrap(),
            x.ncols(),
        );
    }
    // wine: wine_reg_rs31_msl5
    {
        let d = wine::load();
        let x = d.features();
        let y = d.targets();
        let mut m = DecisionTreeRegressor::new()
            .with_min_samples_leaf(5)
            .with_random_state(31);
        m.fit(&x, y).unwrap();
        assert_tree_matches(
            "wine_reg_rs31_msl5",
            &[6, 9, -2, 1, -2, -2, 12, 3, -2, -2, 9, -2, -2],
            &[
                1.5750000476837158,
                3.8249999284744263,
                -2.0,
                1.6749999523162842,
                -2.0,
                -2.0,
                724.5,
                17.75,
                -2.0,
                -2.0,
                3.549999952316284,
                -2.0,
                -2.0,
            ],
            &[
                0.5973046332533771,
                0.17481789802289294,
                0.0,
                0.019991670137442963,
                0.1599999999999997,
                0.0,
                0.2499256837098692,
                0.018175582990397787,
                0.15999999999999992,
                0.0,
                0.06035379812695109,
                0.15999999999999992,
                0.0,
            ],
            &[
                0.0,
                0.0017281702115266564,
                0.0,
                0.001746353820654543,
                0.0,
                0.0,
                0.6398190141626557,
                0.0,
                0.0,
                0.12318154006265075,
                0.0,
                0.0,
                0.23352492174251227,
            ],
            m.tree().unwrap(),
            &m.feature_importances().unwrap(),
            x.ncols(),
        );
    }
}
