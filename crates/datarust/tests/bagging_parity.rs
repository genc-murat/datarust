#[path = "ensemble_fixtures.rs"]
mod ensemble_fixtures;

use datarust::ensemble::{
    BaggingClassifier, BaggingMaxFeatures, BaggingMaxSamples, BaggingRegressor,
};
use datarust::traits::{PredictProba, Predictor};
use datarust::Matrix;
use ensemble_fixtures::{BAG_CLF_FIXTURES, BAG_REG_FIXTURES};

fn assert_bits(actual: f64, expected: f64, ctx: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{ctx}: got {:?} (0x{:016x}), want {:?} (0x{:016x})",
        actual,
        actual.to_bits(),
        expected,
        expected.to_bits()
    );
}

fn parse_max_samples(raw: &str) -> BaggingMaxSamples {
    if raw.contains('.') {
        BaggingMaxSamples::Fraction(raw.parse().unwrap())
    } else {
        BaggingMaxSamples::Count(raw.parse().unwrap())
    }
}

fn parse_max_features(raw: &str) -> BaggingMaxFeatures {
    if raw.contains('.') {
        BaggingMaxFeatures::Fraction(raw.parse().unwrap())
    } else {
        BaggingMaxFeatures::Count(raw.parse().unwrap())
    }
}

#[test]
fn bagging_classifier_parity() {
    for fixture in BAG_CLF_FIXTURES {
        println!("{}", fixture.name);
        let x = Matrix::from_flat(fixture.rows, fixture.cols, fixture.x.to_vec()).unwrap();
        let max_depth = if fixture.base_max_depth == 0 {
            None
        } else {
            Some(fixture.base_max_depth)
        };

        let mut bag = BaggingClassifier::new()
            .with_n_estimators(fixture.n_estimators)
            .with_max_samples(parse_max_samples(fixture.max_samples))
            .with_max_features(parse_max_features(fixture.max_features))
            .with_bootstrap(fixture.bootstrap)
            .with_bootstrap_features(fixture.bootstrap_features)
            .with_max_depth(max_depth)
            .with_random_state(fixture.seed);

        bag.fit(&x, fixture.y).unwrap();

        // Per-estimator predictions on feature subsets
        assert_eq!(bag.estimators().len(), fixture.n_estimators);
        assert_eq!(bag.estimators_features().len(), fixture.n_estimators);
        for i in 0..fixture.n_estimators {
            let x_sub = x
                .select_columns(bag.estimators_features()[i].as_slice())
                .unwrap();
            let est_preds = bag.estimators()[i].predict(&x_sub).unwrap();
            let want = fixture.est_preds[i];
            for (j, (&got, &w)) in est_preds.iter().zip(want.iter()).enumerate() {
                assert_bits(got, w, &format!("{} est {i} row {j}", fixture.name));
            }
        }

        // Per-estimator samples
        for (i, (got, want)) in bag
            .estimators_samples()
            .iter()
            .zip(fixture.est_samples.iter())
            .enumerate()
        {
            assert_eq!(got.as_slice(), *want, "{} est_samples[{i}]", fixture.name);
        }

        // Per-estimator features
        for (i, (got, want)) in bag
            .estimators_features()
            .iter()
            .zip(fixture.est_features.iter())
            .enumerate()
        {
            assert_eq!(got.as_slice(), *want, "{} est_features[{i}]", fixture.name);
        }

        // Ensemble predictions
        let preds = bag.predict(&x).unwrap();
        for (i, (&got, &w)) in preds.iter().zip(fixture.preds.iter()).enumerate() {
            assert_bits(got, w, &format!("{} preds[{i}]", fixture.name));
        }

        // Ensemble probabilities
        let proba = bag.predict_proba(&x).unwrap();
        let got = proba.as_slice();
        let want = fixture.proba;
        for i in 0..got.len() {
            assert_bits(got[i], want[i], &format!("{} proba[{i}]", fixture.name));
        }

        // Verify is_fitted after fit
        assert!(bag.is_fitted());
    }
}

#[test]
fn bagging_regressor_parity() {
    for fixture in BAG_REG_FIXTURES {
        println!("{}", fixture.name);
        let x = Matrix::from_flat(fixture.rows, fixture.cols, fixture.x.to_vec()).unwrap();
        let max_depth = if fixture.base_max_depth == 0 {
            None
        } else {
            Some(fixture.base_max_depth)
        };

        let mut bag = BaggingRegressor::new()
            .with_n_estimators(fixture.n_estimators)
            .with_max_samples(parse_max_samples(fixture.max_samples))
            .with_max_features(parse_max_features(fixture.max_features))
            .with_bootstrap(fixture.bootstrap)
            .with_bootstrap_features(fixture.bootstrap_features)
            .with_max_depth(max_depth)
            .with_random_state(fixture.seed);

        bag.fit(&x, fixture.y).unwrap();

        // Per-estimator predictions
        for i in 0..fixture.n_estimators {
            let x_sub = x
                .select_columns(bag.estimators_features()[i].as_slice())
                .unwrap();
            let est_preds = bag.estimators()[i].predict(&x_sub).unwrap();
            let want = fixture.est_preds[i];
            for (j, (&got, &w)) in est_preds.iter().zip(want.iter()).enumerate() {
                assert_bits(got, w, &format!("{} est {i} row {j}", fixture.name));
            }
        }

        // Per-estimator samples
        for (i, (got, want)) in bag
            .estimators_samples()
            .iter()
            .zip(fixture.est_samples.iter())
            .enumerate()
        {
            assert_eq!(got.as_slice(), *want, "{} est_samples[{i}]", fixture.name);
        }

        // Per-estimator features
        for (i, (got, want)) in bag
            .estimators_features()
            .iter()
            .zip(fixture.est_features.iter())
            .enumerate()
        {
            assert_eq!(got.as_slice(), *want, "{} est_features[{i}]", fixture.name);
        }

        // Ensemble predictions
        let preds = bag.predict(&x).unwrap();
        for (i, (&got, &w)) in preds.iter().zip(fixture.preds.iter()).enumerate() {
            assert_bits(got, w, &format!("{} preds[{i}]", fixture.name));
        }
    }
}
