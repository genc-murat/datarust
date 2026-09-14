//! Decision-tree workflow: synthetic two-feature classification data →
//! grow a `DecisionTreeClassifier` → inspect the fitted `TreeStructure`
//! (depth, leaves, splits, importances) → evaluate on held-out points →
//! compare Gini vs entropy and a depth-constrained tree.
//!
//! Scenario: learn an axis-aligned rule (points are "accepted" when inside a
//! rectangle) — the classic problem decision trees solve exactly, and a
//! sanity check that the learned splits recover the true boundaries.
//!
//! Run: `cargo run --example decision_tree`

use datarust::matrix::Matrix;
use datarust::traits::{PredictProba, Predictor};
use datarust::tree::{Criterion, DecisionTreeClassifier, DecisionTreeRegressor, MaxFeatures};
use datarust::Rng;

/// A deterministic point is "accepted" when it falls inside the rectangle
/// `[2, 6] × [2, 6]` (the rule a tree should rediscover).
fn label(x: f64, y: f64) -> f64 {
    if (2.0..=6.0).contains(&x) && (2.0..=6.0).contains(&y) {
        1.0
    } else {
        0.0
    }
}

/// Uniform points in the unit square scaled to `[0, 8]²`, labeled by the rule.
fn make_dataset(n: usize, seed: u64) -> (Matrix, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, yy) = (rng.next_range(0.0, 8.0), rng.next_range(0.0, 8.0));
        rows.push(vec![x, yy]);
        y.push(label(x, yy));
    }
    (Matrix::new(rows).unwrap(), y)
}

fn accuracy(model: &DecisionTreeClassifier, x: &Matrix, y: &[f64]) -> f64 {
    let pred = model.predict(x).unwrap();
    pred.iter().zip(y).filter(|(p, t)| p == t).count() as f64 / y.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x_train, y_train) = make_dataset(400, 7);
    let (x_test, y_test) = make_dataset(200, 8);
    println!("=== Decision tree on a rectangle rule ===");
    println!(
        "train: {} points, {} positive | test: {} points",
        y_train.len(),
        y_train.iter().filter(|&&v| v == 1.0).count(),
        y_test.len()
    );

    // 1. Unlimited tree: learns the rule perfectly on axis-aligned data.
    let mut clf = DecisionTreeClassifier::new().with_random_state(7);
    clf.fit(&x_train, &y_train)?;
    let tree = clf.tree().expect("fitted");
    println!("\n--- Unlimited tree ---");
    println!(
        "nodes: {}, depth: {}, leaves: {}",
        tree.n_nodes(),
        tree.get_depth(),
        tree.n_leaves()
    );
    let f0 = tree.node_feature(0).expect("internal root");
    let t0 = tree.node_threshold(0).unwrap();
    println!("root split: feature {f0} <= {t0:.4}");
    println!("feature importances: {:?}", clf.feature_importances()?);
    println!(
        "train accuracy: {:.3} | test accuracy: {:.3}",
        accuracy(&clf, &x_train, &y_train),
        accuracy(&clf, &x_test, &y_test)
    );

    // 2. Probabilities come from the leaf's class frequencies.
    let proba = clf.predict_proba(&x_test)?;
    println!("\n--- predict_proba (first 3 rows) ---");
    for i in 0..3 {
        println!(
            "row {i}: p(class 0) = {:.3}, p(class 1) = {:.3}",
            proba.get(i, 0),
            proba.get(i, 1)
        );
    }

    // 3. Depth-1 stump: one split only — still captures most of the signal.
    let mut stump = DecisionTreeClassifier::new()
        .with_max_depth(Some(1))
        .with_random_state(7);
    stump.fit(&x_train, &y_train)?;
    println!("\n--- Depth-1 stump ---");
    println!(
        "test accuracy: {:.3} (unlimited tree: {:.3})",
        accuracy(&stump, &x_test, &y_test),
        accuracy(&clf, &x_test, &y_test)
    );

    // 4. Gini vs entropy: identical here — both recover the same rectangle.
    for criterion in [Criterion::Gini, Criterion::Entropy] {
        let mut c = DecisionTreeClassifier::new()
            .with_criterion(criterion)
            .with_random_state(7);
        c.fit(&x_train, &y_train)?;
        println!(
            "criterion {:<8} → depth {}, test accuracy {:.3}",
            criterion,
            c.tree().unwrap().get_depth(),
            accuracy(&c, &x_test, &y_test)
        );
    }

    // 5. Regression: predict the x-coordinate from both features.
    let (x, _) = make_dataset(200, 9);
    let y_reg: Vec<f64> = (0..x.nrows()).map(|i| x.get(i, 0)).collect();
    let mut reg = DecisionTreeRegressor::new().with_random_state(7);
    reg.fit(&x, &y_reg)?;
    let pred = reg.predict(&x)?;
    let mse: f64 = pred
        .iter()
        .zip(y_reg.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / y_reg.len() as f64;
    println!("\n--- DecisionTreeRegressor ---");
    println!("in-sample MSE reconstructing x from (x, y): {mse:.2e} (pure leaves → ~0)",);

    // 6. Per-split feature strategies: All | Sqrt | Log2 | Count(n).
    // Sqrt considers sqrt(n_features) features at every split, the
    // random-forest-style default for classifiers.
    let sqrt_tree = DecisionTreeClassifier::new()
        .with_max_features(MaxFeatures::Sqrt)
        .with_random_state(7);
    println!(
        "\nwith_max_features(Sqrt) on {} features: {}",
        x_train.ncols(),
        if sqrt_tree.is_fitted() {
            "fitted"
        } else {
            "not fitted yet"
        }
    );

    println!("\nAll trees were seeded with random_state=7 — refitting reproduces them exactly.");
    Ok(())
}
