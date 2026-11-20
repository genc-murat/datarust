//! Hyperparameter search tour: `GridSearchCV`, `RandomizedSearchCV`,
//! `validation_curve`, and `learning_curve` on a supervised pipeline with
//! nested `step__param` addressing.
//!
//! Scenario: a mildly nonlinear target regressed by `Ridge`, tuned end-to-end
//! through a `SupervisedPipeline` (StandardScaler + Ridge) — so the search
//! picks both the regularization strength and a preprocessing flag at once.
//!
//! Run: `cargo run --example hyperparameter_search`

use datarust::linear_model::Ridge;
use datarust::matrix::Matrix;
use datarust::metrics::regression::r2_score;
use datarust::model_selection::{
    learning_curve, validation_curve, GridSearchCV, KFold, ParamDistribution, RandomizedSearchCV,
};
use datarust::scaler::StandardScaler;
use datarust::traits::{ParamValue, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::SupervisedPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Deterministic dataset: y = 2*x0 + sin(x1) + small interaction ──
    let rows: Vec<Vec<f64>> = (0..60)
        .map(|i| {
            let i = i as f64;
            vec![i / 6.0, (i / 8.0).cos() * 3.0, (i % 7.0)]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| 2.0 * r[0] + 1.5 * r[1] + 0.1 * r[2])
        .collect();
    let x = Matrix::new(rows)?;

    let make_pipeline = || {
        SupervisedPipeline::new(Ridge::new()).push(
            "scaler",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
    };

    // ── 2. Grid search over the pipeline (estimator + preprocessing) ──────
    let mut gs = GridSearchCV::new(
        make_pipeline(),
        vec![
            (
                "estimator__alpha",
                vec![
                    ParamValue::Float(10.0),
                    ParamValue::Float(1.0),
                    ParamValue::Float(0.1),
                    ParamValue::Float(0.01),
                ],
            ),
            (
                "scaler__with_mean",
                vec![ParamValue::Bool(true), ParamValue::Bool(false)],
            ),
        ],
        r2_score,
    )
    .with_cv(KFold::new().with_n_splits(5));

    gs.fit(&x, &y)?;
    let results = gs.cv_results().expect("fit populated cv_results");

    println!("=== GridSearchCV: 8 candidates ===");
    for (i, combo) in results.params.iter().enumerate() {
        println!(
            "rank {:>2}  mean R² {:.4} ± {:.4}  {:?}",
            results.rank_test_score[i],
            results.mean_test_score[i],
            results.std_test_score[i],
            combo
        );
    }
    println!(
        "\nbest params: {:?}\nbest mean CV R²: {:.4}",
        gs.best_params().unwrap(),
        gs.best_score().unwrap()
    );

    // The refit winner predicts new samples directly.
    let best = gs.best_estimator().expect("refit=true by default");
    let x_new = Matrix::new(vec![vec![10.0, 3.0, 2.0]])?;
    println!("prediction for x_new: {:.4}", best.predict(&x_new)?[0]);

    // ── 3. Randomized search over continuous alpha ────────────────────────
    let mut rs = RandomizedSearchCV::new(
        make_pipeline(),
        vec![("estimator__alpha", ParamDistribution::Uniform(1e-4, 10.0))],
        r2_score,
    )
    .with_n_iter(15)
    .with_random_state(42)
    .with_cv(KFold::new().with_n_splits(5));

    rs.fit(&x, &y)?;
    println!("\n=== RandomizedSearchCV: 15 sampled alphas ===");
    println!(
        "best params: {:?}\nbest mean CV R²: {:.4}",
        rs.best_params().unwrap(),
        rs.best_score().unwrap()
    );

    // ── 4. Validation curve: how alpha trades bias vs. fit ───────────────
    // Shuffled folds: the rows are generated in sorted-x order, and both
    // curve diagnostics assume exchangeable samples (like sklearn's examples).
    let curve_cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42);

    let (train, test) = validation_curve(
        &make_pipeline(),
        &x,
        &y,
        "estimator__alpha",
        &[
            ParamValue::Float(0.01),
            ParamValue::Float(0.1),
            ParamValue::Float(1.0),
            ParamValue::Float(10.0),
        ],
        &curve_cv,
        r2_score,
    )?;

    println!("\n=== validation_curve (estimator__alpha) ===");
    for (i, alpha) in [0.01, 0.1, 1.0, 10.0].iter().enumerate() {
        println!(
            "alpha {:>6}: train {:.4}  test {:.4}",
            alpha,
            train[i].iter().sum::<f64>() / train[i].len() as f64,
            test[i].iter().sum::<f64>() / test[i].len() as f64
        );
    }

    // ── 5. Learning curve: would more data help? ─────────────────────────
    let (sizes, train, test) = learning_curve(
        &make_pipeline(),
        &x,
        &y,
        &[0.25, 0.5, 0.75, 1.0],
        &curve_cv,
        r2_score,
    )?;

    println!("\n=== learning_curve ===");
    for (i, size) in sizes.iter().enumerate() {
        println!(
            "n_train {:>3}: train {:.4}  test {:.4}",
            size,
            train[i].iter().sum::<f64>() / train[i].len() as f64,
            test[i].iter().sum::<f64>() / test[i].len() as f64
        );
    }

    Ok(())
}
