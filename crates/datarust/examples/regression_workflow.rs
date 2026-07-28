//! End-to-end regression workflow: data generation → train/test split → scaling
//! (fit on train only) → Ridge training → test R² → K-fold cross-validation.
//!
//! Scenario: house-price prediction. Estimate price from area (m²), number of
//! rooms, and building age. The true relationship is
//! `y = 3·area + 2·rooms − 1.5·age + 50 + noise`.
//!
//! Run: `cargo run --example regression_workflow`

use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;
use datarust::model_selection::{cross_val_score, KFold};
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

/// Simple deterministic PRNG (xorshift64) — a fixed seed yields the same noise
/// sequence on every run, so results are reproducible.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// Uniform random number in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Use the upper bits to produce a value in [0, 1).
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximately normal distribution with mean 0 and standard deviation
    /// `sigma` (a simplified Box–Muller transform).
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Generate synthetic data ─────────────────────────────────────
    // True (known) coefficients — we check whether the model can recover them.
    let true_coef = [3.0, 2.0, -1.5];
    let true_intercept = 50.0;

    let n = 120; // number of samples
    let mut rng = Rng::new(42);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        // Features: area 50–250 m², rooms 1–6, age 0–50 years.
        let area = 50.0 + rng.next_f64() * 200.0;
        let rooms = 1.0 + (rng.next_f64() * 5.0).round();
        let age = rng.next_f64() * 50.0;
        rows.push(vec![area, rooms, age]);
        // Price = true relationship + small but measurable noise.
        // Signal range ~600 units; noise std ~80 → the signal dominates, so the
        // model achieves a high R², but not a perfect one (realistic).
        let price = true_intercept
            + true_coef[0] * area
            + true_coef[1] * rooms
            + true_coef[2] * age
            + rng.normal(80.0);
        y.push(price);
    }
    let x = Matrix::new(rows)?;
    println!("=== House-Price Regression ===");
    println!(
        "Data: {n} samples, {} features (area, rooms, age)",
        x.ncols()
    );
    println!("True coefficients: {:?}", true_coef);
    println!("True intercept:    {true_intercept}\n");

    // ── 2. Train/test split (deterministic seed) ───────────────────────
    let (x_tr, x_te, y_tr, y_te) = datarust::model_selection::TrainTestSplit::new()
        .with_test_size(0.25)
        .with_random_state(7)
        .split(&x, &y)?;
    println!("Split: {} train / {} test\n", x_tr.nrows(), x_te.nrows());

    // ── 3. Scaling — fit ONLY on train (prevents data leakage) ─────────
    // Fit the StandardScaler on the training data, then transform both train
    // and test with the same parameters. Including test statistics in the fit
    // would inflate the model's apparent generalization performance.
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_tr)?;
    let x_tr_s = scaler.transform(&x_tr)?;
    let x_te_s = scaler.transform(&x_te)?;

    // ── 4. Train the Ridge regression ──────────────────────────────────
    let mut model = Ridge::new().with_alpha(1.0);
    model.fit(&x_tr_s, &y_tr)?;

    println!("=== Trained Ridge (alpha=1.0) ===");
    println!("Learned coefficients: {:?}", model.coef());
    println!("Learned intercept:    {:.4}", model.intercept());
    // Note: because coefficients live in the scaled feature space, they won't
    // match true_coef exactly; only their signs and relative magnitudes align.
    println!();

    // ── 5. Test performance ────────────────────────────────────────────
    let preds_te = model.predict(&x_te_s)?;
    let r2_te = r2_score(&y_te, &preds_te)?;
    // Regressor::score also returns R² — show consistency between the two.
    let r2_score_method = model.score(&x_te_s, &y_te)?;
    println!("=== Test Performance ===");
    println!("R² (r2_score fn) : {r2_te:.4}");
    println!("R² (model.score) : {r2_score_method:.4}");
    println!("(R² closer to 1.0 is better; 0 means no better than the mean.)\n");

    // ── 6. K-fold cross-validation ─────────────────────────────────────
    // A single train/test split can be luck-dependent. K-fold CV partitions the
    // data into K folds, tests each in turn, and yields a more robust estimate.
    // cross_val_score clones and refits the model on every fold, so we pass the
    // raw (unscaled) X — each fold could embed its own StandardScaler via a
    // pipeline, but we keep it simple here.
    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(1);
    let scores = cross_val_score(&Ridge::new().with_alpha(1.0), &x, &y, &cv, r2_score)?;
    let mean_r2 = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("=== 5-Fold Cross-Validation ===");
    print!("Fold R² scores: ");
    for s in &scores {
        print!("{s:.4}  ");
    }
    println!();
    println!("Mean CV R²: {mean_r2:.4}");

    Ok(())
}
