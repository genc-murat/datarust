//! Model persistence: save a trained SupervisedPipeline to JSON, then reload
//! it without refitting and produce identical predictions.
//!
//! Scenario: production deployment. The model is trained on one machine,
//! written to disk (JSON), then loaded into a service and serves predictions
//! without needing access to the training data again.
//!
//! This example requires the `serde` feature:
//!   `cargo run --example model_persistence --features serde`

use datarust::decomposition::{PCAComponents, PCA};
use datarust::linear_model::Ridge;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Regressor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!(
        "This example requires the `serde` feature.\n\
         Run it with:\n  \
         cargo run --example model_persistence --features serde"
    );
}

#[cfg(feature = "serde")]
struct Rng {
    state: u64,
}

#[cfg(feature = "serde")]
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Generate synthetic regression data ──────────────────────────
    // 6 features; the true relationship depends on the first 4, the last 2 are
    // noise.
    let n = 100;
    let mut rng = Rng::new(7);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..6).map(|_| rng.normal(1.0)).collect();
        let target = 1.5 * row[0] - 2.0 * row[1] + 0.7 * row[2] + row[3] + rng.normal(0.3);
        rows.push(row);
        y.push(target);
    }
    let x = Matrix::new(rows)?;
    println!("=== Model Persistence ===");
    println!("Data: {n} samples, {} features\n", x.ncols());

    // ── 2. Build a SupervisedPipeline: scale → PCA → Ridge ──────────────
    // A preprocessing chain (StandardScaler + PCA) and a final estimator
    // (Ridge). with_estimator turns the preprocessing Pipeline into a
    // SupervisedPipeline.
    let mut pipeline = Pipeline::new()
        .push(
            "scaler",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .push(
            "pca",
            TransformerKind::PCA(PCA::new(PCAComponents::Count(4))),
        )
        .with_estimator(Ridge::new().with_alpha(1.0));

    pipeline.fit(&x, &y)?;
    let preds_original = pipeline.predict(&x)?;
    let r2 = pipeline.score(&x, &y)?;
    println!("Training complete.");
    println!("Training R²: {r2:.4}");
    println!(
        "Pipeline fitted? {}",
        if pipeline.is_fitted() { "yes" } else { "no" }
    );
    println!();

    // ── 3. Save to disk as JSON ─────────────────────────────────────────
    // save_json writes all parameters — including fitted state (scaler
    // mean/std, PCA eigenvectors, Ridge coefficients) — to pretty-printed JSON.
    let path = std::env::temp_dir().join("datarust_model_persistence_demo.json");
    // Remove any leftover file from a previous run.
    let _ = std::fs::remove_file(&path);
    datarust::serialize::save_json(&pipeline, &path)?;
    let file_size = std::fs::metadata(&path)?.len();
    println!("Model saved: {}", path.display());
    println!("File size: {file_size} bytes\n");

    // JSON preview (first ~400 chars) — shows the fitted parameters serialized.
    let json_str = std::fs::read_to_string(&path)?;
    let preview: String = json_str.chars().take(400).collect();
    println!("=== JSON Preview ===\n{preview}...\n");

    // ── 4. Load the model and predict without refitting ────────────────
    // load_json restores the original type. Importantly, the restored pipeline
    // arrives with `is_fitted() == true` — no training data is needed.
    let restored: datarust::pipeline::SupervisedPipeline<Ridge> =
        datarust::serialize::load_json(&path)?;
    println!(
        "Restored model fitted? {}",
        if restored.is_fitted() { "yes" } else { "no" }
    );

    // ── 5. Compare predictions from the original and restored models ────
    let preds_restored = restored.predict(&x)?;
    let max_diff = preds_original
        .iter()
        .zip(preds_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Max difference between original and restored predictions: {max_diff:.2e}");
    assert!(
        max_diff < 1e-10,
        "restored model produced different predictions"
    );
    println!("✓ The restored model produces identical predictions to the original.");

    // ── 6. Cleanup ─────────────────────────────────────────────────────
    let _ = std::fs::remove_file(&path);
    println!("\nTemporary file cleaned up.");

    Ok(())
}
