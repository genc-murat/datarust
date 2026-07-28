//! Ridge (L2) vs Lasso (L1) regularization comparison.
//!
//! Scenario: in an 8-feature dataset, only 3 features carry signal, one is
//! collinear (which would strain LinearRegression), and the rest are pure
//! noise. Compares how Ridge and Lasso behave at different alpha values:
//! Ridge shrinks all coefficients (but zeroes none), while Lasso drives the
//! irrelevant ones to exactly zero, performing implicit feature selection.
//!
//! Run: `cargo run --example regularization_comparison`

use datarust::linear_model::{Lasso, Ridge};
use datarust::traits::{Predictor, Regressor};
use datarust::Matrix;

/// Simple deterministic PRNG (xorshift64) for reproducible data.
struct Rng {
    state: u64,
}

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

/// R² and nonzero-coefficient count for a fitted model.
struct FitSummary {
    r2: f64,
    nonzero: usize,
    coefs: Vec<f64>,
}

/// Generic helper: produce a summary for any Regressor-like type (coef/score).
/// Lasso and Ridge share the same `score` and `coef` signatures.
trait Inspect {
    fn coef(&self) -> &[f64];
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError>;
}

impl Inspect for Ridge {
    fn coef(&self) -> &[f64] {
        self.coef()
    }
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError> {
        Regressor::score(self, x, y)
    }
}

impl Inspect for Lasso {
    fn coef(&self) -> &[f64] {
        self.coef()
    }
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError> {
        // Lasso's inherent `score` method — not the Regressor trait, but the
        // signature matches.
        Lasso::score(self, x, y)
    }
}

fn summarize(
    model: &dyn Inspect,
    x: &Matrix,
    y: &[f64],
) -> Result<FitSummary, Box<dyn std::error::Error>> {
    let coefs = model.coef().to_vec();
    let nonzero = coefs.iter().filter(|c| c.abs() > 1e-10).count();
    Ok(FitSummary {
        r2: model.score(x, y)?,
        nonzero,
        coefs,
    })
}

fn print_coefs(coefs: &[f64]) -> String {
    let parts: Vec<String> = coefs
        .iter()
        .map(|c| {
            if c.abs() < 1e-10 {
                "   0.00".to_string()
            } else {
                format!("{:7.3}", c)
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Generate synthetic data ─────────────────────────────────────
    // 8 features:
    //   x0, x1, x2 → informative (true coefficients 2, 3, -1)
    //   x3         → informative (coefficient 1)
    //   x4, x5     → pure noise (coefficient 0)
    //   x6         → x0 + x1 + small noise (collinear — makes X'X singular)
    //   x7         → pure noise (coefficient 0)
    let true_coef = [2.0, 3.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let n = 150;
    let mut rng = Rng::new(99);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = rng.normal(1.0);
        let x1 = rng.normal(1.0);
        let x2 = rng.normal(1.0);
        let x3 = rng.normal(1.0);
        let x4 = rng.normal(1.0);
        let x5 = rng.normal(1.0);
        let x6 = x0 + x1 + rng.normal(0.05); // collinear
        let x7 = rng.normal(1.0);
        let row = vec![x0, x1, x2, x3, x4, x5, x6, x7];
        let target: f64 = row
            .iter()
            .zip(true_coef.iter())
            .map(|(xi, ci)| xi * ci)
            .sum::<f64>()
            + rng.normal(0.5);
        rows.push(row);
        y.push(target);
    }
    let x = Matrix::new(rows)?;
    println!("=== Ridge (L2) vs Lasso (L1) Comparison ===");
    println!("Data: {n} samples, {} features", x.ncols());
    println!("True coefficients: {}", print_coefs(&true_coef));
    println!("Note: x6 = x0 + x1 (collinear); x4, x5, x7 are pure noise.\n");

    // ── 2. Ridge alpha sweep ────────────────────────────────────────────
    // The Ridge (L2) penalty is ||β||²; a large alpha shrinks all coefficients
    // freely but drives none exactly to zero. Because of the collinearity,
    // LinearRegression would be singular here; Ridge always solves.
    println!("── Ridge (L2): α‖β‖² penalty ──");
    println!(
        "{:<8} {:<8} {:<48} {:<10}",
        "Alpha", "R²", "Coefficients", "Nonzero"
    );
    for &alpha in &[0.01, 1.0, 100.0] {
        let mut m = Ridge::new().with_alpha(alpha);
        m.fit(&x, &y)?;
        let s = summarize(&m, &x, &y)?;
        println!(
            "{:<8.2} {:<8.4} {:<48} {:<10}",
            alpha,
            s.r2,
            print_coefs(&s.coefs),
            s.nonzero
        );
    }
    println!(
        "→ Ridge shrinks all coefficients but does not zero out the noise features (x4, x5, x7).\n"
    );

    // ── 3. Lasso alpha sweep ────────────────────────────────────────────
    // The Lasso (L1) penalty is ||β||₁; soft-thresholding drives some
    // coefficients EXACTLY to zero — this provides implicit feature selection.
    println!("── Lasso (L1): α‖β‖₁ penalty ──");
    println!(
        "{:<8} {:<8} {:<48} {:<10}",
        "Alpha", "R²", "Coefficients", "Nonzero"
    );
    for &alpha in &[0.01, 0.5, 5.0] {
        let mut m = Lasso::new().with_alpha(alpha).with_max_iter(2000);
        m.fit(&x, &y)?;
        let s = summarize(&m, &x, &y)?;
        println!(
            "{:<8.2} {:<8.4} {:<48} {:<10}",
            alpha,
            s.r2,
            print_coefs(&s.coefs),
            s.nonzero
        );
    }
    println!("→ Lasso drives the noise features (x4, x5, x7) EXACTLY to zero at large alpha = automatic feature selection.\n");

    // ── 4. Summary interpretation ───────────────────────────────────────
    println!("=== Interpretation ===");
    println!(
        "• Ridge: no coefficient is zeroed, but all shrink as α grows. Predictions are stable."
    );
    println!("• Lasso: irrelevant features are excluded (sparse model). Ideal for interpretability and feature selection.");
    println!("• Collinearity (x6 ≈ x0+x1): LinearRegression would be singular here; both Ridge and Lasso solve without issue.");

    Ok(())
}
