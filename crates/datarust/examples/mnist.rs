//! MNIST Digits Classification Example (Synthetic Mock)
//!
//! Run: `cargo run --example mnist`

use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::TrainTestSplit;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

// Simple PRNG for synthetic data
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MNIST Digits Classification ===");
    
    // 1. Generate synthetic digits-like data (64 pixels, 10 classes)
    let n = 500;
    let mut rng = Rng(2025);
    let mut num_rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    
    for _ in 0..n {
        let label = (rng.next_f64() * 10.0) as u8;
        let label = label.min(9);
        y.push(label as f64);
        
        let mut row = Vec::with_capacity(64);
        for i in 0..64 {
            // Give features some correlation with the label
            let noise = rng.next_f64() * 0.5;
            let signal = if i % 10 == label as usize { 1.0 } else { 0.0 };
            row.push(signal + noise);
        }
        num_rows.push(row);
    }
    
    let x = Matrix::new(num_rows)?;
    println!("Synthetic Data: {} samples, {} features, 10 classes", x.nrows(), x.ncols());

    // 2. Preprocess & Split
    let (x_tr, x_te, y_tr, y_te) = TrainTestSplit::new()
        .with_test_size(0.2)
        .with_shuffle(true)
        .with_random_state(42)
        .split(&x, &y)?;
    let mut scaler = StandardScaler::new();
    let x_tr_scaled = scaler.fit_transform(&x_tr)?;
    let x_te_scaled = scaler.transform(&x_te)?;

    // 3. Train
    let mut model = LogisticRegression::new().with_solver(LogisticSolver::Svd);
    model.fit(&x_tr_scaled, &y_tr)?;

    // 4. Predict
    let preds = model.predict(&x_te_scaled)?;
    let acc = accuracy_score(&y_te, &preds)?;
    println!("Test Accuracy: {:.2}%", acc * 100.0);

    Ok(())
}
