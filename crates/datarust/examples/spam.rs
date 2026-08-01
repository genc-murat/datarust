//! Spam Detection Classification Example (Synthetic Mock)
//!
//! Run: `cargo run --example spam`

use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};
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
    println!("=== Spam Email Classification ===");

    // 1. Generate synthetic text-like features
    // Simulating TF-IDF features for 50 common words (some spammy, some hammy)
    let n = 1000;
    let n_features = 50;
    let mut rng = Rng(1337);
    let mut num_rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        let is_spam = if rng.next_f64() > 0.7 { 1.0 } else { 0.0 }; // 30% spam
        y.push(is_spam);

        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            // first 10 words are spammy (e.g., "free", "win", "viagra")
            // next 10 are hammy (e.g., "meeting", "code", "attached")
            // remaining 30 are neutral
            let base_freq = rng.next_f64() * 0.1;
            let freq = if (is_spam == 1.0 && j < 10) || (is_spam == 0.0 && (10..20).contains(&j)) {
                base_freq + rng.next_f64() * 0.5
            } else {
                base_freq
            };
            row.push(freq);
        }
        num_rows.push(row);
    }

    let x = Matrix::new(num_rows)?;
    println!(
        "Synthetic TF-IDF Data: {} samples, {} features",
        x.nrows(),
        x.ncols()
    );

    let spam_count = y.iter().filter(|&&v| v == 1.0).count();
    println!(
        "Class distribution: {} Spam, {} Ham",
        spam_count,
        n - spam_count
    );

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
    let prec = precision_score(&y_te, &preds)?;
    let rec = recall_score(&y_te, &preds)?;
    let f1 = f1_score(&y_te, &preds)?;

    println!("=== Evaluation ===");
    println!("Accuracy : {:.4}", acc);
    println!("Precision: {:.4}", prec);
    println!("Recall   : {:.4}", rec);
    println!("F1 Score : {:.4}", f1);

    Ok(())
}
