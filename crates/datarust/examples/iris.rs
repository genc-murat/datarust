//! Iris Classification Example
//!
//! Run: `cargo run --example iris --features datasets`

use datarust::datasets::iris;
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::TrainTestSplit;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Iris Classification ===");
    let data = iris::load();
    let x = data.features();
    let y = data.targets();

    println!("Dataset: {} samples, {} features", x.nrows(), x.ncols());

    // 80/20 train-test split
    let (x_train, x_test, y_train, y_test) = TrainTestSplit::new()
        .with_test_size(0.2)
        .with_shuffle(true)
        .with_random_state(42)
        .split(&x, y)?;

    // Scale features
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    // Train Logistic Regression model
    let mut model = LogisticRegression::new().with_solver(LogisticSolver::Svd);
    model.fit(&x_train_scaled, &y_train)?;

    // Predict and evaluate
    let preds = model.predict(&x_test_scaled)?;
    let acc = accuracy_score(&y_test, &preds)?;
    println!("Test Accuracy: {:.2}%", acc * 100.0);

    Ok(())
}
