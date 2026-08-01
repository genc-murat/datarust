//! Housing Price Regression Example (Synthetic Mock)
//!
//! Run: `cargo run --example housing`

use datarust::linear_model::LinearRegression;
use datarust::metrics::regression::{mean_squared_error, r2_score};
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
    println!("=== Housing Price Regression ===");

    // 1. Generate synthetic housing data
    // Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    let n = 1000;
    let mut rng = Rng(42);
    let mut num_rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for _ in 0..n {
        let med_inc = 1.0 + rng.next_f64() * 10.0;
        let house_age = 1.0 + rng.next_f64() * 50.0;
        let ave_rooms = 1.0 + rng.next_f64() * 10.0;
        let ave_bedrms = 1.0 + rng.next_f64() * 3.0;
        let population = 10.0 + rng.next_f64() * 3000.0;
        let ave_occup = 1.0 + rng.next_f64() * 5.0;
        let latitude = 32.0 + rng.next_f64() * 10.0;
        let longitude = -124.0 + rng.next_f64() * 10.0;

        num_rows.push(vec![
            med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude,
        ]);

        // Price is roughly dependent on MedInc, HouseAge, and Rooms
        let price = 0.5 * med_inc + 0.01 * house_age + 0.1 * ave_rooms + rng.next_f64() * 0.5;
        y.push(price);
    }

    let x = Matrix::new(num_rows)?;
    println!(
        "Synthetic Data: {} samples, {} features",
        x.nrows(),
        x.ncols()
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
    let mut model = LinearRegression::new();
    model.fit(&x_tr_scaled, &y_tr)?;

    // 4. Predict
    let preds = model.predict(&x_te_scaled)?;
    let mse = mean_squared_error(&y_te, &preds, true)?;
    let r2 = r2_score(&y_te, &preds)?;

    println!("Test Mean Squared Error: {:.4}", mse);
    println!("Test R2 Score: {:.4}", r2);

    Ok(())
}
