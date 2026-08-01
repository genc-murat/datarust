//! Titanic Survival Classification Example (Synthetic Mock)
//!
//! Run: `cargo run --example titanic`

use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::{HandleUnknown, OneHotEncoder};
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::TrainTestSplit;
use datarust::scaler::StandardScaler;
use datarust::traits::Predictor;
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::{Matrix, StrMatrix};

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
    println!("=== Titanic Survival Classification ===");
    
    // 1. Generate synthetic titanic-like data
    // Numeric: Age, Fare, SibSp, Parch
    // Categorical: Sex, Pclass, Embarked
    let n = 800;
    let mut rng = Rng(1912);
    let mut num_rows = Vec::with_capacity(n);
    let mut cat_rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    
    let sexes = ["male", "female"];
    let pclasses = ["1", "2", "3"];
    let embarked = ["C", "Q", "S"];
    
    for _ in 0..n {
        let age = 1.0 + rng.next_f64() * 70.0;
        let fare = 5.0 + rng.next_f64() * 100.0;
        let sibsp = (rng.next_f64() * 4.0) as f64;
        let parch = (rng.next_f64() * 3.0) as f64;
        
        let sex = sexes[(rng.next_f64() * 2.0) as usize % 2];
        let pclass = pclasses[(rng.next_f64() * 3.0) as usize % 3];
        let emb = embarked[(rng.next_f64() * 3.0) as usize % 3];
        
        num_rows.push(vec![age, fare, sibsp, parch]);
        cat_rows.push(vec![sex, pclass, emb]);
        
        // Survival probability depends on sex and class mostly
        let mut score = 0.0;
        if sex == "female" { score += 2.0; }
        if pclass == "1" { score += 1.5; }
        if pclass == "3" { score -= 1.0; }
        if age < 10.0 { score += 1.0; }
        
        let survived = if score + (rng.next_f64() * 2.0 - 1.0) > 1.0 { 1.0 } else { 0.0 };
        y.push(survived);
    }
    
    let numeric = Matrix::new(num_rows)?;
    let categorical = StrMatrix::from_strings(cat_rows)?;
    let table = Table::new(numeric, categorical)?;
    
    println!("Synthetic Data: {} samples", n);
    println!("Numeric features: Age, Fare, SibSp, Parch");
    println!("Categorical features: Sex, Pclass, Embarked");

    // 2. Preprocess: ColumnTransformer
    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Drop)
        .add_numeric(
            "num_scaled",
            vec![0, 1, 2, 3], // Age, Fare, SibSp, Parch
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_categorical(
            "cat_ohe",
            vec![0, 1, 2], // Sex, Pclass, Embarked
            CategoricalTransformerKind::OneHotEncoder(
                OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore),
            ),
        );
        
    let x = ct.fit_transform(&table)?;
    println!("Feature matrix after preprocessing: {} × {}", x.nrows(), x.ncols());

    // 3. Split
    let (x_tr, x_te, y_tr, y_te) = TrainTestSplit::new()
        .with_test_size(0.2)
        .with_shuffle(true)
        .with_random_state(42)
        .split(&x, &y)?;

    // 4. Train Logistic Regression
    let mut model = LogisticRegression::new().with_solver(LogisticSolver::Svd);
    model.fit(&x_tr, &y_tr)?;

    // 5. Predict & Evaluate
    let preds = model.predict(&x_te)?;
    let acc = accuracy_score(&y_te, &preds)?;
    println!("Test Accuracy: {:.2}%", acc * 100.0);

    Ok(())
}
