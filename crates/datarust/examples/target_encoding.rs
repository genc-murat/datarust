//! Supervised target encoding with ColumnTransformer, demonstrating
//! TargetEncoder and fit_with_target.
//!
//! Run: `cargo run --example target_encoding`

use datarust::compose::{ColumnTransformer, Table};
use datarust::encoder::{OneHotEncoder, TargetEncoder};
use datarust::imputer::ImputeStrategy;
use datarust::imputer::SimpleImputer;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::FeatureNames;
use datarust::TargetTransformerKind;
use datarust::{Matrix, StrMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build dataset ───────────────────────────────────────────────
    // Numeric: age, income (with missing value)
    let numeric = Matrix::new(vec![
        vec![25.0, 50000.0],
        vec![30.0, 60000.0],
        vec![35.0, f64::NAN],
        vec![40.0, 80000.0],
        vec![45.0, 90000.0],
    ])?;

    // Categorical: city, department
    let categorical = StrMatrix::from_strings(vec![
        vec!["Istanbul", "Engineering"],
        vec!["Ankara", "Sales"],
        vec!["Izmir", "Engineering"],
        vec!["Istanbul", "Sales"],
        vec!["Ankara", "Engineering"],
    ])?;

    // Target: house price
    let y = vec![300000.0, 250000.0, 350000.0, 275000.0, 400000.0];

    let table = Table::new(numeric, categorical)?;

    // ── 2. Build ColumnTransformer with target spec ────────────────────
    let mut ct = ColumnTransformer::new()
        // Impute missing income (col 1)
        .add_numeric(
            "income_imputed",
            vec![1],
            TransformerKind::SimpleImputer(SimpleImputer::new(ImputeStrategy::Mean)),
        )
        // Scale age (col 0)
        .add_numeric(
            "age_scaled",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        // One-hot encode city (col 0 of categorical)
        .add_categorical(
            "city_encoded",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        )
        // Target-encode department (col 1 of categorical) — needs fit_with_target
        .add_target(
            "dept_te",
            vec![1],
            TargetTransformerKind::TargetEncoder(TargetEncoder::new(5.0)?),
        );

    // fit() would error on Target specs — must use fit_with_target()
    ct.fit_with_target(&table, &y)?;

    // ── 3. Transform ──────────────────────────────────────────────────
    let transformed = ct.transform(&table)?;
    println!("=== Target Encoder + ColumnTransformer ===");
    println!(
        "Shape: {} rows × {} cols",
        transformed.nrows(),
        transformed.ncols()
    );
    for i in 0..transformed.nrows() {
        let row: Vec<String> = transformed
            .row(i)
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect();
        println!("  row {}: [{}]", i, row.join(", "));
    }

    // ── 4. Feature names ─────────────────────────────────────────────
    let names = ct.feature_names_out(Some(&["age".into(), "income".into()]));
    println!("\nFeature names: {:?}", names);

    Ok(())
}
