//! End-to-end preprocessing pipeline with datarust.
//!
//! Run: `cargo run --example basic_preprocessing`
//! (optionally with `--features serde,rayon` for serialization and parallelism)

use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::{HandleUnknown, OneHotEncoder};
use datarust::imputer::{ImputeStrategy, SimpleImputer};
use datarust::scaler::{MinMaxScaler, StandardScaler};
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::FeatureNames;
use datarust::{Matrix, StrMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build a mixed-type dataset ──────────────────────────────────
    // Numeric: age, salary, kids
    let numeric = Matrix::new(vec![
        vec![25.0, 50_000.0, 0.0],
        vec![30.0, 60_000.0, 2.0],
        vec![35.0, 70_000.0, 1.0],
        vec![40.0, f64::NAN, 3.0],
    ])?;
    // Categorical: city, education
    let categorical = StrMatrix::from_strings(vec![
        vec!["Istanbul", "Bachelor"],
        vec!["Ankara", "Master"],
        vec!["Izmir", "PhD"],
        vec!["Istanbul", "Bachelor"],
    ])?;
    let table = Table::new(numeric, categorical)?;

    // ── 2. Build a ColumnTransformer ───────────────────────────────────
    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Drop)
        // Impute missing salary (col 1) with the mean
        .add_numeric(
            "salary_imputed",
            vec![1],
            TransformerKind::SimpleImputer(SimpleImputer::new(ImputeStrategy::Mean)),
        )
        // Scale age (col 0) with StandardScaler
        .add_numeric(
            "age_scaled",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        // Scale kids (col 2) with MinMaxScaler
        .add_numeric(
            "kids_scaled",
            vec![2],
            TransformerKind::MinMaxScaler(MinMaxScaler::new()),
        )
        // One-hot encode city (col 0 of categorical)
        .add_categorical(
            "city_encoded",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(
                OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore),
            ),
        );

    // ── 3. Fit & transform ────────────────────────────────────────────
    let result = ct.fit_transform(&table)?;

    println!("=== Preprocessing Result ===");
    println!("Shape: {} rows × {} cols", result.nrows(), result.ncols());
    println!("Data:");
    for i in 0..result.nrows() {
        let row: Vec<String> = result.row(i).iter().map(|v| format!("{:.4}", v)).collect();
        println!("  row {}: [{}]", i, row.join(", "));
    }

    // ── 4. Feature names ──────────────────────────────────────────────
    let names = ct.feature_names_out(Some(&["age".into(), "salary".into(), "kids".into()]));
    println!("\nFeature names: {:?}", names);

    // ── 5. Serialization (only with `--features serde`) ──────────────
    #[cfg(feature = "serde")]
    {
        let json = datarust::serialize::to_json(&ct)?;
        println!("\n=== Serialized (JSON preview) ===");
        println!("{}", &json[..json.len().min(500)]);
        let restored: ColumnTransformer = datarust::serialize::from_json(&json)?;
        let re = restored.transform(&table)?;
        assert_eq!(re, result);
        println!("Round-trip serialization: OK");
    }

    #[cfg(not(feature = "serde"))]
    println!("\n(serde feature not enabled — skipping serialization demo)");

    Ok(())
}
