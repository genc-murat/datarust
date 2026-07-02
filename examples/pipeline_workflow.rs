//! Pipeline workflow: chain transformers and test inverse-transform round-trip.
//!
//! Run: `cargo run --example pipeline_workflow`

use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::OneHotEncoder;
use datarust::pipeline::Pipeline;
use datarust::scaler::{MinMaxScaler, StandardScaler};
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::FeatureNames;
use datarust::Transformer;
use datarust::{Matrix, StrMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Numeric-only pipeline ──────────────────────────────────────
    let x = Matrix::new(vec![
        vec![1.0, 500.0],
        vec![2.0, 300.0],
        vec![3.0, 800.0],
        vec![4.0, 100.0],
    ])?;

    let mut pipeline = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .push("norm", TransformerKind::MinMaxScaler(MinMaxScaler::new()));

    let transformed = pipeline.fit_transform(&x)?;
    println!("=== Pipeline (StandardScaler → MinMaxScaler) ===");
    println!("Input:  {}×{}", x.nrows(), x.ncols());
    println!("Output: {}×{}", transformed.nrows(), transformed.ncols());
    for i in 0..transformed.nrows() {
        let row: Vec<String> = transformed
            .row(i)
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect();
        println!("  [{}]", row.join(", "));
    }
    println!();

    // ── 2. ColumnTransformer with categorical data ────────────────────
    let numeric = Matrix::new(vec![
        vec![10.0, 1000.0],
        vec![20.0, 2000.0],
        vec![30.0, 3000.0],
    ])?;
    let categorical = StrMatrix::from_column(["X", "Y", "Z"])?;
    let table = Table::new(numeric, categorical)?;

    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Passthrough)
        .add_numeric(
            "scaled",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_categorical(
            "cat",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        );

    let out = ct.fit_transform(&table)?;
    println!("=== ColumnTransformer (passthrough) ===");
    println!("Output: {} rows × {} cols", out.nrows(), out.ncols());
    for i in 0..out.nrows() {
        let row: Vec<String> = out.row(i).iter().map(|v| format!("{:.4}", v)).collect();
        println!("  [{}]", row.join(", "));
    }

    // Feature names
    let names = ct.feature_names_out(None);
    println!("\nFeature names: {:?}", names);

    // ── 3. Multi-output (table) transform ────────────────────────────
    let output = ct.fit_transform_to_table(&table)?;
    println!("\n=== Multi-output ===");
    println!(
        "Numeric: {}×{} | Categorical: {}×{}",
        output.numeric.nrows(),
        output.numeric.ncols(),
        output.categorical.nrows(),
        output.categorical.ncols(),
    );

    Ok(())
}
