//! Type inference: how `infer_column` decides numeric vs categorical, and how
//! `parse_numeric_column` turns string cells into f64 with missing handling.
//!
//! Useful when you build a typed table yourself or want to understand why a
//! column was classified the way it was.
//!
//! Run with: `cargo run --example type_inference -p datarust-profile`
//!
//! No features required.

use datarust_profile::infer;
use datarust_profile::ColumnType;

fn main() {
    // A handful of columns, each exercising a different inference outcome.
    let columns: &[(&str, &[&str])] = &[
        ("clean numeric", &["1", "2", "3", "4", "5"]),
        ("numeric with missing", &["1", "2", "NA", "4", ""]),
        ("floats", &["1.5", "2.7", "3.14", "-0.001", "1e3"]),
        ("purely categorical", &["red", "green", "blue", "red"]),
        ("mostly numeric, one word", &["1", "2", "3", "four", "5"]),
        ("all missing", &["", "NA", "null", "-"]),
        ("ids", &["user-001", "user-002", "user-003"]),
        ("dates", &["2024-01-01", "2024-01-02", "2024-01-03"]),
    ];

    println!("┌─ column ──────────────────────┬─ inferred type ─┬─ parsed sample ──────────┐");
    println!("├───────────────────────────────┼─────────────────┼──────────────────────────┤");

    for (name, cells) in columns {
        let owned: Vec<String> = cells.iter().map(|s| s.to_string()).collect();
        let kind = infer::infer_column(&owned);

        // For numeric columns, show the parsed values (NaN = missing).
        let sample = match kind {
            ColumnType::Numeric => {
                let parsed = infer::parse_numeric_column(&owned);
                let rendered: Vec<String> = parsed
                    .iter()
                    .map(|v| {
                        if v.is_finite() {
                            format!("{}", v)
                        } else {
                            "NaN".into()
                        }
                    })
                    .collect();
                rendered.join(", ")
            }
            ColumnType::Categorical => "(categorical — not parsed)".into(),
        };

        println!("│ {:<29} │ {:<15} │ {:<24} │", name, kind, sample);
    }
    println!("└───────────────────────────────┴─────────────────┴──────────────────────────┘");

    // The rule, in one line:
    println!("\nrule: a column is Numeric iff every NON-MISSING cell parses as f64.");
    println!("      presence of a single non-parseable cell (\"four\", \"red\", a date)");
    println!("      forces the whole column to Categorical.");
}
