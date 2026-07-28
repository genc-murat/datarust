//! End-to-end example: profile a mixed table, write JSON + HTML reports.
//!
//! Run with: `cargo run --example profile_table --features serde`

use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table, report};

fn main() -> datarust_profile::Result<()> {
    // Numeric block: age and income, with one missing income (NaN).
    let numeric = Matrix::from_rows(vec![
        vec![25.0, 45_000.0],
        vec![40.0, 82_000.0],
        vec![31.0, f64::NAN],
        vec![40.0, 82_000.0], // duplicate of row 1
    ])?;

    // Categorical block: city and subscription tier.
    let categorical = StrMatrix::from_strings(vec![
        vec!["Istanbul", "basic"],
        vec!["Ankara", "premium"],
        vec!["Izmir", "basic"],
        vec!["Ankara", "premium"],
    ])?;

    let names = vec![
        "age".to_string(),
        "income".to_string(),
        "city".to_string(),
        "tier".to_string(),
    ];

    let profile = profile_table(Some(&numeric), Some(&categorical), &names)?;

    println!(
        "Profiled {} rows x {} columns (~{})",
        profile.n_rows, profile.n_columns, profile.memory_bytes
    );
    println!(
        "Duplicate rows: {} ({:.1}%)",
        profile.duplicate_rows,
        profile.duplicate_fraction * 100.0
    );
    for col in &profile.columns {
        println!(
            "  {:<8} {:<11} missing={:.0}%",
            col.name,
            col.column_type,
            col.missing_fraction * 100.0
        );
    }

    let json = report::to_json(&report::JsonReport::from_profile(&profile))?;
    std::fs::write("profile.json", json)?;
    println!("Wrote profile.json");

    let html = report::to_html(&profile);
    std::fs::write("profile.html", html)?;
    println!("Wrote profile.html");

    Ok(())
}
