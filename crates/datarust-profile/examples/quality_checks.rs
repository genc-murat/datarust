//! Data-quality checks: configure thresholds, run all six checks, and print
//! each finding with its severity.
//!
//! Run with: `cargo run --example quality_checks -p datarust-profile`
//!
//! No features required.

use datarust::{Matrix, StrMatrix};
use datarust_profile::profile_table;
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;
use datarust_profile::Severity;

fn main() -> datarust_profile::Result<()> {
    // A deliberately messy table that triggers several quality flags:
    //   numeric block: [constant, gaps, leak]
    //     - "constant": zero-variance numeric column (ConstantColumn)
    //     - "gaps":     high-missingness numeric column — 6/9 missing (HighMissing)
    //     - "leak":     a clean column with one injected outlier (Outliers)
    //   categorical block: [user_id, tier]
    //     - "user_id":  unique per row → cardinality ≈ rows (NearUnique)
    //     - "tier":     dominated by "basic" — 9/10 = 0.9 (Imbalance)
    //   plus a duplicated row (DuplicateRows)
    let numeric = Matrix::from_rows(vec![
        vec![5.0, f64::NAN, 10.0],
        vec![5.0, f64::NAN, 11.0],
        vec![5.0, f64::NAN, 10.5],
        vec![5.0, f64::NAN, 10.2],
        vec![5.0, f64::NAN, 9.8],
        vec![5.0, 4.0, 10.1],
        vec![5.0, 5.0, 10.3],
        vec![5.0, 6.0, 10.0],
        vec![5.0, 7.0, 10.4],
        vec![5.0, 4.0, 10.1],  // duplicate of row index 5
        vec![5.0, 8.0, 999.0], // outlier in "leak"
    ])?;

    let categorical = StrMatrix::from_strings(vec![
        vec!["user-01", "basic"],
        vec!["user-02", "basic"],
        vec!["user-03", "basic"],
        vec!["user-04", "basic"],
        vec!["user-05", "basic"],
        vec!["user-06", "basic"],
        vec!["user-07", "basic"],
        vec!["user-08", "basic"],
        vec!["user-09", "basic"],
        vec!["user-06", "basic"], // duplicate row → same user_id
        vec!["user-10", "premium"],
    ])?;

    let names = vec![
        "constant".into(),
        "gaps".into(),
        "leak".into(),
        "user_id".into(),
        "tier".into(),
    ];

    let profile = profile_table(Some(&numeric), Some(&categorical), &names)?;

    println!(
        "Dataset: {} rows × {} columns\n",
        profile.n_rows, profile.n_columns
    );

    // Run with the default thresholds first.
    println!("── Default thresholds ──");
    print_findings(&run_checks(&profile, &Thresholds::default()));

    // Now tune thresholds to surface findings the defaults miss:
    //   - lower missing_fraction so "gaps" flags despite ~45% missing
    //   - lower outlier_fraction so a single extreme value flags
    //   - lower imbalance_ratio so "tier" flags at 90% domination
    println!("\n── Tuned thresholds (missing 0.4, outlier 0.02, imbalance 0.85) ──");
    let tuned = Thresholds {
        missing_fraction: 0.4,
        outlier_fraction: 0.02,
        imbalance_ratio: 0.85,
        ..Thresholds::default()
    };
    print_findings(&run_checks(&profile, &tuned));

    Ok(())
}

fn print_findings(findings: &[datarust_profile::QualityIssue]) {
    if findings.is_empty() {
        println!("  (no findings)");
        return;
    }
    for issue in findings {
        let scope = issue.column.as_deref().unwrap_or("(dataset)");
        let sev = match issue.severity {
            Severity::Critical => "CRIT",
            Severity::Warning => "WARN",
            Severity::Info => "info",
        };
        let kind = format!("{:?}", issue.kind);
        println!("  [{sev}] {kind:<16} {scope:<10}  {}", issue.message);
    }
}
