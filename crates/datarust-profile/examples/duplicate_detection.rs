//! Duplicate-row detection: how exact duplicates are counted across numeric,
//! categorical, and mixed tables, and how the fraction drives the
//! `DuplicateRows` quality finding.
//!
//! Run with: `cargo run --example duplicate_detection -p datarust-profile`
//!
//! No features required.

use datarust::{Matrix, StrMatrix};
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;
use datarust_profile::{profile_matrix, profile_str_matrix, profile_table};

fn main() -> datarust_profile::Result<()> {
    // ── Numeric: two exact duplicates out of six rows. ────────────────────
    let clean = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
    ])?;

    let messy = Matrix::from_rows(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![1.0, 10.0], // dup of row 0
        vec![3.0, 30.0],
        vec![2.0, 20.0], // dup of row 1
        vec![4.0, 40.0],
    ])?;

    println!("── numeric Matrix ──");
    report("clean", &profile_matrix(&clean, None)?);
    report("messy", &profile_matrix(&messy, None)?);

    // ── Categorical: duplicates carry across string equality. ─────────────
    let s = StrMatrix::from_strings(vec![
        vec!["a", "x"],
        vec!["b", "y"],
        vec!["a", "x"], // dup
        vec!["c", "z"],
    ])?;

    println!("\n── string StrMatrix ──");
    report("strings", &profile_str_matrix(&s, None)?);

    // ── Mixed: a row is a duplicate only if BOTH blocks match. ────────────
    let num = Matrix::from_rows(vec![vec![1.0], vec![1.0], vec![1.0]])?;
    let cat = StrMatrix::from_strings(vec![vec!["x"], vec!["y"], vec!["x"]])?;

    println!("\n── mixed table (both blocks must match for a duplicate) ──");
    let p = profile_table(Some(&num), Some(&cat), &["n".into(), "c".into()])?;
    report("mixed", &p);
    println!("  → rows 0 and 2 match on BOTH numeric (1) and categorical (\"x\"),");
    println!("    so they ARE duplicates. A match on one block alone is not enough.");

    // ── Quality finding at the dataset level. ─────────────────────────────
    println!("\n── DuplicateRows finding (messy table, default threshold) ──");
    for issue in run_checks(&profile_matrix(&messy, None)?, &Thresholds::default()) {
        if matches!(issue.kind, datarust_profile::QualityKind::DuplicateRows) {
            println!("  [{:?}] {}", issue.severity, issue.message);
        }
    }

    Ok(())
}

fn report(label: &str, p: &datarust_profile::DatasetProfile) {
    println!(
        "  {:<8} {} rows, {} duplicates ({:.1}%)",
        label,
        p.n_rows,
        p.duplicate_rows,
        p.duplicate_fraction * 100.0
    );
}
