//! Missing-value handling: how NaN, empty strings, and NA markers are
//! recognised, counted, and flagged.
//!
//! Two parallel views of the same data:
//!   - a numeric `Matrix`, where missing = non-finite (NaN / ±inf)
//!   - a string `StrMatrix`, where missing = recognised text markers
//!
//! Run with: `cargo run --example missing_values -p datarust-profile`
//!
//! No features required.

use datarust::{Matrix, StrMatrix};
use datarust_profile::infer;
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;
use datarust_profile::{profile_matrix, profile_str_matrix};

fn main() -> datarust_profile::Result<()> {
    // ── Numeric view: NaN is the only missing signal. ─────────────────────
    // Column "temp" has 3 missing of 8 rows (37.5%).
    let m = Matrix::from_rows(vec![
        vec![21.0],
        vec![22.0],
        vec![f64::NAN],
        vec![23.0],
        vec![f64::NAN],
        vec![24.0],
        vec![22.5],
        vec![f64::NAN],
    ])?;

    let p = profile_matrix(&m, Some(&["temp".into()]))?;
    let col = &p.columns[0];
    println!("── numeric Matrix ──");
    println!(
        "  {} rows, {} missing ({:.1}%)",
        col.count,
        col.missing_count,
        col.missing_fraction * 100.0
    );
    let n = col.numeric.as_ref().unwrap();
    println!("  mean over present = {:.2} (NaN excluded)\n", n.mean);

    // ── String view: many markers count as missing. ───────────────────────
    // The infer module recognises "", "NA", "N/A", "null", "NaN", "None",
    // "-", "?" (case-insensitive for the alphabetic forms).
    let s = StrMatrix::from_strings(vec![
        vec!["21"],
        vec!["22"],
        vec![""], // empty
        vec!["23"],
        vec!["N/A"], // explicit marker
        vec!["24"],
        vec!["null"], // explicit marker
        vec!["22.5"],
        vec!["NA"], // explicit marker
    ])?;

    let p = profile_str_matrix(&s, Some(&["temp".into()]))?;
    let col = &p.columns[0];
    println!("── string StrMatrix (same data, textual markers) ──");
    println!(
        "  {} rows, {} missing ({:.1}%)",
        col.count,
        col.missing_count,
        col.missing_fraction * 100.0
    );

    // ── Which markers does is_missing recognise? ──────────────────────────
    println!("\n── recognised missing markers ──");
    for candidate in [
        "", "  ", "NA", "n/a", "NULL", "null", "NaN", "None", "-", "?", "missing",
    ] {
        println!(
            "  {:<10} → {}",
            format!("{:?}", candidate),
            infer::is_missing(candidate)
        );
    }

    // ── HighMissing quality check ─────────────────────────────────────────
    // Lower the threshold so a 37.5% missing column flags as a warning.
    println!("\n── quality check (threshold lowered to 0.3) ──");
    let tuned = Thresholds {
        missing_fraction: 0.3,
        ..Thresholds::default()
    };
    for issue in run_checks(&p, &tuned) {
        println!("  [{:?}] {}", issue.kind, issue.message);
    }

    Ok(())
}
