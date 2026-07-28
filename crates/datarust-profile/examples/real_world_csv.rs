//! A realistic end-to-end profiling run on CSV-like customer data: profile
//! → inspect columns → run quality checks → render HTML + JSON reports.
//!
//! Demonstrates the full workflow a user would follow after loading a CSV
//! (here inlined as string rows, since the `csv` feature is not yet built).
//!
//! Run with: `cargo run --example real_world_csv --features serde -p datarust-profile`

use datarust::StrMatrix;
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;
use datarust_profile::report;
use datarust_profile::{profile_str_matrix, ColumnType};

fn main() -> datarust_profile::Result<()> {
    // Inlined rows as a CSV reader would yield them: strings, with mixed
    // numeric/categorical columns, missing markers, and a duplicate row.
    let rows = vec![
        vec!["customer_id", "age", "income", "city", "plan", "churn"],
        vec!["C001", "34", "58000", "Istanbul", "pro", "no"],
        vec!["C002", "45", "92000", "Ankara", "basic", "no"],
        vec!["C003", "29", "NA", "Izmir", "basic", "yes"],
        vec!["C004", "52", "110000", "Istanbul", "pro", "no"],
        vec!["C005", "41", "67000", "Bursa", "basic", "no"],
        vec!["C006", "38", "74000", "Ankara", "pro", "yes"],
        vec!["C007", "NA", "45000", "Izmir", "basic", "no"],
        vec!["C008", "60", "130000", "Istanbul", "enterprise", "no"],
        vec!["C009", "33", "55000", "NA", "basic", "yes"],
        vec!["C002", "45", "92000", "Ankara", "basic", "no"], // duplicate
    ];

    // Split header (column names) from data rows.
    let header: Vec<String> = rows[0].iter().map(|s| s.to_string()).collect();
    let data: Vec<Vec<String>> = rows[1..]
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();

    let table = StrMatrix::from_strings(data)?;
    let profile = profile_str_matrix(&table, Some(&header))?;

    // ── Dataset overview ──────────────────────────────────────────────────
    println!("═══ Customer data profile ═══");
    println!(
        "  {} rows × {} columns  (≈{} in memory)",
        profile.n_rows,
        profile.n_columns,
        human_bytes(profile.memory_bytes)
    );
    println!(
        "  {} duplicate rows ({:.1}%)\n",
        profile.duplicate_rows,
        profile.duplicate_fraction * 100.0
    );

    // ── Per-column digest ─────────────────────────────────────────────────
    println!("── columns ──");
    for col in &profile.columns {
        let kind = match col.column_type {
            ColumnType::Numeric => {
                if let Some(n) = &col.numeric {
                    format!(
                        "numeric  μ={:.0} σ={:.0} [{:.0}, {:.0}] outliers={}",
                        n.mean, n.std, n.five.min, n.five.max, n.outlier_count
                    )
                } else {
                    "numeric (all missing)".into()
                }
            }
            ColumnType::Categorical => {
                if let Some(c) = &col.categorical {
                    format!(
                        "categ    unique={} top={:?} ({:.0}%)",
                        c.unique,
                        c.top,
                        c.imbalance_ratio * 100.0
                    )
                } else {
                    "categorical (all missing)".into()
                }
            }
        };
        println!(
            "  {:<13} {:<9} missing={:>2} ({:>4.1}%)  {}",
            col.name,
            col.column_type,
            col.missing_count,
            col.missing_fraction * 100.0,
            kind
        );
    }

    // ── Quality findings ──────────────────────────────────────────────────
    println!("\n── quality findings (default thresholds) ──");
    let findings = run_checks(&profile, &Thresholds::default());
    if findings.is_empty() {
        println!("  (none)");
    } else {
        for issue in &findings {
            let scope = issue.column.as_deref().unwrap_or("dataset");
            println!("  [{:?}] {}: {}", issue.severity, scope, issue.message);
        }
    }

    // ── Reports ───────────────────────────────────────────────────────────
    // JSON carries both the profile and the findings; HTML renders a
    // self-contained card-grid report. Both are written to disk.
    let json = report::to_json(&report::JsonReport::from_profile(&profile))?;
    std::fs::write("customer_profile.json", &json)?;
    println!("\n  → customer_profile.json ({} bytes)", json.len());

    let html = report::to_html(&profile);
    std::fs::write("customer_profile.html", &html)?;
    println!("  → customer_profile.html ({} bytes)", html.len());

    Ok(())
}

fn human_bytes(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1} MiB", n as f64 / 1_048_576.0)
    } else if n >= 1_000 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{} B", n)
    }
}
