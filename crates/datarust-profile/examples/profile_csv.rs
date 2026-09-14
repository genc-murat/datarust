//! One-call CSV profiling: read a CSV file, profile it, and render an HTML
//! report plus quality findings — the `csv` feature in six lines.
//!
//! Run with: `cargo run --example profile_csv -p datarust-profile --features csv`

use datarust_profile::profile::DatasetProfile;
use datarust_profile::quality::Thresholds;
use datarust_profile::report;

fn main() -> datarust_profile::Result<()> {
    // Write a small demo CSV so the example is self-contained.
    let dir = std::env::temp_dir().join("datarust-profile-example");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("customers.csv");
    std::fs::write(
        &path,
        "age,income,city,plan\n\
         34,58000,Istanbul,pro\n\
         45,92000,Ankara,basic\n\
         29,NA,Izmir,basic\n\
         52,110000,Istanbul,pro\n\
         41,67000,Bursa,basic\n\
         38,74000,Ankara,pro\n",
    )
    .map_err(datarust_profile::ProfileError::Io)?;

    // ── One call ───────────────────────────────────────────────────────
    let profile = DatasetProfile::from_csv(&path)?;

    println!(
        "═ {} rows × {} columns ({} duplicate rows)",
        profile.n_rows, profile.n_columns, profile.duplicate_rows
    );
    for col in &profile.columns {
        let detail = match (&col.numeric, &col.categorical) {
            (Some(n), _) => format!(
                "μ={:.1} σ={:.1} [{:.0}, {:.0}]",
                n.mean, n.std, n.five.min, n.five.max
            ),
            (_, Some(c)) => format!("{} unique, top {:?}", c.unique, c.top),
            _ => "(all missing)".into(),
        };
        println!(
            "  {:<8} {:<11} missing={:.0}%  {}",
            col.name,
            col.column_type,
            col.missing_fraction * 100.0,
            detail
        );
    }

    let findings = datarust_profile::run_checks(&profile, &Thresholds::default());
    println!("\nfindings:");
    for issue in &findings {
        println!("  [{:?}] {}", issue.severity, issue.message);
    }

    let html = report::to_html(&profile);
    let out = dir.join("customers.html");
    std::fs::write(&out, html)?;
    println!("\n→ {}", out.display());
    Ok(())
}
