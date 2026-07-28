//! Custom reports: bring your own findings to `to_html_with`, and write JSON
//! straight to a file with `to_json_file`.
//!
//! The built-in `run_checks` uses opinionated thresholds. This example shows
//! how to filter, augment, or replace those findings before rendering — e.g.
//! to surface only the Critical/Warning items, or to add a custom note.
//!
//! Run with: `cargo run --example custom_report --features serde -p datarust-profile`

use datarust::Matrix;
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::{QualityIssue, QualityKind, Thresholds};
use datarust_profile::report;
use datarust_profile::types::ColumnType;
use datarust_profile::{profile_matrix, Severity};

fn main() -> datarust_profile::Result<()> {
    // Two numeric columns: one clean, one with an extreme outlier that
    // drags the mean far above the median.
    let m = Matrix::from_rows(vec![
        vec![10.0, 100.0],
        vec![12.0, 105.0],
        vec![11.0, 102.0],
        vec![13.0, 98.0],
        vec![10.0, 101.0],
        vec![14.0, 50000.0], // extreme outlier in column 1
    ])?;

    let p = profile_matrix(&m, Some(&["latency_ms".into(), "bytes".into()]))?;

    // ── 1. Built-in findings with default thresholds. ─────────────────────
    let builtin = run_checks(&p, &Thresholds::default());
    println!("── built-in findings ({}) ──", builtin.len());
    for issue in &builtin {
        println!("  [{:?}] {}", issue.severity, issue.message);
    }

    // ── 2. Filtered: keep only warnings and criticals. ────────────────────
    let serious: Vec<QualityIssue> = builtin
        .iter()
        .filter(|i| matches!(i.severity, Severity::Critical | Severity::Warning))
        .cloned()
        .collect();
    println!("\n── filtered to Warning/Critical ({}) ──", serious.len());

    // ── 3. Augmented: add a custom domain-specific finding. ───────────────
    let mut custom = serious.clone();
    // A hand-rolled finding the built-in checks don't cover: flag a numeric
    // column whose max sits far above its Q3 — a quick proxy for an extreme
    // upper tail that the IQR outlier count alone under-reports.
    for col in &p.columns {
        if col.column_type != ColumnType::Numeric {
            continue;
        }
        if let Some(n) = &col.numeric {
            let iqr = n.five.q3 - n.five.q1;
            if iqr > 0.0 {
                let reach = (n.five.max - n.five.q3) / iqr;
                if reach > 3.0 {
                    custom.push(QualityIssue {
                        kind: QualityKind::Outliers, // reuse an existing kind
                        severity: Severity::Warning,
                        column: Some(col.name.clone()),
                        message: format!(
                            "{}: max reaches {:.1}×IQR above Q3 — extreme upper tail",
                            col.name, reach
                        ),
                    });
                }
            }
        }
    }
    println!(
        "\n── augmented with custom mean-vs-median check ({}) ──",
        custom.len()
    );
    for issue in &custom {
        println!(
            "  [{:?}] {}: {}",
            issue.severity,
            issue.column.as_deref().unwrap_or("-"),
            issue.message
        );
    }

    // ── 4. Render: HTML with the custom finding set, JSON to a file. ──────
    let html = report::to_html_with(&p, &custom);
    std::fs::write("custom_report.html", &html)?;
    println!("\n  → custom_report.html ({} bytes)", html.len());

    let json_report = report::JsonReport {
        profile: &p,
        quality: custom.clone(),
    };
    report::to_json_file(&json_report, "custom_report.json")?;
    println!("  → custom_report.json");

    Ok(())
}
