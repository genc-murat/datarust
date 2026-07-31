//! Example demonstrating relationships & interaction analysis (Pearson correlation, Cramér's V, point-biserial, target leakage).

use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table_with_target, report, run_checks, Thresholds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construct synthetic dataset with collinear features and a binary target
    let numeric = Matrix::from_rows(vec![
        vec![1.0, 2.0, 10.0],
        vec![2.0, 4.1, 20.0],
        vec![3.0, 6.0, 15.0],
        vec![4.0, 8.0, 25.0],
        vec![5.0, 10.2, 30.0],
        vec![6.0, 12.0, 12.0],
    ])?;

    let categorical = StrMatrix::from_strings(vec![
        vec!["cat_A".to_string(), "yes".to_string()],
        vec!["cat_A".to_string(), "yes".to_string()],
        vec!["cat_B".to_string(), "no".to_string()],
        vec!["cat_B".to_string(), "no".to_string()],
        vec!["cat_B".to_string(), "no".to_string()],
        vec!["cat_A".to_string(), "yes".to_string()],
    ])?;

    let names = vec![
        "feature_x".to_string(),
        "feature_x_scaled".to_string(),
        "feature_y".to_string(),
        "group".to_string(),
        "target".to_string(),
    ];

    let profile = profile_table_with_target(Some(&numeric), Some(&categorical), &names, "target")?;

    println!(
        "Dataset: {} rows x {} columns",
        profile.n_rows, profile.n_columns
    );

    if let Some(rels) = &profile.relationships {
        if let Some(p) = &rels.pearson {
            println!("\nPearson Correlation Matrix:");
            for (i, name_i) in p.labels.iter().enumerate() {
                for (j, name_j) in p.labels.iter().enumerate() {
                    println!("  {} <-> {}: {:.3}", name_i, name_j, p.values[i][j]);
                }
            }
        }

        if let Some(cv) = &rels.cramers_v {
            println!("\nCramér's V Matrix:");
            for (i, name_i) in cv.labels.iter().enumerate() {
                for (j, name_j) in cv.labels.iter().enumerate() {
                    println!("  {} <-> {}: {:.3}", name_i, name_j, cv.values[i][j]);
                }
            }
        }

        if !rels.point_biserial.is_empty() {
            println!("\nPoint-Biserial Correlations:");
            for entry in &rels.point_biserial {
                println!(
                    "  {} (binary) <-> {}: {:.3}",
                    entry.categorical, entry.numeric, entry.correlation
                );
            }
        }
    }

    let issues = run_checks(&profile, &Thresholds::default());
    println!("\nData Quality Findings ({} issues):", issues.len());
    for issue in &issues {
        println!(
            "  [{}] [{:?}] {}",
            issue.severity, issue.kind, issue.message
        );
    }

    let html = report::to_html_with(&profile, &issues);
    println!("\nGenerated HTML report ({} bytes)", html.len());

    Ok(())
}
