//! Categorical inspection: type inference, cardinality, top values, and the
//! imbalance ratio — the share of the most frequent value.
//!
//! Run with: `cargo run --example categorical_inspection -p datarust-profile`
//!
//! No features required.

use datarust::StrMatrix;
use datarust_profile::profile_str_matrix;

fn main() -> datarust_profile::Result<()> {
    // A string table mixing numeric-looking and genuinely categorical columns.
    // - "age":       every non-empty cell parses as f64 → inferred numeric
    // - "city":      categorical, moderately balanced
    // - "tier":      categorical, heavily imbalanced ("basic" dominates)
    // - "signup":    a date-like column that does NOT parse as f64 → categorical
    let s = StrMatrix::from_strings(vec![
        vec!["25", "Istanbul", "basic", "2024-01-15"],
        vec!["40", "Ankara", "premium", "2023-11-03"],
        vec!["31", "Izmir", "basic", "2024-02-20"],
        vec!["28", "Istanbul", "basic", "NA"], // missing date
        vec!["52", "Istanbul", "basic", "2024-03-08"],
        vec!["37", "Ankara", "basic", "2024-01-27"],
        vec!["44", "Izmir", "premium", "2024-04-01"],
        vec!["29", "Istanbul", "basic", "2024-02-11"],
    ])?;

    let names: Vec<String> = ["age", "city", "tier", "signup"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let p = profile_str_matrix(&s, Some(&names))?;

    println!("Inferred {} rows × {} columns\n", p.n_rows, p.n_columns);

    for col in &p.columns {
        println!("── {} ──", col.name);
        println!("  type:    {}", col.column_type);
        println!(
            "  missing: {} ({:.0}%)",
            col.missing_count,
            col.missing_fraction * 100.0
        );

        if let Some(n) = &col.numeric {
            // Inferred-numeric column (age): show the numeric summary briefly.
            println!("  mean:   {:.1}", n.mean);
            println!("  range:  [{:.0}, {:.0}]", n.five.min, n.five.max);
        }

        if let Some(c) = &col.categorical {
            println!("  unique: {}", c.unique);
            println!(
                "  top:    {:?} ({} rows, {:.0}% of data)",
                c.top,
                c.freq,
                c.imbalance_ratio * 100.0
            );

            // The imbalance ratio is the share of the top value. Values near
            // 1.0 mean a single value dominates — a data-quality smell for
            // features, and exactly what the Imbalance check flags.
            let verdict = if c.imbalance_ratio >= 0.95 {
                "⚠  dominated by one value"
            } else if c.imbalance_ratio >= 0.7 {
                "·  skewed toward the top value"
            } else {
                "✓  reasonably balanced"
            };
            println!("  balance: {}", verdict);

            // Top values give a frequency table for charts / quick inspection.
            println!("  top values:");
            for (value, count) in &c.top_values {
                println!("    {:<12} ×{}", value, count);
            }
        }
        println!();
    }

    Ok(())
}
