//! Numeric distribution profiling: mean/std, five-number summary, skewness,
//! kurtosis, histogram, and IQR outliers.
//!
//! Run with: `cargo run --example numeric_distribution -p datarust-profile`
//!
//! No features required — this reads the profile fields directly.

use datarust::Matrix;
use datarust_profile::profile_matrix;

fn main() -> datarust_profile::Result<()> {
    // A synthetic column with a visible right skew: most values cluster low,
    // one extreme value pulls the tail.
    let m = Matrix::from_rows(vec![
        vec![10.0],
        vec![12.0],
        vec![11.0],
        vec![13.0],
        vec![10.0],
        vec![14.0],
        vec![12.0],
        vec![11.0],
        vec![150.0], // outlier — well above the IQR fence
    ])?;

    let p = profile_matrix(&m, Some(&["reaction_ms".into()]))?;
    let col = &p.columns[0];
    let n = col.numeric.as_ref().expect("numeric column");

    println!(
        "Column: {} ({} rows, {} missing)\n",
        col.name, col.count, col.missing_count
    );

    // Central tendency + spread.
    println!("  mean   = {:.2}", n.mean);
    println!("  std    = {:.2}\n", n.std);

    // Five-number summary (min / Q1 / median / Q3 / max).
    println!("  five-number summary:");
    println!("    min    = {:.1}", n.five.min);
    println!("    Q1     = {:.1}", n.five.q1);
    println!("    median = {:.1}", n.five.median);
    println!("    Q3     = {:.1}", n.five.q3);
    println!("    max    = {:.1}\n", n.five.max);

    // Distributional shape (v0.2).
    //   skew ≈ 0  → symmetric; large positive → right tail (our case).
    //   excess kurtosis ≈ 0 → normal-like; positive → heavy-tailed.
    println!("  shape:");
    println!(
        "    skewness  = {:+.3}  ({})",
        n.skewness,
        describe_skew(n.skewness)
    );
    println!(
        "    kurtosis  = {:+.3}  ({})\n",
        n.kurtosis,
        describe_kurtosis(n.kurtosis)
    );

    // Histogram — equal-width bins, Sturges' rule for the count.
    println!("  histogram ({} bins):", n.histogram.nbins());
    let max_count = n.histogram.max_count().max(1);
    for (i, &count) in n.histogram.counts.iter().enumerate() {
        let lo = n.histogram.edges.get(i).copied().unwrap_or(f64::NAN);
        let hi = n.histogram.edges.get(i + 1).copied().unwrap_or(f64::NAN);
        let bar_len = (count as f64 / max_count as f64 * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    [{:>7.1}, {:>7.1})  {:>3}  {}", lo, hi, count, bar);
    }
    println!();

    // Outliers — values beyond the Tukey IQR fences.
    println!("  outliers (IQR rule):");
    println!(
        "    {} found ({:.1}% of values)\n",
        n.outlier_count,
        n.outlier_fraction * 100.0
    );

    Ok(())
}

fn describe_skew(s: f64) -> &'static str {
    if s > 0.5 {
        "right-skewed (long upper tail)"
    } else if s < -0.5 {
        "left-skewed (long lower tail)"
    } else {
        "roughly symmetric"
    }
}

fn describe_kurtosis(k: f64) -> &'static str {
    if k > 1.0 {
        "leptokurtic (heavy-tailed, peaked)"
    } else if k < -1.0 {
        "platykurtic (light-tailed, flat)"
    } else {
        "mesokurtic (near-normal tails)"
    }
}
