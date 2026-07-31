# Stop Profiling Columns in Isolation: Relationships, Collinearity, and Target Leakage in Pure Rust

*How datarust-profile v0.3 adds Pearson correlations, Cramér's V, point-biserial analysis, target-leakage detection, and correlation heatmaps — with zero dependencies.*

---

Single-column statistics only tell half the story. You can know the mean, median, skewness, and IQR outliers of every column in your table, and still train a model that collapses due to severe multicollinearity — or worse, ship a model with over-optimistic 99% test accuracy because a feature secretly leaked the target label.

In Python, detecting column interactions is easy: `ydata-profiling` builds correlation matrices and heatmaps out of the box. But if you're building a Rust CLI, a WASM web widget, or a microservice, pulling in Python or heavyweight C bindings just to check pairwise column correlations ruins Rust's deployment promise.

That's why we built [**datarust-profile**](https://crates.io/crates/datarust-profile). Following our [v0.2 release](https://datarust.dev/blog/medium-profile-v0.2/) (which added distributional depth, histograms, and IQR outlier detection), **v0.3** introduces pairwise relationship analysis, target-leakage hints, and dependency-free correlation heatmaps — built directly on `datarust::stats` with **zero external dependencies**.

Here is what v0.3 brings to the Rust data ecosystem.

---

## 1. The Three Layers of Pairwise Relationships

Not all column pairs interact the same way. A numeric column vs a numeric column requires linear correlation; two categorical columns require non-parametric association; and a binary category vs a continuous numeric column requires group-mean separation.

`datarust-profile` v0.3 handles all three automatically inside a structured `Relationships` block.

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::profile_table;

let numeric = Matrix::from_rows(vec![
    vec![25.0, 50000.0],
    vec![40.0, 85000.0],
    vec![31.0, 62000.0],
    vec![52.0, 110000.0],
])?;

let categorical = StrMatrix::from_strings(vec![
    vec!["basic", "no"],
    vec!["premium", "yes"],
    vec!["basic", "no"],
    vec!["premium", "yes"],
])?;

let profile = profile_table(
    Some(&numeric),
    Some(&categorical),
    &["age".into(), "income".into(), "tier".into(), "churn".into()],
)?;

if let Some(rels) = &profile.relationships {
    // 1. Pearson matrix over numeric columns
    if let Some(pearson) = &rels.pearson {
        println!("Pearson r(age, income) = {:.3}", pearson.values[0][1]);
    }

    // 2. Cramér's V matrix over categorical columns
    if let Some(cramers) = &rels.cramers_v {
        println!("Cramér's V(tier, churn) = {:.3}", cramers.values[0][1]);
    }

    // 3. Point-biserial correlations (binary categorical ⇄ numeric)
    for pb in &rels.point_biserial {
        println!("Point-biserial r_pb({}, {}) = {:.3}",
            pb.categorical, pb.numeric, pb.correlation);
    }
}
```

### Layer 1: Pearson Correlation Matrix (Numeric ⇄ Numeric)

For numeric columns, `datarust-profile` routes row-major slice data through `datarust::stats::correlation_matrix`. It produces a symmetric $p \times p$ matrix of Pearson $r \in [-1.0, 1.0]$ coefficients.

When any pair of feature columns exhibits $|r| \ge 0.95$, the quality engine raises a `QualityKind::HighCorrelation` finding. High collinearity inflates coefficient variance in linear models and distorts feature importance scores; identifying it early tells you which redundant feature to drop before training.

### Layer 2: Cramér's V Matrix (Categorical ⇄ Categorical)

Numeric correlation doesn't work on categorical strings like `"tier"` (`"basic"` vs `"premium"`) or `"region"` (`"north"` vs `"south"`).

`v0.3` implements a pure-Rust contingency table and Chi-Squared ($\chi^2$) test to calculate **Cramér's V**:

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(k-1, r-1)}}$$

Cramér's V measures nominal association in $[0.0, 1.0]$, where $0.0$ means complete independence and $1.0$ means complete association. Missing markers (`"NA"`, `"null"`, `""`) are safely excluded using `infer::is_missing()`.

### Layer 3: Point-Biserial Correlation (Binary Categorical ⇄ Numeric)

When a categorical column has exactly 2 unique values (e.g. `churn = "yes"` / `"no"` or `gender = "M"` / `"F"`), standard correlation cannot be directly applied. `v0.3` calculates the **Point-Biserial correlation coefficient** $r_{pb}$ between the binary indicator and every numeric column:

$$r_{pb} = \frac{M_1 - M_0}{S_y} \sqrt{\frac{n_0 n_1}{n(n-1)}}$$

where $M_0$ and $M_1$ are the group means, $S_y$ is the sample standard deviation ($ddof=1$) of the combined sample, and $n_0, n_1$ are the group counts. This detects whether continuous measurements differ significantly across binary group memberships.

---

## 2. Target Leakage: Catching the Silent CV Killer

Target leakage is one of the most dangerous bugs in data science. It occurs when a predictor feature incorporates information about the target variable that will not be available when the model is deployed. 

For instance, an `account_closed_date` or `refund_processed_amount` feature will be perfectly correlated with `churn = 1`. In cross-validation, your model achieves 0.99 ROC-AUC, but in production, it fails completely.

`datarust-profile` v0.3 introduces target-aware profiling entry points:

```rust
use datarust_profile::profile_table_with_target;

// Designate "churn" as the target column:
let profile = profile_table_with_target(
    Some(&numeric_matrix),
    Some(&categorical_matrix),
    &["age".into(), "income".into(), "account_closed_flag".into(), "churn".into()],
    "churn",
)?;
```

When a target column is declared, `quality::run_checks` evaluates every feature's correlation or association against the target. If any feature column crosses `thresholds.target_leakage` (default `0.90`), it emits a `Critical` severity finding:

```text
[Critical] account_closed_flag: Suspected target leakage: feature 'account_closed_flag' has strong correlation with target 'churn' (r = 1.000)
```

Catching leakage at the profiling stage — before running pipeline fitting or feature selection — saves hours of wasted debugging.

---

## 3. Configurable Quality Engine: 8 Core Checks

With v0.3, `quality::run_checks` expands from 6 checks to **8 data-quality findings**:

| Check | Scope | Default Threshold | Description |
|---|---|---|---|
| `HighMissing` | Column | `0.50` | Missing fraction $\ge 50\%$ (escalates to Critical at $90\%$). |
| `ConstantColumn` | Numeric | `1e-12` | Variance near zero; column carries no signal. |
| `NearUnique` | Categorical | `0.98` | High cardinality ratio; likely a raw ID column. |
| `DuplicateRows` | Dataset | $>0$ | Exact-duplicate rows detected. |
| `Outliers` | Numeric | `0.05` | Values outside Tukey IQR fences ($Q1 - 1.5\cdot IQR, Q3 + 1.5\cdot IQR$). |
| `Imbalance` | Categorical | `0.95` | Single value covers $\ge 95\%$ of rows. |
| **`HighCorrelation`** | Numeric Pair | `0.95` | Pearson $\|r\| \ge 0.95$; collinear feature redundancy. |
| **`TargetLeakage`** | Feature | `0.90` | Feature-target correlation or Cramér's $V \ge 0.90$. |

All thresholds are fully customizable via `Thresholds`:

```rust
use datarust_profile::quality::{run_checks, Thresholds};

let mut t = Thresholds::default();
t.high_correlation = 0.90; // stricter collinearity threshold
t.target_leakage = 0.85;   // sensitive target-leakage detection

let issues = run_checks(&profile, &t);
```

---

## 4. Pure CSS Correlation Heatmaps in HTML Reports

The `report::to_html(&profile)` generator creates a single self-contained `.html` file. In v0.3, the report adds a **Relationships & Interaction** section featuring:

- **Pearson Heatmap Table**: Colored `<td style="background-color: ...">` cells using a diverging Red ($-1.0$) $\rightarrow$ White ($0.0$) $\rightarrow$ Blue ($+1.0$) palette.
- **Cramér's V Heatmap Table**: Colored cells using a sequential White ($0.0$) $\rightarrow$ Purple ($1.0$) palette.
- **Point-Biserial Table**: Compact table summarizing binary-categorical ⇄ numeric correlations.
- **Full Dark Mode Support**: CSS `@media (prefers-color-scheme: dark)` styling for low-light review.

```
┌─────────────────────────────────────────────────────────────┐
│ Relationships & Interaction                                 │
├─────────────────────────────────────────────────────────────┤
│ Pearson correlation matrix (numeric)                        │
│             age       income     monthly_charges            │
│   age       1.00       0.42           0.08                  │
│   income    0.42       1.00           0.89  [Blue]          │
│   charges   0.08       0.89           1.00                  │
│                                                             │
│ Cramér's V matrix (categorical)                             │
│             tier      region                                │
│   tier      1.00       0.76  [Purple]                       │
│   region    0.76       1.00                                 │
└─────────────────────────────────────────────────────────────┘
```

Because it uses pure CSS inline styles, there are no JavaScript dependencies, no external CDN calls, and no security risks when serving or opening generated reports.

---

## 5. End-to-End Example

Here is how you can use `datarust-profile` v0.3 in your Rust project today:

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table_with_target, report, run_checks, Thresholds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Prepare data
    let numeric = Matrix::from_rows(vec![
        vec![25.0, 50.0],
        vec![40.0, 120.0],
        vec![35.0, 110.0],
        vec![50.0, 180.0],
    ])?;

    let categorical = StrMatrix::from_strings(vec![
        vec!["basic".to_string(), "no".to_string()],
        vec!["premium".to_string(), "yes".to_string()],
        vec!["premium".to_string(), "yes".to_string()],
        vec!["premium".to_string(), "yes".to_string()],
    ])?;

    let names = vec![
        "age".to_string(),
        "charge".to_string(),
        "plan".to_string(),
        "churn".to_string(),
    ];

    // 2. Profile with target hint
    let profile = profile_table_with_target(
        Some(&numeric),
        Some(&categorical),
        &names,
        "churn",
    )?;

    // 3. Evaluate data quality
    let findings = run_checks(&profile, &Thresholds::default());
    println!("Found {} quality issues:", findings.len());
    for issue in &findings {
        println!(" - [{:?}] {}", issue.severity, issue.message);
    }

    // 4. Generate HTML report
    let html = report::to_html_with(&profile, &findings);
    std::fs::write("profile_report.html", html)?;
    println!("Report saved to profile_report.html");

    Ok(())
}
```

---

## What's Next on the Roadmap?

With **v0.3 (Relationships & Interaction)** complete, work begins on **v0.4 (Data Loading & Ergonomics)**:

- **`csv` feature**: One-call profiling straight from CSV file paths (`from_csv("data.csv")`).
- **`cli` binary**: A command-line tool (`datarust-profile data.csv -o report.html`).
- **Streaming single-pass profiling**: Calculate moments (Welford) and quantiles (reservoir sampling / t-digest) in a single pass without loading whole files into RAM.

Check out the full [ROADMAP.md](https://github.com/genc-murat/datarust/blob/main/crates/datarust-profile/ROADMAP.md) to see where we're headed.

---

## Getting Started

Add `datarust-profile` to your `Cargo.toml`:

```toml
[dependencies]
datarust = "0.6"
datarust-profile = "0.3"
```

For JSON support, enable the `serde` feature:

```toml
[dependencies]
datarust-profile = { version = "0.3", features = ["serde"] }
```

- **Crate on crates.io**: [crates.io/crates/datarust-profile](https://crates.io/crates/datarust-profile)
- **API Documentation**: [docs.rs/datarust-profile](https://docs.rs/datarust-profile)
- **GitHub Repository**: [github.com/genc-murat/datarust](https://github.com/genc-murat/datarust)
- **Interactive Web Book**: [datarust.dev/docs/profile/README/](https://datarust.dev/docs/profile/README/)

Give it a try on your datasets, and let us know what you think!
