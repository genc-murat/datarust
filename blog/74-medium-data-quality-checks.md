# Your Data Has Problems You Don't See

*One function call finds missing values, outliers, duplicates, and constant columns — before they break your model*

---

I generated 203 customer records with ages, incomes, cities, and plans. Then I ran `datarust-profile` on them. It found 8 quality issues in the first pass.

A column with 15% missing values. Another with 12 extreme outliers. A third that was all zeros — present in every row, contributing nothing. Three duplicate rows. A near-unique identifier that inflates cardinality. And two categorical columns so dominated by one value they carry almost no signal.

Every one of these problems would silently degrade a model trained on this data.

## The Setup

A synthetic customer dataset with 203 rows and 10 columns — a mix of numeric fields (age, income, transactions) and categorical fields (city, plan, churn, user IDs). Then I injected the problems:

| Issue | Column | Severity |
|---|---|---|
| 15% missing values | age | High |
| 9% missing values | income | Moderate |
| 12 extreme outliers | income | High |
| Constant (all zeros) | zero_flag | Critical |
| Near-unique (200/203 values) | user_id_str | High |
| 3 duplicate rows | (dataset) | Moderate |
| 60% dominated by "Istanbul" | city | Moderate |
| 79% dominated by "no" | churn | Moderate |

## Step 1: Profile the Dataset

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table, ColumnType};

let profile = profile_table(
    Some(&numeric_block),   // age, income, ...
    Some(&cat_block),       // city, plan, ...
    &column_names,
)?;

println!("{} rows × {} columns  ({} bytes)",
    profile.n_rows, profile.n_columns, profile.memory_bytes);
println!("{} duplicate rows ({:.1}%)",
    profile.duplicate_rows, profile.duplicate_fraction * 100.0);
```

Output:

```
Dataset: 203 rows × 10 columns  (29232 bytes in memory)
Duplicate rows: 3 (1.5%)
```

Each column gets its own profile — mean, std, five-number summary, skewness, kurtosis, histogram, outliers for numeric columns; cardinality, top value, imbalance ratio for categorical:

```
           age  numeric  μ=43.6 σ=13.8  [20, 70]  outliers=0  missing=15%
        income  numeric  μ=308K σ=787K  [26K, 4.4M] outliers=12  missing=9%
        ...
          city  categ    unique=5  top="Istanbul" (60%)
         churn  categ    unique=2  top="no" (79%)
   user_id_str  categ    unique=200 top="u-0000" (1%)
```

The missing fraction, the outlier count, the imbalance ratio — these are the numbers that tell you whether a column is usable.

## Step 2: Run Quality Checks

The profile is descriptive. Quality checks are *opinionated* — they apply configurable thresholds and flag columns that cross them.

```rust
use datarust_profile::quality::{checks::run_checks, Thresholds};

let findings = run_checks(&profile, &Thresholds::default());
```

### Default thresholds

```
  [info] Outliers         income    12 outliers (6.5%) beyond IQR fences
  [WARN] ConstantColumn   zero_flag near-zero variance; effectively constant
  [info] NearUnique       user_id_str  200 unique / 203 rows; likely identifier
  [info] DuplicateRows   (dataset)  3 of 203 rows are exact duplicates
```

Four issues found immediately. The constant column (`zero_flag`) gets a `Warning` severity. Outliers, near-unique identifiers, and duplicates get `Info` — these are common enough that a default shouldn't be alarmist.

The default thresholds are conservative:

| Check | Default | Purpose |
|---|---|---|
| `HighMissing` | > 50% missing | Flag only severe missingness |
| `ConstantColumn` | variance < 1e-12 | True constants only |
| `NearUnique` | > 98% unique values | Identifiers like user IDs |
| `Outliers` | > 5% beyond IQR fences | Flag columns with many outliers |
| `Imbalance` | > 95% dominated by one value | Extreme categorical skew |
| `DuplicateRows` | always | Exact duplicates are always suspicious |

### Tuned thresholds

For stricter detection, adjust the thresholds:

```rust
let tuned = Thresholds {
    missing_fraction: 0.08,   // flag > 8% missing
    outlier_fraction: 0.04,   // flag > 4% outliers
    imbalance_ratio: 0.50,    // flag > 50% domination
    ..Thresholds::default()
};
```

Now six issues surface:

```
  [WARN] HighMissing      age       14.8% of values are missing
  [WARN] HighMissing      income     8.9% of values are missing
  [info] Outliers         income    12 outliers (6.5%) beyond IQR fences
  [WARN] ConstantColumn   zero_flag near-zero variance; effectively constant
  [CRIT] Imbalance        city      'Istanbul' covers 60.1% of rows
  [CRIT] Imbalance        churn     'no' covers 78.8% of rows
  [info] NearUnique       user_id_str  200 unique / 203 rows; likely identifier
  [info] DuplicateRows   (dataset)  3 of 203 rows are exact duplicates
```

`Imbalance` now flags city and churn as `Critical`. A column where 60% of values are the same category carries almost no signal for a model — especially if the remaining 40% is split across 4 other categories. The imbalance ratio quantifies what your gut already suspects.

`HighMissing` catches age (15%) and income (9%) — enough missing data that you need to decide whether to impute, drop, or investigate the collection pipeline.

Note what didn't fire: the `spike` column (40% zeros, 60% uniform). A bimodal distribution masquerading as "normal enough" — mean 32.6, std 33.1 — isn't caught by any check. The histogram shows it, but no rule flags it. This is a genuine limitation: profiles describe, they don't diagnose every shape pathology.

## Step 3: Generate a Report

The HTML report is a single self-contained file — inline CSS, no JavaScript, no external dependencies:

```rust
use datarust_profile::report;

std::fs::write("report.html", report::to_html_with(&profile, &findings))?;
```

Each column becomes a card:

- **Numeric cards** show the summary statistics, a CSS mini-histogram (pure divs, no JS), the five-number summary, skewness, kurtosis, and outlier count.
- **Categorical cards** show cardinality, the top value, imbalance ratio, and a frequency bar chart.

Findings appear in a severity-colored list at the top. The whole thing opens in any browser, renders in dark mode, and is 15 KB for a 10-column dataset.

## Putting It All Together

```rust
use datarust::StrMatrix;
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;
use datarust_profile::{profile_str_matrix, report};

fn main() -> datarust_profile::Result<()> {
    // Load your data as a StrMatrix (rows of strings, no header)
    let data = StrMatrix::from_strings(rows)?;
    let profile = profile_str_matrix(&data, Some(&column_names))?;

    // Run quality checks with tuned thresholds
    let tuned = Thresholds {
        missing_fraction: 0.05,
        imbalance_ratio: 0.60,
        ..Thresholds::default()
    };
    let findings = run_checks(&profile, &tuned);

    // Print findings
    for issue in &findings {
        let col = issue.column.as_deref().unwrap_or("(dataset)");
        println!("[{:?}] {}: {}", issue.severity, col, issue.message);
    }

    // Write HTML report
    std::fs::write("report.html", report::to_html_with(&profile, &findings))?;

    Ok(())
}
```

## What a Profile Can't Tell You

The profile and quality checks find structural issues — missing data, outliers, constants, duplicates, imbalance, near-unique identifiers. These are the problems that break models silently.

They don't find semantic issues — a column labeled "income" that actually holds expenses, or a `999` sentinel value that should have been `null`. They don't find distribution drift between training and production. They don't tell you whether your data is *correct* — only whether it's *well-formed*.

But finding the structural problems first saves hours of debugging later. A constant column and a near-unique identifier in the same dataset: one contributes nothing, the other inflates cardinality. Both look fine to a model until you check the numbers.

Every data project starts the same way: you open a file and hope it's clean. With a one-call profile and configurable quality checks, you don't have to hope.

```bash
cargo add datarust-profile
```

```rust
use datarust::StrMatrix;
use datarust_profile::profile_str_matrix;

let profile = profile_str_matrix(&your_data, None)?;
// 8 findings before you write a single feature
```
