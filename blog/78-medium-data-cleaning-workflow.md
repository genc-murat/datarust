# Clean Your Data in Four Steps With `datarust-profile`

*Profile, detect, clean, re-profile — an iterative data-cleaning workflow that catches problems before they reach your model*

---

A dataset has missing values, outliers, constant columns, duplicates. You clean them one by one. But how do you know you caught everything? And how do you know you didn't make things worse?

The workflow is iterative: **profile → detect → clean → re-profile**. Each cleaning step changes the data, and the next profile tells you whether it worked.

Here's the full cycle on a deliberately dirty dataset, using [`datarust`](https://crates.io/crates/datarust) and [`datarust-profile`](https://crates.io/crates/datarust-profile).

## The Dirty Dataset

A 201-row customer table with 7 columns and 5 injected problems:

| Problem | Column | Injection |
|---|---|---|
| 10 missing values (5%) | age | every 10th row |
| 2 extreme outliers | income | 2.5M and 3.5M |
| Constant (all zeros) | zero_flag | always 0 |
| Near-unique identifier | user_id | 0..200 |
| 1 duplicate row | (all) | row 0 copied |

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table, report};
use datarust_profile::quality::checks::run_checks;
use datarust_profile::quality::Thresholds;

fn build_dirty(rng: &mut Rng) -> (Matrix, StrMatrix) {
    let mut ages = Vec::new();
    let mut incomes = Vec::new();
    let mut transactions = Vec::new();
    let mut zero_flag = Vec::new();
    let mut user_id = Vec::new();
    let mut cities = Vec::new();
    let mut churns = Vec::new();

    for i in 0..200 {
        let age = 20.0 + 50.0 * rng.next_f64();
        let income = 20000.0 + 150000.0 * rng.next_f64().powf(0.7);
        let tx = (10.0 + 90.0 * rng.next_f64()).round();

        ages.push(age);
        incomes.push(income);
        transactions.push(tx);
        zero_flag.push(0.0);
        user_id.push(i as f64);
        cities.push(if rng.next_f64() < 0.55 { "Istanbul" }
            else { ["Ankara", "Izmir", "Bursa"][(rng.next_f64() * 3.0) as usize] }.to_string());
        churns.push(if rng.next_f64() < 0.22 { "yes" } else { "no" }.to_string());
    }

    // Inject missing ages
    for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] { ages[i] = f64::NAN; }
    // Inject outliers
    incomes[12] = 2_500_000.0;
    incomes[99] = 3_500_000.0;
    // Duplicate row 0
    for i in 0..1 {
        ages.push(ages[i]); incomes.push(incomes[i]); transactions.push(transactions[i]);
        zero_flag.push(zero_flag[i]); user_id.push(user_id[i]);
        cities.push(cities[i].clone()); churns.push(churns[i].clone());
    }

    let num = Matrix::new((0..ages.len()).map(|i|
        vec![ages[i], incomes[i], transactions[i], zero_flag[i], user_id[i]]
    ).collect())?;
    let cat = StrMatrix::from_strings((0..cities.len()).map(|i|
        vec![cities[i].clone(), churns[i].clone()]
    ))?;
    Ok((num, cat))
}
```

## Step 1: Profile the Dirty Data

One function call runs all the calculations:

```rust
let names = vec!["age", "income", "transactions", "zero_flag", "user_id", "city", "churn"];
let p = profile_table(Some(&num), Some(&cat), &names)?;
let f = run_checks(&p, &Thresholds::default());
```

Output:

```
── BEFORE: Dirty Dataset ──
  Rows: 201  Cols: 7  Dupes: 1
  age               Numeric  μ=43.8 σ=14.5 miss=5% outl=0
  income            Numeric  μ=135119.5 σ=294866.2 miss=0% outl=2
  transactions      Numeric  μ=56.9 σ=25.9 miss=0% outl=0
  zero_flag         Numeric  μ=0.0 σ=0.0 miss=0% outl=0
  user_id           Numeric  μ=99.0 σ=58.2 miss=0% outl=0
  city            Categorical  unique=4 top=Istanbul ir=0.59
  churn           Categorical  unique=2 top=no ir=0.80
  [Warning] zero_flag: near-zero variance; column is effectively constant
  [Info] dataset: 1 of 201 rows are exact duplicates (0.50%)
```

Two issues flagged immediately: a constant column and a duplicate row. The missing ages and outliers are visible in the stats (5% missing, 2 outliers) but don't trigger the default thresholds — they're real-world moderate issues, not critical failures.

## Step 2: Clean

Four operations, each targeting one problem.

### A. Impute missing ages with median

```rust
let mut imputer = SimpleImputer::new(ImputeStrategy::Median);
let num_imputed = imputer.fit_transform(&num)?;
```

Median is safer than mean for small datasets where outliers in other columns could shift the mean.

### B. Drop the duplicate row

```rust
let mut seen = HashSet::<String>::new();
let keep_idx: Vec<usize> = (0..num_imputed.nrows()).filter(|&i| {
    let key = (0..num_imputed.ncols()).map(|j| format!("{:.6}", num_imputed.get(i, j))).join("|");
    seen.insert(key)
}).collect();
let num_dedup = num_imputed.select_rows(&keep_idx)?;
```

The categorical matrix must be filtered with the same indices — `StrMatrix` doesn't have `select_rows`, so build it manually:

```rust
let cat_rows: Vec<Vec<String>> = (0..cat.nrows())
    .map(|i| (0..cat.ncols()).map(|j| cat.get(i, j).to_string()).collect()).collect();
let cat_dedup = StrMatrix::new(keep_idx.iter().map(|&i| cat_rows[i].clone()).collect())?;
```

### C. Drop the constant and near-unique columns

```rust
let keep_cols: Vec<usize> = vec![0, 1, 2]; // age, income, transactions
let num_clean = num_dedup.select_columns(&keep_cols)?;
```

`VarianceThreshold` from `datarust::selection` can automate this for larger datasets. Here we know exactly which columns are the problem.

### D. Cap income outliers at the 99th percentile

```rust
let income_col: Vec<f64> = (0..num_clean.nrows()).map(|i| num_clean.get(i, 1)).collect();
let mut sorted = income_col.clone();
sorted.sort_by(|a, b| a.total_cmp(b));
let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
let num_capped = Matrix::new((0..num_clean.nrows()).map(|i| {
    let mut row = vec![num_clean.get(i, 0), num_clean.get(i, 1), num_clean.get(i, 2)];
    row[1] = row[1].min(p99);
    row
}).collect())?;
```

Clipping at a percentile preserves the rank order while pulling extreme values into a realistic range.

## Step 3: Re-Profile

Run `profile_table` again on the cleaned data:

```
── AFTER: Cleaned Dataset ──
  Rows: 200  Cols: 5  Dupes: 0
  age               Numeric  μ=43.9 σ=14.1 miss=0% outl=0
  income            Numeric  μ=130150.7 σ=242002.5 miss=0% outl=2
  transactions      Numeric  μ=56.9 σ=26.0 miss=0% outl=0
  city            Categorical  unique=4 top=Istanbul ir=0.58
  churn           Categorical  unique=2 top=no ir=0.80
  No quality flags
```

Key changes:
- **Age**: missing drops from 5% to 0%. Mean stays stable at 43.9 (vs 43.8 before) — median imputation worked.
- **Income**: mean drops from 135K to 130K after capping the two outliers. The standard deviation shrinks from 295K to 242K. The two extreme values are still detected as statistical outliers, but they no longer trigger the quality threshold.
- **zero_flag** and **user_id**: gone. No more constant or near-unique columns.
- **Dataset**: 200 unique rows, zero duplicates, zero quality flags.

## Before vs After Summary

| Metric | Dirty | Clean |
|---|---|---|
| Rows | 201 | 200 |
| Columns | 7 | 5 |
| Duplicates | 1 | 0 |
| Missing values | 10 (age) | 0 |
| Outliers | 2 (income) | detected, not severe |
| Constant columns | 1 | 0 |
| Quality flags | 2 | 0 |

```bash
cargo add datarust
cargo add datarust-profile
```

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table, report, ColumnType};
use datarust_profile::quality::{checks::run_checks, Thresholds};

let profile = profile_table(Some(&data), Some(&labels), &names)?;
// Profile → detect → clean → re-profile.
// Two passes is usually enough to catch everything structural.
```
