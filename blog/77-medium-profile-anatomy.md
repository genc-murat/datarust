# The Anatomy of a DatasetProfile

*Every field of datarust-profile's output, from n_rows to top_values — what it means and when you need it*

---

`datarust-profile` returns a single struct: `DatasetProfile`. Inside it are 5 top-level fields and a vector of `ColumnProfile` entries. Each `ColumnProfile` contains either `NumericStats` (with a `FiveNumber` and a `Histogram`) or `CategoricalStats`. That's 7 structs, 40+ fields.

You don't need all of them. But knowing what exists — and what each field tells you — is the difference between "I ran a profile" and "I understand my data."

This is a reference. Here's every field, with real numbers from a 202-row customer dataset.

## DatasetProfile — The Top Level

```json
{
  "n_rows": 202,
  "n_columns": 7,
  "memory_bytes": 21008,
  "duplicate_rows": 2,
  "duplicate_fraction": 0.0099,
  "columns": [ /* ... */ ]
}
```

| Field | Value | What it tells you |
|---|---|---|
| `n_rows` | 202 | Row count. If this doesn't match expectations, check your data source. |
| `n_columns` | 7 | Column count. Useful for a quick shape sanity check. |
| `memory_bytes` | 21008 | Rough in-memory size of the profiled cells. 21 KB for 202×7 cells ≈ 15 bytes/cell. |
| `duplicate_rows` | 2 | Exact duplicates count. 2 rows in this dataset appear twice. |
| `duplicate_fraction` | 0.0099 | 2/202 = 0.99% of rows are duplicates. Low enough to ignore, but worth checking if they're legitimate repetitions or data-entry errors. |

The duplicate detection is an O(n²) scan comparing every row against every previously seen row. For 202 rows it's instant. For 2 million rows, you'd want a hash-based approach — this is a known limitation.

## ColumnProfile — One Per Column

Each column gets one of these:

```
name:             age
column_type:      Numeric
count:            202
missing_count:    10
missing_fraction: 0.0495
```

| Field | Example | What it tells you |
|---|---|---|
| `name` | `"age"` | Column name from your header. Falls back to `x0`, `x1`, ... if you don't supply names. |
| `column_type` | `Numeric` | Inferred type: `Numeric` (every non-missing cell parses as f64) or `Categorical`. |
| `count` | 202 | Total cells including missing. Always equals `n_rows`. |
| `missing_count` | 10 | How many cells are NaN (for numeric) or empty/marker strings (for categorical). |
| `missing_fraction` | 0.0495 | 10/202 = 4.95% missing. Above 5% you should investigate why. |

## NumericStats — For Numeric Columns

```
mean:            43.12
std:             14.75
skewness:        0.17
kurtosis:        -1.24
outlier_count:   0
outlier_fraction: 0.0000
```

| Field | Example | What it tells you |
|---|---|---|
| `mean` | 43.12 | Arithmetic mean of non-missing values. Sensitive to outliers. |
| `std` | 14.75 | Sample standard deviation (ddof=1). With mean 43 and std 15, the 68% range is roughly [28, 58]. |
| `skewness` | 0.17 | Fisher-Pearson skewness. Near 0 → symmetric. Positive → right tail. `income` below has skew=8.5. |
| `kurtosis` | -1.24 | Excess kurtosis. Near 0 → normal tails. Negative → light tails (uniform-like). `income` below has kurt=100. |
| `outlier_count` | 0 | Values beyond Tukey IQR fences (Q1 − 1.5×IQR, Q3 + 1.5×IQR). 0 for age — no extreme ages. |
| `outlier_fraction` | 0.0000 | Fraction of values flagged as outliers. |

Compare `age` (skew=0.17, kurt=−1.24, outliers=0) with `income` (skew=8.50, kurt=100.42, outliers=1). The income column has a single extreme value at 999,999 that dominates the kurtosis statistic and creates a long empty tail in the histogram. One row creates `kurtosis = 100`.

### FiveNumber

```
min:           24279.05
q1:            79590.22
median:        110272.67
q3:            140611.86
max:           999999.00
```

| Field | Example | What it tells you |
|---|---|---|
| `min` | 24,279 | Minimum non-missing value. |
| `q1` | 79,590 | 25th percentile. 50% of incomes are between 79,590 and 140,612. |
| `median` | 110,273 | 50th percentile. The middle value. More robust than mean for skewed data. |
| `q3` | 140,612 | 75th percentile. |
| `max` | 999,999 | Maximum. The massive gap between Q3 (140k) and max (1M) confirms the outlier. |

The IQR is `140612 − 79590 = 61022`. The outlier fence is `140612 + 1.5 × 61022 = 232145`. The 999,999 value is 12.6× the Q3 — well beyond the fence.

### Histogram

```
  [24279.05, 132692.38): 139  ████████████████████████████████████████
  [132692.38, 241105.71):  62  ██████████████████
  [241105.71, 349519.03):   0
  [349519.03, 457932.36):   0
  [457932.36, 566345.69):   0
  [566345.69, 674759.02):   0
  [674759.02, 783172.34):   0
  [783172.34, 891585.67):   0
  [891585.67, 999999.00):   1  █
```

| Field | What it tells you |
|---|---|
| `edges` | Bin boundaries. Sturges' rule determines the count. For 202 rows: ceil(log2(202) + 1) = 9 bins. |
| `counts` | Values per bin. The income histogram shows 139 values in the first bin, 62 in the second, and then 7 empty bins before a single value in the last bin. |

The equal-width bins reveal the outlier problem immediately: 7 empty bins between the main distribution and the extreme value. A histogram with this shape means the column is dominated by a single extreme observation.

## CategoricalStats — For Categorical Columns

```
unique:          4
top:             "Istanbul"
freq:            115
imbalance_ratio: 0.5693
top_values:
          Istanbul  ×115
             Bursa  ×33
             Izmir  ×30
            Ankara  ×24
```

| Field | Example | What it tells you |
|---|---|---|
| `unique` | 4 | Distinct non-missing values. 4 cities in the dataset. |
| `top` | `"Istanbul"` | The most frequent value. |
| `freq` | 115 | How many times the top value appears. |
| `imbalance_ratio` | 0.5693 | 115/202 = 56.9%. Istanbul dominates, but not overwhelmingly. Compare with `churn` below (76.7% "no"). |
| `top_values` | `[(Istanbul, 115), (Bursa, 33), ...]` | Up to 8 most frequent values with counts, descending. Useful for bar charts or quick inspection. |

The `imbalance_ratio` is the share of the top value. Values above 0.95 trigger the `Imbalance` quality check at default thresholds. At 0.57 for city and 0.77 for churn, these columns are skewed but not alarmingly so.

For `churn`:

```
unique:          2
top:             "no"
freq:            155
imbalance_ratio: 0.7673
top_values:
                no  ×155
               yes  ×47
```

Two unique values (binary), 77% "no", 23% "yes". This is a 77/23 split — moderately imbalanced. A model predicting "no" for everyone would get 77% accuracy, which sounds good but is useless. The imbalance ratio quantifies this.

## What to Check First

You don't need to read every field. Here's what to scan:

1. **`duplicate_fraction`** — above 0.01? Check which rows are duplicated.
2. **`missing_fraction` per column** — above 0.05? Investigate why data is missing.
3. **`column_type`** — any column inferred as numeric that shouldn't be (zip codes, IDs)?
4. **`outlier_count`** — any column with outliers above 1-2%? The IQR rule catches tail risk.
5. **`imbalance_ratio`** — above 0.90 for categorical features? That column carries almost no signal.
6. **`skewness`** — above +1.0 or below −1.0 for numeric features? Consider a log transform.
7. **`kurtosis`** — above +5.0? Extreme values will dominate your model.

The profile gives you all of these numbers in one function call. You don't have to write a loop, compute a quantile, or remember the IQR formula. It's there, column by column, before you fit a single model.

```bash
cargo add datarust-profile
```

```rust
use datarust::Matrix;
use datarust_profile::profile_matrix;

let m = Matrix::new(rows)?;
let p = profile_matrix(&m, None)?;

// Check the first column
println!("{} missing of {} rows", p.columns[0].missing_count, p.n_rows);
if let Some(n) = &p.columns[0].numeric {
    println!("skewness {:.2}, outliers {}", n.skewness, n.outlier_count);
}
```

7 structs, 40+ fields, one function call.
