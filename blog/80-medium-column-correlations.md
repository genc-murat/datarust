# Your Columns Are Talking to Each Other — and You're Not Listening

*datarust-profile v0.3 computes Pearson, Cramér's V, and point-biserial correlations automatically — so you can find redundant features before they waste your model's capacity*

---

I had a dataset with 12 columns. Two of them were `temperature` and `ice_cream_sales`. I didn't notice they were correlated. My model did — it assigned `ice_cream_sales` a feature importance of 0.42, second only to the target. What I didn't realize: `temperature` already explained 98% of `ice_cream_sales` variance. I was feeding my model the same information twice.

Multicollinearity doesn't always break your model. But it inflates variance of coefficient estimates, makes feature importance unreliable, and wastes capacity on redundant signal. The fix is simple: check pairwise correlations before you train. The problem is that most profiling tools don't do this — you have to write the correlation matrix yourself, separately, after profiling.

`datarust-profile` v0.3 changes that. The `relationships` block computes three types of pairwise correlations automatically:

- **Pearson** for numeric ↔ numeric
- **Cramér's V** for categorical ↔ categorical
- **Point-biserial** for binary categorical ↔ numeric

No extra code. No separate correlation step. It's just there in the profile.

---

## Pearson Correlation: When One Feature Is Just a Scaled Copy of Another

```rust
use datarust::Matrix;
use datarust_profile::profile_table;

let numeric = Matrix::from_rows(vec![
    vec![1.0, 2.0, 10.0],
    vec![2.0, 4.0, 20.0],
    vec![3.0, 6.0, 15.0],
    vec![4.0, 8.0, 25.0],
    vec![5.0, 10.0, 30.0],
])?;

let names = vec!["temperature".into(), "ice_cream_sales".into(), "humidity".into()];
let profile = profile_table(Some(&numeric), None, &names)?;
```

The profile automatically computes the Pearson correlation matrix over all numeric columns. Access it through `profile.relationships`:

```rust
if let Some(rels) = &profile.relationships {
    if let Some(pearson) = &rels.pearson {
        for (i, a) in pearson.labels.iter().enumerate() {
            for (j, b) in pearson.labels.iter().enumerate() {
                if j > i {
                    println!("  {} <-> {}: r={:.3}", a, b, pearson.values[i][j]);
                }
            }
        }
    }
}
```

Output:

```
  temperature <-> ice_cream_sales: r=1.000
  temperature <-> humidity: r=0.900
  ice_cream_sales <-> humidity: r=0.900
```

`temperature` and `ice_cream_sales` are perfectly correlated (r=1.000). In this dataset, `ice_cream_sales` is literally `temperature * 2`. Feeding both to a model is redundant — the model has to learn that relationship on its own, wasting capacity and making coefficient estimates less stable.

The Pearson matrix lives in `relationships.pearson`. It's a symmetric matrix with labels on both axes. The diagonal is always 1.0 (a variable correlated with itself). You only care about the upper triangle.

---

## Cramér's V: Correlation for Categories

Numeric correlation is straightforward. But what about categorical columns? `city` and `plan` — are they associated? You can't compute Pearson on strings.

Cramér's V measures the strength of association between two categorical variables. It ranges from 0 (no association) to 1 (perfect association). Under the hood, `datarust-profile` builds a contingency table, computes chi-squared, and normalizes by the maximum possible chi-squared.

```rust
use datarust::StrMatrix;
use datarust_profile::profile_table;

let cat = StrMatrix::from_strings(vec![
    vec!["Istanbul".to_string(), "premium".to_string()],
    vec!["Istanbul".to_string(), "basic".to_string()],
    vec!["Ankara".to_string(), "basic".to_string()],
    vec!["Izmir".to_string(), "basic".to_string()],
    vec!["Ankara".to_string(), "premium".to_string()],
])?;

let names = vec!["city".into(), "plan".into()];
let profile = profile_table(None, Some(&cat), &names)?;

if let Some(rels) = &profile.relationships {
    if let Some(cv) = &rels.cramers_v {
        for (i, a) in cv.labels.iter().enumerate() {
            for (j, b) in cv.labels.iter().enumerate() {
                if j > i {
                    println!("  {} <-> {}: V={:.3}", a, b, cv.values[i][j]);
                }
            }
        }
    }
}
```

Output:

```
  city <-> plan: V=0.408
```

V=0.408 is a moderate association. In this data, Istanbul has a mix of premium and basic users, while Ankara and Izmir skew basic. The association is real but not strong enough to drop either column.

If you see V > 0.7, one category is almost perfectly predicted by the other. That's a candidate for removal — or for creating a combined feature if the interaction matters.

---

## Point-Biserial: When One Column Is Binary

Most datasets have binary columns — `churned`, `is_premium`, `has_discount`. These are categorical, but you want to know how they correlate with your numeric features. Pearson doesn't work here (one variable isn't numeric). Cramér's V works but loses direction (it doesn't tell you *which* category maps to higher values).

Point-biserial correlation solves this. It's the correlation between a binary categorical variable and a continuous numeric variable. It tells you not just *whether* they're associated, but *which direction*:

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::profile_table;

let numeric = Matrix::from_rows(vec![
    vec![1.0, 100.0],
    vec![1.0, 150.0],
    vec![0.0, 300.0],
    vec![0.0, 250.0],
    vec![1.0, 120.0],
])?;

let cat = StrMatrix::from_strings(vec![
    vec!["churned".to_string()],
    vec!["churned".to_string()],
    vec!["active".to_string()],
    vec!["active".to_string()],
    vec!["churned".to_string()],
])?;

let names = vec!["loyalty_score".into(), "revenue".into(), "status".into()];
let profile = profile_table(Some(&numeric), Some(&cat), &names)?;

if let Some(rels) = &profile.relationships {
    for entry in &rels.point_biserial {
        println!(
            "  {} (binary) <-> {}: r={:.3}",
            entry.categorical, entry.numeric, entry.correlation
        );
    }
}
```

Output:

```
  status (binary) <-> loyalty_score: r=-1.000
  status (binary) <-> revenue: r=0.957
```

Two findings here:

1. `status` ↔ `loyalty_score`: r=-1.000. Churned users have low loyalty scores. Perfect negative correlation — this feature is almost a direct encoding of the target.

2. `status` ↔ `revenue`: r=0.957. Active users generate more revenue. Strong positive correlation.

Point-biserial entries are listed in `relationships.point_biserial` as a flat vector. Each entry has `categorical` (the binary column name), `numeric` (the continuous column name), and `correlation` (the r value).

---

## Connecting to Quality Checks

The relationships block isn't just for manual inspection. `run_checks` uses it automatically:

```rust
use datarust_profile::{run_checks, Thresholds};

let issues = run_checks(&profile, &Thresholds::default());
for issue in &issues {
    println!("  [{}] {}", issue.severity, issue.message);
}
```

Two checks use the relationships data:

- **`HighCorrelation`** — flags any numeric pair with |r| ≥ 0.95. Severity: Warning.
- **`TargetLeakage`** — flags features correlated with the designated target (requires `profile_table_with_target`). Severity: Critical.

For the temperature/ice_cream_sales example:

```
  [warning] High Pearson correlation between 'temperature' and 'ice_cream_sales' (r = 1.000)
```

For target leakage (using `profile_table_with_target` with `days_since_login` as target):

```
  [critical] Suspected target leakage: feature 'income' has strong correlation with target 'days_since_login' (r = 0.905)
  [critical] Suspected target leakage: feature 'plan' has high point-biserial correlation with target 'days_since_login' (r = 0.995)
```

You can tune the thresholds:

```rust
let issues = run_checks(&profile, &Thresholds {
    high_correlation: 0.90,   // flag pairs with |r| >= 0.90 instead of 0.95
    target_leakage: 0.85,     // flag features with |r| >= 0.85 instead of 0.90
    ..Default::default()
});
```

---

## When to Drop, When to Keep

Not every high correlation means you should drop a column. Here's my rule of thumb:

| Correlation | Action |
|---|---|
| r > 0.99 | Almost certainly redundant. Drop one. |
| r > 0.95 | Likely redundant. Check if both add value in cross-validation. |
| r > 0.85 | Potentially redundant. Keep if domain knowledge says they capture different aspects. |
| r < 0.85 | Probably fine. Two features can be correlated and still both useful. |

For Cramér's V, the thresholds are similar but the interpretation differs: V > 0.7 means one category almost perfectly predicts the other.

For point-biserial, a |r| > 0.90 between a binary column and a numeric feature is suspicious — especially if the binary column is the target or derived from it.

---

## The Complete Picture

The relationships block runs automatically. You don't call it explicitly — it's part of `profile_table`, `profile_matrix`, and their `_with_target` variants. If you have 2+ numeric columns, you get Pearson. If you have 2+ categorical columns, you get Cramér's V. If you have both, you get point-biserial for every binary/numeric pair.

```rust
let profile = profile_table(
    Some(&numeric_data),
    Some(&categorical_data),
    &column_names,
)?;

// All three are available immediately
let pearson = &profile.relationships.as_ref().unwrap().pearson;
let cramers_v = &profile.relationships.as_ref().unwrap().cramers_v;
let point_biserial = &profile.relationships.as_ref().unwrap().point_biserial;
```

If you only have numeric data, `relationships.pearson` exists but `cramers_v` is `None` and `point_biserial` is empty. If you only have categorical data, `cramers_v` exists but `pearson` is `None`. If you have one column total, `relationships` itself is `None`.

The HTML report includes color-coded heatmaps for Pearson (red-white-blue diverging) and Cramér's V (white-purple sequential), plus a point-biserial table. Open the report in a browser and the correlations are visually obvious.

---

## Summary

`datarust-profile` v0.3's `relationships` block answers three questions automatically:

1. **Which numeric columns are redundant?** → Pearson correlation matrix
2. **Which categorical columns are associated?** → Cramér's V matrix
3. **Which binary columns predict which numeric features?** → Point-biserial entries

No extra code. No separate correlation step. The data is in `profile.relationships`, the quality checks flag problems automatically, and the HTML report visualizes everything. Check correlations before you train — your model's coefficients will thank you.
