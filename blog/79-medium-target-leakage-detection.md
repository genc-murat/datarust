# The Feature That Secrets Your Target: Catching Leakage Before It Breaks Your Model

*A single profiling call detects features that silently leak the target label — saving you from 99% accuracy that means nothing in production*

---

Your model scores 99.2% accuracy on the test set. You deploy it. It fails within a week.

The culprit: a feature column that contains information about the target variable. An `account_closed_date` that only exists when `churn = 1`. A `refund_amount` that is zero for non-churners. A `post_purchase_flag` that is literally the target in disguise.

Cross-validation doesn't catch this — the leakage is present in both training and test splits. Feature importance scores don't catch it — the model correctly identifies the leaky feature as highly predictive. The only thing that catches it is checking whether any feature is *suspiciously correlated with the target before you train*.

That's what `datarust-profile` v0.3's target-leakage detection does. One function call, before any model fitting, flags features that correlate with the target above a configurable threshold.

---

## The Scenario

A subscription dataset with 200 rows. Four numeric features, two categorical features, and a binary target `churn`. One of the numeric features — `days_since_last_login` — is a near-perfect mirror of the target: customers who churned haven't logged in for 78–92 days, while active users logged in within 2–5 days.

The point-biserial correlation between `days_since_last_login` and `churn` is 0.995. A model trained on this data will achieve near-perfect accuracy. But `days_since_last_login` is only available *after* a customer has already decided to leave — it's a post-hoc artifact, not a predictive feature.

Without target-aware profiling, you'd never notice. With it, the finding appears before you write a single line of model code.

---

## Step 1: Profile With a Target Hint

`datarust-profile` v0.3 introduces `profile_table_with_target` — same as `profile_table`, but with a designated target column name:

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::profile_table_with_target;

let numeric = Matrix::from_rows(vec![
    vec![25.0, 52000.0, 3.0,  2.0],
    vec![34.0, 68000.0, 12.0, 85.0],
    vec![41.0, 45000.0, 7.0,  3.0],
    vec![29.0, 72000.0, 15.0, 92.0],
    vec![52.0, 38000.0, 1.0,  5.0],
    vec![38.0, 81000.0, 9.0,  78.0],
])?;

let categorical = StrMatrix::from_strings(vec![
    vec!["basic".to_string(),    "no".to_string()],
    vec!["premium".to_string(),  "yes".to_string()],
    vec!["basic".to_string(),    "no".to_string()],
    vec!["premium".to_string(),  "yes".to_string()],
    vec!["basic".to_string(),    "no".to_string()],
    vec!["premium".to_string(),  "yes".to_string()],
])?;

let names = vec![
    "age".into(),
    "income".into(),
    "transactions".into(),
    "days_since_last_login".into(),
    "plan".into(),
    "churn".into(),
];

let profile = profile_table_with_target(
    Some(&numeric),
    Some(&categorical),
    &names,
    "churn",   // <-- designate the target column
)?;
```

The `target_column` field on the profile is now set to `Some("churn")`. This is all you need — the profiling step itself doesn't change. The target hint only affects what `run_checks` evaluates downstream.

---

## Step 2: Run Quality Checks

```rust
use datarust_profile::run_checks;
use datarust_profile::Thresholds;

let findings = run_checks(&profile, &Thresholds::default());
```

When a target column is declared, `run_checks` evaluates three correlation pathways against the target:

| Pathway | What it checks | Default threshold |
|---|---|---|
| Pearson $r$ | Numeric feature vs numeric target | $\|r\| \ge 0.90$ |
| Cramér's $V$ | Categorical feature vs categorical target | $V \ge 0.90$ |
| Point-biserial $r_{pb}$ | Numeric feature vs binary categorical target | $\|r_{pb}\| \ge 0.90$ |

For our dataset, `days_since_last_login` (numeric) vs `churn` (binary categorical) goes through the point-biserial pathway. The correlation is 0.995 — above the 0.90 threshold:

```
[Critical] days_since_last_login: Suspected target leakage:
  feature 'days_since_last_login' has high point-biserial
  correlation with target 'churn' (r = 0.995)
```

`Critical` severity. Not a suggestion — a warning that this feature will inflate your metrics and mislead your model.

The tool also catches `income` (r = 0.932) and `plan` (Cramér's V = 1.0) as leaking features. In a real dataset, finding multiple leakage signals at once is common — it means several columns were constructed from post-event data.

---

## How Point-Biserial Works

Standard Pearson correlation doesn't work when one variable is binary (`"yes"`/`"no"`) and the other is continuous. Point-biserial correlation handles exactly this case.

It maps the binary category to 0 and 1, then computes:

$$r_{pb} = \frac{M_1 - M_0}{S_y} \sqrt{\frac{n_0 \cdot n_1}{n \cdot (n - 1)}}$$

where $M_0$ and $M_1$ are the group means, $S_y$ is the pooled standard deviation, and $n_0, n_1$ are the group sizes.

In our example:
- Churned group mean (`days_since_last_login`): 85.0
- Active group mean: 3.3
- The gap is enormous — $r_{pb} = 0.995$

This is exactly the kind of feature that looks predictive but is actually a data-collection artifact. The customer didn't churn *because* they stopped logging in; they stopped logging in *because* they already churned. The temporal ordering is reversed.

---

## The Three Leakage Pathways

### Numeric Feature vs Numeric Target

When both the feature and target are continuous (e.g., `income` vs `annual_spend`), leakage is detected via the Pearson correlation matrix. Any feature-target pair with $|r| \ge 0.90$ is flagged.

### Categorical Feature vs Categorical Target

When both are categorical (e.g., `plan_type` vs `churn`), Cramér's V measures association. A $V \ge 0.90$ means the feature categories almost perfectly predict the target categories.

### Numeric Feature vs Binary Target (Point-Biserial)

The most common case in classification: a continuous feature vs a yes/no target. Point-biserial detects group-mean separation that's too clean to be real.

---

## Customizing the Threshold

The default 0.90 threshold is conservative — it only flags near-perfect correlation. For more sensitive detection:

```rust
let mut t = Thresholds::default();
t.target_leakage = 0.80;   // flag features with |r| >= 0.80

let findings = run_checks(&profile, &t);
```

With a 0.80 threshold, even moderately correlated features surface. This is useful when you're auditing a dataset for the first time and want to see everything that's suspicious — not just the obvious leakage.

For production pipelines where you want to be strict:

```rust
let strict = Thresholds {
    target_leakage: 0.95,   // only flag near-perfect leakage
    ..Thresholds::default()
};
```

---

## What Leakage Detection Doesn't Catch

Target-leakage detection finds *statistical* correlation between features and the target. It doesn't find:

1. **Temporal leakage** — a feature that's computed from future data (e.g., using next month's revenue to predict this month's churn). The feature values are valid numbers; the problem is *when* they were measured.

2. **Indirect leakage** — a feature that's a downstream consequence of the target but has low direct correlation. For example, `support_ticket_count` might only be 0.3 correlated with `churn`, but it's still a post-hoc artifact.

3. **Encoding leakage** — a categorical feature whose labels encode the target (e.g., `churned_yes` as a category name).

These require domain knowledge and temporal reasoning. Statistical correlation is a necessary but not sufficient check. It catches the obvious cases — the ones that would otherwise waste days of debugging.

---

## Putting It Together

```rust
use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_table_with_target, report, run_checks, Thresholds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let numeric = Matrix::from_rows(vec![
        vec![25.0, 52000.0, 3.0,  2.0],
        vec![34.0, 68000.0, 12.0, 85.0],
        vec![41.0, 45000.0, 7.0,  3.0],
        vec![29.0, 72000.0, 15.0, 92.0],
        vec![52.0, 38000.0, 1.0,  5.0],
        vec![38.0, 81000.0, 9.0,  78.0],
    ])?;

    let categorical = StrMatrix::from_strings(vec![
        vec!["basic".to_string(),    "no".to_string()],
        vec!["premium".to_string(),  "yes".to_string()],
        vec!["basic".to_string(),    "no".to_string()],
        vec!["premium".to_string(),  "yes".to_string()],
        vec!["basic".to_string(),    "no".to_string()],
        vec!["premium".to_string(),  "yes".to_string()],
    ])?;

    let names = vec![
        "age".into(), "income".into(), "transactions".into(),
        "days_since_last_login".into(), "plan".into(), "churn".into(),
    ];

    // Profile with target hint
    let profile = profile_table_with_target(
        Some(&numeric), Some(&categorical), &names, "churn",
    )?;

    // Check for leakage with a sensitive threshold
    let mut t = Thresholds::default();
    t.target_leakage = 0.80;
    let findings = run_checks(&profile, &t);

    for issue in &findings {
        let col = issue.column.as_deref().unwrap_or("(dataset)");
        println!("[{:?}] {}: {}", issue.severity, col, issue.message);
    }

    // Generate HTML report with correlation heatmaps
    let html = report::to_html_with(&profile, &findings);
    std::fs::write("leakage_report.html", html)?;

    Ok(())
}
```

Output:

```
[Critical] plan: Suspected target leakage: categorical feature 'plan'
  has high Cramér's V with target 'churn' (V = 1.000)
[Critical] income: Suspected target leakage: feature 'income'
  has high point-biserial correlation with target 'churn' (r = 0.932)
[Critical] days_since_last_login: Suspected target leakage:
  feature 'days_since_last_login' has high point-biserial
  correlation with target 'churn' (r = 0.995)
```

Three findings. The tool catches every feature that's too closely tied to the target — whether through point-biserial (numeric vs binary) or Cramér's V (categorical vs categorical).

---

## When to Run This

Run target-leakage detection **before** any model training, feature selection, or cross-validation. It's a one-line addition to your profiling step:

```rust
let profile = profile_table_with_target(
    Some(&numeric), Some(&categorical), &names, "churn",
)?;
let issues = run_checks(&profile, &Thresholds::default());
```

If zero findings come back, you can proceed with confidence. If a `Critical` finding appears, you know exactly which feature to investigate — before it contaminates your metrics.

The profiling step is already the first thing you do with any dataset. Adding a target hint costs nothing and catches the silent killer that cross-validation misses.

---

```toml
[dependencies]
datarust = "0.6"
datarust-profile = "0.3"
```

- **Crate**: [crates.io/crates/datarust-profile](https://crates.io/crates/datarust-profile)
- **Docs**: [docs.rs/datarust-profile](https://docs.rs/datarust-profile)
- **GitHub**: [github.com/genc-murat/datarust](https://github.com/genc-murat/datarust)
