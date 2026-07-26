# The Rule Said 30 Days or More. My Feature Said Day 30 Was No.

*A practical datarust guide to strict thresholds, inclusive business rules, boundary tests, missing values, and the hundred customers who became ineligible for exactly one day.*

---

The eligibility rule looked unambiguous:

```text
The account must be at least 30 days old.
```

I turned account age into a binary feature with one line:

```rust
Binarizer::new().threshold(30.0)
```

Accounts younger than 30 days became `0`. Older accounts became `1`. The feature flowed through the pipeline, the matrix had the expected shape, and nothing returned an error.

On day 30, however, an account was still `0`.

At `30.000001` days it became `1`.

The code implemented “more than 30.” The policy said “30 or more.” One character separated those statements:

```text
value > 30
value >= 30
```

That character rejected every customer whose stored account age was exactly 30 days.

It was easy to dismiss this as an edge case until I looked at the input distribution. The upstream job exported completed whole days, so exact integers were not rare at all. Day 30 was an entire daily batch.

Let's reproduce the boundary with [datarust](https://crates.io/crates/datarust), measure its effect on 300 renewal decisions, and make the business comparison explicit instead of hoping a convenient transformer means the same thing as the policy sentence.

## Binarizer uses a strict comparison

`Binarizer` converts every value into either `0.0` or `1.0`.

Its rule is:

```text
value > threshold  -> 1.0
otherwise          -> 0.0
```

Equality belongs to the second branch.

With a threshold of `30.0`:

```text
29.999999 -> 0
30.000000 -> 0
30.000001 -> 1
```

That is a coherent and documented contract. The transformer is not miscalculating the number 30. I supplied a strict-threshold tool for an inclusive-threshold rule.

The dangerous part is that both interpretations sound similar in casual conversation:

- “over 30 days” means `> 30`
- “at least 30 days” means `>= 30`

A generic feature name such as `mature_account` can hide the distinction from everyone downstream.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new eligibility_boundary
cd eligibility_boundary
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix,
};
use datarust::scaler::Binarizer;
use datarust::traits::Transformer;
use datarust::Matrix;

fn values(x: &Matrix) -> Vec<f64> {
    (0..x.nrows())
        .map(|row| x.get(row, 0))
        .collect()
}

fn at_least(
    x: &Matrix,
    boundary: f64,
) -> Result<Matrix, Box<dyn std::error::Error>> {
    let rows = x
        .rows_ref()
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| {
                    if value >= boundary { 1.0 } else { 0.0 }
                })
                .collect()
        })
        .collect();
    Ok(Matrix::new(rows)?)
}

fn report(
    name: &str,
    truth: &[f64],
    predicted: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let cm = confusion_matrix(truth, predicted)?;
    let eligible_recall =
        cm[1][1] as f64 / (cm[1][0] + cm[1][1]) as f64;

    println!("{name}");
    println!("  confusion: {cm:?}");
    println!(
        "  accuracy:  {:.3}",
        accuracy_score(truth, predicted)?,
    );
    println!("  eligible recall: {eligible_recall:.3}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boundary = Matrix::new(vec![
        vec![29.0],
        vec![29.999],
        vec![30.0],
        vec![30.000_001],
        vec![31.0],
    ])?;
    let truth = vec![0.0, 0.0, 1.0, 1.0, 1.0];

    let mut strict = Binarizer::new().threshold(30.0);
    let strict_boundary = strict.fit_transform(&boundary)?;
    let inclusive_boundary = at_least(&boundary, 30.0)?;

    println!("Boundary audit");
    println!("days       expected   Binarizer(30)   >= 30");
    for row in 0..boundary.nrows() {
        println!(
            "{:<10.6} {:>8.0} {:>17.0} {:>8.0}",
            boundary.get(row, 0),
            truth[row],
            strict_boundary.get(row, 0),
            inclusive_boundary.get(row, 0),
        );
    }

    let mut cohort_rows = Vec::new();
    let mut cohort_truth = Vec::new();
    for (days, eligible) in [
        (29.0, 0.0),
        (30.0, 1.0),
        (31.0, 1.0),
    ] {
        for _ in 0..100 {
            cohort_rows.push(vec![days]);
            cohort_truth.push(eligible);
        }
    }

    let cohort = Matrix::new(cohort_rows)?;
    let strict_cohort = strict.transform(&cohort)?;
    let inclusive_cohort = at_least(&cohort, 30.0)?;

    println!();
    println!("Three hundred account renewals");
    report(
        "Binarizer threshold 30",
        &cohort_truth,
        &values(&strict_cohort),
    )?;
    report(
        "Explicit days >= 30",
        &cohort_truth,
        &values(&inclusive_cohort),
    )?;
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
Boundary audit
days       expected   Binarizer(30)   >= 30
29.000000         0                 0        0
29.999000         0                 0        0
30.000000         1                 0        1
30.000001         1                 1        1
31.000000         1                 1        1

Three hundred account renewals
Binarizer threshold 30
  confusion: [[100, 0], [100, 100]]
  accuracy:  0.667
  eligible recall: 0.500
Explicit days >= 30
  confusion: [[100, 0], [0, 200]]
  accuracy:  1.000
  eligible recall: 1.000
```

The boundary row is the only disagreement in the five-value audit.

In the operational cohort, that one numeric value represents 100 people.

## An edge value can be a busy place

Our cohort contains:

```text
100 accounts at 29 days -> not eligible
100 accounts at 30 days -> eligible
100 accounts at 31 days -> eligible
```

The strict rule correctly rejects day 29 and correctly accepts day 31. It rejects all 100 day-30 accounts.

In datarust's confusion matrix, rows are true classes and columns are predicted classes:

```text
[[true negatives,  false positives],
 [false negatives, true positives ]]
```

So this result:

```text
[[100, 0],
 [100, 100]]
```

means the rule produced no false approvals but 100 false rejections. It found only half of the 200 genuinely eligible accounts.

The overall accuracy of `66.7%` is not caused by noise, model uncertainty, or a difficult optimization problem. It is a precise implementation of the wrong comparison operator.

Boundary values become especially crowded when upstream systems:

- truncate timestamps to completed days
- round currency to cents
- store percentages as whole integers
- group ages into completed years
- schedule evaluations at midnight
- cap measurements at a contractual limit

In continuous mathematics, landing on one exact point can seem unlikely. In business data, preprocessing and policy calendars manufacture exact points constantly.

## Why lowering the threshold to 29 is a fragile fix

If account age is guaranteed to be a whole number, this produces the desired mapping:

```rust
Binarizer::new().threshold(29.0)
```

For integer values:

```text
29 > 29 -> false
30 > 29 -> true
```

But the code now says “strictly greater than 29” while the policy says “at least 30.” Those happen to agree only because of an unstated discrete-input invariant.

The moment fractional days arrive, an account at `29.5` becomes eligible too early.

The workaround also becomes awkward for money and measurements. To express “at least $10.00,” should the threshold be `$9.99`, `$9.999999`, or the largest floating-point number below ten? Each answer embeds assumptions about units and precision.

I would use the shifted-threshold technique only when the domain is explicitly discrete, the step size is part of the schema, and tests enforce it. Even then, a name such as `completed_days_gt_29` should make the actual operation visible.

## Epsilon does not define the policy

Another tempting repair is:

```rust
Binarizer::new().threshold(30.0 - 1e-9)
```

This may make `30.0` pass. It also makes some values below 30 pass, and the width of that exception is now determined by a magic floating-point constant.

An epsilon is appropriate when the domain has a documented measurement tolerance. It is not a general substitute for `>=`.

For currency, I prefer integer minor units when possible:

```text
1000 cents rather than 10.00 binary floating-point dollars
```

For durations, I compare timestamps or integer duration units under an explicit rounding policy. For physical measurements, I document whether the threshold refers to the raw observation, a rounded display value, or a tolerance band.

The question is not “How close is this float to 30?” The question is “What did the policy owner mean by 30?”

## Make inclusive comparisons explicit

The `at_least` helper in the experiment writes the rule directly:

```rust
if value >= boundary { 1.0 } else { 0.0 }
```

That explicit comparison is often preferable for deterministic policy features. It can run during feature construction before the rows become a `Matrix`, or live in a named transformation step whose code and version are deployed with the model.

A name can carry the contract too:

```text
account_age_days_gte_30
```

That is less elegant than `mature_account`, but it is far easier to audit six months later.

If I wrap the comparison in a reusable custom transformer, I also preserve its code identity and version with the pipeline. Saving only the downstream coefficients is not enough: changing `>=` to `>` changes the meaning of every fitted coefficient that consumes this feature.

## Binarizer is still the right tool for strict rules

There is nothing inherently wrong with `Binarizer`'s contract.

It is a natural fit for rules such as:

```text
more than 3 failed attempts
temperature above 80°C
balance greater than $0
score strictly higher than the control limit
```

The important word is “more than,” “above,” or “greater than.” Equality should belong to the zero side.

For example:

```rust
let mut high_temperature =
    Binarizer::new().threshold(80.0);
```

This creates `1.0` only when the temperature is strictly above 80. A reading of exactly 80 remains `0.0` by design.

Tools become dangerous when their short names replace the business sentence in our heads. I now put the sentence and the operator beside each other during review.

## Threshold in the units you think you have

Comparison direction is only half of the contract. The feature's units matter too.

Suppose account age is standardized before binarization:

```text
raw days -> StandardScaler -> Binarizer(threshold = 30)
```

The binarizer no longer receives days. It receives standard deviations from the training mean. A threshold of 30 now means 30 standard deviations, which will probably turn every ordinary row into zero.

If the policy is written in days, evaluate it while the value is still in days:

```text
raw days -> inclusive 30-day rule -> binary feature
```

The resulting zero/one indicator generally does not need ordinary numeric scaling. If raw age is also useful to the model, keep both features deliberately rather than overwriting one and forgetting which units remain.

The same warning applies to logarithms, power transforms, and unit conversion. A threshold of `100` means something different in dollars, cents, standardized coordinates, and log space.

## Missing is not below the boundary

There is one more quiet branch in floating-point comparisons:

```rust
f64::NAN > 30.0   // false
f64::NAN >= 30.0  // false
```

If a missing account age reaches the binarizer, it becomes `0.0` under the strict comparison. The explicit inclusive helper above would also return `0.0` unless it checks for missingness first.

That does not mean the account is known to be too young. It means the comparison could not establish eligibility.

Depending on the application, a missing value may need to:

- stop the decision and request the source field
- follow a separately approved fallback policy
- be imputed before the comparison
- produce an additional `age_missing` indicator
- route to human review

Silently treating “unknown” as “not eligible” may be conservative, but it is still a policy decision. It should not emerge accidentally from how `NaN` comparisons work.

## Test the fence, not only the field

Random test data often exercises the broad regions on either side of a threshold and misses equality entirely.

For every rule boundary, I now write a small table with at least:

```text
just below
exactly equal
just above
missing
minimum and maximum valid values
```

For discrete domains, I test the adjacent representable values:

```text
29, 30, 31 completed days
999, 1000, 1001 cents
```

For time rules, I also test:

- one instant before the cutoff
- the exact cutoff instant
- one instant after it
- daylight-saving and time-zone conversion, when relevant
- leap days and month-end behavior for calendar-based policies

These are not merely unit tests for syntax. They are executable examples of the policy.

I ask the policy owner to confirm the expected middle row. “Exactly equal” is where vague language becomes a decision.

## Monitor the boundary population

A correct operator can still be attached to the wrong input definition.

In production I record counts around important thresholds:

```text
age < 30
age = 30
age > 30
age missing
```

A sudden spike at exactly 30 might indicate a new rounding rule, a batch default, or upstream clipping. A disappearance of fractional ages might mean a source changed from elapsed hours to completed days.

I also compare the binary feature rate before and after pipeline releases. If eligibility moves sharply while the underlying population appears stable, boundary semantics and feature units are among the first things I inspect.

Aggregate monitoring cannot prove that individual decisions are correct, but it can expose when a supposedly tiny edge begins carrying a large share of traffic.

## Keep deterministic rules out of the model's imagination

If eligibility is defined exactly by policy, I do not ask a statistical model to rediscover it from examples.

A model might learn a soft approximation around day 30. Regularization can move the effective boundary. Class imbalance can favor one side. A retrain can shift it again.

That may be appropriate when age is one predictive signal among many. It is not appropriate when `>= 30` is a contractual gate.

I prefer to separate the layers:

```text
policy rule: deterministic and auditable
model score: statistical and uncertain
decision: explicit combination of both
```

The model can estimate risk or expected value after the account qualifies. It should not reinterpret a hard eligibility clause unless that is an intentional product change.

## One character, one daily cohort

The transformer ran correctly. The matrix shape was correct. The output contained only zeros and ones. None of those facts told me that the feature matched the sentence written by the business.

`Binarizer::threshold(30.0)` means strictly greater than 30. It does not mean at least 30.

In our small experiment, the mismatch turned 100 eligible day-30 accounts into false negatives. Accuracy fell to `66.7%`, and only half of eligible accounts passed. The “edge case” occupied one third of the batch.

The durable fix was not a clever epsilon. It was to make the contract visible:

- use `Binarizer` when strict `>` semantics are intended
- write an explicit `>=` transformation for inclusive rules
- define units, rounding, missingness, and time behavior
- test below, equal, and above every important boundary
- monitor how much real traffic lands on the fence

A threshold is not only a number. It is a number, an operator, a unit, and a policy about equality.
