# The Missing Value Wasn't Zero. My Model Treated It Like One.

*A practical datarust experiment with mean, median, and KNN imputation — measured against values we deliberately hid.*

---

Missing data creates a strangely persuasive temptation.

You have a numeric matrix. One value is absent. The model wants a number. Zero is a number.

Problem solved.

Except zero is rarely an empty box. It means zero dollars, zero degrees, zero purchases, zero millimeters, or zero years. Replacing “we do not know” with “we measured none” does not remove ambiguity. It turns ambiguity into a confident lie.

I've seen this happen because a CSV parser needed a default, because a database column was nullable, and because a model pipeline simply refused to accept `NaN`. The code became easier to run. The predictions became harder to trust.

In this article, we'll use [**datarust**](https://crates.io/crates/datarust) to compare five ways of handling missing numeric values:

1. Fill with zero
2. Fill with the training-column mean
3. Fill with the training-column median
4. Use five nearest neighbors with uniform weights
5. Use five nearest neighbors weighted by distance

The useful twist is that we know the correct answers. We will generate a complete dataset, hide 20% of the test cells, impute them, and compare each result with the values we removed.

Then we'll send every completed matrix through the same regression model to see whether better imputation actually helps the downstream task.

## A controlled missing-data experiment

Our synthetic dataset has six correlated numeric features created from three hidden signals. Each column has a different average — around `-5`, `10`, `20`, `30`, `50`, or `100` — but a similar amount of variation.

That detail is deliberate.

Different averages make zero a visibly bad default. Similar spreads keep distance-based comparison fair: one column does not dominate KNN merely because its unit varies a thousand times more than another.

We will create 300 complete training rows and 100 complete test rows. Only the test copy gets masked. The original test matrix stays private and is used solely to measure imputation error.

Create a new Rust application:

```sh
cargo new missing_values
cd missing_values
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::imputer::{
    ImputeStrategy, KnnImputer, KnnWeights, SimpleImputer,
};
use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        sigma
            * (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn make_data(rng: &mut Rng, n: usize) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);

    for _ in 0..n {
        let a = rng.normal(1.0);
        let b = rng.normal(1.0);
        let c = rng.normal(1.0);

        let row = vec![
            10.0 + a + rng.normal(0.08),
            20.0 + 0.9 * a + 0.2 * b + rng.normal(0.08),
            -5.0 + b + rng.normal(0.08),
            30.0 + 0.8 * b + 0.2 * c + rng.normal(0.08),
            100.0 + c + rng.normal(0.08),
            50.0 + 0.7 * a - 0.4 * c + rng.normal(0.08),
        ];

        let target = 2.0 * row[0]
            - 1.5 * row[2]
            + 0.8 * row[4]
            + rng.normal(0.4);

        rows.push(row);
        targets.push(target);
    }

    (Matrix::new(rows).unwrap(), targets)
}

fn hide_values(complete: &Matrix) -> (Matrix, Vec<bool>) {
    let mut masked = complete.clone();
    let mut hidden = vec![false; complete.nrows() * complete.ncols()];

    for i in 0..complete.nrows() {
        for j in 0..complete.ncols() {
            // A deterministic 20% mask keeps the example reproducible.
            if (i * 7 + j * 11) % 5 == 0 {
                masked.set(i, j, f64::NAN);
                hidden[i * complete.ncols() + j] = true;
            }
        }
    }

    (masked, hidden)
}

fn zero_fill(x: &Matrix) -> Matrix {
    let mut out = x.clone();

    for i in 0..out.nrows() {
        for j in 0..out.ncols() {
            if out.get(i, j).is_nan() {
                out.set(i, j, 0.0);
            }
        }
    }

    out
}

fn hidden_rmse(truth: &Matrix, filled: &Matrix, hidden: &[bool]) -> f64 {
    let mut squared_error = 0.0;
    let mut count = 0;

    for (index, &was_hidden) in hidden.iter().enumerate() {
        if was_hidden {
            let difference =
                truth.as_slice()[index] - filled.as_slice()[index];
            squared_error += difference * difference;
            count += 1;
        }
    }

    (squared_error / count as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let (train_x, train_y) = make_data(&mut rng, 300);
    let (test_truth, test_y) = make_data(&mut rng, 100);
    let (test_missing, hidden) = hide_values(&test_truth);

    let hidden_count = hidden.iter().filter(|&&value| value).count();

    let zero = zero_fill(&test_missing);

    let mut mean_imputer =
        SimpleImputer::new(ImputeStrategy::Mean);
    mean_imputer.fit(&train_x)?;
    let mean = mean_imputer.transform(&test_missing)?;

    let mut median_imputer =
        SimpleImputer::new(ImputeStrategy::Median);
    median_imputer.fit(&train_x)?;
    let median = median_imputer.transform(&test_missing)?;

    let mut knn_uniform =
        KnnImputer::new(5, KnnWeights::Uniform);
    knn_uniform.fit(&train_x)?;
    let uniform = knn_uniform.transform(&test_missing)?;

    let mut knn_distance =
        KnnImputer::new(5, KnnWeights::Distance);
    knn_distance.fit(&train_x)?;
    let distance = knn_distance.transform(&test_missing)?;

    // Train one downstream model on complete training data. Every imputation
    // strategy is evaluated with this exact same fitted model.
    let mut model = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .with_estimator(Ridge::new().with_alpha(1.0));
    model.fit(&train_x, &train_y)?;

    println!(
        "Hidden cells: {hidden_count} of {} (20.0%)",
        hidden.len()
    );
    println!("method          hidden RMSE   downstream R²");

    for (name, filled) in [
        ("zero", &zero),
        ("mean", &mean),
        ("median", &median),
        ("KNN uniform", &uniform),
        ("KNN distance", &distance),
    ] {
        let predictions = model.predict(filled)?;
        println!(
            "{name:<13} {:>11.3}   {:>13.3}",
            hidden_rmse(&test_truth, filled, &hidden),
            r2_score(&test_y, &predictions)?,
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6, the output is:

```text
Hidden cells: 120 of 600 (20.0%)
method          hidden RMSE   downstream R²
zero               48.213        -315.524
mean                0.793           0.858
median              0.788           0.859
KNN uniform         0.272           0.961
KNN distance        0.245           0.962
```

The distance-weighted KNN imputer reconstructs the hidden values most accurately and gives the downstream model the best predictions.

Zero does something more dramatic: it turns a healthy regression model into a catastrophe.

## Why zero is not neutral

One of our features is centered near `100`. Replacing a missing value with `0` says the observation is roughly 100 standard deviations away from its normal range. The Ridge model does not see an empty field. It sees an extreme measurement and responds accordingly.

That is why the zero-filled test set produces `R² = -315.524`. Negative `R²` means the predictions are worse than always returning the test target mean. Very negative means they are much worse.

This result is intentionally severe because the feature offsets make the mistake easy to see. In standardized data, zero might happen to equal the mean. In a count feature, zero may be a genuine value. In a sparse bag-of-words matrix, an absent term is naturally zero.

The point is not “never use zero.” It is:

> Use zero only when zero represents the missing observation correctly, not because the numeric type needs a value.

Sometimes a sentinel such as `-999` feels safer because it looks obviously artificial. To most estimators, it is simply an even more extreme number.

## Mean imputation is boring in a useful way

`SimpleImputer` learns one fill value per column:

```rust
let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
imputer.fit(&train_x)?;
let completed = imputer.transform(&test_missing)?;
```

Mean imputation gives every missing cell the training-column average. It does not use the rest of the row. Two people with completely different observed features receive the same replacement in a missing column.

That sounds crude because it is crude. It is also fast, deterministic, easy to explain, and surprisingly hard to beat when features have weak relationships or the missing fraction is small.

In our experiment, hidden-value RMSE falls from `48.213` with zero fill to `0.793` with the mean. Downstream `R²` recovers to `0.858`.

Mean imputation also pulls variance toward the center. If 30% of a column is missing and every absent value becomes exactly the mean, the completed column looks less variable than the underlying data. Correlations can weaken too.

It is a baseline, not a reconstruction of reality.

## Median helps when the average is easy to bully

Median imputation changes only the statistic:

```rust
let mut imputer = SimpleImputer::new(ImputeStrategy::Median);
```

The median is less sensitive to outliers and skewed distributions. A handful of enormous transactions can drag mean purchase value upward; the median stays closer to the typical observation.

Our synthetic columns are close to symmetric and contain no major outliers, so mean and median are nearly tied:

```text
mean RMSE:   0.793
median RMSE: 0.788
```

That is the result I would expect. Choosing median does not automatically improve an imputer. It changes the assumption from “missing values resemble the average” to “missing values resemble the middle observation.” The data distribution decides whether that distinction matters.

`SimpleImputer` also supports `MostFrequent` and `Constant(value)`. The most-frequent strategy can make sense for discrete numeric codes, though categorical strings belong in the categorical preprocessing path rather than being disguised as arbitrary numbers.

## KNN uses the rest of the row

The six features in this experiment are correlated. If column 0 is missing but column 1 is present, column 1 tells us quite a lot because both depend strongly on the same hidden signal.

KNN imputation uses that structure:

1. Compare the incomplete row with stored training rows over features observed in both.
2. Find the nearest `k` rows.
3. Fill each missing column from the neighbors that have that value.

In datarust:

```rust
let mut imputer = KnnImputer::new(5, KnnWeights::Distance);
imputer.fit(&train_x)?;
let completed = imputer.transform(&test_missing)?;
```

Distances are computed only over co-observed features. datarust scales the squared distance by the ratio of total features to co-observed features, so a comparison based on two known values is not treated exactly like one based on all six.

Uniform weighting gives every selected neighbor the same vote. Distance weighting gives closer rows more influence:

```text
KNN uniform RMSE:  0.272
KNN distance RMSE: 0.245
```

Both dramatically outperform column statistics here because the features were built to share information. Distance weighting adds a small improvement because the closest local patterns are especially informative.

If the columns were independent, there would be no useful neighborhood structure and mean imputation could be just as good — while being much cheaper.

## Distance has units, even when the code does not mention them

KNN is sensitive to feature scale. A column that varies from `0` to `100,000` can dominate a column that varies from `0` to `1`, even if the smaller feature is more predictive of the missing value.

Our features have different offsets but similar spreads. Offsets cancel when distances are calculated, so KNN sees comparable variation. Real datasets often do not give you that convenience.

Normally, you would normalize observed values using statistics learned from training data. There is a practical wrinkle: standard scaling and missingness must agree on a workflow. In datarust v0.6, `StandardScaler` expects complete input, so it cannot directly transform a matrix that still contains `NaN` values.

Options include:

- Put features into comparable domain units before they enter the matrix.
- Use a NaN-aware scaling step upstream, fitted on training observations only.
- Start with simple imputation, scale, and accept that KNN no longer has missing cells to refine.
- Use KNN only on a subset whose feature scales are already comparable.

Do not silently run raw KNN on kilometers, dollars, percentages, and event counts and assume “nearest” has a useful meaning.

## The imputer must not study the test answers

The evaluation design matters as much as the algorithm.

Our experiment keeps two versions of the test data:

```text
test_truth   → complete, used only for scoring hidden cells
test_missing → masked, passed to each fitted imputer
```

Every imputer learns only from `train_x`:

```rust
mean_imputer.fit(&train_x)?;
let mean = mean_imputer.transform(&test_missing)?;
```

If we called `fit_transform` on the combined train and test matrix, the mean and median would include test values. KNN would search test rows as reference neighbors. The evaluation would look better because the preprocessing had already studied the data it was supposed to simulate seeing for the first time.

This is data leakage wearing a preprocessing badge.

In cross-validation, the same rule applies: fit the imputer inside each training fold, then transform that fold's validation rows. A global imputation pass before cross-validation contaminates every fold with statistics from its held-out section.

## Better reconstruction helped the model — this time

We measure two outcomes:

- RMSE on the exact cells we hid
- `R²` from a Ridge model trained on complete training rows

The ordering is consistent: methods that reconstruct hidden cells better also preserve the downstream regression signal better.

That relationship is not guaranteed.

An imputer can achieve low average reconstruction error while damaging the one feature most important to the model. Conversely, a rough imputation may preserve enough decision-relevant structure for predictions to remain stable.

Choose an imputer using the actual downstream validation metric when prediction is the goal. Hidden-cell reconstruction is a useful diagnostic when ground truth can be simulated, not a substitute for end-to-end evaluation.

Our setup also gives the model an advantage: training rows are complete. In a real project, training data may contain missing values too. `SimpleImputer` learns statistics from available values, and `KnnImputer` can store an incomplete reference matrix, but you should test the real missingness pattern rather than assuming this clean-reference result will transfer.

## Missingness may itself be information

Imputation fills the hole, but it can erase the fact that a hole existed.

Suppose income is more often missing for new accounts, or a lab test is ordered only for higher-risk patients. The absence of a value may carry predictive information even after the numeric feature is imputed.

A common approach is to add a missingness indicator before filling:

```text
income        income_was_missing
52,000        0
mean_fill     1
```

That lets the model distinguish an observed mean-like value from an inserted mean. It can improve predictions, but it also deserves scrutiny: the indicator may encode a process artifact, access disparity, or policy that changes after deployment.

More fundamentally, ask why data is missing:

- **Missing completely at random:** absence is unrelated to observed or missing values.
- **Missing at random:** absence can be explained by observed variables.
- **Missing not at random:** absence depends on the missing value itself or an unobserved cause.

Mean and KNN imputation do not magically resolve these mechanisms. If high-income respondents systematically skip an income question, neighbors from observed incomes may still give a biased picture.

An imputer completes a matrix. It does not repair the data-generating process.

## The operational tradeoff

Simple imputation stores a small vector of fill values and transforms each cell once. KNN stores the reference dataset and compares new incomplete rows with candidate neighbors. That costs more memory and substantially more computation.

For a batch of millions of rows, the difference matters. datarust can parallelize KNN distance work with the optional `rayon` feature, but parallelism does not change the underlying search cost.

I would usually begin with mean or median, establish an end-to-end baseline, and earn the complexity of KNN with validation results. In this experiment, reducing hidden-cell RMSE from about `0.79` to `0.25` and raising downstream `R²` from `0.86` to `0.96` is a strong case. In another dataset, the gain may be too small to justify storing and searching the reference matrix.

Also plan for failure cases:

- A column that is entirely missing cannot provide a learned mean or median.
- A row with no co-observed features cannot be compared with KNN reference rows.
- New input must have the same feature count and ordering as training.
- Imputation ranges should be validated; mathematically plausible values may violate domain constraints.

datarust returns errors for these structural cases. Your application still needs to decide how to report, reject, or route the affected record.

## What I would take to production

The experiment tells a clean story:

| Method | Hidden-cell RMSE | Downstream R² | Main tradeoff |
|---|---:|---:|---|
| Zero | 48.213 | -315.524 | Fast, but semantically wrong here |
| Mean | 0.793 | 0.858 | Simple, smooths variation |
| Median | 0.788 | 0.859 | Robust to outliers |
| KNN uniform | 0.272 | 0.961 | Better local estimates, slower |
| KNN distance | 0.245 | 0.962 | Best here, same search cost |

I would not carry the winning row of that table blindly into another project. I would carry the workflow:

1. Preserve missing values as missing until the preprocessing boundary.
2. Fit imputers on training data only.
3. Start with a transparent baseline.
4. Test richer methods when features contain useful correlations.
5. Evaluate both reconstruction and the real downstream task.
6. Keep missingness indicators when absence may matter.
7. Monitor missing rates and patterns after deployment.

The most important step happens before any of those: deciding that “unknown” deserves to remain honest long enough to be handled deliberately.

A model can work with an estimate.

It cannot question a zero that arrived pretending to be a measurement.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [SimpleImputer and KnnImputer guide](https://genc-murat.github.io/datarust/guide/imputers.html).*
