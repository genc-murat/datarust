# My Cross-Validation Was 100% Accurate. One Fold Couldn't Even Train.

*A practical datarust experiment with KFold, StratifiedKFold, rare classes, and the difference between splitting every row and testing every class.*

---

The cross-validation report looked perfect.

```text
fold 1: 100% accuracy
fold 2: 100% accuracy
fold 3: 100% accuracy
fold 4: 100% accuracy
```

Then fold 5 failed before making a prediction.

The first four scores had not discovered an excellent classifier. Their test sets contained no positive examples. The fifth training set contained no positive examples. Every row participated in validation exactly once, yet the experiment had almost completely failed to evaluate the problem I cared about.

This is one of the uncomfortable truths behind K-fold cross-validation: distributing rows evenly is not the same thing as distributing classes sensibly.

For a balanced regression problem, ordinary `KFold` is often a reasonable starting point. For rare-event classification, the splitter becomes part of the experiment.

Let's make that visible with [datarust](https://crates.io/crates/datarust).

## A dataset with 5% positives — sorted at export time

The example contains 500 rows:

```text
475 negative examples
 25 positive examples
```

The two numeric features carry useful signal. Positive rows are shifted away from negatives, though the classes still overlap.

There is one operational detail that matters enormously: the rows are sorted by label. All negatives come first and all positives appear at the end.

That sounds like a deliberately hostile setup, and it is. It is also not difficult to encounter. A query may contain `ORDER BY status`; reviewed fraud cases may be appended in a batch; a spreadsheet may be grouped by outcome; an export tool may sort categories for readability.

The model does not know any of that. The splitter only sees row indices.

I compare three five-fold strategies:

1. `KFold` with its default `shuffle = false`
2. Shuffled `KFold` with a fixed random seed
3. Shuffled `StratifiedKFold` with the same seed

Each valid fold fits a fresh `StandardScaler` and `LogisticRegression` pipeline using only that fold's training rows.

Here is the complete program:

```rust
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::model_selection::{KFold, StratifiedKFold};
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::traits::Predictor;
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 11) as f64 / (1_u64 << 53) as f64)
            .max(f64::MIN_POSITIVE)
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn count_positive(y: &[f64]) -> usize {
    y.iter().filter(|&&label| label == 1.0).count()
}

fn metrics(
    y: &[f64],
    pred: &[f64],
) -> (f64, Option<f64>, Option<f64>) {
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut fn_ = 0_usize;
    let mut correct = 0_usize;

    for (&actual, &predicted) in y.iter().zip(pred) {
        if actual == predicted {
            correct += 1;
        }
        match (actual == 1.0, predicted == 1.0) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
            (false, false) => {}
        }
    }

    let precision = (tp + fp > 0)
        .then(|| tp as f64 / (tp + fp) as f64);
    let recall = (tp + fn_ > 0)
        .then(|| tp as f64 / (tp + fn_) as f64);

    (
        correct as f64 / y.len() as f64,
        precision,
        recall,
    )
}

fn show(value: Option<f64>) -> String {
    value.map_or_else(
        || "  n/a".to_string(),
        |v| format!("{v:5.2}"),
    )
}

fn evaluate(
    name: &str,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
    x: &Matrix,
    y: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{name}");
    println!("fold  train +  test +   accuracy  precision  recall");

    let mut accuracies = Vec::new();
    let mut recalls = Vec::new();

    for (fold, (train_idx, test_idx)) in
        folds.into_iter().enumerate()
    {
        let x_train = x.select_rows(&train_idx)?;
        let x_test = x.select_rows(&test_idx)?;
        let y_train: Vec<f64> =
            train_idx.iter().map(|&i| y[i]).collect();
        let y_test: Vec<f64> =
            test_idx.iter().map(|&i| y[i]).collect();
        let train_positive = count_positive(&y_train);
        let test_positive = count_positive(&y_test);

        let mut model = Pipeline::new()
            .push(
                "scale",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .with_estimator(
                LogisticRegression::new()
                    .with_solver(LogisticSolver::Svd)
                    .with_max_iter(200),
            );

        match model.fit(&x_train, &y_train) {
            Ok(()) => {
                let pred = model.predict(&x_test)?;
                let (accuracy, precision, recall) =
                    metrics(&y_test, &pred);
                accuracies.push(accuracy);
                if let Some(value) = recall {
                    recalls.push(value);
                }
                println!(
                    "{:>4}  {:>7}  {:>6}   {:>8.3}  {}      {}",
                    fold + 1,
                    train_positive,
                    test_positive,
                    accuracy,
                    show(precision),
                    show(recall),
                );
            }
            Err(error) => println!(
                "{:>4}  {:>7}  {:>6}   cannot fit: {error}",
                fold + 1,
                train_positive,
                test_positive,
            ),
        }
    }

    if !accuracies.is_empty() {
        let mean =
            accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        println!("mean accuracy over fitted folds: {mean:.3}");
    }
    if !recalls.is_empty() {
        let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        println!(
            "mean recall over folds containing positives: {mean:.3}"
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let mut rows = Vec::new();
    let mut y = Vec::new();

    // The export is sorted: 475 negatives, then 25 positives.
    for _ in 0..475 {
        rows.push(vec![rng.normal(), rng.normal()]);
        y.push(0.0);
    }
    for _ in 0..25 {
        rows.push(vec![
            4.0 + rng.normal(),
            4.0 + rng.normal(),
        ]);
        y.push(1.0);
    }

    let x = Matrix::new(rows)?;
    println!(
        "Dataset: {} rows, {} positives ({:.1}%)",
        y.len(),
        count_positive(&y),
        100.0 * count_positive(&y) as f64 / y.len() as f64,
    );

    let plain: Vec<_> = KFold::new()
        .with_n_splits(5)
        .split(y.len())?
        .collect();
    evaluate("KFold, no shuffle", plain, &x, &y)?;

    let shuffled: Vec<_> = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42)
        .split(y.len())?
        .collect();
    evaluate("KFold, shuffled", shuffled, &x, &y)?;

    let stratified: Vec<_> = StratifiedKFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42)
        .split(&y)?
        .collect();
    evaluate("StratifiedKFold", stratified, &x, &y)?;

    Ok(())
}
```

The actual output is the part worth studying:

```text
Dataset: 500 rows, 25 positives (5.0%)

KFold, no shuffle
fold  train +  test +   accuracy  precision  recall
   1       25       0      1.000    n/a        n/a
   2       25       0      1.000    n/a        n/a
   3       25       0      1.000    n/a        n/a
   4       25       0      1.000    n/a        n/a
   5        0      25   cannot fit: invalid input:
                        LogisticRegression requires at least 2 distinct classes
mean accuracy over fitted folds: 1.000

KFold, shuffled
fold  train +  test +   accuracy  precision  recall
   1       25       0      1.000    n/a        n/a
   2       18       7      0.980   1.00       0.71
   3       17       8      0.940   1.00       0.25
   4       20       5      0.990   1.00       0.80
   5       20       5      0.980   1.00       0.60
mean accuracy over fitted folds: 0.978
mean recall over folds containing positives: 0.591

StratifiedKFold
fold  train +  test +   accuracy  precision  recall
   1       20       5      1.000   1.00       1.00
   2       20       5      0.980   1.00       0.60
   3       20       5      0.970   1.00       0.40
   4       20       5      1.000   1.00       1.00
   5       20       5      1.000   1.00       1.00
mean accuracy over fitted folds: 0.990
mean recall over folds containing positives: 0.800
```

## Plain KFold fulfilled the row contract

Nothing in ordinary `KFold` promises to preserve class ratios. Without shuffling, it divides the row indices into consecutive blocks.

Our 500-row dataset becomes five test folds of 100 rows each. Because the positive rows occupy indices 475 through 499, the first four test folds contain only negatives. The final fold contains all 25 positives.

Every row is used for testing exactly once. No rows overlap between a fold's train and test sets. The mechanics are correct.

The experiment is still useless.

The first four folds report perfect accuracy because their only question is, “Can the model recognize negatives?” Precision and recall for the positive class are not zero; they are undefined because no positive event occurred and the model raised no positive alert.

The last fold has the opposite problem. All positives are held out, so its training data contains one class. Logistic regression cannot learn a boundary between two classes when only one class is present, and datarust correctly returns an error.

Averaging the four successful accuracy values would produce `1.000`. That number is mathematically tidy and scientifically indefensible.

## Shuffling removed the ordering, not the sampling risk

Turning on shuffle is a major improvement:

```rust
let cv = KFold::new()
    .with_n_splits(5)
    .with_shuffle(true)
    .with_random_state(42);
```

The positives are no longer trapped at the end. Four folds now contain rare events and every training set contains both classes.

But random allocation does not guarantee balance. With this seed, the test-fold positive counts are:

```text
0, 7, 8, 5, 5
```

One fold still cannot measure positive recall. Another gets 8 of the dataset's 25 positives, while its training set receives only 17. Class prevalence moves from 0% in one test fold to 8% in another, even though the full dataset is 5% positive.

This is not proof that shuffling is bad. It is proof that random does not mean representative. With rare classes, chance has fewer events to distribute.

The fixed seed is important for reproducibility, but it does not make a particular random partition statistically privileged. Seed 42 is repeatable; it is not sacred.

## StratifiedKFold preserved the question in every fold

`StratifiedKFold` groups indices by class and distributes each class across the folds. With 25 positives and five folds, every test set receives exactly five positives:

```text
train positives: 20
test positives:   5
```

Each training fold retains the dataset's 5% positive rate. Each test fold can measure precision and recall. No fold receives an easier evaluation problem merely because the rare class disappeared.

The datarust API makes the difference explicit. `KFold::split` takes a row count because it does not inspect labels:

```rust
for (train_idx, test_idx) in kfold.split(x.nrows())? {
    // ...
}
```

`StratifiedKFold::split` takes the targets because class membership determines the partition:

```rust
for (train_idx, test_idx) in stratified.split(&y)? {
    // ...
}
```

That small type-level difference describes the entire purpose of stratification.

## Stratification did not improve the classifier

The stratified run reports mean recall of `0.800`, while the shuffled KFold run reports `0.591` over the four folds where recall exists.

I would not conclude that stratification made the model better. The fitted training subsets and tested observations differ between the two schemes. Cross-validation is measuring models fitted on different data, so the scores are estimates with sampling variation.

What stratification improved is the *experiment design*:

- Every fold trains on both classes.
- Every fold tests on both classes.
- Class prevalence remains comparable across folds.
- The same metrics have meaning in every row of the report.

It reduces one avoidable source of score variance. It does not tune the probability threshold, rebalance the loss, create new positive examples, calibrate probabilities, or guarantee good recall.

The model and the evaluation protocol have different jobs.

## Five folds require at least five minority examples

Stratification cannot manufacture data.

If a dataset contains only three positive examples and I request five folds, at least two test folds must receive zero positives. Round-robin assignment cannot place one event into every fold when fewer events than folds exist.

My practical rule is:

```text
n_splits <= count of the rarest class
```

That is only the minimum for defined per-class metrics. One positive per fold still produces an extremely noisy recall estimate: it can only be `0` or `1`. For useful uncertainty, I want substantially more minority examples in each test fold.

When the rare class is genuinely tiny, repeated cross-validation, fewer folds, bootstrap analysis, uncertainty intervals, or a carefully protected holdout may be more honest than printing a precise-looking mean from fragile partitions.

## The scaler belongs inside each fold

Notice that the example creates and fits a fresh pipeline inside the loop:

```rust
let mut model = Pipeline::new()
    .push(
        "scale",
        TransformerKind::StandardScaler(StandardScaler::new()),
    )
    .with_estimator(LogisticRegression::new());

model.fit(&x_train, &y_train)?;
```

The scaler learns its mean and standard deviation only from the fold's training rows. Fitting one scaler on all 500 rows before cross-validation would let each test fold influence its own preprocessing.

Stratifying the labels while leaking feature statistics would fix one experimental boundary and break another.

The reusable unit is the whole preprocessing-and-model pipeline, not the final estimator alone.

## When stratification is the wrong answer

If rows are ordered by time, blindly shuffling or stratifying can leak the future into the past. A churn model evaluated on randomly mixed customer months may look excellent and fail when asked to predict next quarter. Time-aware validation must respect chronology even when class counts become inconvenient.

Grouped data creates a similar constraint. Multiple rows from the same patient, customer, device, or household should often remain in the same fold. Otherwise the model may be tested on an entity it effectively saw during training.

In those cases, the problem is not “KFold or StratifiedKFold?” It is defining the real independence boundary first — time, group, geography, batch, or deployment environment — and preserving class balance only where that boundary allows it.

Stratification is a tool, not permission to scramble structure that matters.

## What I now print before any score

The most useful addition to this experiment was not another metric. It was these two columns:

```text
train +
test +
```

Before trusting a cross-validation summary, I inspect the class count in every partition. A mean accuracy can hide an invalid fold. A standard deviation cannot explain a metric that was undefined. A successful API call cannot guarantee that each fold asked the business question.

For this dataset, the story is simple:

- Plain KFold produced four meaningless perfect scores and one model that could not train.
- Shuffling broke the label ordering but still left one test fold without a positive.
- Stratification gave every fold five positive examples and made all per-class metrics evaluable.

Cross-validation is often described as “use every row for both training and testing.” That definition is mechanically correct and incomplete.

For imbalanced classification, I need something stronger:

> Use every row — while making sure every fold still contains the problem the model is supposed to solve.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
