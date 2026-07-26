# My Random Labels Scored 69.5% Accuracy.

*A practical datarust demonstration of feature-selection leakage, accidental correlation, and why preprocessing belongs inside cross-validation.*

---

I built a classifier on data with no signal.

Not weak signal. Not noisy signal. No signal.

The feature matrix contained independent random numbers. The labels came from a separate random-number generator. Nothing in one could predict the other beyond chance.

The cross-validation accuracy was `69.5%`.

For a balanced binary target, that is the sort of result that earns a second experiment, a slide in a meeting, and possibly an optimistic sentence containing the word “promising.” It was also completely fake.

The classifier had not discovered structure in randomness. I had allowed feature selection to read the validation answers before cross-validation began.

This is one of the most dangerous forms of data leakage because the final code can look responsible:

- There is a feature selector.
- There is five-fold cross-validation.
- No target column is visibly copied into the matrix.
- Every API call succeeds.

The operations are individually reasonable. Their order invalidates the measurement.

In this article, we'll recreate the mistake with [**datarust**](https://crates.io/crates/datarust), then fix it by moving `SelectKBest` into a supervised pipeline. We will also increase the number of pure-noise features from 20 to 1,000 and watch the leaky score improve as the dataset becomes less informative.

That last part feels rude because it is exactly how accidental discovery works.

## The experiment has no hidden signal

We generate 200 rows and up to 1,000 numeric features. Every feature is sampled independently from a standard normal distribution.

The binary labels come from a second RNG with a different seed:

```text
X = random noise
y = independent random coin flips
```

No honest model should generalize meaningfully above 50% accuracy. A finite sample will move around that number — 47%, 53%, perhaps a little farther — but there is no relationship to learn.

For each feature count, we compare two workflows:

**Leaky workflow:** select the 10 features most associated with `y` using all 200 rows, then cross-validate logistic regression on that already-selected matrix.

**Honest workflow:** place `SelectKBest` before logistic regression inside the pipeline, so each fold selects features using only its own training rows.

Create a Rust project:

```sh
cargo new selection_leakage
cd selection_leakage
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::{cross_val_score, KFold};
use datarust::pipeline::Pipeline;
use datarust::selection::{ScoreFunc, SelectKBest};
use datarust::traits::Transformer;
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

    fn normal(&mut self) -> f64 {
        let u = self.next_f64().max(f64::MIN_POSITIVE);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut feature_rng = Rng::new(2026);
    let mut label_rng = Rng::new(99);

    let rows: Vec<Vec<f64>> = (0..200)
        .map(|_| {
            (0..1_000)
                .map(|_| feature_rng.normal())
                .collect()
        })
        .collect();
    let y: Vec<f64> = (0..200)
        .map(|_| {
            if label_rng.next_f64() < 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let all_x = Matrix::new(rows)?;
    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42);

    println!("features  leaky CV accuracy  honest CV accuracy");

    for n_features in [20, 100, 1_000] {
        let columns: Vec<usize> = (0..n_features).collect();
        let x = all_x.select_columns(&columns)?;

        // WRONG: the selector sees every label before cross-validation.
        let mut global_selector =
            SelectKBest::new(ScoreFunc::FClassif, 10)?;
        global_selector.fit_with_numeric_labels(&x, &y)?;
        let globally_selected = global_selector.transform(&x)?;

        let classifier = LogisticRegression::new()
            .with_solver(LogisticSolver::Svd)
            .with_max_iter(200);
        let leaky_scores = cross_val_score(
            &classifier,
            &globally_selected,
            &y,
            &cv,
            accuracy_score,
        )?;

        // RIGHT: each CV clone fits SelectKBest on its training fold.
        let honest = Pipeline::new()
            .push(
                "select",
                TransformerKind::SelectKBest(
                    SelectKBest::new(ScoreFunc::FClassif, 10)?,
                ),
            )
            .with_estimator(
                LogisticRegression::new()
                    .with_solver(LogisticSolver::Svd)
                    .with_max_iter(200),
            );
        let honest_scores = cross_val_score(
            &honest,
            &x,
            &y,
            &cv,
            accuracy_score,
        )?;

        println!(
            "{n_features:>8}  {:>17.3}  {:>18.3}",
            mean(&leaky_scores),
            mean(&honest_scores),
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seeds above:

```text
features  leaky CV accuracy  honest CV accuracy
      20              0.550               0.450
     100              0.660               0.550
    1000              0.695               0.530
```

The honest scores hover around chance.

The leaky scores improve as we add noise.

## `SelectKBest` did what it was asked to do

`ScoreFunc::FClassif` calculates an ANOVA F statistic between every feature and the class labels. Features whose group means differ more strongly receive higher scores.

In data with real signal, this can identify useful predictors. In pure noise, every true association is zero, but observed sample associations are not exactly zero. Some columns will look a little related to the label by chance.

With 20 noise features, the strongest accident is modest. With 1,000 noise features, we get 1,000 chances to find an impressive accident and keep the best 10.

This is the multiple-comparisons problem in practical clothing. If you flip enough coins, one sequence will look suspiciously intentional.

The selector is not malfunctioning. It correctly ranks the relationships in the sample it sees. The error is treating the winning sample correlations as if they were selected independently of the validation rows used to score them.

Feature selection is model fitting whenever it reads the target.

## The leaky workflow let validation labels choose the columns

Here is the incorrect order:

```rust
global_selector.fit_with_numeric_labels(&x, &y)?;
let selected = global_selector.transform(&x)?;

let scores = cross_val_score(
    &classifier,
    &selected,
    &y,
    &cv,
    accuracy_score,
)?;
```

Before the first fold exists, `SelectKBest` has already inspected every label. That includes labels that will later appear in validation folds.

Suppose feature 417 happens to be unusually high for positive rows in this finite dataset. It is selected globally because of that relationship. When fold 3 treats some rows as validation data, those validation labels have already contributed to choosing feature 417.

The classifier does not see validation labels during `fit`. It does not need to. The feature space arrived carrying information about them.

This is why leakage can survive a code review that looks only for `model.fit(x_test, y_test)`. Information can cross the boundary through imputation statistics, category vocabularies, scaling, PCA, feature selection, threshold tuning, or manual exploratory decisions.

The boundary applies to every learned choice, not only the final estimator.

## The pipeline creates the right experimental boundary

The honest workflow wraps selection and classification together:

```rust
let honest = Pipeline::new()
    .push(
        "select",
        TransformerKind::SelectKBest(
            SelectKBest::new(ScoreFunc::FClassif, 10)?,
        ),
    )
    .with_estimator(LogisticRegression::new());
```

`cross_val_score` clones this complete supervised pipeline for each fold. During `fit`, datarust passes the fold's training targets to `SelectKBest`, transforms those training features, and then fits logistic regression.

The validation rows are transformed using the selector fitted on training rows only. Their labels are used only by `accuracy_score` after prediction.

```text
fold training X,y
      ↓
fit SelectKBest on training labels
      ↓
transform training X
      ↓
fit LogisticRegression

fold validation X
      ↓
transform with the fitted selector
      ↓
predict
      ↓
compare with validation y
```

The pipeline is not just convenient composition. It is an executable statement about which observations are allowed to influence which fitted state.

## Why honest accuracy is not exactly 50%

With 1,000 features, the honest mean accuracy is `0.530`. With 20 features, it is `0.450`.

Neither contradicts the claim that the data contains no signal.

We have only 200 samples and five validation folds. Random outcomes vary. A fair coin does not produce exactly 50 heads in every 100 flips, and a chance classifier does not return exactly `0.500` in every experiment.

Repeat the entire data generation with different seeds and the honest scores will move around chance. Sometimes one run will look better than expected. That is why repeated evaluation, confidence intervals, and an untouched test set matter when differences are small.

The leaky pattern is different: accuracy rises systematically as the global selector receives more noise features to search. It is not evidence of learnable structure. It is evidence of a larger search space for coincidences.

## Selecting on all training data is eventually correct

There is an important nuance here.

After cross-validation has selected the workflow and hyperparameters, we normally fit the final selector and model on all available training data:

```rust
let mut final_model = Pipeline::new()
    .push(
        "select",
        TransformerKind::SelectKBest(
            SelectKBest::new(ScoreFunc::FClassif, 10)?,
        ),
    )
    .with_estimator(LogisticRegression::new());

final_model.fit(&x_train, &y_train)?;
let predictions = final_model.predict(&x_test)?;
```

That is not leakage. The entire training set is allowed to shape the final model. The independent test set is not.

The rule is relative to the evaluation boundary:

- During cross-validation, each fold's validation rows must remain unseen by every fitted step.
- During final fitting, all training rows may be used.
- During final evaluation, test rows must remain unseen until prediction.

“Never fit feature selection on all data” is too vague. “Never let evaluation rows influence feature selection” is the actual rule.

## Unsupervised preprocessing can leak too

`SelectKBest` makes the problem obvious because it consumes `y`. Unsupervised transformers do not get a free pass.

A scaler fitted on all rows sees validation means and variances. PCA fitted globally sees validation covariance structure. An imputer learns held-out column statistics. A variance threshold sees which features vary in the validation set.

These leaks are often smaller because they do not read labels directly, but they still make the validation transform more informed than a production transform would be.

The safe default is simple:

> If a preprocessing step has `fit`, put it inside the validation loop.

Pure row-local operations that learn no state — for example, a fixed mathematical transform — can be applied independently. Anything that estimates parameters or chooses columns belongs to the fitted pipeline.

## The feature search can extend beyond the code

Suppose the pipeline is perfectly honest, but we run it 200 times:

- Try different feature subsets
- Change random seeds
- Swap scoring functions
- Inspect validation errors
- Keep the most flattering result

The validation set begins to guide human feature selection. We can overfit it through repeated decisions even when no single API call leaks.

This is why a final untouched test set remains useful. It evaluates the complete development process, including decisions made after looking at cross-validation results.

If the dataset is small and model comparison is extensive, nested cross-validation gives an outer estimate while an inner loop chooses features and hyperparameters.

Leakage is not exclusively a software bug. It is an information-flow problem, and humans are part of the flow.

## Feature selection is still useful

None of this argues against `SelectKBest`.

Feature selection can:

- Reduce fitting time and model size
- Remove clearly irrelevant columns
- Improve generalization in high-dimensional, small-sample settings
- Make downstream coefficients easier to inspect
- Reduce noise before an estimator

datarust supports three scoring functions:

- `FClassif` for continuous features and categorical labels
- `Chi2` for non-negative features
- `MutualInformation` for histogram-estimated nonlinear dependence

Each has assumptions. Chi-square rejects negative inputs. Mutual information needs enough data for stable histogram estimates. ANOVA F scoring focuses on class-mean separation and may miss nonlinear relationships.

The selected feature count `k` is also a hyperparameter. Choosing `k = 10` because it sounds compact is still a modeling decision. Compare candidate values inside the same leakage-safe validation structure.

And preserve feature names. A support mask is much more useful when it tells you `temperature_delta` survived rather than `x417`.

## The practical result

The experiment contains no predictive relationship, yet global feature selection creates this table:

| Candidate noise features | Leaky CV accuracy | Honest CV accuracy |
|---:|---:|---:|
| 20 | 0.550 | 0.450 |
| 100 | 0.660 | 0.550 |
| 1,000 | 0.695 | 0.530 |

Adding 980 useless columns improved the leaky result by 14.5 percentage points.

The honest pipeline correctly refuses to turn that abundance of coincidence into repeatable performance.

The habits I would keep are straightforward:

1. Split evaluation data before any learned preprocessing.
2. Put supervised feature selection inside the pipeline.
3. Fit every stateful transformer within each cross-validation fold.
4. Tune `k` and the scoring method through the same validation process.
5. Keep a final test set outside repeated development decisions.
6. Be suspicious when adding noise makes validation better.

The classifier was never 69.5% accurate at predicting random labels.

The experiment was 69.5% accurate at predicting answers it had already partially seen.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), with `SelectKBest` covered in the [API reference](https://genc-murat.github.io/datarust/api.html).*
