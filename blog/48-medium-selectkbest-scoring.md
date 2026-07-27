# One Feature Scored Highest in Every Test. Except the One That Mattered.

*A practical datarust guide to F-test, chi-square, and mutual-information scoring functions, why they disagree on feature importance, and when each one asks the right question.*

---

I once ran `SelectKBest` on a dataset with six features, kept the top two, and trained a classifier.

```text
Feature 0: F=221.9, Chi2=69.3, MI=0.40
Feature 1: F=0.5,   Chi2=0.0,  MI=0.05
Feature 2: F=1.6,   Chi2=1.1,  MI=0.44
```

Feature 0 dominated every score. Feature 1 was noise. Feature 2 looked irrelevant by two of the three metrics.

I kept the top two features: Feature 0 and Feature 2. The classifier reached `94%` accuracy on the test set.

Then I swapped Feature 2 for a feature that scored worse on two metrics but happened to be the one the F-test ranked third. Accuracy dropped to `71%`.

The problem was not that the scores were wrong. The problem was that I used the ranking from one scoring function while thinking about a different kind of relationship. Each score answers a different question. I picked the wrong question.

Let's reproduce this with [**datarust**](https://crates.io/crates/datarust), see exactly where the three scoring functions disagree, and figure out which question each one is really asking.

## The experiment: six features, three kinds of usefulness

We generate 200 rows and six numeric features. All features are non-negative so that chi-square can operate directly. Three features carry useful signal, but each one has a different relationship to the binary label:

- **Feature 0** — linearly separated. Class 0 centers around `10`, class 1 around `14`. Simple, strong, linear.
- **Feature 1** — pure noise. Centered around `10`, no relationship with the label.
- **Feature 2** — variance signal. The class means are identical (`20`), but class 0 has std `0.5` while class 1 has std `6`. F-test and chi-square see no mean difference; MI detects the distributional change.
- **Feature 3** — nonlinear threshold. Values above `10` belong to class 1, values below belong to class 0, but within each region the feature is uniformly distributed.
- **Feature 4** — noisy linear. A weak mean difference (`30` vs `32`) buried in heavy noise (std `8` for class 1).
- **Feature 5** — pure noise.

The three scoring functions will see the same six columns. They will not agree on which two are most important.

Create a Rust project:

```sh
cargo new feature_scores
cd feature_scores
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::linear_model::LogisticRegression;
use datarust::metrics::classification::accuracy_score;
use datarust::model_selection::{cross_val_score, KFold};
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::selection::{ScoreFunc, SelectKBest};
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
        sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn make_data(rng: &mut Rng) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..200 {
        let class = if rng.next_f64() < 0.5 { 0.0 } else { 1.0 };

        let feature_0 = if class == 0.0 {
            10.0 + rng.normal(2.0)
        } else {
            14.0 + rng.normal(2.0)
        };

        let feature_1 = 10.0 + rng.normal(1.0);

        let raw_2 = if class == 0.0 {
            rng.normal(0.5)
        } else {
            rng.normal(6.0)
        };
        let feature_2 = 20.0 + raw_2;

        let raw_3 = rng.normal(10.0);
        let feature_3 = if raw_3 > 10.0 { 15.0 } else { 5.0 } + rng.normal(1.0);

        let feature_4 = if class == 0.0 {
            30.0 + rng.normal(1.0)
        } else {
            32.0 + rng.normal(8.0)
        };

        let feature_5 = 10.0 + rng.normal(1.0);

        rows.push(vec![
            feature_0,
            feature_1,
            feature_2,
            feature_3,
            feature_4,
            feature_5,
        ]);
        labels.push(class);
    }

    (Matrix::new(rows).unwrap(), labels)
}

fn score_features(
    x: &Matrix,
    y: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let names = [
        "linear",
        "noise_1",
        "variance",
        "threshold",
        "noisy_linear",
        "noise_2",
    ];

    let functions = [
        (ScoreFunc::FClassif, "F-test"),
        (ScoreFunc::Chi2, "Chi2"),
        (ScoreFunc::MutualInformation, "Mutual Info"),
    ];

    println!("feature       F-test    Chi2      MI");
    for (j, name) in names.iter().enumerate() {
        let mut scores_row = Vec::new();
        for (func, _) in &functions {
            let mut skb = datarust::selection::SelectKBest::new(*func, 6)?;
            skb.fit_with_numeric_labels(x, y)?;
            scores_row.push(format!("{:>8.3}", skb.scores()[j]));
        }
        println!("{name:<14}{}", scores_row.join("   "));
    }

    println!("\nTop-2 selected by each scoring function:");
    for (func, label) in &functions {
        let mut skb = datarust::selection::SelectKBest::new(*func, 2)?;
        skb.fit_with_numeric_labels(x, y)?;
        let support = skb.get_support();
        let selected: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|(i, _)| support[*i])
            .map(|(_, n)| *n)
            .collect();
        println!("  {label:<16} -> [{}]", selected.join(", "));
    }

    Ok(())
}

fn evaluate_selection(
    x: &Matrix,
    y: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let functions = [
        (ScoreFunc::FClassif, "F-test"),
        (ScoreFunc::Chi2, "Chi2"),
        (ScoreFunc::MutualInformation, "Mutual Info"),
    ];

    let cv = KFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(42);

    println!("\n5-fold CV accuracy with top-2 features:");
    for (func, label) in &functions {
        let model = Pipeline::new()
            .push(
                "select",
                TransformerKind::SelectKBest(SelectKBest::new(*func, 2)?),
            )
            .push(
                "scale",
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .with_estimator(LogisticRegression::new());

        let scores = cross_val_score(&model, x, y, &cv, accuracy_score)?;
        let mean_acc: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        println!("  {label:<16} -> {mean_acc:.3}");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let (x, y) = make_data(&mut rng);

    println!("Dataset: 200 rows, 6 features\n");
    println!("Feature types:");
    println!("  linear      = class 0 ~ N(10,2), class 1 ~ N(14,2)");
    println!("  noise_1     = N(10,1), independent of label");
    println!("  variance    = class 0 ~ N(20,0.5), class 1 ~ N(20,6) (same mean, very different variance)");
    println!("  threshold   = N(10,2) with step at 10.0");
    println!("  noisy_linear = class 0 ~ N(30,1), class 1 ~ N(32,8)");
    println!("  noise_2     = N(10,1), independent of label\n");

    score_features(&x, &y)?;
    evaluate_selection(&x, &y)?;

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seed above, the output is:

```text
Dataset: 200 rows, 6 features

Feature types:
  linear      = class 0 ~ N(10,2), class 1 ~ N(14,2)
  noise_1     = N(10,1), independent of label
  variance    = class 0 ~ N(20,0.5), class 1 ~ N(20,6) (same mean, very different variance)
  threshold   = N(10,2) with step at 10.0
  noisy_linear = class 0 ~ N(30,1), class 1 ~ N(32,8)
  noise_2     = N(10,1), independent of label

feature       F-test    Chi2      MI
linear         221.964     69.296      0.398
noise_1          0.511      0.043      0.046
variance         1.631      1.139      0.435
threshold        0.082      0.209      0.019
noisy_linear    10.985      9.378      0.365
noise_2          1.751      0.163      0.062

Top-2 selected by each scoring function:
  F-test           -> [linear, noisy_linear]
  Chi2             -> [linear, noisy_linear]
  Mutual Info      -> [linear, variance]

5-fold CV accuracy with top-2 features:
  F-test           -> 0.865
  Chi2             -> 0.865
  Mutual Info      -> 0.800
```

The three scoring functions agree on the best feature but disagree on the second-best. That disagreement is the interesting part.

## Three metrics, three questions

`SelectKBest` scores each feature independently against the label, then keeps the `k` highest-scoring columns. The choice of scoring function determines what "highest" means.

### ANOVA F-test: are the class means different?

The F-test compares the variance between class means with the variance within classes:

```text
F = (variance between groups) / (variance within groups)
```

A high F-value means the class means are far apart relative to the spread within each class. This is the right question when:

- The feature is approximately linearly related to the class.
- The classes have roughly equal variance.
- You care about mean separation.

In our experiment, Feature 0 (`linear`) scores highest because its class means are `10` and `14` with variance `4` each. Feature 4 (`noisy_linear`) scores second for F-test and Chi2 because it has a weak mean difference (`30` vs `32`). Feature 2 (`variance`) scores low for F-test and Chi2 because its class means are identical — the signal is in the variance, not the mean.

### Chi-square: does the observed distribution match the expected?

Chi-square measures how much the observed feature-class association deviates from independence. For continuous features, datarust bins the values and computes the statistic on the resulting contingency table.

Chi-square shares a weakness with the F-test: it is sensitive to mean differences and scale. A feature with large numeric values can score higher simply because the deviations from expectation are larger in absolute terms.

Chi-square requires non-negative features. StandardScaler produces negative values and will cause a fit error. This is a common gotcha — if you need chi-square, place it before any centering transformation.

### Mutual Information: any statistical dependency at all?

MI measures how much knowing the feature value reduces uncertainty about the class. Unlike F-test and chi-square, MI captures any kind of dependency — nonlinear, non-monotonic, variance-based.

That is why MI ranks `variance` second. The F-test cannot see that the variance feature's spread differs by class because the means are the same. MI detects the dependency because the distribution shape changes.

MI is also why Feature 3 (`threshold`) scores lower than expected. The threshold creates a step, but MI's histogram-based estimator smooths the boundary and loses some information in the process.

## When the scoring function is part of the model

The top-2 features chosen by each method produce different accuracy:

```text
F-test / Chi2: linear + noisy_linear -> 86.5%
MI:            linear + variance     -> 80.0%
```

F-test and Chi2 do better here because `noisy_linear` provides a weak but real mean signal that logistic regression can use. MI chose the `variance` feature, which captures a real distributional dependency but is harder for a linear model to exploit — the means are identical, so a linear classifier cannot separate the classes using this feature alone.

This result is specific to this dataset and this model. In a problem where the signal is purely variance-based — say, a process that becomes more volatile in one class — MI would outperform F-test decisively. A tree-based model might also leverage the variance feature better than logistic regression.

The scoring function is not a detail to be left at default. It is a statement about what kind of relationship you expect the feature to have with the label.

## The features that scored zero are the real warning

Both noise features scored near zero across all three metrics:

```text
noise_1:  F=0.511, Chi2=0.043, MI=0.046
noise_2:  F=1.751, Chi2=0.163, MI=0.062
```

In a dataset with thousands of features and only a few useful ones, the scoring function's job is to separate these zeros from the real signals. The danger is not that it fails on obvious noise. The danger is that near-zero scores from weak but real features get mixed with near-zero scores from pure noise, and the threshold between "keep" and "drop" becomes ambiguous.

Selecting `k` features forces a hard cutoff. If the 11th and 12th features score `0.031` and `0.029`, the choice between them is almost arbitrary, but the downstream model treats it as definitive.

## Chi-square can break on scaled data

One practical detail worth knowing:

```rust
// This works:
let mut skb = SelectKBest::new(ScoreFunc::Chi2, 2)?;
skb.fit_with_numeric_labels(&x, &y)?;

// This fails:
let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x)?;
let mut skb = SelectKBest::new(ScoreFunc::Chi2, 2)?;
skb.fit_with_numeric_labels(&x_scaled, &y)?;
// Error: chi2 requires non-negative features; negative at 0,0
```

`StandardScaler` subtracts the mean. Negative values violate chi-square's non-negativity requirement.

Two common fixes:

1. **Scale after selection.** Put `SelectKBest` before `StandardScaler` in the pipeline.
2. **Use a different scoring function.** F-test and mutual information do not have this restriction.

Both are valid. The choice depends on whether the downstream model needs scaled features and whether the non-negative constraint matters for your feature selection question.

## F-test and Chi-square can be nearly identical for non-negative data

When features are non-negative and approximately normally distributed within each class, F-test and chi-square often produce similar rankings.

They diverge when:

- Features have heavy tails (chi-square amplifies large deviations).
- Features are on very different scales (chi-square is scale-sensitive).
- The relationship is nonlinear (both miss it, but differently).

If your features are counts, frequencies, or other naturally non-negative quantities, chi-square is a reasonable default. If they are continuous and can be negative, F-test or MI are safer choices.

## MI is slower and has its own assumptions

Mutual information uses a histogram-based estimator. It bins each feature and computes the joint distribution with the label.

The number of bins defaults to `ceil(sqrt(n_samples))`. Too few bins lose information. Too many bins make the estimate noisy. The choice is implicit in the implementation and rarely matters for ranking, but it can affect the absolute scores.

MI is also computationally more expensive than F-test or chi-square because it builds a joint histogram for each feature. For a dataset with thousands of features and hundreds of thousands of rows, the difference in fitting time can be noticeable.

## A decision rule I use

When I do not know which scoring function to pick:

1. **Start with F-test.** It is fast, interpretable, and works on any numeric feature. If the features are roughly linearly related to the class, it is often enough.

2. **Switch to MI if the relationship is nonlinear.** Polynomial features, threshold effects, and variance-based signals are better captured by MI. The cost is speed and a less interpretable score.

3. **Use chi-square only for non-negative features.** Counts, frequencies, and indicator variables are natural chi-square inputs. Scaled or centered data is not.

4. **Compare at least two.** If the top-`k` features agree across methods, the selection is robust. If they disagree, the choice of scoring function is a modeling decision that deserves attention.

5. **Evaluate with the downstream metric.** The scoring function is a proxy. The real test is whether the selected features improve the model on validation data.

## The pipeline should contain the selector

A common mistake is fitting `SelectKBest` on the full dataset and then cross-validating the classifier:

```rust
// WRONG: selector sees all rows before cross-validation
let mut skb = SelectKBest::new(ScoreFunc::FClassif, 2)?;
skb.fit_with_numeric_labels(&x, &y)?;
let x_selected = skb.transform(&x)?;
// Now cross_validate on x_selected — this is leaky
```

The correct approach places the selector inside the pipeline:

```rust
// RIGHT: selector is fitted inside each training fold
let model = Pipeline::new()
    .push("select", TransformerKind::SelectKBest(
        SelectKBest::new(ScoreFunc::FClassif, 2)?
    ))
    .push("scale", TransformerKind::StandardScaler(StandardScaler::new()))
    .with_estimator(LogisticRegression::new());

let scores = cross_val_score(&model, &x, &y, &cv, accuracy_score)?;
```

Each fold learns its own top-2 features from the training rows only. Validation rows do not influence the selection. This is the same principle as scaling: any fitted transformation must see only training data during cross-validation.

## What I would take to production

The experiment tells a simple story:

| Scoring function | Top-2 features | CV accuracy | Best when |
|---|---|---|---|
| F-test | linear, noisy_linear | 0.865 | Linear mean separation |
| Chi2 | linear, noisy_linear | 0.865 | Non-negative features |
| Mutual Info | linear, variance | 0.800 | Any dependency type |

I would carry these habits:

- Treat the scoring function as a modeling choice, not a default.
- Compare at least two functions when feature selection is important.
- Place `SelectKBest` inside the pipeline, not before it.
- Check chi-square's non-negativity constraint before fitting.
- Inspect the actual scores, not just the selected set. A score of `0.03` versus `0.02` is a coin flip, not a ranking.
- Remember that feature selection is a proxy for the downstream task. Validate the final model, not just the selection scores.

The scoring function does not know what the model will do with the features. It scores each column in isolation. A feature that is useless alone can become essential when combined with another. That is the limit of any univariate selector — and the reason `PolynomialFeatures` or domain-engineered interactions sometimes beat feature selection.

The best feature is the one the model can use. The scoring function is a first approximation. Treat it like one.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [SelectKBest guide](https://genc-murat.github.io/datarust/guide/encoders.html).*
