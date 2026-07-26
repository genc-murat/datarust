# Neither Feature Predicted the Label. Their Product Predicted Every One.

*A practical datarust guide to interaction effects, univariate feature selection, PolynomialFeatures, and the pipeline order that discarded a relationship before the model could see it.*

---

I scored two features against a binary label.

```text
signal_a score: 0.0
signal_b score: 0.0
```

Neither feature knew anything about the class on its own. A univariate selector kept one because I had asked it to keep the “best” feature, even though best meant winning a zero–zero tie.

The classifier reached `50%` accuracy.

Then I created one additional column:

```text
signal_a × signal_b
```

Its score was `115.0`. The same selector kept it, the same logistic regression model received it, and test accuracy became `100%`.

Nothing in the raw data changed. The label was never leaked into a feature. The only difference was whether I generated interactions before or after feature selection.

The first pipeline asked:

> Which raw feature is useful by itself?

The second asked:

> Which raw or interaction feature is useful by itself?

Those questions produced different candidate sets, and one of them made the true relationship impossible to represent.

Let's build the failure with [datarust](https://crates.io/crates/datarust), inspect the selected feature names, and look at the two confusion matrices.

## The label lives in the quadrant

Our dataset has two numeric features. Their signs determine the class:

```text
signal_a   signal_b   class
positive   positive     1
positive   negative     0
negative   positive     0
negative   negative     1
```

Class one means the signs agree. Class zero means they disagree.

This is the familiar XNOR pattern. It has a simple multiplicative description:

```text
class = 1 when signal_a × signal_b > 0
class = 0 when signal_a × signal_b < 0
```

But each raw feature is individually balanced.

Among positive `signal_a` rows, half belong to class zero and half to class one. The same is true for negative `signal_a`, positive `signal_b`, and negative `signal_b`. Knowing either coordinate alone provides no information about whether the signs match.

Training uses positive magnitudes from these sets:

```text
signal_a: 1, 2, 3, 4
signal_b: 1, 2, 3
```

For every magnitude pair, we generate all four sign combinations. That gives 48 balanced training rows.

The test set uses different magnitudes:

```text
signal_a: 0.5, 1.5, 2.5, 3.5
signal_b: 0.75, 1.75
```

Its 32 rows obey the same sign rule without repeating the training coordinates.

We compare two supervised pipelines:

```text
Pipeline A
SelectKBest(k=1) -> PolynomialFeatures -> LogisticRegression

Pipeline B
PolynomialFeatures -> SelectKBest(k=1) -> LogisticRegression
```

The components are almost identical. Their order decides whether the selector has an interaction column available to score.

Here is the complete Rust program:

```rust
use datarust::linear_model::{
    LogisticRegression, LogisticSolver,
};
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix,
};
use datarust::pipeline::{
    Pipeline, SupervisedPipeline,
};
use datarust::polynomial::PolynomialFeatures;
use datarust::selection::{ScoreFunc, SelectKBest};
use datarust::traits::{FeatureNames, Predictor};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn quadrant_data(
    a_values: &[f64],
    b_values: &[f64],
) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();

    for &a in a_values {
        for &b in b_values {
            for &(sign_a, sign_b) in &[
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ] {
                let x0 = sign_a * a;
                let x1 = sign_b * b;
                rows.push(vec![x0, x1]);
                labels.push(
                    if x0 * x1 > 0.0 { 1.0 } else { 0.0 }
                );
            }
        }
    }

    (Matrix::new(rows).unwrap(), labels)
}

fn classifier() -> LogisticRegression {
    LogisticRegression::new()
        .with_solver(LogisticSolver::Svd)
        .with_max_iter(200)
}

fn selector_state(
    model: &SupervisedPipeline<LogisticRegression>,
) -> (&[f64], &[bool]) {
    match model
        .transformers()
        .get_step("select")
        .unwrap()
    {
        TransformerKind::SelectKBest(selector) => {
            (selector.scores(), selector.get_support())
        }
        _ => unreachable!(),
    }
}

fn report(
    name: &str,
    model: &SupervisedPipeline<LogisticRegression>,
    x_test: &Matrix,
    y_test: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let predictions = model.predict(x_test)?;
    let names = model
        .transformers()
        .feature_names_out(Some(&[
            "signal_a".to_string(),
            "signal_b".to_string(),
        ]));
    let (scores, support) = selector_state(model);

    println!("{name}");
    println!("selector scores: {scores:?}");
    println!("selector support: {support:?}");
    println!("features reaching model: {names:?}");
    println!(
        "test accuracy: {:.3}",
        accuracy_score(y_test, &predictions)?
    );
    println!(
        "confusion matrix: {:?}\n",
        confusion_matrix(y_test, &predictions)?
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x_train, y_train) = quadrant_data(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 2.0, 3.0],
    );
    let (x_test, y_test) = quadrant_data(
        &[0.5, 1.5, 2.5, 3.5],
        &[0.75, 1.75],
    );

    let mut select_then_interact = Pipeline::new()
        .push(
            "select",
            TransformerKind::SelectKBest(
                SelectKBest::new(
                    ScoreFunc::FClassif,
                    1,
                )?,
            ),
        )
        .push(
            "interact",
            TransformerKind::PolynomialFeatures(
                PolynomialFeatures::new(2)
                    .interaction_only(true)
                    .include_bias(false),
            ),
        )
        .with_estimator(classifier());
    select_then_interact.fit(&x_train, &y_train)?;

    let mut interact_then_select = Pipeline::new()
        .push(
            "interact",
            TransformerKind::PolynomialFeatures(
                PolynomialFeatures::new(2)
                    .interaction_only(true)
                    .include_bias(false),
            ),
        )
        .push(
            "select",
            TransformerKind::SelectKBest(
                SelectKBest::new(
                    ScoreFunc::FClassif,
                    1,
                )?,
            ),
        )
        .with_estimator(classifier());
    interact_then_select.fit(&x_train, &y_train)?;

    println!(
        "train rows: {}, test rows: {}\n",
        x_train.nrows(),
        x_test.nrows()
    );

    report(
        "select raw -> create interactions",
        &select_then_interact,
        &x_test,
        &y_test,
    )?;
    report(
        "create interactions -> select",
        &interact_then_select,
        &x_test,
        &y_test,
    )?;

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

This is the output I measured against the current datarust codebase:

```text
train rows: 48, test rows: 32

select raw -> create interactions
selector scores: [0.0, 0.0]
selector support: [true, false]
features reaching model: ["signal_a"]
test accuracy: 0.500
confusion matrix: [[0, 16], [0, 16]]

create interactions -> select
selector scores: [0.0, 0.0, 115.0]
selector support: [false, false, true]
features reaching model: ["signal_a signal_b"]
test accuracy: 1.000
confusion matrix: [[16, 0], [0, 16]]
```

The first pipeline removed one half of the relationship. No later step could reconstruct it.

## SelectKBest scored what existed at that moment

`SelectKBest` with `FClassif` calculates an ANOVA F score independently for each input column. It compares how much the class means differ with how much values vary inside each class.

For `signal_a`, both class means are zero. Each class contains the same positive and negative magnitudes. The same symmetry holds for `signal_b`.

Therefore:

```text
selector scores: [0.0, 0.0]
```

The selector is not claiming the dataset has no signal. It is making a narrower statement:

> Neither column currently presented to me separates the classes by itself.

That statement is correct.

We configured `k = 1`, so the selector must still keep one column. datarust breaks equal-score ties by the lower column index, producing:

```text
selector support: [true, false]
features reaching model: ["signal_a"]
```

This is a useful reminder that “selected” does not always mean “informative.” Sometimes it means the API was required to return exactly `k` winners and one feature won a tie among equally unhelpful candidates.

I inspect the actual scores, not only the support mask.

## A deleted feature cannot participate later

After selection, Pipeline A passes one column into `PolynomialFeatures`:

```text
signal_a
```

The transformer is configured with:

```rust
PolynomialFeatures::new(2)
    .interaction_only(true)
    .include_bias(false)
```

With only one input feature, there is no two-feature interaction to create. Repeated powers such as `signal_a²` are excluded by `interaction_only(true)`, and the bias column is disabled. The logistic model receives only `signal_a`.

No estimator can infer the missing product from one coordinate. Rows with the same `signal_a` sign can belong to either class depending on the removed `signal_b` sign.

The fitted logistic model predicts class one for every test row:

```text
confusion matrix: [[0, 16], [0, 16]]
```

It correctly classifies the 16 class-one rows and misses all 16 class-zero rows. Balanced accuracy happens to equal ordinary accuracy here: `0.500`.

The failure did not originate in logistic regression. The representation reaching the estimator no longer contained the answer.

## Interaction first made the relationship visible

Pipeline B gives both raw columns to `PolynomialFeatures`. The generated feature names are conceptually:

```text
signal_a
signal_b
signal_a signal_b
```

The third name represents the product.

Now `SelectKBest` sees three candidates:

```text
selector scores: [0.0, 0.0, 115.0]
```

The raw coordinates remain individually useless. The product has negative values for class zero and positive values for class one, so its class means are clearly separated.

The selector keeps exactly that column:

```text
selector support: [false, false, true]
features reaching model: ["signal_a signal_b"]
```

Logistic regression only needs a threshold at zero in this one-dimensional space. It classifies all 32 test rows correctly, including magnitudes not used during fitting.

We did not replace the linear estimator with a nonlinear one. We changed the feature space so the nonlinear relationship in raw coordinates became a linear boundary in the product coordinate.

That is often what feature engineering does: it moves complexity out of the decision rule and into the representation.

## Univariate does not mean weak

Univariate selection can be extremely useful. If one sensor is strongly associated with failure and 10,000 other columns are noise, scoring columns independently can reduce the search space quickly and transparently.

Its limitation is structural:

```text
score(feature_j, target)
```

Each score is computed without asking how `feature_j` behaves jointly with another feature.

Pure interactions are the sharpest counterexample. Neither input has marginal information, yet the pair determines the answer.

Changing from `FClassif` to a nonlinear univariate score does not automatically repair this exact dataset. Mutual information between either raw coordinate and the class is also zero because they are genuinely independent. The information appears only in the joint pair or in a derived product feature.

The issue is not simply that ANOVA is too linear. The issue is that the selector receives one column at a time.

## The safe order depends on the intended candidate space

It would be too broad to conclude that polynomial expansion must always precede selection.

With `p` raw features, degree-two expansion can create approximately `p² / 2` pairwise terms. Starting with 10,000 columns would make all-pairs generation absurd before the selector had a chance to help.

There are several reasonable designs:

- Keep a broader raw shortlist, then generate interactions among the survivors.
- Generate only domain-approved interactions rather than every possible pair.
- Add known ratios, products, or differences as named source features.
- Use a model family that can learn relevant interactions without explicit polynomial expansion.
- Compare interaction strategies inside leakage-safe validation.

Pipeline A failed because `k = 1` guaranteed that a two-feature relationship could not survive. If it kept both raw features, the later polynomial step could create their product.

So I ask a practical question before aggressive selection:

> What is the smallest group of raw variables required to express the relationships I believe may exist?

If the answer includes pairs or groups, selecting isolated columns too early is risky.

## Both learned steps still belong inside validation

This controlled example uses a fixed train/test boundary. In model development, the candidate order and `k` should be evaluated with cross-validation on the training set.

`SelectKBest` consumes labels, so it must fit only on each fold's training rows. Putting it in datarust's `SupervisedPipeline` ensures the target-aware fitting happens inside the pipeline.

`PolynomialFeatures` does not inspect labels. Its transformation rule is deterministic once the input width and configuration are known. Keeping it inside the same pipeline still preserves the exact training-to-serving graph and feature order.

The honest validation flow for Pipeline B is:

```text
fold training rows
    -> fit interaction schema
    -> create interaction columns
    -> score/select using fold training labels
    -> fit logistic regression

fold validation rows
    -> apply the fitted interaction schema
    -> apply the fitted support mask
    -> predict
```

Creating all interactions and selecting them globally before cross-validation would let validation labels influence the support mask. The representation would be powerful for two reasons: one legitimate, one leaked.

Pipeline order and evaluation boundaries are separate concerns. Both must be correct.

## Feature names made the silent loss obvious

Both pipelines produced a one-column matrix for the estimator.

Shape alone could not explain the 50-point accuracy difference:

```text
Pipeline A model input: 32 × 1
Pipeline B model input: 32 × 1
```

Propagated feature names could:

```text
Pipeline A: ["signal_a"]
Pipeline B: ["signal_a signal_b"]
```

This is why I log the semantic schema after every selection or expansion step. A final matrix width tells me how many columns survived. It does not tell me whether the interaction, category, sensor, or time-derived feature I cared about is among them.

For an interaction pipeline, I keep:

- input feature names and order,
- generated feature names,
- selector scores,
- support mask,
- final model-input names,
- selected-feature stability across folds,
- validation results for candidate orders and values of `k`.

That turns “the model got worse” into a traceable representation change.

## The zero scores were true and incomplete

The raw selector reported:

```text
[0.0, 0.0]
```

It was not broken. Neither coordinate had a marginal relationship with the class.

The mistake was treating marginal relevance as the only kind of relevance that could exist.

Once the product column entered the candidate set, the same selector immediately found it:

```text
[0.0, 0.0, 115.0]
```

The same model then moved from an all-positive confusion matrix to a perfect diagonal one.

That result leaves me with a useful review question whenever a compact feature-selection stage sits early in a pipeline:

> Am I removing weak variables, or am I removing ingredients before their useful relationship has been constructed?

In this experiment, each ingredient was useless alone.

Together, they predicted every label.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
