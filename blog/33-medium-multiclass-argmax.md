# My Model Had Three Probability Columns. It Never Predicted the Third Class.

*A practical datarust guide to multiclass logistic regression, probability-column order, argmax decisions, confusion matrices, and the 99% accuracy that hid an unused class.*

---

The model returned three probabilities for every row.

For one rare-class example, they were:

```text
class 0: 0.000
class 1: 0.894
class 2: 0.106
```

Class two was not assigned zero probability. The model knew it existed.

It still never predicted class two.

Hard multiclass prediction chooses the class with the largest probability. `0.106` lost to `0.894` on every row where the rare class was plausible, so the third probability column never became the final decision.

The overall test accuracy was `99%`.

The macro F1 score was `0.649`.

Both numbers were correct. One counted all 100 rows equally, most of which belonged to an easy majority class. The other gave each class equal weight, including the class with zero successful predictions.

This is a different failure from a binary classifier using an inconvenient threshold. With three mutually exclusive classes, a probability can be substantial and still lose the argmax competition every time.

Let's make that behavior visible with [datarust](https://crates.io/crates/datarust), including the learned class order, probability matrix, predicted-class counts, and confusion matrix.

## Three labels, one missing distinction

The controlled dataset has one numeric feature and three classes:

```text
class 0: feature values near -2
class 1: feature values near +2
class 2: feature values near +2
```

Class zero is easy to separate. Classes one and two are deliberately indistinguishable from the feature matrix.

Their training counts differ:

```text
class 0: 180 rows
class 1:  18 rows
class 2:   2 rows
```

At the positive feature values shared by classes one and two, the model repeatedly sees class one nine times more often. It can learn that empirical proportion. It cannot invent a feature that explains which otherwise identical positive row belongs to class two.

The test set preserves the same proportions:

```text
class 0: 90 rows
class 1:  9 rows
class 2:  1 row
```

That single rare row makes the accuracy arithmetic especially sharp. Correctly classify the first 99 and miss the last one, and accuracy becomes `0.990`.

Here is the complete Rust program:

```rust
use datarust::linear_model::{
    LogisticRegression, LogisticSolver,
};
use datarust::metrics::classification::{
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
};
use datarust::traits::Predictor;
use datarust::Matrix;

fn add_rows(
    rows: &mut Vec<Vec<f64>>,
    labels: &mut Vec<f64>,
    count: usize,
    class: f64,
) {
    for i in 0..count {
        let x = if class == 0.0 {
            -2.1 + 0.2 * (i % 2) as f64
        } else {
            1.9 + 0.2 * (i % 2) as f64
        };
        rows.push(vec![x]);
        labels.push(class);
    }
}

fn dataset(counts: [usize; 3]) -> (Matrix, Vec<f64>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();

    for (class, count) in counts.into_iter().enumerate() {
        add_rows(
            &mut rows,
            &mut labels,
            count,
            class as f64,
        );
    }

    (Matrix::new(rows).unwrap(), labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x_train, y_train) = dataset([180, 18, 2]);
    let (x_test, y_test) = dataset([90, 9, 1]);

    let mut model = LogisticRegression::new()
        .with_solver(LogisticSolver::Svd)
        .with_max_iter(200);
    model.fit(&x_train, &y_train)?;

    let predictions = model.predict(&x_test)?;
    let probabilities = model.predict_proba(&x_test)?;

    println!(
        "learned class order: {:?}",
        model.classes()
    );
    println!(
        "probability matrix: {} x {}",
        probabilities.nrows(),
        probabilities.ncols()
    );
    println!(
        "confusion matrix: {:?}",
        confusion_matrix(&y_test, &predictions)?
    );
    println!(
        "accuracy:        {:.3}",
        accuracy_score(&y_test, &predictions)?
    );
    println!(
        "macro precision: {:.3}",
        precision_score(&y_test, &predictions)?
    );
    println!(
        "macro recall:    {:.3}",
        recall_score(&y_test, &predictions)?
    );
    println!(
        "macro F1:        {:.3}",
        f1_score(&y_test, &predictions)?
    );

    let predicted_counts: Vec<usize> = model
        .classes()
        .iter()
        .map(|class| {
            predictions
                .iter()
                .filter(|value| *value == class)
                .count()
        })
        .collect();
    println!("predicted counts: {predicted_counts:?}");

    println!("\nrepresentative probabilities");
    for &row in &[0usize, 90, 99] {
        println!(
            "true class {:.0}: \
             [{:.3}, {:.3}, {:.3}] -> predicted {:.0}",
            y_test[row],
            probabilities.get(row, 0),
            probabilities.get(row, 1),
            probabilities.get(row, 2),
            predictions[row],
        );
    }

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

This is the output I measured against the current datarust codebase:

```text
learned class order: [0.0, 1.0, 2.0]
probability matrix: 100 x 3
confusion matrix: [[90, 0, 0], [0, 9, 0], [0, 1, 0]]
accuracy:        0.990
macro precision: 0.633
macro recall:    0.667
macro F1:        0.649
predicted counts: [90, 10, 0]

representative probabilities
true class 0: [1.000, 0.000, 0.000] -> predicted 0
true class 1: [0.000, 0.894, 0.106] -> predicted 1
true class 2: [0.000, 0.894, 0.106] -> predicted 1
```

The model separated negative from positive rows almost perfectly. It had no basis for separating the two positive classes.

## Probability columns follow the learned class order

For multiclass logistic regression, `predict_proba` returns one column per learned class. The column index is not a label that should be guessed from context.

The model exposes the mapping:

```rust
println!("{:?}", model.classes());
```

Our output is:

```text
[0.0, 1.0, 2.0]
```

Therefore a probability row such as:

```text
[0.000, 0.894, 0.106]
```

means:

```text
P(class 0) = 0.000
P(class 1) = 0.894
P(class 2) = 0.106
```

I keep the class array beside any exported probability matrix. A `100 × 3` shape says there are three scores per row. It does not preserve the semantic label attached to each column once those values leave the model API.

This becomes especially important when application labels are later rendered as names such as `normal`, `review`, and `fraud`. If the display layer assumes a different order, every probability can be numerically valid and assigned to the wrong class.

For binary models, datarust offers the convenience method `predict_positive_proba`. That method is intentionally binary. In multiclass work, I use the full matrix and the explicit `classes()` mapping.

## Hard prediction is an argmax decision

`predict` converts each probability row into one class label by selecting the largest entry.

For the rare-class row:

```text
[0.000, 0.894, 0.106]
```

the winner is class one.

The class-two probability is meaningful in the soft output, but hard prediction does not ask whether `0.106` is “large enough to care about.” It asks only whether it is larger than both competing probabilities.

It never is.

This is why the predicted counts are:

```text
[90, 10, 0]
```

The model emits a third probability column on all 100 rows and a third hard label on none of them.

That distinction matters in triage systems. A class might never win first place while still appearing often enough in second place to justify manual review, top-k routing, or a separate business rule. Conversely, turning every nontrivial second-place score into an alert can create an unusable queue.

Probability output and decision policy are separate layers. The default argmax policy is useful, not sacred.

## Accuracy counted rows, not classes

The model gets 99 of 100 test rows right:

```text
90 class-zero rows correct
 9 class-one rows correct
 0 class-two rows correct
--------------------------
99 correct out of 100
```

That produces:

```text
accuracy = 99 / 100 = 0.990
```

Accuracy answers:

> What fraction of all rows received the correct hard label?

If class frequency matches business importance, this may be useful. Here it makes complete failure on class two cost only one percentage point.

The number is not dishonest. It is row-weighted by definition.

Calling the model “99% accurate” without reporting class support and class-level behavior would be the dishonest part.

## Macro metrics gave every class a vote

For multiclass inputs, datarust's `precision_score`, `recall_score`, and `f1_score` return macro averages. The calculation first evaluates each class one-versus-the-rest, then takes the arithmetic mean across classes.

From the confusion matrix, the per-class behavior is:

| Class | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 |
| 1 | 0.900 | 1.000 | 0.947 |
| 2 | 0.000 | 0.000 | 0.000 |

Class one has one false positive: the class-two row assigned to it. Class two has no predicted positives and no true positives, so datarust handles the zero denominators as zero.

Average those three class values equally:

```text
macro precision = (1.000 + 0.900 + 0.000) / 3 = 0.633
macro recall    = (1.000 + 1.000 + 0.000) / 3 = 0.667
macro F1        = (1.000 + 0.947 + 0.000) / 3 = 0.649
```

Macro F1 does not care that class zero owns 90% of the rows. Each class contributes one third of the final value.

That makes it a useful counterweight when every class matters. It can also be noisy when a class has only one test example. Metric choice does not solve inadequate evaluation support.

## The confusion matrix located the missing class

datarust returns a matrix where rows are true classes and columns are predicted classes:

```text
                 predicted
              0    1    2
true 0       90    0    0
true 1        0    9    0
true 2        0    1    0
```

Two details are immediately visible:

- The third prediction column sums to zero. Class two is never emitted.
- The single class-two row lands in the class-one column.

A scalar metric cannot show that direction of confusion. The model might instead have confused class two with class zero, or split it across both alternatives. Those errors could have very different operational consequences.

I print predicted-class counts beside the confusion matrix because an empty prediction column is an important deployment signal. A model can technically support a class while functionally never selecting it.

## The probabilities exposed an information problem

The representative class-one and class-two rows are identical in feature space:

```text
true class 1: x near +2
true class 2: x near +2
```

The model returns the same probabilities for both:

```text
[0.000, 0.894, 0.106]
```

This is not merely a threshold problem. There is no decision boundary that can assign two identical feature rows to different labels consistently.

The probabilities reflect the relative frequency observed in that region. Class one is much more common, so it wins.

Resampling class two more heavily could raise its probability. A class-specific decision rule could make it win sooner. Those changes would move which positive rows receive which label, but they would not reveal which indistinguishable row truly belongs to class two.

If both classes matter, the durable repair is new information:

- a feature that captures the missing distinction,
- a corrected label definition,
- a temporal or contextual signal available at prediction time,
- or a decision process that acknowledges the ambiguity.

Weighting can change the cost of a mistake. It cannot create separation absent from the features.

## A custom decision rule needs its own evaluation

Suppose the application wants to review any row with:

```text
P(class 2) >= 0.10
```

Our rare-class row would enter review. So would all nine class-one rows in the same positive region, because they receive the same `0.106` probability.

The review rule would find the rare case with ten candidates:

```text
1 true class-two row
9 class-one rows
```

Whether that is useful depends on review capacity and mistake cost. It is neither automatically better nor worse than argmax.

Once I replace argmax with thresholds, top-k routing, abstention, or manual review, I evaluate the complete policy:

- coverage,
- per-class recall,
- false alerts,
- queue size,
- cost at the chosen operating point,
- and behavior when class prevalence changes.

I do not tune that rule on the final test set and continue calling the result untouched test performance. Decision policy is another model-development choice and needs a validation boundary.

## One rare test row is a fragile estimate

The controlled test set uses one class-two row to make the arithmetic transparent. Missing it gives exactly zero recall for the class and exactly 99% total accuracy.

In a real evaluation, one example is not enough to estimate performance reliably. If the model happened to classify that single row correctly, measured recall would jump from zero to one. Neither result would provide a stable picture of future rare cases.

I would also inspect:

- support per class in every split,
- multiple stratified folds when random splitting is appropriate,
- grouped or chronological boundaries when the application requires them,
- uncertainty intervals for per-class metrics,
- prevalence drift between training and serving,
- and whether every production class existed during training.

Macro averaging makes rare classes visible. It does not manufacture more rare observations.

## What I report for multiclass models

My minimum report is no longer one accuracy value. It includes:

- learned class order,
- true support per class,
- predicted count per class,
- confusion matrix,
- overall accuracy,
- macro precision, recall, and F1,
- probability summaries by true class,
- and the exact rule that turns probabilities into actions.

For this experiment, that report tells a coherent story:

```text
accuracy:             0.990
macro F1:             0.649
class-two test rows:  1
class-two predictions: 0
class-two probability on its row: 0.106
```

The model was excellent at recognizing the negative region, strong at choosing the common positive class, and incapable of distinguishing the rare positive class from its identical neighbor.

“99% accurate” compressed all of that into the least informative sentence available.

## The third column existed only in the soft answer

The probability matrix had the expected width:

```text
100 rows × 3 classes
```

Every row's probabilities summed to one. The class order was explicit. The model API behaved correctly.

The surprise came from treating three probability columns as evidence that all three classes participated in hard predictions.

They did not.

Class two received `0.106` where it was plausible, lost to class one's `0.894`, and finished with zero predicted rows. Accuracy barely noticed. Macro F1 and the confusion matrix did.

So after fitting a multiclass model, I now ask a question that sounds almost too simple:

> Did the model actually predict every class it claims to support?

In this experiment, the answer was sitting in the column sums.

The third probability column was present.

It was just never the largest.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
