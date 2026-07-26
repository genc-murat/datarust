# Every Count Was Non-Negative. StandardScaler Made Chi-Square Refuse Them.

*A practical datarust guide to chi-square feature selection, non-negative inputs, supervised pipeline order, score-sensitive scaling, and the preprocessing step that changed the statistic's question.*

---

The raw table contained counts:

```text
page views
email opens
support tickets
```

Every value was zero or greater.

I standardized the columns, because that was what the rest of the numeric pipeline did. Then I asked chi-square feature selection to keep the two columns most associated with the class label.

It refused to fit:

```text
chi2 requires non-negative features; negative at 0,0
```

My first reaction was that the validation check must be wrong. Counts cannot be negative.

The validation check was right. The values reaching it were no longer counts.

`StandardScaler` had subtracted each training-column mean. Any observation below its column mean became negative. The first raw page-view count was `0`; after standardization it was approximately `-1.14`.

I had not corrupted the data. I had changed its representation into one that no longer satisfied the statistic I chose.

Let's reproduce the failure with [datarust](https://crates.io/crates/datarust), compare two valid repairs, and look at why moving a selector one line earlier can change more than whether the code runs.

## Three count features, two useful signals

Our controlled dataset has 12 rows and three non-negative features:

```text
page_views       associated with class 1
email_opens      same pattern in both classes
support_tickets  associated with class 0
```

The first six rows belong to class `0`; the final six belong to class `1`.

Class `1` has many more page views. Class `0` has more support tickets. Both groups receive the same sequence of email-open counts, so that column carries no association with the label in this example.

We will try four things:

1. Standardize the matrix, then apply chi-square selection.
2. Apply chi-square directly to the raw counts.
3. Apply an ANOVA F-test to the standardized values.
4. Put chi-square before scaling in a target-aware pipeline.

The first path should fail. The remaining paths answer valid, though not identical, statistical questions.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new count_feature_selection
cd count_feature_selection
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::selection::{ScoreFunc, SelectKBest};
use datarust::traits::{FeatureNames, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn data() -> (Matrix, Vec<f64>) {
    let rows = vec![
        vec![0.0, 0.0, 4.0],
        vec![1.0, 1.0, 5.0],
        vec![0.0, 2.0, 4.0],
        vec![1.0, 3.0, 6.0],
        vec![2.0, 4.0, 5.0],
        vec![0.0, 5.0, 4.0],
        vec![8.0, 0.0, 0.0],
        vec![9.0, 1.0, 1.0],
        vec![10.0, 2.0, 0.0],
        vec![8.0, 3.0, 1.0],
        vec![9.0, 4.0, 0.0],
        vec![10.0, 5.0, 1.0],
    ];
    let labels = vec![0.0; 6]
        .into_iter()
        .chain(vec![1.0; 6])
        .collect();
    (Matrix::new(rows).unwrap(), labels)
}

fn selected_names(
    selector: &SelectKBest,
    names: &[String],
) -> Vec<String> {
    selector.feature_names_out(Some(names))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x, y) = data();
    let names = vec![
        "page_views".to_string(),
        "email_opens".to_string(),
        "support_tickets".to_string(),
    ];

    let mut scaler = StandardScaler::new();
    let standardized = scaler.fit_transform(&x)?;
    let raw_minimum = x
        .as_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let standardized_minimum = standardized
        .as_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    println!("Raw minimum:          {raw_minimum:.3}");
    println!(
        "Standardized minimum: {standardized_minimum:.3}"
    );
    println!("First raw row:         {:?}", x.row(0));
    println!(
        "First standardized:    {:?}",
        standardized.row(0),
    );

    let mut broken =
        SelectKBest::new(ScoreFunc::Chi2, 2)?;
    match broken.fit_with_numeric_labels(&standardized, &y) {
        Ok(()) => println!(
            "Unexpectedly fitted chi-square on standardized data"
        ),
        Err(error) => println!(
            "Chi-square after StandardScaler: ERROR: {error}"
        ),
    }

    let mut chi2 =
        SelectKBest::new(ScoreFunc::Chi2, 2)?;
    chi2.fit_with_numeric_labels(&x, &y)?;
    println!();
    println!("Chi-square on raw counts");
    println!(
        "  scores:   [{:.3}, {:.3}, {:.3}]",
        chi2.scores()[0],
        chi2.scores()[1],
        chi2.scores()[2],
    );
    println!(
        "  selected: {:?}",
        selected_names(&chi2, &names),
    );

    let mut f_test =
        SelectKBest::new(ScoreFunc::FClassif, 2)?;
    f_test.fit_with_numeric_labels(&standardized, &y)?;
    println!("F-test on standardized values");
    println!(
        "  scores:   [{:.3}, {:.3}, {:.3}]",
        f_test.scores()[0],
        f_test.scores()[1],
        f_test.scores()[2],
    );
    println!(
        "  selected: {:?}",
        selected_names(&f_test, &names),
    );

    let selector =
        SelectKBest::new(ScoreFunc::Chi2, 2)?;
    let mut valid_order = Pipeline::new()
        .push(
            "chi2",
            TransformerKind::SelectKBest(selector),
        )
        .push(
            "scale",
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        );
    let selected_then_scaled = valid_order
        .fit_transform_with_target(&x, &y)?;

    println!();
    println!(
        "Valid pipeline output shape: {} x {}",
        selected_then_scaled.nrows(),
        selected_then_scaled.ncols(),
    );
    println!(
        "Valid pipeline names: {:?}",
        valid_order.feature_names_out(Some(&names)),
    );

    let selector =
        SelectKBest::new(ScoreFunc::Chi2, 2)?;
    let mut invalid_order = Pipeline::new()
        .push(
            "scale",
            TransformerKind::StandardScaler(
                StandardScaler::new(),
            ),
        )
        .push(
            "chi2",
            TransformerKind::SelectKBest(selector),
        );

    match invalid_order.fit_transform_with_target(&x, &y) {
        Ok(_) => println!(
            "Unexpectedly fitted invalid pipeline"
        ),
        Err(error) => println!(
            "Reversed pipeline: ERROR: {error}"
        ),
    }
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
Raw minimum:          0.000
Standardized minimum: -1.464
First raw row:         [0.0, 0.0, 4.0]
First standardized:    [-1.1401076442575895, -1.4638501094227998, 0.6504869220243743]
Chi-square after StandardScaler: ERROR: invalid input: chi2 requires non-negative features; negative at 0,0

Chi-square on raw counts
  scores:   [43.103, 0.000, 20.161]
  selected: ["page_views", "support_tickets"]
F-test on standardized values
  scores:   [284.091, 0.000, 107.759]
  selected: ["page_views", "support_tickets"]

Valid pipeline output shape: 12 x 2
Valid pipeline names: ["page_views", "support_tickets"]
Reversed pipeline: ERROR: invalid input: chi2 requires non-negative features; negative at 0,0
```

The result is not a mysterious interaction between two libraries. Each step does exactly what its contract says.

The contracts simply do not compose in that order.

## StandardScaler changes the origin

For each feature, standardization computes:

```text
standardized = (value - training_mean) / training_std
```

The raw page-view column is:

```text
0, 1, 0, 1, 2, 0, 8, 9, 10, 8, 9, 10
```

Its mean is positive. Subtracting that mean sends every lower-than-average count below zero.

The first page-view value changes from:

```text
0.0 -> -1.1401
```

The first email-open value changes from:

```text
0.0 -> -1.4639
```

Negative standardized values do not mean negative page views. They mean “below the training mean, measured in training standard deviations.”

That is often an excellent representation for a linear model. It is not a collection of non-negative observed quantities anymore.

The scaler did not merely make the numbers smaller. It moved the origin and changed their interpretation.

## Why chi-square needs non-negative features

The chi-square selector asks whether the amount of each feature observed across classes differs from the amount expected under independence.

For a feature and class, the contribution has the familiar form:

```text
(observed - expected)² / expected
```

In datarust's selector, the observed quantity is the sum of that feature's values within a class. Expected quantities are derived from the feature total and class proportions.

This construction needs non-negative feature values. Counts, frequencies, and non-negative intensities can be accumulated into meaningful observed totals. Centered values can cancel each other, produce negative class sums, and destroy the count-like interpretation of “observed.”

That is why datarust checks the matrix before scoring and stops at the first negative cell:

```text
negative at 0,0
```

Failing loudly is useful here. A selector that silently produced numbers from invalid inputs would make the pipeline look healthy while the statistical meaning had already disappeared.

## Raw chi-square found the two intended features

Applied directly to the non-negative data, chi-square returns:

```text
page_views       43.103
email_opens       0.000
support_tickets  20.161
```

`page_views` differs sharply between the two classes. `support_tickets` also differs, in the opposite direction. Chi-square measures association rather than direction, so both receive positive scores.

`email_opens` has the same sequence `0..5` in each class. Its class totals equal the totals expected under independence, so its score is zero.

With `k = 2`, datarust keeps:

```text
["page_views", "support_tickets"]
```

The selector is supervised: it needs the class labels to compute these scores. That fact matters later when we place it inside validation folds.

## Selecting first, then scaling, preserves both contracts

The valid pipeline order is:

```text
non-negative counts
        |
        v
SelectKBest(Chi2)
        |
        v
StandardScaler
        |
        v
estimator
```

Chi-square sees the raw non-negative representation it requires. Only the two retained columns are then standardized for an estimator that may benefit from centered, comparable scales.

In datarust, `SelectKBest` needs target values, so the preprocessing pipeline is fitted with:

```rust
valid_order.fit_transform_with_target(&x, &y)?;
```

The output contains 12 rows and two columns, and feature-name propagation confirms their identity:

```text
["page_views", "support_tickets"]
```

For an end-to-end model, attach the estimator with `with_estimator(...)`. The resulting supervised pipeline passes labels to target-aware transformers during fitting and applies only the learned transformation during prediction.

That separation prevents production rows from influencing which features were selected.

## Reversing the order changes the question before it fails

The invalid pipeline is:

```text
non-negative counts
        |
        v
StandardScaler
        |
        v
SelectKBest(Chi2)
```

By the time chi-square receives the matrix, each column has approximately zero mean. The entries describe deviations above and below an average, not accumulated event amounts.

Even if the negative-value guard were removed, those centered class sums would not answer the original chi-square question.

This is why I do not treat the exception as an inconvenience to work around. It is evidence that the intended statistic and the supplied representation disagree.

## An F-test is compatible with centered continuous values

If the features should be interpreted as continuous measurements rather than count-like quantities, `ScoreFunc::FClassif` may be a better candidate.

The ANOVA F-test compares between-class variation with within-class variation. Negative feature values are valid, and standardization does not invalidate the calculation.

In our example, the F-test on standardized data selects the same two columns:

```text
page_views       284.091
email_opens        0.000
support_tickets  107.759
```

The numerical scores are not comparable with the chi-square scores. They come from different formulas and answer different questions. Their agreement on the selected names is a property of this clean synthetic dataset, not a general guarantee.

F-tests also carry assumptions. They focus on differences in class means relative to within-class variance and are best matched to approximately continuous features under a suitable sampling design. A strongly nonlinear dependency can exist even when class means are similar.

Choosing `FClassif` only to avoid an error would repeat the original mistake in a different form. I choose it when its statistical question matches the feature and problem.

## Mutual information is another question, not a universal escape hatch

`ScoreFunc::MutualInformation` can capture more general dependencies between a feature and the class label. The current datarust implementation bins each numeric feature and estimates mutual information from the resulting joint histogram.

It accepts values across a numeric range, including negative standardized values. But it introduces different choices:

- histogram resolution depends on sample count
- small datasets can produce unstable estimates
- binning can hide or invent apparent structure
- a larger score still does not reveal the direction of association

Mutual information is valuable when nonlinear dependency matters. It is not “chi-square without the annoying restriction.”

The score function should follow the data-generating meaning:

```text
non-negative count-like features -> consider Chi2
continuous mean differences       -> consider FClassif
broader discrete dependency       -> consider MutualInformation
```

Then validate the selected features and downstream model across folds rather than trusting a single ranking.

## Non-negative scaling can still change the ranking

A common suggestion is to replace `StandardScaler` with `MinMaxScaler` or `MaxAbsScaler` before chi-square. That can keep training values non-negative when the raw data is non-negative.

It also changes the chi-square score.

If one feature is multiplied by a positive constant `c`, both its observed and expected totals are multiplied by `c`. Its contribution becomes:

```text
(c × observed - c × expected)²
-------------------------------- = c × original contribution
         c × expected
```

So chi-square feature scores are sensitive to per-feature scale. Dividing every column by a different maximum or range can reorder the features.

That does not automatically make scaled chi-square invalid. It means the scaling policy becomes part of the statistical question. Binary indicators, raw event counts, normalized frequencies, and per-user rates represent different evidence even when all remain non-negative.

There is another production trap with `MinMaxScaler`: an unseen value below the training minimum can transform to a negative number if output is not clipped. A chi-square selector is normally used only during fitting, but any design that refits selection on a shifted batch needs to re-check its domain assumptions.

I do not insert a non-negative scaler merely to silence the guard. I first decide what quantity chi-square should accumulate.

## Imputation happens before the same contract check

Missing-value handling can also introduce invalid values.

If a non-negative count column is imputed with its training mean or median, the fill remains non-negative. A constant sentinel such as `-1`, however, breaks chi-square immediately:

```text
missing count -> -1 -> invalid Chi2 input
```

Even a non-negative fill changes class totals and can affect feature scores. If missingness differs by class, that change may be substantial.

For count features I decide explicitly whether missing means:

- a genuine zero event count
- an unknown count requiring imputation
- a failed measurement requiring a missingness indicator
- a row that should not enter this selector

Zero and unknown are not interchangeable just because both can keep a matrix non-negative.

A robust order might be:

```text
validate schema
-> handle missing values under a documented policy
-> verify finite, non-negative Chi2 inputs
-> fit Chi2 selection on training labels only
-> scale selected features if the estimator needs it
-> fit estimator
```

## Feature selection must stay inside validation

Fixing the operation order does not fix data leakage automatically.

Chi-square reads the labels. If I score all features on the complete dataset and then cross-validate a model using the chosen subset, every validation fold has already influenced which columns exist. With enough noisy features, that can create impressive validation performance from chance associations.

The selector belongs inside each training fold:

```text
fold training rows:
    fit imputation
    fit Chi2 selection
    fit scaling
    fit estimator

fold validation rows:
    transform with fitted steps
    predict
    score
```

datarust's supervised pipeline is designed for this target-aware flow. When cloned and evaluated through cross-validation, each fold learns its own selection mask without reading that fold's validation labels.

After choosing the complete configuration, fit one final pipeline on all approved training data and apply it unchanged to the untouched test set and production rows.

Step order protects statistical assumptions. Fold boundaries protect evaluation honesty. I need both.

## Validate meanings, not only ranges

Before using `ScoreFunc::Chi2`, I now audit more than `minimum >= 0`.

I record:

- feature names and units
- whether values are counts, frequencies, rates, or arbitrary magnitudes
- minimum, maximum, and missing-value count
- how zeros were created
- any per-feature scaling applied before selection
- class totals used in scoring
- selected names and scores for each validation fold

The range check catches obvious invalid input. The meaning check catches a subtler problem: a column can be non-negative and still be a poor match for chi-square.

An arbitrary ID, an ordinal code, or a temperature shifted upward by 100 can all be non-negative. That does not turn them into meaningful observed counts.

Non-negativity is necessary for this implementation. It is not sufficient evidence that the statistic is appropriate.

## The error message was the useful result

I began with three innocent count columns and a familiar preprocessing habit: standardize everything numeric.

After scaling, the minimum value moved from `0.000` to `-1.464`. Chi-square stopped at the first cell because the matrix no longer satisfied its contract.

Two valid paths emerged:

- preserve count meaning, apply chi-square first, then scale the selected columns
- treat the inputs as continuous features and choose a compatible score such as `FClassif`

Both selected `page_views` and `support_tickets` in our controlled data. They did so with different formulas, assumptions, and score values.

The lesson was larger than “put these two steps in the other order.” A preprocessing step can change the meaning of a feature before it changes its range. Once that happens, the next statistic may be answering a different question—or no valid question at all.

Every transformer has an output contract. Every selector has an input contract. A pipeline works when those contracts meet in the middle.
