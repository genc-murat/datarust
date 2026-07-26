# Every Value Appeared Once. “Most Frequent” Chose the Minimum.

*A practical datarust guide to mode-imputation ties, deterministic fallbacks, continuous features, fragile training snapshots, and the missing temperature whose replacement jumped three degrees after one new row.*

---

The imputer said the most frequent temperature was `18`.

The training values were:

```text
18, 19, 20, 21
```

Every value appeared exactly once.

There was no single most frequent temperature. There was a four-way tie, and `18` was simply the smallest candidate.

I reversed the training rows:

```text
21, 20, 19, 18
```

The fill value stayed `18`. This was not an accidental “first row wins” behavior. The choice was deterministic.

Then one additional observation arrived:

```text
18, 19, 20, 21, 21
```

Now `21` genuinely appeared most often. Every missing temperature transformed from `18` to `21` after a one-row data refresh.

In a fixed downstream model, that three-degree imputation change moved the prediction by 15 units.

Nothing crashed. `MostFrequent` followed its contract in both snapshots. My mistake was reading the word “most” as evidence that one value had actually dominated the data.

Let's reproduce the behavior with [datarust](https://crates.io/crates/datarust), compare it with mean and median filling, and make ties visible before a deterministic fallback becomes an invisible business rule.

## A mode is not always one value

The mode of a sample is the value or values with the highest frequency.

For this column:

```text
value  count
18     1
19     1
20     1
21     1
```

all four values are modes under the mathematical definition. A software API that must return one scalar needs a tie-breaking rule.

datarust's numeric mode implementation uses a clear deterministic policy:

```text
highest count wins
if counts tie, smallest value wins
```

That policy is reproducible and independent of row order. It also means a continuous column with all unique observations receives its minimum as the “most frequent” fill.

The output is valid. Whether it is meaningful depends on the feature.

## The complete Rust experiment

Create a small application and add datarust:

```sh
cargo new mode_imputation
cd mode_imputation
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::imputer::{
    ImputeStrategy, SimpleImputer,
};
use datarust::linear_model::LinearRegression;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

fn fitted_imputer(
    values: &[f64],
    strategy: ImputeStrategy,
) -> Result<SimpleImputer, Box<dyn std::error::Error>> {
    let x = Matrix::new(
        values
            .iter()
            .map(|&value| vec![value])
            .collect(),
    )?;
    let mut imputer = SimpleImputer::new(strategy);
    imputer.fit(&x)?;
    Ok(imputer)
}

fn filled_value(
    imputer: &SimpleImputer,
) -> Result<f64, Box<dyn std::error::Error>> {
    let missing = Matrix::new(vec![vec![f64::NAN]])?;
    Ok(imputer.transform(&missing)?.get(0, 0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tied = [18.0, 19.0, 20.0, 21.0];
    let reversed = [21.0, 20.0, 19.0, 18.0];
    let revised = [18.0, 19.0, 20.0, 21.0, 21.0];

    let tied_mode = fitted_imputer(
        &tied,
        ImputeStrategy::MostFrequent,
    )?;
    let tied_mean = fitted_imputer(
        &tied,
        ImputeStrategy::Mean,
    )?;
    let tied_median = fitted_imputer(
        &tied,
        ImputeStrategy::Median,
    )?;
    let reversed_mode = fitted_imputer(
        &reversed,
        ImputeStrategy::MostFrequent,
    )?;
    let revised_mode = fitted_imputer(
        &revised,
        ImputeStrategy::MostFrequent,
    )?;

    println!("Four unique temperatures: {tied:?}");
    println!(
        "  MostFrequent fill: {:.1}",
        filled_value(&tied_mode)?,
    );
    println!(
        "  Mean fill:         {:.1}",
        filled_value(&tied_mean)?,
    );
    println!(
        "  Median fill:       {:.1}",
        filled_value(&tied_median)?,
    );
    println!("Reversed row order: {reversed:?}");
    println!(
        "  MostFrequent fill: {:.1}",
        filled_value(&reversed_mode)?,
    );
    println!("One repeated value: {revised:?}");
    println!(
        "  MostFrequent fill: {:.1}",
        filled_value(&revised_mode)?,
    );

    let binary_tie = fitted_imputer(
        &[0.0, 1.0, 0.0, 1.0],
        ImputeStrategy::MostFrequent,
    )?;
    println!(
        "Binary tie [0, 1, 0, 1] -> fill {:.1}",
        filled_value(&binary_tie)?,
    );

    let calibration_x = Matrix::new(
        (15..=25)
            .map(|temperature| vec![temperature as f64])
            .collect(),
    )?;
    let calibration_y: Vec<f64> = (15..=25)
        .map(|temperature| {
            5.0 * temperature as f64 + 10.0
        })
        .collect();
    let mut model = LinearRegression::new();
    model.fit(&calibration_x, &calibration_y)?;

    let tied_input = Matrix::new(vec![vec![
        filled_value(&tied_mode)?,
    ]])?;
    let revised_input = Matrix::new(vec![vec![
        filled_value(&revised_mode)?,
    ]])?;
    let prediction_tied = model.predict(&tied_input)?[0];
    let prediction_revised =
        model.predict(&revised_input)?[0];

    println!();
    println!(
        "Downstream prediction from missing temperature"
    );
    println!(
        "  tied-mode artifact:    {prediction_tied:.1}"
    );
    println!(
        "  revised-mode artifact: {prediction_revised:.1}"
    );
    println!(
        "  prediction shift:      {:.1}",
        prediction_revised - prediction_tied,
    );
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
Four unique temperatures: [18.0, 19.0, 20.0, 21.0]
  MostFrequent fill: 18.0
  Mean fill:         19.5
  Median fill:       19.5
Reversed row order: [21.0, 20.0, 19.0, 18.0]
  MostFrequent fill: 18.0
One repeated value: [18.0, 19.0, 20.0, 21.0, 21.0]
  MostFrequent fill: 21.0
Binary tie [0, 1, 0, 1] -> fill 0.0

Downstream prediction from missing temperature
  tied-mode artifact:    100.0
  revised-mode artifact: 115.0
  prediction shift:      15.0
```

The smallest value wins both the continuous and binary ties. One extra observation removes the continuous tie and moves the learned fill to the largest value.

## Deterministic does not mean data-supported

Reversing the row order proves something useful:

```text
[18, 19, 20, 21] -> 18
[21, 20, 19, 18] -> 18
```

The imputer is not sensitive to ingestion order. Re-running the same snapshot produces the same fitted state.

That is good engineering behavior. It should not be confused with strong statistical evidence.

For the tied snapshot, the data supports four equally frequent candidates. The implementation chooses one through a documented ordering rule. The choice is deterministic because production systems need repeatable output, not because `18` is empirically more common.

I now use two separate words in reviews:

```text
selected     -> the implementation returned this value
dominant     -> the data clearly favored this value
```

Our fill is selected. It is not dominant.

## Continuous measurements rarely have a useful exact mode

Mode imputation is most natural for discrete values that repeat:

```text
0, 0, 0, 1, 1
small, small, medium, small
device type 2, device type 2, device type 4
```

Temperature is continuous in principle. With sufficient measurement precision, two readings that are physically similar may still be different floating-point values:

```text
19.9998
20.0001
20.0004
```

An exact-value frequency count treats them as three categories. If most or all observations are unique, tie-breaking rather than concentration determines the fill.

Rounding first can create repeated values, but then the rounding resolution becomes part of the model:

```text
20.04 -> 20.0 at one decimal
20.04 -> 20.04 at two decimals
```

Different resolutions can produce different modes. I only quantize a continuous feature when that resolution has a domain meaning, such as the actual sensor precision—not merely to make mode imputation return something convenient.

## Mean and median answer different questions

For the four unique temperatures, mean and median both return `19.5`:

```text
Mean fill:   19.5
Median fill: 19.5
```

They are central summaries rather than common observed values.

That makes them look more intuitive here, but neither recovers the unknown temperature. Mean assumes squared-error-style centrality is useful. Median assumes absolute-error-style centrality and is more resistant to extremes. Both can be poor when the distribution is multimodal, segmented, or missing systematically.

The comparison is not:

```text
mode bad, mean good
```

It is:

```text
choose a statistic whose question matches the feature
```

For a continuous temperature, mean or median is usually a more defensible baseline than an exact-value mode. For a repeated discrete state, most frequent may preserve a real common category.

## One extra row created a three-degree jump

The revised snapshot contains one additional `21`:

```text
18 -> count 1
19 -> count 1
20 -> count 1
21 -> count 2
```

The mode is now unique. `MostFrequent` correctly changes from the tie fallback `18` to the genuinely most common `21`.

The discontinuity is still operationally important:

```text
one added row
-> fill changes by 3 degrees
-> every missing cell changes together
```

A mean would move from `19.5` to `19.8`. A median would move from `19.5` to `20.0`. The mode jumps because its objective depends on the identity of the highest count, not on gradual movement of all values.

This can be exactly what I want for a categorical state. It can be unnecessarily brittle for a continuous measurement.

## The downstream model amplified the imputation change

The experiment fits a separate calibration model with the exact relationship:

```text
prediction = 5 × temperature + 10
```

This model is deliberately simple. It remains fixed while we pass it the two possible imputed temperatures:

```text
temperature 18 -> prediction 100
temperature 21 -> prediction 115
```

The 15-unit shift is not a fitting accident. It is the model's five-unit slope multiplied by the three-degree change in preprocessing.

In a real pipeline, the impact depends on downstream sensitivity. A tree might send the two fills to different branches. A logistic model might cross a decision threshold. A nonlinear transformation could amplify or compress the gap.

This is why I test predictions, not only learned fill values. A preprocessing change that looks small in input units can be large in decision units.

## Binary ties can hide a policy decision

For the binary values:

```text
0, 1, 0, 1
```

both states occur twice. datarust selects the smaller value:

```text
fill = 0
```

If zero means “device off,” “customer did not consent,” or “test negative,” this fallback has business meaning. The imputer did not reason that zero was safer or more ethical. Numeric ordering resolved a frequency tie.

For binary flags, I ask directly:

- Is missing truly equivalent to false?
- Should unknown become a third state?
- Is one fallback conservative, and who approved that policy?
- Would a missingness indicator help the model distinguish observed zero from imputed zero?
- Should the decision stop rather than guess?

An automatic smallest-value tie-break is reproducible. It is not a substitute for those answers.

## Numeric category codes make “smallest” arbitrary

`SimpleImputer` operates on numeric matrices. A discrete numeric code may look like:

```text
1 = email
2 = phone
3 = chat
```

If all three tie, the mode fallback is code `1`, or email. Rename the codes and the fallback changes even though the underlying category frequencies do not.

The smallest code has no inherent categorical priority unless the encoding contract says it does.

For genuine string categories, I prefer a categorical preprocessing path with an explicit missing token or policy. If numeric codes are unavoidable, I persist the codebook and test tie behavior in decoded business names.

I do not let `1 < 2 < 3` silently become “email is the preferred fallback.”

## Exact floating-point identity matters

Most-frequent imputation groups values by their stored numeric identity. Two values that print similarly may still differ internally:

```text
20.0
20.0000000001
```

If upstream calculations, unit conversions, or parsing routes create tiny differences, an expected mode can fragment into many singletons.

Conversely, rounding or clipping can manufacture a large repeated value at a boundary. A sensor capped at `100` may make `100` the mode because every larger physical measurement is stored as the same maximum.

Before using exact numeric frequency, I inspect:

- the number of distinct values
- measurement resolution
- rounding and clipping rules
- top frequency and runner-up frequency
- how much traffic lands on hard bounds

The histogram needs a data-generating explanation, not only a tallest bar.

## Measure the mode margin

A single fill value hides how secure its selection was.

I record at least:

```text
top_count
second_count
number_of_tied_top_values
observed_non_missing_count
selected_fill
```

A useful diagnostic is the mode margin:

```text
mode_margin = top_count - second_count
```

For the tied snapshot:

```text
top_count = 1
second_count = 1
margin = 0
top tie count = 4
```

For the revised snapshot:

```text
top_count = 2
second_count = 1
margin = 1
top tie count = 1
```

The revised mode is unique, but it is still supported by only two observations. Uniqueness and reliability are different properties.

For a large production feature, I may also track the top-frequency proportion and its change over time.

## Fit inside the validation boundary

Like mean and median, most-frequent imputation learns state from data. It belongs inside each cross-validation fold.

If I fit the imputer on the complete dataset before validation, the validation rows influence which value fills missing training and validation cells. With close counts, a few held-out rows can break a tie or change the winning mode.

The honest flow is:

```text
fold training rows:
    fit mode and tie policy
    transform training rows
    fit model

fold validation rows:
    transform with training-fold mode
    predict
    score
```

I also record the learned fill per fold. If it jumps among several values while scores look stable, the deployed artifact may depend strongly on which rows happen to be included in the final fit.

That instability is evidence worth keeping, not noise to hide behind a mean score.

## Never refit the imputer on a production request batch

The fitted fill value is part of the model artifact.

If a service refits `MostFrequent` on each incoming batch, one user's transformed value depends on who else arrived in the same batch. A batch containing an extra `21` can change every missing temperature from `18` to `21`.

That destroys replayability and makes online behavior sensitive to traffic composition.

The production contract should be:

```text
fit imputer on approved training data
persist fill values with the model
transform future rows without refitting
monitor drift separately
```

When a retraining job intentionally updates the imputer, the fill change belongs in the model-version diff and release validation.

## Ties deserve an explicit policy

Smallest-wins is a reasonable library default because it is deterministic. A high-stakes application may need a different policy.

Options include:

- reject fitting when the top frequency is tied
- use a configured domain fallback
- create an explicit unknown or missing category
- use mean or median for a continuous feature
- estimate within a reviewed group when global populations differ
- add a missingness indicator
- route the case to manual review

Randomly selecting among tied values usually makes debugging and replay harder. If stochastic imputation is scientifically justified, I use an explicit seeded distributional method and propagate uncertainty rather than hiding randomness inside a nominally simple mode.

The important step is turning a default into a conscious contract.

## Missingness still carries the harder question

Even a strongly dominant mode does not explain why a value is missing.

If temperature readings fail more often during overheating, filling them with the ordinary operating mode systematically erases the very condition we care about. If one device model reports less reliably, the fill can become a proxy for device population.

I ask:

```text
Would this value be missing for the same reasons in training and production?
```

Mode stability cannot repair a changing collection process. It only makes one numeric replacement repeatable.

Validation should reproduce realistic missingness patterns, including their relationship with time, device, user segment, and target outcome.

## The smallest value was an implementation answer

For `[18, 19, 20, 21]`, datarust filled the missing temperature with `18`. Reversing the rows preserved that answer because ties are resolved by the smallest value.

Adding one more `21` changed the learned fill to `21`. A fixed downstream slope turned that three-degree move into a 15-unit prediction change.

No arithmetic was wrong. The surprise came from information the single fill value did not display:

```text
Was the mode unique?
How many observations supported it?
How close was the runner-up?
Did numeric ordering decide a business fallback?
Was exact-value frequency meaningful for this feature?
```

My safeguards now are straightforward:

- reserve mode imputation mainly for genuinely discrete, repeated values
- inspect distinct counts and mode margins
- define tie behavior explicitly
- decode numeric categories before interpreting the fallback
- test downstream prediction sensitivity
- fit and persist the imputer inside the model pipeline
- monitor learned fills across folds and model versions
- treat missingness as a data-process question, not only a blank cell

“Most frequent” names a strategy. It does not guarantee that the sample contained one clear winner.
