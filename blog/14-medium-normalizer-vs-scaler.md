# My Most Active User Became Identical to My Least Active One

*A practical datarust guide to row normalization, feature scaling, cosine similarity, and the information we deliberately throw away when magnitude stops mattering.*

---

I had two users who could not have looked more different on a dashboard.

```text
             views   likes   purchases
casual          10       2           1
power user    1000     200         100
```

One had performed 13 actions. The other had performed 1,300.

Then I applied L2 normalization, and they became exactly the same row.

```text
[0.975900, 0.195180, 0.097590]
```

At first, that can feel like a catastrophic loss of information. Sometimes it is. In another kind of model, it is precisely the point.

The two users have the same behavioral *composition*: ten views for every two likes and one purchase. If I am building a similarity system based on taste or action mix, their direction may matter more than their activity volume. If I am predicting customer value, erasing the hundred-fold magnitude difference would be absurd.

This is why `Normalizer` is not just another scaler with a slightly different formula. It operates across each row, answers a different question, and removes information that ordinary feature scaling preserves.

Let's make the distinction concrete with [datarust](https://crates.io/crates/datarust).

## A tiny behavioral recommender

Imagine four existing user profiles. Each row contains counts for views, likes, and purchases:

```text
casual   [  10,   2,   1]
power    [1000, 200, 100]
browser  [  50,  40,   5]
buyer    [  50,   5,  40]
```

A new user's profile is:

```text
query    [50, 10, 5]
```

The query has the same `10:2:1` ratio as both the casual and power users. Raw Euclidean distance may not recognize that because it measures absolute count differences. L2 normalization should.

I also include `StandardScaler` to show why column scaling and row normalization are not substitutes.

Here is the complete Rust program:

```rust
use datarust::scaler::{Norm, Normalizer, StandardScaler};
use datarust::traits::Transformer;
use datarust::Matrix;

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let names = ["casual", "power", "browser", "buyer"];
    let candidates = Matrix::new(vec![
        vec![10.0, 2.0, 1.0],
        vec![1000.0, 200.0, 100.0],
        vec![50.0, 40.0, 5.0],
        vec![50.0, 5.0, 40.0],
    ])?;
    let query = Matrix::new(vec![vec![50.0, 10.0, 5.0]])?;

    let mut l2 = Normalizer::new(Norm::L2);
    l2.fit(&candidates)?;
    let candidates_l2 = l2.transform(&candidates)?;
    let query_l2 = l2.transform(&query)?;

    let mut standard = StandardScaler::new();
    standard.fit(&candidates)?;
    let candidates_standard = standard.transform(&candidates)?;
    let query_standard = standard.transform(&query)?;

    println!(
        "candidate   raw distance   standard distance   L2 distance   cosine"
    );
    for (i, name) in names.iter().enumerate() {
        println!(
            "{name:<9}   {:12.4}   {:17.4}   {:11.4}   {:6.4}",
            distance(candidates.row(i), query.row(0)),
            distance(
                candidates_standard.row(i),
                query_standard.row(0),
            ),
            distance(candidates_l2.row(i), query_l2.row(0)),
            cosine(candidates.row(i), query.row(0)),
        );
    }

    println!("\nL2-normalized rows [views, likes, purchases]");
    for (i, name) in names.iter().enumerate() {
        println!(
            "{name:<9} [{:.6}, {:.6}, {:.6}]",
            candidates_l2.get(i, 0),
            candidates_l2.get(i, 1),
            candidates_l2.get(i, 2),
        );
    }
    println!(
        "query     [{:.6}, {:.6}, {:.6}]",
        query_l2.get(0, 0),
        query_l2.get(0, 1),
        query_l2.get(0, 2),
    );

    Ok(())
}
```

This is the actual output:

```text
candidate   raw distance   standard distance   L2 distance   cosine
casual           40.9878              0.1704        0.0000   1.0000
power           973.4603              4.0480        0.0000   1.0000
browser          30.0000              0.3694        0.4714   0.8889
buyer            35.3553              0.8843        0.5732   0.8357

L2-normalized rows [views, likes, purchases]
casual    [0.975900, 0.195180, 0.097590]
power     [0.975900, 0.195180, 0.097590]
browser   [0.778499, 0.622799, 0.077850]
buyer     [0.778499, 0.077850, 0.622799]
query     [0.975900, 0.195180, 0.097590]
```

Three perfectly valid representations tell three different stories.

## Raw distance preferred the wrong behavior for the right arithmetic

In the original count space, the query is closest to `browser`:

```text
query -> browser: 30.0000
query -> casual:  40.9878
query -> power:   973.4603
```

The arithmetic is correct. Browser has exactly 50 views, matching the query's largest component, and its other count differences happen to produce the smallest Euclidean distance.

But browser's behavioral mix is different. It has 40 likes for 50 views, while the query has 10. Casual is smaller in every dimension, yet its three counts are in exactly the same proportions as the query.

Raw Euclidean distance answers:

> Which user has the closest absolute action counts?

It does not answer:

> Which user distributes their actions in the most similar way?

The first question may be perfect for capacity planning or customer-value tiers. It may be wrong for a taste profile.

## StandardScaler worked down the columns

`StandardScaler` learns one mean and standard deviation for each feature across the candidate users:

```text
views       -> centered and scaled across users
likes       -> centered and scaled across users
purchases   -> centered and scaled across users
```

It makes a one-standard-deviation difference in purchases comparable to a one-standard-deviation difference in views. That is column-wise scaling.

In this experiment, standardized distance selects the casual user at `0.1704`, while the power user remains far away at `4.0480`. Their common ratios do not make them identical because StandardScaler preserves where each row sits relative to the population's feature distributions.

This is often what I want for regression, clustering, and distance-based modeling where both user magnitude and differences between feature units matter.

It also means the transform depends on the fitted population. Add different training users and the column means and standard deviations change. The same raw profile may receive different standardized coordinates after retraining.

## Normalizer worked across each row

L2 normalization divides every component of a row by that row's Euclidean norm:

```text
                    x
normalized(x) = ---------
                  ||x||₂
```

For the casual user:

```text
||[10, 2, 1]||₂ = sqrt(10² + 2² + 1²) = sqrt(105)
```

For the power user, every component is 100 times larger, so the norm is also 100 times larger. The factors cancel:

```text
normalize([10, 2, 1])
    == normalize([1000, 200, 100])
```

The query is five times the casual row, so it lands on the same unit vector too. Their L2 distance becomes zero and cosine similarity becomes one.

`Normalizer` therefore answers a direction question:

> If magnitude is ignored, which rows point in the same direction?

Unlike StandardScaler, it does not learn population statistics. Its `fit` step records the expected feature count, but one user's normalized values do not depend on the other users. Adding an extreme new training row does not change the normalized representation of existing rows.

## Why L2 distance and cosine similarity agree

Once two nonzero rows have L2 norm 1, their Euclidean distance is directly related to cosine similarity:

```text
||u - v||² = 2 - 2 cos(θ)
```

Higher cosine similarity means smaller Euclidean distance on the unit sphere. Ranking L2-normalized rows by Euclidean distance therefore gives the same order as ranking the original nonzero rows by cosine similarity.

That connection is why L2 normalization appears so often in document retrieval, embeddings, content similarity, and user-profile matching. A very long document and a short document can share the same term distribution; a highly active user and a casual user can express the same preference direction.

It is also why applying normalization without intending cosine-like geometry can be a serious modeling mistake.

## L1, L2, and Max preserve different shapes

datarust supports three row norms:

```rust
let mut l1 = Normalizer::new(Norm::L1);
let mut l2 = Normalizer::new(Norm::L2);
let mut max = Normalizer::new(Norm::Max);
```

For `[10, 2, 1]`, they produce:

```text
L1   [0.769231, 0.153846, 0.076923]
L2   [0.975900, 0.195180, 0.097590]
Max  [1.000000, 0.200000, 0.100000]
```

With nonnegative counts, L1 normalization makes the row sum to one. The result reads naturally as a composition: about 76.9% views, 15.4% likes, and 7.7% purchases.

L2 normalization puts the row on the unit Euclidean sphere, which connects naturally to cosine similarity.

Max normalization divides by the largest absolute component. The dominant feature becomes `1`, and every other value becomes a ratio relative to it.

All three make `[10, 2, 1]` identical to `[1000, 200, 100]`. They differ in the coordinate system used to describe that shared shape.

## The zero row is a special case

What is the direction of a user who performed no actions?

```text
[0, 0, 0]
```

Mathematically, it has no direction and cannot be divided by a nonzero norm. datarust leaves an all-zero row unchanged for L1, L2, and Max normalization.

That is a safe numeric behavior, but it is not a semantic answer. In a recommender, a zero profile is a cold-start user. Its zero vector should not quietly be treated as an ordinary point with meaningful similarity.

I usually handle it explicitly with a fallback strategy, onboarding preferences, popularity prior, or a separate “no history” path.

## Magnitude cannot be recovered afterward

There is no unique inverse for row normalization.

These rows all produce the same normalized result:

```text
[10, 2, 1]
[50, 10, 5]
[1000, 200, 100]
```

Once only the direction remains, the original length is unknowable. That is why datarust's `Normalizer` does not offer a meaningful `inverse_transform`.

If both composition and volume matter, I preserve volume before normalizing. For nonnegative counts, a useful design is:

```text
normalized behavior: L1 or L2 representation of the action mix
activity feature:    log(1 + views + likes + purchases)
```

The model can then learn whether two users with the same proportions but different total activity should behave differently. We remove the accidental dominance of magnitude without making magnitude unavailable.

That extra feature is not a technical workaround. It expresses the actual hypothesis: direction and length may carry separate information.

## Comparable components still matter

Even row normalization assumes the components belong in one meaningful vector.

In this example, one purchase contributes the same raw squared magnitude as one view before L2 normalization. The product team may reasonably believe that a purchase should count ten or one hundred times more.

I can apply domain weights before normalization:

```text
[views, likes, 20 * purchases]
```

But that changes the angles between every user profile. The weight is part of the similarity definition and should be validated, not hidden in preprocessing.

Mixing age, annual income, country code, and click count into one row and normalizing it would be even harder to defend. A row is not automatically a coherent vector merely because its values fit inside a `Matrix`.

## When I use each operation

I use feature scaling when columns have different units or dispersions and the model should still care about a row's overall position and magnitude. `StandardScaler`, `RobustScaler`, and `MinMaxScaler` belong to that family, with different assumptions about their column distributions.

I use row normalization when the direction or composition of each sample is the object of interest and magnitude should not dominate similarity. Text vectors, interaction profiles, spectra, and some count-based representations are common examples.

Sometimes I use both, but order matters. Standardizing columns first can create negative values and center features around the population mean; normalizing afterward then measures the direction of those deviations. That is not the same geometry as normalizing raw counts. I only chain them when I can describe that geometry in plain language.

## The information loss is the feature

The surprising result from this experiment is easy to state:

- Raw distance chose `browser`, whose total counts happened to be closest.
- Standard scaling chose `casual` but kept the power user far away.
- L2 normalization made `casual`, `power`, and the query identical.

None of those answers is universally correct.

The right answer depends on what “similar” means in the application. Similar total behavior? Similar position relative to the population? Similar proportions regardless of volume?

Normalization is useful precisely because it forgets magnitude. The danger begins when that forgetting happens without a decision.

Before I normalize a row now, I ask a blunt question:

> If a user becomes one hundred times more active without changing their proportions, should the model consider them unchanged?

If the answer is yes, row normalization may be the right geometry.

If the answer is no, the vector's length is not noise. It is data.

---

*The complete example and its reported output were run against the current datarust codebase. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
