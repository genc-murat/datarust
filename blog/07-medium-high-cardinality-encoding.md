# One Categorical Column Became 500 Numeric Columns. That Wasn't the Bug.

*A practical datarust guide to dense one-hot, sparse CSR, frequency encoding, and the production meaning of an unknown category.*

---

The original column looked harmless:

```text
product_id
product_000
product_071
product_250
```

One column. Ten thousand rows. Just strings.

Then I one-hot encoded it and got a `10,000 × 500` matrix.

Five million floating-point cells appeared because a product catalog had 500 distinct values. Almost every cell was zero. The transformation was mathematically ordinary and operationally ridiculous in exactly the way high-cardinality categorical data often is.

The instinctive reaction is to blame one-hot encoding. Sometimes that is fair. Sometimes the real mistake is using a dense representation for sparse information. And sometimes replacing one-hot encoding with a single numeric column saves memory by silently changing the question the model can answer.

In this article, we'll use [**datarust**](https://crates.io/crates/datarust) to compare three representations of the same product category:

- Dense one-hot encoding
- Sparse one-hot encoding in CSR form
- Normalized frequency encoding

We will measure their shapes and approximate numeric-buffer sizes, then pass in a common product, a rare product, and a product the encoder has never seen.

The code is straightforward. The semantics are where people get hurt.

## A deliberately long-tailed catalog

Our training data contains 10,000 orders across 500 products.

Every product appears at least once. Products `000` through `099` form the popular head of the catalog and appear 96 times each. Products `100` through `499` form the long tail and appear once each.

This is cleaner than a real commerce dataset, but it gives us the behavior we want to inspect:

- One-hot encoding cares only which category a row contains.
- Frequency encoding cares how often that category appeared during fitting.
- An unknown product has no learned category or frequency.

Create a Rust project:

```sh
cargo new categorical_encoding
cd categorical_encoding
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::encoder::{
    FrequencyEncoder, HandleUnknown, OneHotEncoder, UnknownFrequency,
};
use datarust::StrMatrix;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut product_ids = Vec::with_capacity(10_000);

    // Guarantee that all 500 products appear at least once.
    for id in 0..500 {
        product_ids.push(format!("product_{id:03}"));
    }

    // The remaining 9,500 orders belong to the 100 popular products.
    for row in 500..10_000 {
        let id = (row - 500) % 100;
        product_ids.push(format!("product_{id:03}"));
    }

    let products = StrMatrix::from_column(product_ids)?;

    // Learn one category-to-column mapping, then render it both densely and
    // sparsely. Unknown categories will become all-zero rows.
    let mut one_hot =
        OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore);
    let dense = one_hot.fit_transform(&products)?;
    let sparse = one_hot.transform_sparse(&products)?;

    // Replace each product with its proportion in the training data.
    let mut frequency = FrequencyEncoder::new(true)
        .handle_unknown(UnknownFrequency::Zero);
    let frequencies = frequency.fit_transform(&products)?;

    // Approximate the numeric storage. This excludes Vec/string/HashMap
    // allocation overhead and focuses on the transformed matrix buffers.
    let dense_bytes =
        dense.nrows() * dense.ncols() * std::mem::size_of::<f64>();
    let sparse_bytes = sparse.nnz()
        * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>())
        + (sparse.nrows() + 1) * std::mem::size_of::<usize>();
    let frequency_bytes = frequencies.nrows()
        * frequencies.ncols()
        * std::mem::size_of::<f64>();

    println!(
        "Rows: {}, categories: {}",
        products.nrows(),
        one_hot.n_output_cols()
    );
    println!(
        "Dense one-hot:  {} x {}, approximately {:.2} MiB",
        dense.nrows(),
        dense.ncols(),
        mib(dense_bytes)
    );
    println!(
        "Sparse one-hot: {} x {}, nnz {}, density {:.4}%, \
         approximately {:.2} MiB",
        sparse.nrows(),
        sparse.ncols(),
        sparse.nnz(),
        sparse.density() * 100.0,
        mib(sparse_bytes)
    );
    println!(
        "Frequency:      {} x {}, approximately {:.2} MiB",
        frequencies.nrows(),
        frequencies.ncols(),
        mib(frequency_bytes)
    );

    // Compare a popular product, a singleton, and an unseen product.
    let incoming = StrMatrix::from_column([
        "product_000",
        "product_250",
        "product_999",
    ])?;
    let incoming_one_hot = one_hot.transform_sparse(&incoming)?;
    let incoming_frequency = frequency.transform(&incoming)?;

    println!("\nIncoming values");
    for row in 0..incoming.nrows() {
        println!(
            "{:<11} one-hot nnz={}  frequency={:.4}",
            incoming.get(row, 0),
            incoming_one_hot.row_nz(row).count(),
            incoming_frequency.get(row, 0),
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
Rows: 10000, categories: 500
Dense one-hot:  10000 x 500, approximately 38.15 MiB
Sparse one-hot: 10000 x 500, nnz 10000, density 0.2000%, approximately 0.23 MiB
Frequency:      10000 x 1, approximately 0.08 MiB

Incoming values
product_000 one-hot nnz=1  frequency=0.0096
product_250 one-hot nnz=1  frequency=0.0001
product_999 one-hot nnz=0  frequency=0.0000
```

The smallest representation is frequency encoding. The sparse one-hot representation is only a little larger.

They do not contain the same information.

## One-hot encoding preserved identity

One-hot encoding creates one output column per learned category:

```text
              product_000  product_001  ...  product_250  ...
product_000        1            0                  0
product_250        0            0                  1
```

The model can learn a separate coefficient for `product_000` and `product_250`. It does not assume an ordering and does not claim that product 250 is “more product” than product 0.

That is exactly what we want when category identity may have a category-specific relationship with the target.

The number of columns is not an implementation accident. It is the cost of preserving 500 independent identities in a linear numeric representation.

If we encoded product IDs as integers instead:

```text
product_000 → 0
product_250 → 250
```

a linear model would treat the distance between 0 and 250 as meaningful. It could conclude that product 250 is halfway between products 100 and 400. Unless the categories have a genuine order, that geometry is invented.

One-hot encoding is wide because it refuses to invent that order.

## Dense storage was the actual absurdity

Every row contains exactly one product. Therefore, every one-hot row contains exactly one `1.0` and 499 zeros.

The dense matrix stores all five million values:

```text
10,000 rows × 500 columns × 8 bytes = 40,000,000 bytes
```

That is about `38.15 MiB` for the flat `f64` buffer, excluding allocation metadata and the encoder's category maps.

The density is only `0.2%`.

CSR sparse storage keeps three pieces:

- The nonzero values
- The column index of each nonzero value
- Row pointers describing where each row begins

For this matrix, datarust stores 10,000 nonzero values instead of five million dense cells. Our approximate buffer calculation is `0.23 MiB`, over 160 times smaller than the dense numeric buffer.

The logical shape is still `10,000 × 500`. Sparse storage did not reduce the number of features or merge any products. It represented the same zeros honestly by not storing them.

That distinction is useful:

> Sparse one-hot solves a storage problem. It does not solve a high-dimensional modeling problem.

An estimator still needs to handle 500 possible coefficients. The downstream API must also accept `SparseMatrix`; converting back to dense before training brings the memory cost right back.

## Frequency encoding changed the question

Frequency encoding replaces a category with its training count or proportion:

```rust
let mut frequency = FrequencyEncoder::new(true);
```

With normalized proportions:

```text
product_000 → 0.0096
product_250 → 0.0001
```

Ten thousand rows now require one numeric column rather than 500. The transformed buffer is about `0.08 MiB`.

That compression is possible because frequency encoding discards identity. Every popular product with the same count receives `0.0096`. Every singleton receives `0.0001`. Products 250 and 417 become indistinguishable even though they are different items.

The model can learn that popular products behave differently from rare products. It cannot learn that one rare product has a high return rate while another rare product has a low one.

Frequency encoding is appropriate when prevalence itself contains useful signal, or when category identity is too granular and popularity is a meaningful summary. It is not “one-hot but smaller.”

It is a different feature.

That makes it worth naming clearly in a real feature schema. `product_frequency` tells the truth. Calling the value `product_id_encoded` hides the assumption.

## Rare categories collide by design

Our long tail contains 400 singleton products. Frequency encoding maps all of them to the same number.

This can be helpful. A model with one training example for a product cannot estimate a reliable product-specific coefficient anyway. Treating all singletons as similarly rare may reduce variance.

It can also erase a real distinction. A newly launched flagship product and an obsolete replacement part may both appear once for completely different reasons.

Common alternatives include:

- Group rare categories into an explicit `OTHER` bucket before one-hot encoding
- Use domain hierarchy, such as product family instead of SKU
- Combine identity features with frequency features
- Use smoothed target encoding with leakage-safe validation
- Use a hashing scheme with a fixed output width and accepted collisions

datarust gives us one-hot, frequency, ordinal, and target encoders. The correct choice still depends on what distinctions the application needs to preserve.

## Unknown categories are a product decision

`product_999` did not exist during fitting.

We configured one-hot encoding with:

```rust
.handle_unknown(HandleUnknown::Ignore)
```

The unknown product becomes an all-zero one-hot row. No learned category column is activated.

We configured frequency encoding with:

```rust
.handle_unknown(UnknownFrequency::Zero)
```

The unknown product receives frequency `0.0`, below every category seen during training.

Both behaviors let production continue, but they mean different things:

- One-hot all-zero means “none of the known categories.”
- Frequency zero means “not observed in the fitting sample.”

Neither tells the model what the new product will do.

The default `OneHotEncoder` strategy is `HandleUnknown::Error`, which can be the safer contract when unseen values indicate broken schema, unexpected upstream behavior, or a missing retraining process. Silently ignoring them favors availability over strictness.

I would choose explicitly and monitor the rate of unknown categories. One unknown during a product launch is normal. Forty percent unknown means the model is operating outside its learned vocabulary.

An all-zero row also needs care when dropping a reference category. If a known dropped category and an unknown category both produce no active bit, the downstream model cannot distinguish them without an additional indicator.

## Fit vocabulary on training data only

The encoder must learn categories and frequencies from the training split:

```rust
one_hot.fit(&train_categories)?;
let validation = one_hot.transform(&validation_categories)?;
```

Fitting on the complete dataset before splitting leaks validation and test vocabulary into training. With frequency encoding, it also leaks how common each category is in held-out data.

This may not reveal the target directly, but it makes the evaluation less representative of deployment. A product that should be unknown during validation becomes known because the encoder peeked.

The category mapping is fitted state. Treat it like scaler statistics or model coefficients.

When the encoder is serialized, version it with:

- Feature name and order
- Category normalization rules
- Training window
- Unknown-category policy
- Model version

Changing `Product-001` to `product_001` upstream can turn a familiar product into an unknown one without changing the matrix shape.

## Target encoding is powerful enough to leak the answer

High-cardinality features often lead to target encoding. Instead of frequency, a category is replaced with its smoothed target mean:

```text
product_042 → average conversion rate for product_042
```

datarust provides `TargetEncoder` with a smoothing factor that pulls low-count categories toward the global target mean. This can preserve outcome-related category differences in one column.

It also creates a more direct leakage risk because the encoder reads `y` during fitting.

Imagine a category that occurs once. With no smoothing, its target mean is that row's target. If we fit and transform the same training row, the encoded feature can contain its own answer.

A leakage-safe training workflow uses out-of-fold encoding:

1. Split training data into folds.
2. For each fold, fit `TargetEncoder` on the other folds.
3. Transform the held-out fold.
4. Assemble those out-of-fold values to train the model.
5. Fit one final encoder on all training rows for validation, test, and production transforms.

Smoothing reduces singleton overconfidence but does not replace that validation design. datarust exposes the encoding primitive; the caller owns cross-fitting.

Target encoding can be excellent. It simply deserves more suspicion than a method whose name contains the word “target.”

## The right encoder depends on what should remain learnable

Here is the comparison I would use:

| Encoding | Output width | Preserves identity? | Handles long tail by | Main risk |
|---|---:|---|---|---|
| Dense one-hot | 500 | Yes | Separate columns | Mostly zeros, high memory |
| Sparse one-hot | 500 logical | Yes | Separate columns | Estimator must support sparse input |
| Frequency | 1 | No | Collapsing equal counts | Different categories collide |
| Target | 1 | Partly, through outcome mean | Smoothing | Target leakage |
| Ordinal | 1 | As ordered ranks | Imposed order | Invented distance if order is false |

For a low-cardinality field such as plan type with four stable values, dense one-hot is often the boring correct answer.

For thousands of categories where identity matters, sparse one-hot may preserve the right information at a practical storage cost.

For a feature where popularity itself matters, frequency encoding may be more honest than pretending every category deserves an independent coefficient.

For a truly ordered feature such as `low / medium / high`, an ordinal encoder with an explicit order expresses information one-hot would ignore.

For high-cardinality supervised problems, target encoding may offer a strong compact signal if it is smoothed and cross-fitted correctly.

There is no universally best categorical encoder because each one asserts a different geometry.

## The 500 columns were telling us something

Our measured result is dramatic:

| Representation | Shape | Density | Approx. numeric buffers |
|---|---:|---:|---:|
| Dense one-hot | 10,000 × 500 | 0.2% | 38.15 MiB |
| Sparse one-hot | 10,000 × 500 | 0.2% | 0.23 MiB |
| Frequency | 10,000 × 1 | 100% | 0.08 MiB |

It would be easy to declare frequency encoding the winner because its matrix is smallest.

That would be like choosing a photograph format by deleting the colors and celebrating the file size.

Sparse one-hot retained product identity while eliminating almost all zero storage. Frequency encoding compressed further by retaining only popularity. Both can be correct, depending on the question.

The useful workflow is:

1. Decide what categorical distinctions the model must learn.
2. Measure cardinality and long-tail behavior on training data.
3. Choose representation and unknown policy explicitly.
4. Validate with the downstream task, not matrix width alone.
5. Monitor vocabulary drift after deployment.

One column becoming 500 was not the bug.

The bug would have been compressing it back to one and pretending nothing changed.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [categorical encoder guide](https://genc-murat.github.io/datarust/guide/encoders.html).*
