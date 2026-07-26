# My Dataset Had No Labels. Rust Still Found the Customers I Was Looking For.

*A practical guide to customer segmentation with StandardScaler, KMeans, and silhouette scoring in datarust — including the parts clustering tutorials usually skip.*

---

Supervised machine learning starts with a comforting assumption: somewhere, somebody has already written down the correct answer.

This customer churned. That transaction was fraudulent. This house sold for $420,000. Give the model enough examples with labels attached, and it can learn the relationship between the inputs and the answer.

Then someone sends you a customer table with no answer column and asks a wonderfully vague question:

*“Can we find the different kinds of customers in here?”*

I've been on both sides of that request. It sounds simple in a meeting. In practice, “different kinds” can mean spending level, purchase frequency, recency, product preference, geography, or whatever pattern the algorithm happens to find first.

This is where clustering is useful — and where it is very easy to produce a colorful chart that says less than everyone hopes.

In this article, we'll build a small customer-segmentation workflow with [**datarust**](https://crates.io/crates/datarust), a pure-Rust preprocessing and classical-ML library. We will:

1. Represent customer behavior as a numeric matrix
2. Scale features so dollars do not overpower everything else
3. Compare several values of `k`
4. Evaluate them without ground-truth labels
5. Translate cluster centers back into human units
6. Assign a new customer to a fitted segment

The code is small. The decisions around it are the real subject.

## The scenario: three numbers per customer

Imagine an online shop. For every customer, we calculate three features over the last year:

- **Annual spend** in dollars
- **Number of orders** placed
- **Recency**, measured as days since the last order

Those three values are not a complete description of a human being. They are a behavioral snapshot, which is exactly what we want here. The goal is not to claim that customers *are* three numbers; it is to find useful purchasing patterns in three clearly defined measurements.

Create a new Rust application and add datarust:

```sh
cargo new customer_segments
cd customer_segments
cargo add datarust
```

Then replace `src/main.rs` with the following:

```rust
use datarust::cluster::{metrics::silhouette_score, KMeans};
use datarust::scaler::StandardScaler;
use datarust::traits::{Clusterer, Transformer};
use datarust::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // annual_spend ($), orders_last_year, days_since_last_order
    let raw = Matrix::new(vec![
        vec![120.0,  2.0, 110.0],
        vec![180.0,  3.0,  95.0],
        vec![220.0,  4.0,  80.0],
        vec![150.0,  2.0, 120.0],
        vec![260.0,  5.0,  70.0],
        vec![200.0,  3.0,  88.0],
        vec![310.0,  5.0,  65.0],
        vec![140.0,  2.0, 105.0],
        vec![780.0, 12.0,  30.0],
        vec![920.0, 15.0,  24.0],
        vec![1100.0, 17.0, 20.0],
        vec![1250.0, 19.0, 16.0],
        vec![860.0, 13.0,  28.0],
        vec![1050.0, 16.0, 18.0],
        vec![1350.0, 21.0, 14.0],
        vec![980.0, 15.0,  22.0],
        vec![3200.0, 38.0,  7.0],
        vec![3800.0, 44.0,  4.0],
        vec![4500.0, 52.0,  2.0],
        vec![4100.0, 48.0,  3.0],
        vec![5200.0, 60.0,  1.0],
        vec![3500.0, 41.0,  6.0],
        vec![4700.0, 55.0,  2.0],
        vec![3900.0, 46.0,  5.0],
    ])?;

    // KMeans uses distance, so put every feature on a comparable scale.
    let mut scaler = StandardScaler::new();
    let x = scaler.fit_transform(&raw)?;

    // Try several cluster counts and keep the strongest silhouette score.
    println!("k  silhouette  inertia");
    let mut best_k = 0;
    let mut best_score = f64::NEG_INFINITY;

    for k in 2..=5 {
        let mut candidate = KMeans::new()
            .with_n_clusters(k)
            .with_n_init(20)
            .with_random_state(42);

        let labels = candidate.fit_predict(&x)?;
        let score = silhouette_score(&x, &labels)?;

        println!("{k}  {score:>10.3}  {:>7.2}", candidate.inertia());

        if score > best_score {
            best_k = k;
            best_score = score;
        }
    }

    // Fit the selected model once more so we can inspect and use it.
    let mut model = KMeans::new()
        .with_n_clusters(best_k)
        .with_n_init(20)
        .with_random_state(42);

    let labels = model.fit_predict(&x)?;

    // Cluster centers live in standardized space. Convert them back to
    // dollars, order counts, and days so a person can understand them.
    let scaled_centers = Matrix::new(model.cluster_centers().to_vec())?;
    let centers = scaler.inverse_transform(&scaled_centers)?;

    let mut counts = vec![0usize; best_k];
    for &label in &labels {
        counts[label] += 1;
    }

    println!("\nBest k: {best_k} (silhouette {best_score:.3})");
    println!("cluster  customers  spend  orders  recency");

    for (cluster, count) in counts.iter().enumerate() {
        println!(
            "{cluster:>7}  {count:>9}  {:>5.0}  {:>6.1}  {:>7.1}",
            centers.get(cluster, 0),
            centers.get(cluster, 1),
            centers.get(cluster, 2),
        );
    }

    // New data must pass through the same fitted scaler before prediction.
    let new_customer = Matrix::new(vec![vec![1150.0, 18.0, 12.0]])?;
    let new_customer_scaled = scaler.transform(&new_customer)?;
    let segment = model.predict(&new_customer_scaled)?[0];

    println!("\nNew customer belongs to cluster {segment}");
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6, the fixed seed produces:

```text
k  silhouette  inertia
2       0.658    19.49
3       0.747     4.19
4       0.690     2.66
5       0.633     1.34

Best k: 3 (silhouette 0.747)
cluster  customers  spend  orders  recency
      0          8    198     3.2     91.6
      1          8   4112    48.0      3.8
      2          8   1036    16.0     21.5

New customer belongs to cluster 2
```

The result looks almost suspiciously clean because the example data was designed to contain three visible behavioral groups. Real customer data will be noisier, less balanced, and much more interesting. The workflow still holds.

## Why scaling is not optional here

KMeans repeatedly asks one question: which centroid is closest to this row?

“Closest” is measured with Euclidean distance. Without scaling, consider the difference between these two customers:

```text
Customer A: $1,000 spend, 15 orders, 20 days since last order
Customer B: $4,000 spend, 45 orders,  4 days since last order
```

The spending difference is `3,000`. The order difference is `30`. The recency difference is `16`. Squared distance makes that imbalance even more dramatic, so annual spend effectively gets to make the clustering decision by itself.

Maybe spend really is 100 times more important to your business. If so, encode that choice deliberately. Raw units should not make that decision accidentally.

`StandardScaler` transforms each column to roughly zero mean and unit variance:

```rust
let mut scaler = StandardScaler::new();
let x = scaler.fit_transform(&raw)?;
```

Now a meaningful change in recency can influence distance alongside a meaningful change in spend. We have not declared the features equally *valuable*. We have made their numeric units comparable.

There is another important consequence: the scaler becomes part of the fitted system. A new customer must be transformed with these exact learned means and standard deviations. Creating a fresh scaler during prediction would place the new row in a different coordinate system — or, with a single row, collapse every standardized feature to zero.

That is why the example keeps `scaler` alive and calls `transform`, not `fit_transform`, at prediction time.

## Choosing `k` without pretending the answer fell from the sky

KMeans requires the number of clusters up front. This is an awkward API because the question we started with was, “What groups are in the data?” and the algorithm immediately replies, “How many would you like?”

Hard-coding `k = 3` because we want three marketing personas is not discovery. It is arranging the data into the shape of a slide deck.

Instead, the example fits several candidates:

```rust
for k in 2..=5 {
    let mut candidate = KMeans::new()
        .with_n_clusters(k)
        .with_n_init(20)
        .with_random_state(42);

    let labels = candidate.fit_predict(&x)?;
    let score = silhouette_score(&x, &labels)?;
}
```

The silhouette score compares two distances for every customer:

- How close is this customer to others in the same cluster?
- How close is this customer to the nearest different cluster?

The result ranges from `-1` to `1`. Values closer to `1` suggest compact, separated groups. Values near `0` suggest overlap. Negative values are a warning that customers may fit another cluster better.

Here, `k = 3` produces the highest score: `0.747`.

We also print inertia, the total squared distance from every point to its assigned center. It drops from `19.49` to `1.34` as `k` increases. That does **not** mean five clusters are automatically better. Inertia nearly always falls when you add clusters; if every customer got a private cluster, it could reach zero and tell us nothing useful.

Silhouette is not an oracle either. It tends to favor compact, well-separated groups and can miss structures that are meaningful to the business. Use it as evidence, then inspect stability, cluster sizes, and actual customer behavior.

## Initialization is why we run KMeans more than once

KMeans is sensitive to its starting centroids. Give it an unlucky starting point and it can settle into a worse local solution even though a better grouping exists.

datarust uses k-means++ initialization by default, spreading the initial centers apart instead of choosing all of them blindly. We also request 20 restarts:

```rust
.with_n_init(20)
```

Each restart begins from a different initialization, and the model keeps the result with the lowest inertia. Twenty is not a magic number; it is a modest insurance policy for a small dataset.

The fixed random state makes this reproducible:

```rust
.with_random_state(42)
```

That matters when you are debugging, writing tests, or comparing feature sets. If the data and configuration stay the same, the result should not change just because the program ran on Tuesday.

## A cluster number is not a customer persona

After fitting, KMeans gives us labels such as `0`, `1`, and `2`. Those numbers have no rank and no permanent semantic meaning. Cluster `0` is not worse than cluster `1`, and a different seed or retraining run may permute the IDs.

The useful information is in the centroids.

Because we trained on standardized data, the raw centroids initially look like z-scores. The example converts them back:

```rust
let scaled_centers = Matrix::new(model.cluster_centers().to_vec())?;
let centers = scaler.inverse_transform(&scaled_centers)?;
```

Now we can read them in business units:

| Cluster | Customers | Avg. spend | Avg. orders | Avg. recency |
|---:|---:|---:|---:|---:|
| 0 | 8 | $198 | 3.2 | 91.6 days |
| 1 | 8 | $4,112 | 48.0 | 3.8 days |
| 2 | 8 | $1,036 | 16.0 | 21.5 days |

Someone on the marketing team might call these “Occasional,” “Champions,” and “Regular.” That can be useful shorthand, but notice where those names came from: a human interpretation of measurable behavior, not from the algorithm.

I would store the interpretation separately from the numeric ID. For example, match segments to sorted centroid properties or a versioned ruleset rather than assuming cluster `1` will always mean “Champions.” Retraining can move or reorder centers.

And I would resist names like “Loyal” unless the features actually measure loyalty. Frequent recent purchases suggest engagement; they do not tell us how a person feels.

## Predicting a segment for a new customer

Once fitted, KMeans can assign a new row to its nearest learned centroid:

```rust
let new_customer = Matrix::new(vec![vec![1150.0, 18.0, 12.0]])?;
let new_customer_scaled = scaler.transform(&new_customer)?;
let segment = model.predict(&new_customer_scaled)?[0];
```

Our new customer lands in cluster `2`, the regular group. That is unsurprising: their spend, order frequency, and recency are all close to that cluster's center.

In an application, this can happen when a profile is refreshed, before a campaign is selected, or as part of a batch segmentation job. The important sequence is always the same:

```text
raw customer features
        ↓
same fitted StandardScaler
        ↓
same fitted KMeans centroids
        ↓
cluster ID → versioned business interpretation
```

Do not silently change the feature window from “last 12 months” to “all time.” Do not switch recency from days to hours without retraining. The compiler can verify three columns; it cannot verify what those columns mean.

## Saving the fitted artifacts

If another process will serve the segments, enable datarust's `serde` feature:

```toml
[dependencies]
datarust = { version = "0.6", features = ["serde"] }
```

Then save the fitted scaler and model:

```rust
datarust::serialize::save_json(&scaler, "customer-scaler.json")?;
datarust::serialize::save_json(&model, "customer-kmeans.json")?;
```

Load both before prediction:

```rust
let scaler: StandardScaler =
    datarust::serialize::load_json("customer-scaler.json")?;
let model: KMeans =
    datarust::serialize::load_json("customer-kmeans.json")?;

let x = scaler.transform(&new_customer)?;
let segment = model.predict(&x)?[0];
```

The two files are one logical model. Version and deploy them together. A perfectly restored KMeans model paired with the wrong scaler is a perfectly reproducible bug.

## What I would check before using this for real

This example is intentionally friendly. Production data will ask harder questions:

- **Are there outliers?** One wholesale account can pull a mean-based centroid far away from ordinary customers. Consider capping extreme values or using `RobustScaler`.
- **Are the clusters stable?** Refit on bootstrap samples or adjacent time periods. If the segments disappear every week, they are not safe foundations for a long-running strategy.
- **Are tiny clusters real?** A group of three rows may be a valuable niche, a data error, or three employees testing checkout.
- **Do the features encode a sensitive proxy?** Geography, device, income, and purchasing power can introduce fairness concerns even when protected attributes are absent.
- **Does segmentation improve an outcome?** A beautiful silhouette score does not prove that segment-specific messaging, recommendations, or service levels help anyone.
- **Does KMeans fit the geometry?** It works best for roughly compact, similarly sized groups. Curved shapes, heavy density differences, and categorical behavior may need another algorithm.

That last point is worth being honest about: datarust v0.6 provides KMeans, not a complete clustering zoo. If the data clearly calls for DBSCAN or hierarchical clustering, use the tool that matches the problem rather than forcing KMeans into it.

## The useful result is not the number `0.747`

The silhouette score told us that three clusters separate this dataset better than the nearby alternatives we tried. That is a useful technical result.

The actual value comes afterward:

- Can a person explain how the groups differ?
- Do the groups remain recognizable next month?
- Can the application assign new customers consistently?
- Does acting on the segmentation make the experience better?

Rust helps with the operational half of that story. The scaler and centroids are concrete types. Shape mistakes return errors. The fitted artifacts can be serialized to readable JSON. The final program compiles into a small binary without a Python runtime or native linear-algebra stack.

But Rust does not rescue a vague question, a careless feature definition, or a segment name invented after looking at three averages.

That is probably the right division of responsibility. Let the library handle distance, fitting, and prediction. Keep the meaning with the humans.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), and the repository includes a [runnable KMeans example](https://github.com/genc-murat/datarust/blob/main/examples/kmeans_clustering.rs).*
