# Nine Sensors Walked Into PCA. Three Signals Came Out.

*A practical datarust guide to scaling, dimensionality reduction, reconstruction, and the difference between “95% of variance” and “95% of meaning.”*

---

Industrial telemetry has a habit of growing sideways.

You begin with motor temperature and current. Someone adds bearing temperature. Then coolant temperature, power draw, RPM, two vibration axes, and a sound-level sensor. A year later, the dashboard has nine lines that seem to rise and fall in groups, and nobody wants to remove any of them because each sensor was expensive enough to earn a meeting.

The columns are different, but the underlying machine may be doing only a few things.

It gets hotter. It works harder. It vibrates more.

That is the kind of redundancy Principal Component Analysis is good at finding. PCA does not know what heat, load, or vibration mean. It looks for directions in the data that explain the most variation, then expresses each row in those new coordinates.

In this article, we'll use [**datarust**](https://crates.io/crates/datarust) to turn nine correlated sensor readings into three principal components. Then we'll reconstruct the original readings and use reconstruction error to examine a row where one sensor disagrees with the others.

The interesting parts are not the matrix multiplication. They are the choices around it:

- Why scaling comes before PCA
- How to choose a component count
- What explained variance actually promises
- Why principal components are not automatically interpretable
- When reconstruction error is useful — and when it is just a number

## Build a small telemetry laboratory

We will generate 300 observations from three hidden processes: heat, load, and vibration. Those hidden values are not included in the dataset. They only influence the nine sensor readings.

For example, motor, bearing, and coolant temperatures all respond strongly to heat. Current, power, and RPM respond mostly to load. The vibration sensors and sound level respond mostly to vibration. Every sensor also gets a little independent noise.

This gives us a controlled version of a real problem: many observed columns, a smaller number of underlying patterns.

Create a Rust project and add datarust:

```sh
cargo new sensor_pca
cd sensor_pca
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use datarust::decomposition::{PCAComponents, PCA};
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;
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
        sigma
            * (-2.0 * u.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn row_rmse(a: &Matrix, b: &Matrix, row: usize) -> f64 {
    let mse = a
        .row(row)
        .iter()
        .zip(b.row(row))
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / a.ncols() as f64;

    mse.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(2026);
    let mut rows = Vec::new();

    for _ in 0..300 {
        let heat = rng.normal(1.0);
        let load = rng.normal(1.0);
        let vibration = rng.normal(1.0);

        rows.push(vec![
            // motor_temp, bearing_temp, coolant_temp
            70.0 + 8.0 * heat + 2.0 * load + rng.normal(0.6),
            55.0 + 6.5 * heat + 0.8 * load + rng.normal(0.5),
            45.0 + 5.5 * heat + 1.2 * load + rng.normal(0.4),
            // current, power, rpm
            15.0 + 3.0 * load + 0.7 * heat + rng.normal(0.3),
            120.0 + 22.0 * load + 4.0 * heat + rng.normal(2.0),
            1500.0 + 120.0 * load - 10.0 * heat + rng.normal(12.0),
            // vibration_x, vibration_y, sound_level
            1.0 + 0.35 * vibration + 0.04 * load + rng.normal(0.03),
            0.8 + 0.30 * vibration - 0.03 * load + rng.normal(0.03),
            65.0 + 5.0 * vibration + 1.2 * load + rng.normal(0.5),
        ]);
    }

    let raw = Matrix::new(rows)?;

    // Put temperatures, RPM, vibration, and power on comparable scales.
    let mut scaler = StandardScaler::new();
    let scaled = scaler.fit_transform(&raw)?;

    // Keep the smallest number of components that explains at least 95% of
    // variance. PCA decides the final component count during fit.
    let mut pca = PCA::new(PCAComponents::Variance(0.95));
    let projected = pca.fit_transform(&scaled)?;

    println!("Input shape:     {} x {}", raw.nrows(), raw.ncols());
    println!(
        "Projected shape: {} x {}",
        projected.nrows(),
        projected.ncols()
    );
    println!("Components kept: {}", pca.n_components());

    let mut cumulative = 0.0;
    for (index, ratio) in pca.explained_variance_ratio().iter().enumerate() {
        cumulative += ratio;
        println!(
            "PC{}: {:>5.1}% (cumulative {:>5.1}%)",
            index + 1,
            ratio * 100.0,
            cumulative * 100.0,
        );
    }

    // Reconstruct the standardized training data and measure information loss.
    let reconstructed = pca.inverse_transform(&projected)?;
    let reconstruction_mse = scaled
        .as_slice()
        .iter()
        .zip(reconstructed.as_slice())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / scaled.as_slice().len() as f64;

    println!(
        "Standardized reconstruction RMSE: {:.3}",
        reconstruction_mse.sqrt()
    );

    // Row 0 is internally consistent. Row 1 reports a very hot motor while
    // every related temperature and load sensor stays normal.
    let samples = Matrix::new(vec![
        vec![
            73.4, 58.01, 47.39, 14.45, 115.4,
            1459.0, 1.058, 0.869, 65.64,
        ],
        vec![
            95.0, 58.01, 47.39, 14.45, 115.4,
            1459.0, 1.058, 0.869, 65.64,
        ],
    ])?;

    let samples_scaled = scaler.transform(&samples)?;
    let samples_projected = pca.transform(&samples_scaled)?;
    let samples_reconstructed = pca.inverse_transform(&samples_projected)?;

    println!(
        "Normal row reconstruction error:       {:.3}",
        row_rmse(&samples_scaled, &samples_reconstructed, 0)
    );
    println!(
        "Inconsistent row reconstruction error: {:.3}",
        row_rmse(&samples_scaled, &samples_reconstructed, 1)
    );

    Ok(())
}
```

Run it:

```sh
cargo run --release
```

With datarust v0.6 and the fixed seed above, the output is:

```text
Input shape:     300 x 9
Projected shape: 300 x 3
Components kept: 3
PC1:  44.6% (cumulative  44.6%)
PC2:  32.2% (cumulative  76.8%)
PC3:  22.8% (cumulative  99.5%)
Standardized reconstruction RMSE: 0.068
Normal row reconstruction error:       0.005
Inconsistent row reconstruction error: 0.679
```

Nine columns became three, and those three components retain `99.5%` of the standardized dataset's variance.

That sounds like magic. It is mostly correlation.

## Scaling decides what PCA thinks is important

PCA looks for directions with the most variance. Our raw RPM values move by hundreds, temperatures move by tens, and vibration readings move by fractions.

Without scaling, RPM and power would dominate the covariance matrix because their numeric units create larger variance. PCA would faithfully conclude that the biggest numbers are the most important directions.

That is mathematically correct and usually not what we mean.

We standardize every feature first:

```rust
let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&raw)?;
```

Afterward, each column has roughly zero mean and unit variance. PCA can now respond to correlation structure instead of the arbitrary choice to measure RPM in revolutions per minute and vibration in millimeters per second.

There are cases where you should not standardize. If every column shares a meaningful unit and larger variance genuinely deserves more influence, raw covariance may be appropriate. The important thing is to choose, not to let unit conventions choose for you.

And, as always, the fitted scaler is part of the model. New telemetry must use `scaler.transform`. Calling `fit_transform` on a new batch would give that batch its own coordinate system and make its PCA projection incompatible with training.

## Asking for variance instead of guessing a column count

datarust lets us select PCA components three ways:

```rust
PCAComponents::Count(3)       // always keep exactly 3
PCAComponents::Variance(0.95) // keep enough for at least 95%
PCAComponents::All            // keep every possible component
```

For this example, I prefer the variance threshold:

```rust
let mut pca = PCA::new(PCAComponents::Variance(0.95));
```

We state the acceptable retention target, and `fit` finds the smallest component count that reaches it. The result happens to be three components, but the code did not know that in advance.

The first component explains `44.6%`, the second brings the cumulative total to `76.8%`, and the third reaches `99.5%`. That sharp elbow is exactly what we expected from data generated by three hidden processes.

Real data rarely sends such a neat thank-you note. You may see a gradual curve where 20 components each contribute a little. In that case, there is no natural low-dimensional summary, and forcing one creates more information loss.

Also notice that our `95%` request returned `99.5%`. PCA keeps whole components. The second component was not enough, and the third carried a large block of additional variance; it cannot keep only one-third of PC3.

## “99.5% of variance” is not “99.5% of meaning”

Explained variance is precise, useful, and often described too casually.

PCA preserves directions where the dataset varies most. It does not know which variation is valuable to your application. A rare fault signal might have low variance. A slowly drifting calibration error might matter enormously while contributing very little to the overall covariance. A high-variance operating mode might be completely routine.

So this statement is valid:

> Three components explain 99.5% of variance in the standardized training data.

This one is not guaranteed:

> Three components preserve 99.5% of everything important about the machine.

If PCA feeds a downstream classifier or regressor, validate that downstream task with and without dimensionality reduction. If it compresses telemetry, evaluate reconstruction error per sensor and per operating regime. If it supports anomaly detection, confirm that historical incidents actually produce useful scores.

Variance is a property of the data distribution. Value is a property of the problem.

## Principal components are directions, not selected sensors

PCA did not choose three of our nine columns. It created three new columns, each a weighted combination of all original standardized features:

```text
PC1 = w₁·motor_temp + w₂·bearing_temp + ... + w₉·sound_level
```

The weights live in `pca.components()`:

```rust
for (index, weights) in pca.components().iter().enumerate() {
    println!("PC{} weights: {:?}", index + 1, weights);
}
```

In our controlled dataset, one component will mostly reflect temperature-related movement, another load, and another vibration. But mixed relationships can rotate those interpretations, especially when components explain similar amounts of variance.

The sign is arbitrary too. If every weight in a component flips sign, the projected scores flip with it, but the component represents the same axis and explains the same variance. “Positive PC2 means high load” is only true after you inspect the fitted weights and commit to that specific model artifact.

If you need original feature names to remain directly interpretable, PCA may be the wrong transformation. Feature selection keeps columns. PCA replaces them.

## Reconstruction makes the loss visible

Projection moves each nine-dimensional row into three-dimensional component space:

```rust
let projected = pca.transform(&scaled)?;
```

Inverse transformation maps it back toward the original standardized space:

```rust
let reconstructed = pca.inverse_transform(&projected)?;
```

The reconstructed row is not the original. It is the best approximation PCA can make using only the retained component directions.

Across the training data, standardized reconstruction RMSE is `0.068`. In plain language, the typical reconstruction error is about seven-hundredths of a feature standard deviation. That is small because our sensors are highly redundant and the independent noise is modest.

Working in standardized space matters for this metric. If we calculated one combined RMSE in raw units, RPM errors would numerically overwhelm vibration errors again. For operational use, I would also inspect per-feature errors after converting back to real units:

```rust
let reconstructed_raw = scaler.inverse_transform(&reconstructed)?;
```

An aggregate score can hide the one sensor whose reconstruction is unacceptable.

## A disagreement between sensors leaves a larger residual

The two sample rows differ in only one place. The normal row describes a slightly warm, lightly loaded machine with consistent temperatures. The second row claims the motor is at `95°C` while bearing temperature, coolant temperature, current, power, and RPM remain unchanged.

That combination does not resemble the relationships PCA learned from normal training rows.

When projected into three dimensions and reconstructed, the normal row returns almost perfectly: error `0.005`. The inconsistent row returns with error `0.679`, more than 100 times larger.

This is the basic idea behind PCA reconstruction-error anomaly detection:

```text
new row
   ↓ scale
principal components
   ↓ inverse transform
expected low-dimensional reconstruction
   ↓ compare
reconstruction residual
```

The retained components model common correlation patterns. A row that cannot be represented well by those patterns leaves a large residual.

But `0.679` is not automatically an alarm threshold. We constructed both the training distribution and the anomaly, so the separation is unusually friendly. A real threshold should be chosen from held-out normal data, known incidents, acceptable false-positive rates, and ideally the cost of missing a fault.

The training set also needs to be mostly representative of normal behavior. If faults are common in the fitting data, PCA may learn them as ordinary variance and reconstruct them quite well.

Reconstruction error is a signal for investigation, not a diagnosis.

## What I would save for production

The scaler and PCA object are one logical transformation. The PCA components were learned in the scaler's standardized coordinate system; pairing them with a different scaler silently changes the meaning of every projection.

Enable serialization:

```toml
[dependencies]
datarust = { version = "0.6", features = ["serde"] }
```

Then save both fitted artifacts:

```rust
datarust::serialize::save_json(&scaler, "sensor-scaler.json")?;
datarust::serialize::save_json(&pca, "sensor-pca.json")?;
```

At inference time:

```rust
let scaler: StandardScaler =
    datarust::serialize::load_json("sensor-scaler.json")?;
let pca: PCA =
    datarust::serialize::load_json("sensor-pca.json")?;

let scaled = scaler.transform(&new_telemetry)?;
let reduced = pca.transform(&scaled)?;
```

Version the two JSON files together with the feature order, units, sampling window, training-data version, and any anomaly threshold. “Nine numeric columns” is not a schema. The program must know which column is motor temperature and whether it arrived in Celsius.

## When PCA helps — and when it gets in the way

PCA is useful when:

- Dense numeric features are strongly correlated
- A smaller representation makes visualization or storage easier
- Collinearity destabilizes a downstream linear model
- You want to denoise by dropping low-variance directions
- Reconstruction residuals are meaningful for the domain

I would be cautious when:

- Original feature interpretability is essential
- Rare, low-variance signals carry the outcome you care about
- Relationships are strongly nonlinear
- The data mixes incompatible units without a scaling decision
- Most input is sparse, such as bag-of-words or TF-IDF features

For sparse text-like matrices, centering for PCA destroys sparsity. datarust's `TruncatedSVD` does not center input and is usually the more natural decomposition there.

For large dense matrices, datarust also offers an optional `matrixmultiply` feature for faster matrix products, plus a randomized PCA solver when you request a small fixed rank. Start with the exact default and measure before adding complexity.

## Three columns are simpler, but not free

Our result is compelling:

| Representation | Columns | Variance retained | Reconstruction RMSE |
|---|---:|---:|---:|
| Original standardized sensors | 9 | 100% | 0 |
| PCA projection | 3 | 99.5% | 0.068 |

We reduced the width by two-thirds while preserving nearly all common variation. We also gained a useful residual for detecting relationships the low-dimensional model cannot explain.

What we gave up is directness. A row of three PCA scores is harder to explain than nine named sensor values. Reconstruction is approximate. The model depends on a fitted scaler, a fixed feature order, and a training distribution that may drift as equipment ages or operating regimes change.

That is the real tradeoff. PCA does not discover that nine sensors were unnecessary. It discovers that, in this dataset, their movement mostly fits inside a three-dimensional space.

Sometimes that is exactly the simplification you need.

Just remember: fewer columns do not mean fewer responsibilities.

```sh
cargo add datarust
```

---

*datarust is MIT-licensed and available on [crates.io](https://crates.io/crates/datarust). Documentation lives at [genc-murat.github.io/datarust](https://genc-murat.github.io/datarust/), including the [PCA and TruncatedSVD guide](https://genc-murat.github.io/datarust/guide/decomposition.html).*
