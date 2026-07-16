//! Standalone benchmark runner for the sklearn comparison table in README.
//!
//! Produces deterministic synthetic data with the same PRNG layout as
//! `benches/compare_sklearn.py` and measures median `fit_transform` time over
//! `N` repetitions for a set of representative transformers.
//!
//! Usage: `cargo run --release --example bench_compare_rust [reps]`

use std::time::Instant;

use datarust::categorical_kind::CategoricalTransformerKind;
use datarust::compose::{ColumnTransformer, Table};
use datarust::decomposition::{PCAComponents, PCA};
use datarust::encoder::OneHotEncoder;
use datarust::linear_model::LinearRegression;
use datarust::pipeline::Pipeline;
use datarust::scaler::{MinMaxScaler, RobustScaler, StandardScaler};
use datarust::traits::{Regressor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::{Matrix, StrMatrix};

// Deterministic xorshift64 PRNG so Rust and Python generate identical data.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    // uniform f64 in [0,1)
    fn next_unit(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }
}

fn make_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
    let mut rng = Rng::new(seed);
    let data: Vec<Vec<f64>> = (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_range(-100.0, 100.0)).collect())
        .collect();
    Matrix::new(data).unwrap()
}

fn make_str_matrix(rows: usize, cols: usize, seed: u64, cardinality: usize) -> StrMatrix {
    let mut rng = Rng::new(seed);
    let data: Vec<Vec<String>> = (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| format!("cat_{}", (rng.next_unit() * cardinality as f64) as usize))
                .collect()
        })
        .collect();
    StrMatrix::new(data).unwrap()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    }
}

/// Run `workload` `reps`+warmup times, return median ms.
fn measure<F: FnMut()>(reps: usize, mut workload: F) -> f64 {
    // one warmup (cache/JIT-equivalent) not counted
    workload();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        workload();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        samples.push(dt);
    }
    median(samples)
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    // (rows, cols) tuples. Keep identical to Python script.
    let sizes: &[(usize, usize)] = &[(1_000, 10), (10_000, 100), (50_000, 200)];
    let cat_sizes: &[(usize, usize)] = &[(1_000, 5), (10_000, 10), (50_000, 20)];

    println!("workload,rows,cols,rust_ms");
    for &(rows, cols) in sizes {
        let x = make_matrix(rows, cols, 42);
        let s = StandardScaler::new();
        let ms = measure(reps, || {
            let mut s = s.clone();
            let _ = s.fit_transform(&x);
        });
        println!("standard_scaler,{rows},{cols},{ms:.4}");

        let m = MinMaxScaler::new();
        let ms = measure(reps, || {
            let mut m = m.clone();
            let _ = m.fit_transform(&x);
        });
        println!("minmax_scaler,{rows},{cols},{ms:.4}");

        let r = RobustScaler::new();
        let ms = measure(reps, || {
            let mut r = r.clone();
            let _ = r.fit_transform(&x);
        });
        println!("robust_scaler,{rows},{cols},{ms:.4}");

        // PCA — components = min(10, cols/2)
        let k = 10.min(cols / 2).max(1);
        let pca = PCA::new(PCAComponents::Count(k));
        let ms = measure(reps, || {
            let mut p = pca.clone();
            let _ = p.fit_transform(&x);
        });
        println!("pca,{rows},{cols},{ms:.4}");

        // LinearRegression — y is a linear combination of the features so the
        // fit always succeeds and mirrors sklearn's LinearRegression on the
        // same deterministic input.
        let x_for_lr = make_matrix(rows, cols, 42);
        let y: Vec<f64> = (0..rows)
            .map(|i| {
                let row_base = i * cols;
                (0..cols)
                    .map(|j| x_for_lr.as_slice()[row_base + j] * (j as f64 + 1.0))
                    .sum::<f64>()
            })
            .collect();
        let ms = measure(reps, || {
            let mut m = LinearRegression::new();
            let _ = m.fit(&x_for_lr, &y);
            let _ = m.predict(&x_for_lr);
        });
        println!("linear_regression,{rows},{cols},{ms:.4}");

        // Pipeline: Standard -> MinMax -> Robust
        let build_pipe = || {
            Pipeline::new()
                .push("s1", TransformerKind::StandardScaler(StandardScaler::new()))
                .push("s2", TransformerKind::MinMaxScaler(MinMaxScaler::new()))
                .push("s3", TransformerKind::RobustScaler(RobustScaler::new()))
        };
        let ms = measure(reps, || {
            let mut p = build_pipe();
            let _ = p.fit_transform(&x);
        });
        println!("pipeline_3scalers,{rows},{cols},{ms:.4}");
    }

    // OneHotEncoder + ColumnTransformer over categorical sizes.
    for &(rows, cols) in cat_sizes {
        let x_str = make_str_matrix(rows, cols, 42, 20);

        let ohe = OneHotEncoder::new();
        let ms = measure(reps, || {
            let mut e = ohe.clone();
            let _ = e.fit_transform(&x_str);
        });
        println!("onehot_encoder,{rows},{cols},{ms:.4}");

        // ColumnTransformer: numeric scaled + categorical one-hot, mixed table.
        let numeric = make_matrix(rows, cols, 7);
        let categorical = make_str_matrix(rows, cols, 11, 15);
        let table = Table::new(numeric, categorical).unwrap();
        let build_ct = || {
            ColumnTransformer::new()
                .add_numeric(
                    "num",
                    (0..cols).collect(),
                    TransformerKind::StandardScaler(StandardScaler::new()),
                )
                .add_categorical(
                    "cat",
                    (cols..cols + cols).collect(),
                    CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
                )
        };
        let ms = measure(reps, || {
            let mut ct = build_ct();
            let _ = ct.fit_transform_to_table(&table);
        });
        println!("column_transformer,{rows},{cols},{ms:.4}");
    }
}
