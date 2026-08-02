use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use datarust::categorical_kind::CategoricalTransformerKind;
use datarust::compose::*;
use datarust::decomposition::*;
use datarust::encoder::*;
use datarust::pipeline::Pipeline;
use datarust::scaler::*;
use datarust::traits::{Clusterer, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::{Matrix, StrMatrix};

/// Deterministic pseudo-random dense matrix with values in roughly `[-1, 1]`.
///
/// The previous `sin(i * cols + j)` generator produced columns that were all
/// linear combinations of `sin(i·cols)` and `cos(i·cols)`, so the data had
/// rank at most 2 regardless of width. That made `XᵀX` singular for the
/// linear-model benchmarks and panicked the Cholesky solver. Pseudo-random
/// values are full-rank with probability 1 and are also a more representative
/// workload for the dense matmul benchmarks.
fn make_matrix(rows: usize, cols: usize) -> Matrix {
    // xorshift64* PRNG, seeded for reproducibility.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let data: Vec<f64> = (0..rows * cols).map(|_| next()).collect();
    Matrix::from_flat(rows, cols, data).unwrap()
}

fn make_str_matrix(rows: usize, cols: usize) -> StrMatrix {
    let data: Vec<Vec<String>> = (0..rows)
        .map(|i| (0..cols).map(|j| format!("cat_{}_{}", j, i % 5)).collect())
        .collect();
    StrMatrix::new(data).unwrap()
}

fn bench_onehot_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("onehot_encoder");
    for (rows, cols) in [(100, 5), (1000, 10)] {
        let x = make_str_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", rows),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    OneHotEncoder::new,
                    |mut ohe| ohe.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_power_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_transformer");
    for (rows, cols) in [(100, 5), (1000, 20)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", rows),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    PowerTransformer::default,
                    |mut pt| pt.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_column_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_transformer");
    let rows = 1000;
    let num_cols = 5;
    let cat_cols = 5;
    let numeric: Matrix = make_matrix(rows, num_cols);
    let categorical: StrMatrix = make_str_matrix(rows, cat_cols);
    let table = Table::new(numeric, categorical).unwrap();
    group.bench_with_input(
        criterion::BenchmarkId::new("fit_transform", rows),
        &table,
        |bencher, tbl| {
            bencher.iter_batched(
                || {
                    ColumnTransformer::new()
                        .add_numeric(
                            "nums",
                            (0..num_cols).collect(),
                            TransformerKind::StandardScaler(StandardScaler::new()),
                        )
                        .add_categorical(
                            "cats",
                            (num_cols..num_cols + cat_cols).collect(),
                            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
                        )
                },
                |mut ct| ct.fit_transform_to_table(tbl),
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

fn bench_matrix_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_matmul");
    for size in [10, 50, 100] {
        let a = make_matrix(size, size);
        let b = make_matrix(size, size);
        group.bench_with_input(
            criterion::BenchmarkId::new("square", size),
            &(a, b),
            |bencher, (aa, bb)| bencher.iter(|| aa.matmul(bb)),
        );
    }
    group.finish();
}

fn bench_matrix_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_transpose");
    for size in [50, 200, 500] {
        let m = make_matrix(size, size);
        group.bench_with_input(
            criterion::BenchmarkId::new("square", size),
            &m,
            |bencher, m| bencher.iter(|| m.transpose()),
        );
    }
    group.finish();
}

fn bench_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_scaler");
    for (rows, cols) in [(100, 10), (1000, 50), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", rows),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    StandardScaler::new,
                    |mut s| s.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_minmax_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_scaler");
    for (rows, cols) in [(100, 10), (1000, 50)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", rows),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    MinMaxScaler::new,
                    |mut s| s.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("pca");
    for (rows, cols, k) in [(50, 10, 3), (200, 20, 5)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", format!("{}x{}->{}", rows, cols, k)),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || PCA::new(PCAComponents::Count(k)),
                    |mut pca| pca.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");
    let x = make_matrix(1000, 20);
    group.bench_with_input(
        criterion::BenchmarkId::new("3_scalers", 1000),
        &x,
        |bencher, x| {
            bencher.iter_batched(
                || {
                    Pipeline::new()
                        .push("s1", TransformerKind::StandardScaler(StandardScaler::new()))
                        .push("s2", TransformerKind::MinMaxScaler(MinMaxScaler::new()))
                        .push("s3", TransformerKind::RobustScaler(RobustScaler::new()))
                },
                |mut pipe| pipe.fit_transform(x),
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

fn bench_robust_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_scaler");
    for (rows, cols) in [(100, 10), (1000, 50)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            criterion::BenchmarkId::new("fit_transform", rows),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    RobustScaler::new,
                    |mut s| s.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_linear_regression(c: &mut Criterion) {
    use datarust::linear_model::LinearRegression;
    use datarust::traits::Predictor;

    let mut group = c.benchmark_group("linear_regression");
    for (rows, cols) in [(1_000, 10), (10_000, 50), (100_000, 100)] {
        let x = make_matrix(rows, cols);
        // Deterministic target derived from the first column so the fit is
        // well-conditioned and always succeeds.
        let y: Vec<f64> = (0..rows)
            .map(|i| ((i as f64) * cols as f64).sin() * 10.0 + (i as f64))
            .collect();

        group.bench_with_input(
            criterion::BenchmarkId::new("fit", format!("{rows}x{cols}")),
            &(&x, &y),
            |bencher, (x, y)| {
                bencher.iter_batched(
                    LinearRegression::new,
                    |mut m| m.fit(x, y),
                    BatchSize::SmallInput,
                )
            },
        );

        // Pre-fit model for the predict benchmark.
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        group.bench_with_input(
            criterion::BenchmarkId::new("predict", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| model.predict(x)),
        );
    }
    group.finish();
}

fn bench_ridge_and_lasso(c: &mut Criterion) {
    use datarust::linear_model::{Lasso, Ridge};
    use datarust::traits::Predictor;

    let mut group = c.benchmark_group("regularized");
    for (rows, cols) in [(1_000, 10), (10_000, 50), (50_000, 100)] {
        let x = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows)
            .map(|i| ((i as f64) * cols as f64).sin() * 10.0 + (i as f64))
            .collect();

        group.bench_with_input(
            criterion::BenchmarkId::new("ridge_fit", format!("{rows}x{cols}")),
            &(&x, &y),
            |bencher, (x, y)| {
                bencher.iter_batched(
                    || Ridge::new().with_alpha(1.0),
                    |mut m| m.fit(x, y),
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            criterion::BenchmarkId::new("lasso_fit", format!("{rows}x{cols}")),
            &(&x, &y),
            |bencher, (x, y)| {
                bencher.iter_batched(
                    || Lasso::new().with_alpha(0.1).with_max_iter(200),
                    |mut m| m.fit(x, y),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_logistic_regression(c: &mut Criterion) {
    use datarust::linear_model::LogisticRegression;
    use datarust::traits::Predictor;

    let mut group = c.benchmark_group("logistic_regression");
    for (rows, cols) in [(1_000, 10), (10_000, 50), (50_000, 100)] {
        let x = make_matrix(rows, cols);
        // Deterministic binary target: threshold a linear combination of features.
        let y: Vec<f64> = (0..rows)
            .map(|i| {
                let s = x.as_slice();
                let base = i * cols;
                let dot: f64 = (0..cols).map(|j| s[base + j] * (j as f64 + 1.0)).sum();
                if dot > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        group.bench_with_input(
            criterion::BenchmarkId::new("fit", format!("{rows}x{cols}")),
            &(&x, &y),
            |bencher, (x, y)| {
                bencher.iter_batched(
                    || LogisticRegression::new().with_max_iter(50),
                    |mut m| m.fit(x, y),
                    BatchSize::SmallInput,
                )
            },
        );

        let mut model = LogisticRegression::new().with_max_iter(50);
        model.fit(&x, &y).unwrap();
        group.bench_with_input(
            criterion::BenchmarkId::new("predict", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| model.predict(x)),
        );
    }
    group.finish();
}

fn bench_kmeans(c: &mut Criterion) {
    use datarust::cluster::KMeans;

    let mut group = c.benchmark_group("kmeans");
    for (rows, cols, k) in [(1_000, 10, 5), (5_000, 20, 10)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_predict", format!("{rows}x{cols}x{k}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || {
                        KMeans::new()
                            .with_n_clusters(k)
                            .with_n_init(3)
                            .with_random_state(0)
                    },
                    |mut km| km.fit_predict(x),
                    BatchSize::SmallInput,
                )
            },
        );

        let mut km = KMeans::new()
            .with_n_clusters(k)
            .with_n_init(3)
            .with_random_state(0);
        km.fit(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("predict", format!("{rows}x{cols}x{k}")),
            &x,
            |bencher, x| bencher.iter(|| km.predict(x)),
        );
    }
    group.finish();
}

fn bench_quantile_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_transformer");
    for (rows, cols) in [(1_000, 50), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || QuantileTransformer::new(1000).unwrap(),
                    |mut qt| qt.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_kbins_discretizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("kbins_discretizer");
    for (rows, cols) in [(1_000, 50), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("quantile_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || KBinsDiscretizer::new(10).unwrap().strategy(BinStrategy::Quantile),
                    |mut kb| kb.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_polynomial_features(c: &mut Criterion) {
    use datarust::polynomial::PolynomialFeatures;

    let mut group = c.benchmark_group("polynomial_features");
    for (rows, cols, deg) in [(1_000, 10, 2), (1_000, 10, 3), (10_000, 5, 2)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}^d{deg}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || PolynomialFeatures::new(deg),
                    |mut pf| pf.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_truncated_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncated_svd");
    for (rows, cols, k) in [(500, 30, 5), (1_000, 50, 10)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}->{k}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || TruncatedSVD::new(k).unwrap(),
                    |mut svd| svd.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_categorical_encoders(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorical_encoders");
    for (rows, cols) in [(1_000, 10), (10_000, 20)] {
        let x = make_str_matrix(rows, cols);

        group.bench_with_input(
            BenchmarkId::new("ordinal_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || OrdinalEncoder::new(OrdinalCategories::Auto),
                    |mut enc| enc.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("frequency_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || FrequencyEncoder::new(true),
                    |mut enc| enc.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("target_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                let y: Vec<f64> = (0..rows).map(|i| (i % 7) as f64).collect();
                bencher.iter_batched(
                    || TargetEncoder::new(1.0).unwrap(),
                    |mut enc| enc.fit_transform(x, &y),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_stats_bulk(c: &mut Criterion) {
    use datarust::stats;

    let mut group = c.benchmark_group("stats_bulk");
    let qs = [0.0, 0.25, 0.5, 0.75, 1.0];
    for (rows, cols) in [(10_000, 100), (100_000, 100)] {
        let x = make_matrix(rows, cols);
        let flat = x.as_slice();
        group.bench_with_input(
            BenchmarkId::new("mean_var_flat", format!("{rows}x{cols}")),
            &flat,
            |bencher, flat| bencher.iter(|| stats::column_mean_var_flat(flat, rows, cols, 1)),
        );
        group.bench_with_input(
            BenchmarkId::new("quantiles_flat", format!("{rows}x{cols}")),
            &flat,
            |bencher, flat| {
                bencher.iter(|| stats::column_quantiles_many_flat(flat, rows, cols, &qs).unwrap())
            },
        );
    }
    group.finish();
}

fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");
    for (rows, cols) in [(10_000, 50), (100_000, 20)] {
        let nested: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| ((i * cols + j) as f64).sin()).collect())
            .collect();
        group.bench_with_input(
            BenchmarkId::new("from_nested", format!("{rows}x{cols}")),
            &nested,
            |bencher, n| bencher.iter(|| Matrix::new(n.clone()).unwrap()),
        );

        let m = make_matrix(rows, cols);
        let sel: Vec<usize> = (0..cols).step_by(2).collect();
        group.bench_with_input(
            BenchmarkId::new("select_columns", format!("{rows}x{cols}")),
            &m,
            |bencher, m| bencher.iter(|| m.select_columns(&sel)),
        );
    }
    group.finish();
}

fn bench_sparse_matrix(c: &mut Criterion) {
    use datarust::matrix::SparseMatrix;

    let mut group = c.benchmark_group("sparse_matrix");
    for (rows, cols) in [(10_000, 100), (50_000, 50)] {
        let nnz = (rows * cols / 100).max(1);
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut triplets = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            triplets.push((next() % rows, next() % cols, 1.0));
        }
        let sp = SparseMatrix::from_triplets(rows, cols, &triplets).unwrap();

        group.bench_with_input(
            BenchmarkId::new("from_triplets", format!("{rows}x{cols}")),
            &triplets,
            |bencher, t| bencher.iter(|| SparseMatrix::from_triplets(rows, cols, t).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("to_dense", format!("{rows}x{cols}")),
            &sp,
            |bencher, sp| bencher.iter(|| sp.to_dense().unwrap()),
        );
    }
    group.finish();
}

fn bench_knn_imputer(c: &mut Criterion) {
    use datarust::imputer::{KnnImputer, KnnWeights};

    let mut group = c.benchmark_group("knn_imputer");
    for (rows, cols) in [(500, 10), (1_000, 20)] {
        let mut flat = make_matrix(rows, cols).as_slice().to_vec();
        // Punch ~10% NaNs deterministically.
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32
        };
        for v in flat.iter_mut() {
            if next() % 10 == 0 {
                *v = f64::NAN;
            }
        }
        let x = Matrix::from_flat(rows, cols, flat).unwrap();
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || KnnImputer::new(5, KnnWeights::Uniform),
                    |mut imp| imp.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_matmul,
    bench_matrix_transpose,
    bench_standard_scaler,
    bench_minmax_scaler,
    bench_robust_scaler,
    bench_power_transformer,
    bench_onehot_encoder,
    bench_column_transformer,
    bench_pca,
    bench_pipeline,
    bench_linear_regression,
    bench_ridge_and_lasso,
    bench_logistic_regression,
    bench_kmeans,
    bench_quantile_transformer,
    bench_kbins_discretizer,
    bench_polynomial_features,
    bench_truncated_svd,
    bench_categorical_encoders,
    bench_stats_bulk,
    bench_matrix_ops,
    bench_sparse_matrix,
    bench_knn_imputer,
);
criterion_main!(benches);
