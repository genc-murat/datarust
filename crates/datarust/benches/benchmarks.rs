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
                // Flip ~10% of the labels deterministically so the data is not
                // perfectly separable. IRLS on exactly separable data diverges
                // and burns all `max_iter` Newton steps at O(n·p²) each, which
                // is not representative of real fits.
                let base_label = if dot > 0.0 { 1.0 } else { 0.0 };
                if i % 10 == 0 {
                    1.0 - base_label
                } else {
                    base_label
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
                    || {
                        KBinsDiscretizer::new(10)
                            .unwrap()
                            .strategy(BinStrategy::Quantile)
                    },
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

        let flat = make_matrix(rows, cols).as_slice().to_vec();
        group.bench_with_input(
            BenchmarkId::new("from_flat", format!("{rows}x{cols}")),
            &flat,
            |bencher, f| bencher.iter(|| Matrix::from_flat(rows, cols, f.clone()).unwrap()),
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

fn bench_silhouette_score(c: &mut Criterion) {
    use datarust::cluster::metrics::silhouette_score;
    let mut group = c.benchmark_group("silhouette_score");
    for (rows, cols, k) in [(1000, 10, 5), (5000, 20, 10)] {
        let x = make_matrix(rows, cols);
        let labels: Vec<usize> = (0..rows).map(|i| i % k).collect();
        group.bench_with_input(
            BenchmarkId::new("score", format!("{rows}x{cols}x{k}")),
            &(&x, &labels),
            |bencher, (x, labels)| bencher.iter(|| silhouette_score(x, labels).unwrap()),
        );
    }
    group.finish();
}

fn bench_simple_imputer(c: &mut Criterion) {
    use datarust::imputer::{ImputeStrategy, SimpleImputer};
    let mut group = c.benchmark_group("simple_imputer");
    for (rows, cols) in [(10_000, 20), (50_000, 20)] {
        let mut flat = make_matrix(rows, cols).as_slice().to_vec();
        // Punch ~10% NaNs
        let mut state: u64 = 0x1234_5678;
        for v in flat.iter_mut() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            if (state >> 32) % 10 == 0 {
                *v = f64::NAN;
            }
        }
        let x = Matrix::from_flat(rows, cols, flat).unwrap();
        for strategy in [
            ImputeStrategy::Mean,
            ImputeStrategy::Median,
            ImputeStrategy::MostFrequent,
            ImputeStrategy::Constant(0.0),
        ] {
            let s_name = format!("{:?}", strategy);
            group.bench_with_input(
                BenchmarkId::new(
                    format!("fit_transform_{}", s_name.split('(').next().unwrap()),
                    format!("{rows}x{cols}"),
                ),
                &x,
                |bencher, x| {
                    bencher.iter_batched(
                        || SimpleImputer::new(strategy.clone()),
                        |mut imp| imp.fit_transform(x),
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    group.finish();
}

fn bench_select_k_best(c: &mut Criterion) {
    use datarust::selection::{ScoreFunc, SelectKBest};
    let mut group = c.benchmark_group("select_k_best");
    for (rows, cols) in [(10_000, 50), (50_000, 50)] {
        let x = make_matrix(rows, cols);
        let labels: Vec<String> = (0..rows).map(|i| format!("class_{}", i % 3)).collect();
        for (name, func) in [
            ("f_classif", ScoreFunc::FClassif),
            ("mutual_info", ScoreFunc::MutualInformation),
        ] {
            group.bench_with_input(
                BenchmarkId::new(format!("fit_transform_{}", name), format!("{rows}x{cols}")),
                &(&x, &labels),
                |bencher, (x, y)| {
                    bencher.iter_batched(
                        || SelectKBest::new(func, 10).unwrap(),
                        |mut skb| {
                            skb.fit_with_labels(x, y).unwrap();
                            skb.transform(x).unwrap()
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    group.finish();
}

fn bench_correlation_matrix(c: &mut Criterion) {
    use datarust::stats;
    let mut group = c.benchmark_group("correlation_matrix_flat");
    for (rows, cols) in [(10_000, 50), (10_000, 100), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        let flat = x.as_slice().to_vec();
        group.bench_with_input(
            BenchmarkId::new("pearson", format!("{rows}x{cols}")),
            &flat,
            |bencher, flat| bencher.iter(|| stats::correlation_matrix_flat(flat, rows, cols)),
        );
    }
    group.finish();
}

fn bench_extra_scalers(c: &mut Criterion) {
    let mut group = c.benchmark_group("extra_scalers");
    for (rows, cols) in [(1_000, 50), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("maxabs_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    MaxAbsScaler::new,
                    |mut s| s.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("normalizer_l2_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || Normalizer::new(Norm::L2),
                    |mut n| n.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("binarizer_fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    Binarizer::new,
                    |mut b| b.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_kbins_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("kbins_uniform");
    for (rows, cols) in [(1_000, 50), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || {
                        KBinsDiscretizer::new(10)
                            .unwrap()
                            .strategy(BinStrategy::Uniform)
                    },
                    |mut kb| kb.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_variance_threshold(c: &mut Criterion) {
    use datarust::selection::VarianceThreshold;
    let mut group = c.benchmark_group("variance_threshold");
    for (rows, cols) in [(10_000, 50), (50_000, 50)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || VarianceThreshold::new(0.0).unwrap(),
                    |mut vt| vt.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_regularized_predict(c: &mut Criterion) {
    use datarust::linear_model::{Lasso, Ridge};
    use datarust::traits::Predictor;

    let mut group = c.benchmark_group("regularized_predict");
    for (rows, cols) in [(10_000, 50), (50_000, 100)] {
        let x = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows)
            .map(|i| ((i as f64) * cols as f64).sin() * 10.0 + (i as f64))
            .collect();

        let mut ridge = Ridge::new().with_alpha(1.0);
        ridge.fit(&x, &y).unwrap();
        group.bench_with_input(
            BenchmarkId::new("ridge_predict", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| ridge.predict(x)),
        );

        let mut lasso = Lasso::new().with_alpha(0.1).with_max_iter(200);
        lasso.fit(&x, &y).unwrap();
        group.bench_with_input(
            BenchmarkId::new("lasso_predict", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| lasso.predict(x)),
        );
    }
    group.finish();
}

fn bench_stats_mode(c: &mut Criterion) {
    use datarust::stats;
    let mut group = c.benchmark_group("stats_mode");
    for n in [10_000, 50_000] {
        // Mostly-distinct continuous values: the worst case for `mode`, which
        // drives the SimpleImputer MostFrequent strategy.
        let col = make_matrix(n, 1).as_slice().to_vec();
        group.bench_with_input(BenchmarkId::new("mode", n), &col, |bencher, col| {
            bencher.iter(|| stats::mode(col))
        });
    }
    group.finish();
}

fn bench_matrix_column_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_column_gather");
    for (rows, cols) in [(10_000, 100), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("col_all", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| (0..cols).map(|j| x.col(j)).collect::<Vec<_>>()),
        );
    }
    group.finish();
}

fn bench_pipeline_imputer(c: &mut Criterion) {
    use datarust::imputer::{ImputeStrategy, SimpleImputer};

    let mut group = c.benchmark_group("pipeline_imputer");
    for (rows, cols) in [(10_000, 50), (50_000, 20)] {
        let mut flat = make_matrix(rows, cols).as_slice().to_vec();
        // Punch ~10% NaNs deterministically.
        let mut state: u64 = 0x1234_5678;
        for v in flat.iter_mut() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            if (state >> 32) % 10 == 0 {
                *v = f64::NAN;
            }
        }
        let x = Matrix::from_flat(rows, cols, flat).unwrap();
        group.bench_with_input(
            BenchmarkId::new("impute_then_scale", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || {
                        Pipeline::new()
                            .push(
                                "imputer",
                                TransformerKind::SimpleImputer(SimpleImputer::new(
                                    ImputeStrategy::Mean,
                                )),
                            )
                            .push(
                                "scaler",
                                TransformerKind::StandardScaler(StandardScaler::new()),
                            )
                    },
                    |mut pipe| pipe.fit_transform(x),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_cross_val(c: &mut Criterion) {
    use datarust::linear_model::LinearRegression;
    use datarust::metrics::regression::r2_score;
    use datarust::model_selection::{cross_val_score, KFold};

    let mut group = c.benchmark_group("cross_val");
    for (rows, cols) in [(1_000, 10), (5_000, 20)] {
        let x = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows)
            .map(|i| ((i as f64) * cols as f64).sin() * 10.0 + (i as f64))
            .collect();
        let cv = KFold::new().with_n_splits(5);
        let model = LinearRegression::new();
        group.bench_with_input(
            BenchmarkId::new("linear_5fold", format!("{rows}x{cols}")),
            &(&x, &y, &cv, &model),
            |bencher, (x, y, cv, model)| {
                bencher.iter(|| cross_val_score(*model, x, y, cv, r2_score))
            },
        );
    }
    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    use datarust::metrics::classification::{accuracy_score, roc_auc_score};
    use datarust::metrics::regression::{mean_squared_error, r2_score};
    let mut group = c.benchmark_group("metrics");
    for rows in [10_000, 100_000] {
        let y_true: Vec<f64> = (0..rows).map(|i| (i % 2) as f64).collect();
        let y_pred: Vec<f64> = (0..rows).map(|i| ((i + 1) % 2) as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("r2_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| r2_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("mean_squared_error", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| mean_squared_error(t, p, true).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("accuracy_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| accuracy_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("roc_auc_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| roc_auc_score(t, p).unwrap()),
        );
    }
    group.finish();
}

fn bench_label_encoder(c: &mut Criterion) {
    use datarust::encoder::LabelEncoder;
    let mut group = c.benchmark_group("label_encoder");
    for (rows, cardinality) in [(10_000, 10), (100_000, 10), (100_000, 10_000)] {
        let labels: Vec<String> = (0..rows)
            .map(|i| format!("class_{}", i % cardinality))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("fit_transform", format!("{rows}x{cardinality}")),
            &labels,
            |bencher, labels| {
                bencher.iter_batched(
                    LabelEncoder::new,
                    |mut le| le.fit_transform(labels.iter().cloned()),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_stats_nested(c: &mut Criterion) {
    use datarust::stats;
    let mut group = c.benchmark_group("stats_nested");
    let qs = [0.0, 0.25, 0.5, 0.75, 1.0];
    for (rows, cols) in [(10_000, 100), (100_000, 20)] {
        let nested: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| ((i * cols + j) as f64).sin()).collect())
            .collect();
        group.bench_with_input(
            BenchmarkId::new("mean_var", format!("{rows}x{cols}")),
            &nested,
            |bencher, n| bencher.iter(|| stats::column_mean_var(n, 1)),
        );
        group.bench_with_input(
            BenchmarkId::new("quantiles", format!("{rows}x{cols}")),
            &nested,
            |bencher, n| bencher.iter(|| stats::column_quantiles_many(n, &qs).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("mode_column", format!("{rows}x{cols}")),
            &nested,
            |bencher, n| bencher.iter(|| stats::mode_column(n)),
        );
    }
    group.finish();
}

fn bench_train_test_split(c: &mut Criterion) {
    use datarust::model_selection::train_test_split;
    let mut group = c.benchmark_group("train_test_split");
    for (rows, cols) in [(10_000, 50), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        let y: Vec<f64> = (0..rows).map(|i| (i % 2) as f64).collect();
        group.bench_with_input(
            BenchmarkId::new("split", format!("{rows}x{cols}")),
            &(&x, &y),
            |bencher, (x, y)| bencher.iter(|| train_test_split(x, y)),
        );
    }
    group.finish();
}

/// Classification metrics that the existing `metrics` group does not cover:
/// confusion-matrix-based scoring (precision/recall/F1/kappa/MCC) and the
/// binary probability metrics (`log_loss`, `average_precision`).
fn bench_metrics_more(c: &mut Criterion) {
    use datarust::metrics::classification::{
        average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
        matthews_corrcoef, precision_score, recall_score,
    };
    let mut group = c.benchmark_group("metrics_more");
    for rows in [10_000, 100_000] {
        // Multiclass: 10 classes with ~10% deterministic label noise so the
        // confusion matrix is not trivially diagonal.
        let y_true: Vec<f64> = (0..rows).map(|i| (i % 10) as f64).collect();
        let y_pred: Vec<f64> = (0..rows)
            .map(|i| {
                if i % 10 == 0 {
                    ((i + 1) % 10) as f64
                } else {
                    (i % 10) as f64
                }
            })
            .collect();
        // Binary {0, 1} targets with distinct scores for the probability metrics.
        let y_bin: Vec<f64> = (0..rows).map(|i| (i % 2) as f64).collect();
        // `(i*31) % rows` is a permutation for these row counts (gcd(31, rows)=1),
        // but ~1/7 of rows share the previous score so the tied-score branches of
        // `average_precision_score` and `log_loss` are exercised too.
        let y_score: Vec<f64> = (0..rows)
            .map(|i: usize| {
                if i % 7 == 0 {
                    ((i.wrapping_sub(1).wrapping_mul(31)) % rows) as f64 / rows as f64
                } else {
                    ((i.wrapping_mul(31)) % rows) as f64 / rows as f64
                }
            })
            .collect();

        // NOTE: the confusion-matrix family (precision/recall/F1/kappa/MCC) all
        // share `confusion_matrix_labeled` as their dominant kernel, so their
        // medians land within noise of each other — kept for coverage, but the
        // `confusion_matrix` row alone explains the shared cost.
        group.bench_with_input(
            BenchmarkId::new("confusion_matrix", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| confusion_matrix(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("precision_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| precision_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("recall_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| recall_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("f1_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| f1_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("cohen_kappa_score", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| cohen_kappa_score(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("matthews_corrcoef", rows),
            &(&y_true, &y_pred),
            |bencher, (t, p)| bencher.iter(|| matthews_corrcoef(t, p).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("log_loss", rows),
            &(&y_bin, &y_score),
            |bencher, (t, p)| bencher.iter(|| log_loss(t, p, 1e-15).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("average_precision_score", rows),
            &(&y_bin, &y_score),
            |bencher, (t, p)| bencher.iter(|| average_precision_score(t, p).unwrap()),
        );
    }
    group.finish();
}

/// Softmax probability paths of `LogisticRegression`, which the existing group
/// only exercises through hard-label `predict`.
fn bench_logistic_predict_proba(c: &mut Criterion) {
    use datarust::linear_model::LogisticRegression;
    use datarust::traits::Predictor;

    let mut group = c.benchmark_group("logistic_predict_proba");

    // Binary: same non-separable generator as bench_logistic_regression.
    let (rows, cols) = (50_000, 100);
    let x = make_matrix(rows, cols);
    let y: Vec<f64> = (0..rows)
        .map(|i| {
            let s = x.as_slice();
            let base = i * cols;
            let dot: f64 = (0..cols).map(|j| s[base + j] * (j as f64 + 1.0)).sum();
            let base_label = if dot > 0.0 { 1.0 } else { 0.0 };
            if i % 10 == 0 {
                1.0 - base_label
            } else {
                base_label
            }
        })
        .collect();
    let mut model = LogisticRegression::new().with_max_iter(50);
    model.fit(&x, &y).unwrap();
    group.bench_with_input(
        BenchmarkId::new("predict_proba_binary", format!("{rows}x{cols}")),
        &x,
        |bencher, x| bencher.iter(|| model.predict_proba(x).unwrap()),
    );
    group.bench_with_input(
        BenchmarkId::new("predict_proba_for_class", format!("{rows}x{cols}")),
        &x,
        |bencher, x| bencher.iter(|| model.predict_proba_for_class(x, 1.0).unwrap()),
    );

    // Multiclass (3 classes, one-vs-rest): smaller so the one-time fit stays cheap.
    let (rows, cols) = (10_000, 50);
    let x3 = make_matrix(rows, cols);
    let y3: Vec<f64> = (0..rows)
        .map(|i| {
            let s = x3.as_slice();
            let base = i * cols;
            let dot: f64 = (0..cols).map(|j| s[base + j] * (j as f64 + 1.0)).sum();
            let base_label = if dot > 0.0 { 1.0 } else { 0.0 };
            if i % 10 == 0 {
                2.0
            } else {
                base_label
            }
        })
        .collect();
    let mut model3 = LogisticRegression::new().with_max_iter(50);
    model3.fit(&x3, &y3).unwrap();
    group.bench_with_input(
        BenchmarkId::new("predict_proba_multiclass", format!("{rows}x{cols}")),
        &x3,
        |bencher, x| bencher.iter(|| model3.predict_proba(x).unwrap()),
    );
    group.finish();
}

/// `transform` on fitted categorical encoders. The existing `categorical_encoders`
/// group only measures `fit_transform`, so production-time encoding of new data
/// was unmeasured.
fn bench_encoder_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_transform");
    for (rows, cols) in [(10_000, 20), (50_000, 20)] {
        let x = make_str_matrix(rows, cols);
        let y: Vec<f64> = (0..rows).map(|i| (i % 7) as f64).collect();

        let mut ohe = OneHotEncoder::new();
        ohe.fit_transform(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("onehot_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| ohe.transform(x).unwrap()),
        );

        let mut ord = OrdinalEncoder::new(OrdinalCategories::Auto);
        ord.fit_transform(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("ordinal_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| ord.transform(x).unwrap()),
        );

        let mut freq = FrequencyEncoder::new(true);
        freq.fit_transform(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("frequency_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| freq.transform(x).unwrap()),
        );

        let mut te = TargetEncoder::new(1.0).unwrap();
        te.fit_transform(&x, &y).unwrap();
        group.bench_with_input(
            BenchmarkId::new("target_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| te.transform(x).unwrap()),
        );
    }
    group.finish();
}

/// `transform` on a fitted `PolynomialFeatures` — isolates the interaction-term
/// expansion from the one-time fit (the existing group only benches `fit_transform`).
fn bench_polynomial_transform(c: &mut Criterion) {
    use datarust::polynomial::PolynomialFeatures;
    let mut group = c.benchmark_group("polynomial_transform");
    for (rows, cols, deg) in [(10_000, 5, 2), (10_000, 5, 3)] {
        let x = make_matrix(rows, cols);
        let mut pf = PolynomialFeatures::new(deg);
        pf.fit(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("transform", format!("{rows}x{cols}^d{deg}")),
            &x,
            |bencher, x| bencher.iter(|| pf.transform(x).unwrap()),
        );
    }
    group.finish();
}

/// `transform`/`inverse_transform` on fitted scalers. The scaler groups only
/// bench `fit_transform`, so the memory-bound production transform path — and
/// `QuantileTransformer`'s per-element interpolation — were never isolated.
fn bench_scaler_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaler_transform");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_matrix(rows, cols);

        let mut s = StandardScaler::new();
        s.fit(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("standard_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| s.transform(x).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("standard_inverse", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| s.inverse_transform(x).unwrap()),
        );

        let mut m = MinMaxScaler::new();
        m.fit(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("minmax_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| m.transform(x).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("minmax_inverse", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| m.inverse_transform(x).unwrap()),
        );

        let mut r = RobustScaler::new();
        r.fit(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("robust_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| r.transform(x).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("robust_inverse", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| r.inverse_transform(x).unwrap()),
        );

        // `inverse_transform` is benched on the raw input, not on transformed
        // data — the values do not matter, this isolates the elementwise unscale
        // path (the same cost regardless of what the input represents).

        let mut q = QuantileTransformer::new(1000).unwrap();
        q.fit(&x).unwrap();
        // NOTE: QuantileTransformer has no inverse_transform (per its docs), so
        // only the forward transform is benched here.
        group.bench_with_input(
            BenchmarkId::new("quantile_transform", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| q.transform(x).unwrap()),
        );
    }
    group.finish();
}

/// Sparse one-hot path (`transform_sparse` / `fit_transform_sparse`), which was
/// never benched — dense one-hot was, but the CSR output path is separate.
fn bench_onehot_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("onehot_sparse");
    for (rows, cols) in [(10_000, 20), (50_000, 20)] {
        let x = make_str_matrix(rows, cols);
        let mut ohe = OneHotEncoder::new();
        ohe.fit_transform(&x).unwrap();
        group.bench_with_input(
            BenchmarkId::new("transform_sparse", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| ohe.transform_sparse(x).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("fit_transform_sparse", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    OneHotEncoder::new,
                    |mut ohe| ohe.fit_transform_sparse(x).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Splitter construction and iteration for `KFold` and `StratifiedKFold` — the
/// only model-selection entry points without coverage (KFold split is lazy, so
/// the iterator is forced inside the closure).
fn bench_model_selection_more(c: &mut Criterion) {
    use datarust::model_selection::{KFold, StratifiedKFold};
    let mut group = c.benchmark_group("model_selection");
    for n in [10_000, 100_000] {
        let kf = KFold::new()
            .with_n_splits(5)
            .with_shuffle(true)
            .with_random_state(0);
        group.bench_with_input(BenchmarkId::new("kfold_5_shuffled", n), &n, |bencher, n| {
            bencher.iter(|| kf.split(*n).unwrap().for_each(drop))
        });

        let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let skf = StratifiedKFold::new().with_n_splits(5);
        group.bench_with_input(
            BenchmarkId::new("stratified_kfold_5", n),
            &y,
            |bencher, y| bencher.iter(|| skf.split(y).unwrap().for_each(drop)),
        );
    }
    group.finish();
}

/// String-column gathering: cloning every cell (`StrMatrix::column`) vs the
/// borrowing `column_refs` view. This is the allocation-vs-borrow tradeoff the
/// profile crate's categorical tallies make on every wide table.
fn bench_strmatrix_column(c: &mut Criterion) {
    let mut group = c.benchmark_group("strmatrix_column");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_str_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("column_clone", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| (0..cols).map(|j| x.column(j)).collect::<Vec<_>>()),
        );
        group.bench_with_input(
            BenchmarkId::new("column_refs", format!("{rows}x{cols}")),
            &x,
            |bencher, x| bencher.iter(|| (0..cols).map(|j| x.column_refs(j)).collect::<Vec<_>>()),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_label_encoder,
    bench_stats_nested,
    bench_train_test_split,
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
    bench_silhouette_score,
    bench_simple_imputer,
    bench_select_k_best,
    bench_metrics,
    bench_correlation_matrix,
    bench_extra_scalers,
    bench_kbins_uniform,
    bench_variance_threshold,
    bench_regularized_predict,
    bench_stats_mode,
    bench_matrix_column_gather,
    bench_pipeline_imputer,
    bench_cross_val,
    bench_metrics_more,
    bench_logistic_predict_proba,
    bench_encoder_transform,
    bench_polynomial_transform,
    bench_scaler_transform,
    bench_onehot_sparse,
    bench_model_selection_more,
    bench_strmatrix_column,
);
criterion_main!(benches);
