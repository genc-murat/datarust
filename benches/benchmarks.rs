use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use datarust::categorical_kind::CategoricalTransformerKind;
use datarust::compose::*;
use datarust::decomposition::*;
use datarust::encoder::*;
use datarust::pipeline::Pipeline;
use datarust::scaler::*;
use datarust::traits::Transformer;
use datarust::transformer_kind::TransformerKind;
use datarust::{Matrix, StrMatrix};

fn make_matrix(rows: usize, cols: usize) -> Matrix {
    let data: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| ((i * cols + j) as f64).sin() * 10.0)
                .collect()
        })
        .collect();
    Matrix::new(data).unwrap()
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
    use datarust::traits::Regressor;

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
    use datarust::traits::Regressor;

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
    use datarust::traits::Regressor;

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
);
criterion_main!(benches);
