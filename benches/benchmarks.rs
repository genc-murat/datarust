use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use datarust::decomposition::*;
use datarust::pipeline::Pipeline;
use datarust::scaler::*;
use datarust::traits::Transformer;
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

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
                    || StandardScaler::new(),
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
                    || MinMaxScaler::new(),
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
                    || RobustScaler::new(),
                    |mut s| s.fit_transform(x),
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
    bench_pca,
    bench_pipeline,
);
criterion_main!(benches);
