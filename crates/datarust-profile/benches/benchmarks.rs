use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use datarust::{Matrix, StrMatrix};
use datarust_profile::{profile_matrix, profile_str_matrix, profile_table, run_checks, Thresholds};

/// Deterministic pseudo-random matrix with values in roughly `[-1, 1]`.
fn make_matrix(rows: usize, cols: usize) -> Matrix {
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

fn bench_profile_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_matrix");
    for (rows, cols) in [(10_000, 20), (100_000, 20), (10_000, 100)] {
        let x = make_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("from_matrix", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || (),
                    |_| profile_matrix(x, None).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_profile_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_table");
    for (rows, n_cols, c_cols) in [(10_000, 10, 10), (100_000, 10, 10)] {
        let numeric = make_matrix(rows, n_cols);
        let categorical = make_str_matrix(rows, c_cols);
        let names: Vec<String> = (0..n_cols + c_cols).map(|j| format!("col{j}")).collect();
        group.bench_with_input(
            BenchmarkId::new("mixed_table", format!("{rows}x{}+{}", n_cols, c_cols)),
            &(&numeric, &categorical, &names),
            |bencher, (num, cat, names)| {
                bencher.iter_batched(
                    || (),
                    |_| profile_table(Some(num), Some(cat), names).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_profile_str_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_str_matrix");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_str_matrix(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("from_str_matrix", format!("{rows}x{cols}")),
            &x,
            |bencher, x| {
                bencher.iter_batched(
                    || (),
                    |_| profile_str_matrix(x, None).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_quality_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_checks");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        let profile = profile_matrix(&x, None).unwrap();
        group.bench_with_input(
            BenchmarkId::new("run_checks", format!("{rows}x{cols}")),
            &profile,
            |bencher, profile| bencher.iter(|| run_checks(profile, &Thresholds::default())),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_profile_matrix,
    bench_profile_table,
    bench_profile_str_matrix,
    bench_quality_checks,
);
criterion_main!(benches);
