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

/// String matrix with `cardinality` distinct values per column (near-unique
/// for `cardinality == rows`), for high-cardinality categorical workloads.
fn make_str_matrix_cardinality(rows: usize, cols: usize, cardinality: usize) -> StrMatrix {
    let data: Vec<Vec<String>> = (0..rows)
        .map(|i| (0..cols).map(|j| format!("id_{}_{}", j, i % cardinality)).collect())
        .collect();
    StrMatrix::new(data).unwrap()
}

/// String matrix of binary columns (two alternating values per column).
fn make_str_matrix_binary(rows: usize, cols: usize) -> StrMatrix {
    let data: Vec<Vec<String>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    if (i + j) % 2 == 0 {
                        "a".to_string()
                    } else {
                        "b".to_string()
                    }
                })
                .collect()
        })
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

fn bench_relationships(c: &mut Criterion) {
    use datarust_profile::Relationships;
    let mut group = c.benchmark_group("relationships");
    // Wide numeric-only table: isolates the Pearson O(p²·n) kernel that every
    // profile runs under the hood.
    for (rows, cols) in [(10_000, 50), (10_000, 100)] {
        let m = make_matrix(rows, cols);
        let cols_vec: Vec<Vec<f64>> = (0..cols).map(|j| m.col(j)).collect();
        let names: Vec<String> = (0..cols).map(|j| format!("c{j}")).collect();
        let mut refs: Vec<(&str, &[f64])> = Vec::with_capacity(cols);
        for (j, c) in cols_vec.iter().enumerate() {
            refs.push((names[j].as_str(), c.as_slice()));
        }
        group.bench_with_input(
            BenchmarkId::new("pearson_wide", format!("{rows}x{cols}")),
            &refs,
            |bencher, refs| {
                bencher.iter(|| Relationships::compute(refs, &[] as &[(&str, &[&str])]))
            },
        );
    }
    // Mixed numeric + categorical: exercises Pearson + Cramér's V.
    for (rows, n_cols, c_cols) in [(10_000, 20, 20)] {
        let num = make_matrix(rows, n_cols);
        let cat = make_str_matrix(rows, c_cols);
        let num_cols: Vec<Vec<f64>> = (0..n_cols).map(|j| num.col(j)).collect();
        let num_names: Vec<String> = (0..n_cols).map(|j| format!("n{j}")).collect();
        let mut num_refs: Vec<(&str, &[f64])> = Vec::with_capacity(n_cols);
        for (j, c) in num_cols.iter().enumerate() {
            num_refs.push((num_names[j].as_str(), c.as_slice()));
        }
        let cat_cols: Vec<Vec<&str>> = (0..c_cols)
            .map(|j| (0..rows).map(|i| cat.get(i, j)).collect())
            .collect();
        let cat_names: Vec<String> = (0..c_cols).map(|j| format!("k{j}")).collect();
        let mut cat_refs: Vec<(&str, &[&str])> = Vec::with_capacity(c_cols);
        for (j, c) in cat_cols.iter().enumerate() {
            cat_refs.push((cat_names[j].as_str(), c.as_slice()));
        }
        group.bench_with_input(
            BenchmarkId::new("mixed", format!("{rows}x{n_cols}+{c_cols}")),
            &(&num_refs, &cat_refs),
            |bencher, (num_refs, cat_refs)| {
                bencher.iter(|| Relationships::compute(num_refs, cat_refs))
            },
        );
    }
    group.finish();
}

fn bench_profile_table_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_table_wide");
    for (rows, n_cols, c_cols) in [(10_000, 30, 30), (50_000, 20, 20)] {
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

fn bench_report_html(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_html");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        let profile = profile_matrix(&x, None).unwrap();
        group.bench_with_input(
            BenchmarkId::new("to_html", format!("{rows}x{cols}")),
            &profile,
            |bencher, profile| bencher.iter(|| datarust_profile::report::to_html(profile)),
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

fn bench_cramers_v_high_cardinality(c: &mut Criterion) {
    use datarust_profile::Relationships;
    let mut group = c.benchmark_group("cramers_v_high_cardinality");
    // High-cardinality categoricals: the worst case for Cramér's V, where the
    // contingency table exceeds the flat-array limit and falls back to the
    // hash-map path keyed by (row_level, col_level).
    for (rows, cols, cardinality) in [(10_000, 10, 100), (10_000, 10, 1_000)] {
        let cat = make_str_matrix_cardinality(rows, cols, cardinality);
        let cat_cols: Vec<Vec<&str>> = (0..cols)
            .map(|j| (0..rows).map(|i| cat.get(i, j)).collect())
            .collect();
        let cat_names: Vec<String> = (0..cols).map(|j| format!("k{j}")).collect();
        let mut cat_refs: Vec<(&str, &[&str])> = Vec::with_capacity(cols);
        for (j, c) in cat_cols.iter().enumerate() {
            cat_refs.push((cat_names[j].as_str(), c.as_slice()));
        }
        group.bench_with_input(
            BenchmarkId::new("cramers_v", format!("{rows}x{cols}x{cardinality}")),
            &cat_refs,
            |bencher, cat_refs| {
                bencher.iter(|| Relationships::compute(&[] as &[(&str, &[f64])], cat_refs))
            },
        );
    }
    group.finish();
}

fn bench_point_biserial(c: &mut Criterion) {
    use datarust_profile::Relationships;
    let mut group = c.benchmark_group("point_biserial");
    // Many binary categoricals crossed with many numeric columns: the
    // (categorical × numeric) pairwise path that dominated profile cost on
    // wide tables before the allocation-free rewrite.
    for (rows, n_cols, c_cols) in [(10_000, 20, 20), (50_000, 10, 10)] {
        let num = make_matrix(rows, n_cols);
        let cat = make_str_matrix_binary(rows, c_cols);
        let num_cols: Vec<Vec<f64>> = (0..n_cols).map(|j| num.col(j)).collect();
        let num_names: Vec<String> = (0..n_cols).map(|j| format!("n{j}")).collect();
        let mut num_refs: Vec<(&str, &[f64])> = Vec::with_capacity(n_cols);
        for (j, c) in num_cols.iter().enumerate() {
            num_refs.push((num_names[j].as_str(), c.as_slice()));
        }
        let cat_cols: Vec<Vec<&str>> = (0..c_cols)
            .map(|j| (0..rows).map(|i| cat.get(i, j)).collect())
            .collect();
        let cat_names: Vec<String> = (0..c_cols).map(|j| format!("k{j}")).collect();
        let mut cat_refs: Vec<(&str, &[&str])> = Vec::with_capacity(c_cols);
        for (j, c) in cat_cols.iter().enumerate() {
            cat_refs.push((cat_names[j].as_str(), c.as_slice()));
        }
        group.bench_with_input(
            BenchmarkId::new("compute", format!("{rows}x{n_cols}+{c_cols}")),
            &(&num_refs, &cat_refs),
            |bencher, (num_refs, cat_refs)| {
                bencher.iter(|| Relationships::compute(num_refs, cat_refs))
            },
        );
    }
    group.finish();
}

fn bench_profile_str_high_cardinality(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_str_high_cardinality");
    // Near-unique string columns (e.g. identifier columns): stresses type
    // inference, per-column categorical tallies, duplicate-row detection, and
    // Cramér's V over huge level sets.
    for (rows, cols, cardinality) in [(10_000, 20, 10_000), (50_000, 10, 50_000)] {
        let x = make_str_matrix_cardinality(rows, cols, cardinality);
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

#[cfg(feature = "serde")]
fn bench_report_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_json");
    for (rows, cols) in [(10_000, 20), (100_000, 20)] {
        let x = make_matrix(rows, cols);
        let profile = profile_matrix(&x, None).unwrap();
        let report = datarust_profile::report::JsonReport::from_profile(&profile);
        group.bench_with_input(
            BenchmarkId::new("to_json", format!("{rows}x{cols}")),
            &report,
            |bencher, report| bencher.iter(|| datarust_profile::report::to_json(report).unwrap()),
        );
    }
    group.finish();
}

#[cfg(feature = "serde")]
criterion_group!(
    benches,
    bench_profile_matrix,
    bench_profile_table,
    bench_profile_str_matrix,
    bench_quality_checks,
    bench_relationships,
    bench_profile_table_wide,
    bench_report_html,
    bench_cramers_v_high_cardinality,
    bench_point_biserial,
    bench_profile_str_high_cardinality,
    bench_report_json,
);
#[cfg(not(feature = "serde"))]
criterion_group!(
    benches,
    bench_profile_matrix,
    bench_profile_table,
    bench_profile_str_matrix,
    bench_quality_checks,
    bench_relationships,
    bench_profile_table_wide,
    bench_report_html,
    bench_cramers_v_high_cardinality,
    bench_point_biserial,
    bench_profile_str_high_cardinality,
);
criterion_main!(benches);
