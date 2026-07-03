#!/usr/bin/env python3
"""Benchmark mirror of examples/bench_compare_rust.rs.

Generates deterministic synthetic data with the *same* PRNG layout as the Rust
runner (xorshift64, seed 42) so both sides transform identical inputs, then
reports median `fit_transform` time in milliseconds.

Usage: python3 benches/compare_sklearn.py [reps]
"""

import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer


# --- xorshift64 PRNG identical to the Rust runner -----------------------------

class XorShift64:
    def __init__(self, seed: int):
        self.state = 0x9E3779B97F4A7C15 if seed == 0 else seed

    def next_u64(self) -> int:
        s = self.state
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = s
        return s

    def next_unit(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next_unit()


def make_matrix(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    rng = XorShift64(seed)
    # Generate in the same iteration order as Rust: row-major, col inner.
    flat = [rng.next_range(-100.0, 100.0) for _ in range(rows * cols)]
    return np.asarray(flat, dtype=np.float64).reshape(rows, cols)


def make_str_array(rows: int, cols: int, seed: int = 42, cardinality: int = 20):
    rng = XorShift64(seed)
    # List-of-lists of strings, same iteration order as Rust.
    data = [
        [f"cat_{int(rng.next_unit() * cardinality)}" for _ in range(cols)]
        for _ in range(rows)
    ]
    # object array so sklearn treats each cell as a string.
    return np.asarray(data, dtype=object)


def median(samples):
    s = sorted(samples)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def measure(reps: int, workload):
    workload()  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        workload()
        dt = (time.perf_counter() - t0) * 1000.0
        samples.append(dt)
    return median(samples)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    sizes = [(1_000, 10), (10_000, 100), (50_000, 200)]
    cat_sizes = [(1_000, 5), (10_000, 10), (50_000, 20)]

    print("workload,rows,cols,sklearn_ms")
    for rows, cols in sizes:
        x = make_matrix(rows, cols, 42)

        ms = measure(reps, lambda: StandardScaler().fit_transform(x))
        print(f"standard_scaler,{rows},{cols},{ms:.4f}")

        ms = measure(reps, lambda: MinMaxScaler().fit_transform(x))
        print(f"minmax_scaler,{rows},{cols},{ms:.4f}")

        ms = measure(reps, lambda: RobustScaler().fit_transform(x))
        print(f"robust_scaler,{rows},{cols},{ms:.4f}")

        k = min(10, max(1, cols // 2))
        ms = measure(reps, lambda: PCA(n_components=k).fit_transform(x))
        print(f"pca,{rows},{cols},{ms:.4f}")

        ms = measure(
            reps,
            lambda: Pipeline(
                [
                    ("s1", StandardScaler()),
                    ("s2", MinMaxScaler()),
                    ("s3", RobustScaler()),
                ]
            ).fit_transform(x),
        )
        print(f"pipeline_3scalers,{rows},{cols},{ms:.4f}")

    for rows, cols in cat_sizes:
        x_str = make_str_array(rows, cols, 42, 20)
        ms = measure(reps, lambda: OneHotEncoder(sparse_output=False).fit_transform(x_str))
        print(f"onehot_encoder,{rows},{cols},{ms:.4f}")

        # ColumnTransformer: numeric scaled + categorical one-hot.
        numeric = make_matrix(rows, cols, 7)
        categorical = make_str_array(rows, cols, 11, 15)
        x_all = np.hstack([numeric, categorical.astype(str)])
        num_idx = list(range(cols))
        cat_idx = list(range(cols, 2 * cols))
        ms = measure(
            reps,
            lambda: ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_idx),
                    ("cat", OneHotEncoder(sparse_output=False), cat_idx),
                ]
            ).fit_transform(x_all),
        )
        print(f"column_transformer,{rows},{cols},{ms:.4f}")


if __name__ == "__main__":
    main()
