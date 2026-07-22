# Installation

## Cargo

```toml
[dependencies]
datarust = "0.5"
```

The default build has **zero external dependencies** — `cargo add datarust` pulls in nothing but the standard library.

## Optional features

All features are opt-in and independent. Enable the ones you need:

```toml
[dependencies]
datarust = { version = "0.5", features = ["serde", "rayon", "matrixmultiply"] }
```

### `serde`

Enables JSON serialization of fitted transformers via `datarust::serialize::{save_json, load_json, to_json, from_json}`. Models serialize to human-readable JSON (not pickle).

```rust
use datarust::serialize::{save_json, load_json};
use datarust::scaler::StandardScaler;
use datarust::traits::Transformer;

let mut scaler = StandardScaler::new();
scaler.fit(&x)?;

// Save
save_json(&scaler, "model.json")?;

// Load into a fresh instance
let restored: StandardScaler = load_json("model.json")?;
let out = restored.transform(&x)?;
```

### `rayon`

Enables parallel column/row operations for large datasets. Speeds up scaler transforms, encoders, and the KNN imputer on inputs above ~4 000 rows.

### `matrixmultiply`

Enables a tuned pure-Rust GEMM (via the `matrixmultiply` crate, **no system BLAS**) for `Matrix::matmul` and centered-covariance computation. Significantly speeds up PCA, TruncatedSVD, and the linear models on large dense inputs. The default build remains zero-external-dependency; opt in with this feature.

### `datasets`

Embeds four classic toy datasets (Iris, Breast Cancer, Wine, Diabetes) as `const` arrays for examples, tests, and onboarding. No file I/O, no network access — the data is compiled into the binary. See the [Datasets guide](./guide/datasets.md).

## Rust version

datarust targets **Rust 1.70+** (the `rust-version` field in `Cargo.toml`). It is tested against stable, beta, and the MSRV in CI.

## Platform support

CI runs on **Ubuntu** and **macOS** (both x86-64 and arm64). WASM should work for the default build (no system calls, no threading without `rayon`), though it is not formally tested.
