# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT `LICENSE` file.
- `rust-toolchain.toml` pinning the stable toolchain.
- GitHub Actions CI (fmt, clippy, test matrix across features and OSes, MSRV 1.70).
- Crate metadata (`repository`, `keywords`, `categories`, `rust-version`, `readme`).
- Crate-level lints: `missing_docs`, `clippy::all`.
- `DatarustError::Io` and `DatarustError::Serde` variants for precise serialization/IO errors.

### Changed
- `serialize::{save_json, load_json}` now accept `AsRef<Path>` instead of `&str` and report
  IO/serde errors via the new dedicated `DatarustError` variants.
- `Matrix` construction from `Vec<Vec<f64>>` is now fallible: the panicking `From<Vec<Vec<f64>>>`
  impl was replaced by `TryFrom<Vec<Vec<f64>>>` (returns `Result`).
- Sorting of floating-point columns now uses `total_cmp`, making it NaN-safe and panic-free.
- `stats::median_sorted` and `stats::quantile` validate their inputs and return `Option` instead
  of panicking on empty slices / out-of-range quantiles.
- `jacobi::eigh` validates the input matrix shape and returns `Option` on malformed input instead
  of panicking.
- Covariance computation is consolidated into a single tested implementation.

### Fixed
- `PowerTransformer` now exposes `inverse_transform` on the `Transformer` trait (the underlying
  `inverse_one` was previously dead code).
- `ColumnTransformer` no longer panics when too few input feature names are supplied.
- `.gitignore` no longer excludes `.github/`, unblocking CI workflows.
- README `OrdinalEncoder` example now matches the real API.
- Stale doc comment claiming categorical passthrough is unsupported (it is implemented).
