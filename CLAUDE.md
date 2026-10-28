# Claude Code Instructions for datarust

## Project Overview
`datarust` is a pure-Rust classical machine learning ecosystem with bit-level scikit-learn parity and automated dataset profiling.

## Commands
- Build workspace: `cargo build --workspace`
- Test workspace: `cargo test --workspace`
- Format code: `cargo fmt --all -- --check`
- Site build: `npm run build && npm run check`

## Architecture & Code Conventions
- Pure safe Rust without default external linear algebra dependencies.
- Use `datarust::Matrix` for all continuous feature operations.
- Transformers implement `Transformer`, estimators implement `Estimator` / `Predictor`.
- Use `datarust_profile` for EDA and automated dataset quality reporting.
