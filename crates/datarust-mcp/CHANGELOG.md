# Changelog

All notable changes to `datarust-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-20

First crates.io release. The crate previously lived in-tree as an unpublished
`0.7.0` placeholder; `0.1.0` marks its public debut.

### Added

- stdio JSON-RPC 2.0 MCP server handling `initialize`,
  `notifications/initialized`, `ping`, `tools/list`, and `tools/call` —
  compatible with Claude Desktop, Cursor, Glama, and Smithery.
- `profile_dataset`: profiles CSV content (descriptive statistics,
  distribution shape, pairwise correlations, data-quality checks), with an
  optional target column for leakage detection. Backed by `datarust-profile`.
- `train_linear_regression`: fits OLS linear regression with scikit-learn
  parity.
- `scale_features`: StandardScaler / MinMaxScaler / RobustScaler.
- `evaluate_metrics`: R², MSE, RMSE, MAE, and accuracy.
