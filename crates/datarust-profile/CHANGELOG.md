# Changelog

All notable changes to `datarust-profile` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-31

This release adds pairwise relationship analysis across numeric and categorical columns, target-leakage detection hints, correlation heatmaps in the HTML report, and two new data-quality checks.

### Added

- **`Relationships` block** on `DatasetProfile`:
  - **Pearson correlation matrix** over numeric columns using `datarust::stats::correlation_matrix`.
  - **Cramér's V association matrix** over categorical columns via pure-Rust contingency tables (zero dependencies).
  - **Point-biserial correlation** between binary categorical columns and numeric columns.
- **Target-leakage hints**:
  - `profile_matrix_with_target` and `profile_table_with_target` entry points.
  - `DatasetProfile::with_target` builder method.
- **Data-quality checks**:
  - `QualityKind::HighCorrelation` — flags pairs of numeric columns with `|r| >= 0.95`.
  - `QualityKind::TargetLeakage` — flags feature columns strongly correlated (`|r| >= 0.90` or `V >= 0.90`) with a designated target column.
- **Correlation heatmaps in HTML report**:
  - Diverging Red–White–Blue palette for Pearson correlation.
  - Sequential White–Purple palette for Cramér's V matrix.
  - Point-biserial summary table.
  - Full dark-mode CSS support.
- New runnable `correlation_analysis` example demonstrating relationship profiling and leakage hints.

## [0.2.0] - 2026-07-31

Distributional depth release: shape statistics, IQR outlier detection, categorical imbalance, and a `_flat` fast path.

### Added

- `skewness` and excess `kurtosis` (Fisher–Pearson) on `NumericStats`.
- `Histogram` (equal-width, Sturges bins) on `NumericStats`.
- IQR outlier detection (`outlier_count`, `outlier_fraction`) and `QualityKind::Outliers`.
- Categorical `imbalance_ratio` (`freq / present`) and `QualityKind::Imbalance`.
- `_flat` fast path in `from_matrix` via `column_mean_var_flat` and `column_quantiles_many_flat`.
- Responsive CSS card-grid layout in HTML report with inline mini-bar charts.

## [0.1.0] - 2026-07-31

Initial release of `datarust-profile`.

### Added

- Core data model: `DatasetProfile`, `ColumnProfile`, `NumericStats`, `CategoricalStats`.
- Entry points: `profile_matrix`, `profile_str_matrix`, `profile_table`.
- Basic statistics: count, missing_count, missing_fraction, mean, std (ddof=1), five-number summary, unique, top, freq.
- Automatic column-type inference (`infer_column`) and missing-value markers.
- Duplicate-row detection.
- Quality checks: `HighMissing`, `ConstantColumn`, `NearUnique`, `DuplicateRows`.
- Self-contained HTML report renderer (`report::to_html`).
- Serde-gated JSON report renderer (`report::to_json`).
