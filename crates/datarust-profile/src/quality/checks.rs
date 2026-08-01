//! Data-quality checks derived from a [`DatasetProfile`].
//!
//! Each [`QualityIssue`] is a single human-readable finding with a severity.
//! The thresholds are conservative defaults; callers may filter the returned
//! list as desired.

use crate::profile::DatasetProfile;
use crate::types::{ColumnType, Severity};

/// The category of a [`QualityIssue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum QualityKind {
    /// A column has a missing fraction at or above the threshold.
    HighMissing,
    /// A numeric column has (near-)zero variance — it carries no signal.
    ConstantColumn,
    /// A categorical column has cardinality equal to the row count (likely an
    /// identifier rather than a feature).
    NearUnique,
    /// The dataset contains exact-duplicate rows.
    DuplicateRows,
    /// A numeric column has values outside the Tukey IQR fences.
    Outliers,
    /// A categorical column is dominated by a single value.
    Imbalance,
    /// A pair of numeric columns is highly correlated (`|r| >= threshold`).
    HighCorrelation,
    /// A feature column is suspiciously highly correlated with the target column.
    TargetLeakage,
}

/// A single data-quality finding.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct QualityIssue {
    /// What kind of issue this is.
    pub kind: QualityKind,
    /// How serious the issue is.
    pub severity: Severity,
    /// Which column the issue concerns, or `None` for dataset-wide findings.
    pub column: Option<String>,
    /// Human-readable description, suitable for direct display in a report.
    pub message: String,
}

/// Thresholds controlling when each check fires.
///
/// All fields are intentionally `pub` so callers can tune them before running
/// [`run_checks`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Thresholds {
    /// Missing fraction at or above which [`QualityKind::HighMissing`] fires.
    pub missing_fraction: f64,
    /// Variance at or below which [`QualityKind::ConstantColumn`] fires.
    pub near_zero_variance: f64,
    /// `unique / n_rows` at or above which [`QualityKind::NearUnique`] fires.
    pub near_unique_ratio: f64,
    /// Outlier fraction at or above which [`QualityKind::Outliers`] fires.
    pub outlier_fraction: f64,
    /// Imbalance ratio (`freq / present`) at or above which
    /// [`QualityKind::Imbalance`] fires.
    pub imbalance_ratio: f64,
    /// Correlation magnitude `|r|` at or above which [`QualityKind::HighCorrelation`] fires.
    pub high_correlation: f64,
    /// Correlation magnitude `|r|` or Cramér's V at or above which [`QualityKind::TargetLeakage`] fires.
    pub target_leakage: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            missing_fraction: 0.5,
            near_zero_variance: 1e-12,
            near_unique_ratio: 0.98,
            outlier_fraction: 0.05,
            imbalance_ratio: 0.95,
            high_correlation: 0.95,
            target_leakage: 0.90,
        }
    }
}

/// Runs all data-quality checks against `profile` using `thresholds`.
pub fn run_checks(profile: &DatasetProfile, thresholds: &Thresholds) -> Vec<QualityIssue> {
    let mut issues = Vec::new();

    for col in &profile.columns {
        if col.missing_fraction >= thresholds.missing_fraction && col.count > 0 {
            issues.push(QualityIssue {
                kind: QualityKind::HighMissing,
                severity: if col.missing_fraction >= 0.9 {
                    Severity::Critical
                } else {
                    Severity::Warning
                },
                column: Some(col.name.clone()),
                message: format!(
                    "{}: {:.1}% of values are missing",
                    col.name,
                    col.missing_fraction * 100.0
                ),
            });
        }

        match col.column_type {
            ColumnType::Numeric => {
                if let Some(n) = &col.numeric {
                    let var = n.std * n.std;
                    if var <= thresholds.near_zero_variance {
                        issues.push(QualityIssue {
                            kind: QualityKind::ConstantColumn,
                            severity: Severity::Warning,
                            column: Some(col.name.clone()),
                            message: format!(
                                "{}: near-zero variance ({:.3e}); column is effectively constant",
                                col.name, var
                            ),
                        });
                    }
                    if n.outlier_count > 0 && n.outlier_fraction >= thresholds.outlier_fraction {
                        issues.push(QualityIssue {
                            kind: QualityKind::Outliers,
                            severity: if n.outlier_fraction >= 0.2 {
                                Severity::Warning
                            } else {
                                Severity::Info
                            },
                            column: Some(col.name.clone()),
                            message: format!(
                                "{}: {} outliers ({:.1}%) beyond IQR fences",
                                col.name,
                                n.outlier_count,
                                n.outlier_fraction * 100.0
                            ),
                        });
                    }
                }
            }
            ColumnType::Categorical => {
                if let Some(c) = &col.categorical {
                    if col.count > 0 {
                        let ratio = c.unique as f64 / col.count as f64;
                        if ratio >= thresholds.near_unique_ratio {
                            issues.push(QualityIssue {
                                kind: QualityKind::NearUnique,
                                severity: Severity::Info,
                                column: Some(col.name.clone()),
                                message: format!(
                                    "{}: {} unique values across {} rows (ratio {:.2}); likely an identifier",
                                    col.name, c.unique, col.count, ratio
                                ),
                            });
                        }
                        if c.imbalance_ratio >= thresholds.imbalance_ratio {
                            issues.push(QualityIssue {
                                kind: QualityKind::Imbalance,
                                severity: Severity::Critical,
                                column: Some(col.name.clone()),
                                message: format!(
                                    "{}: top value '{}' covers {:.1}% of rows",
                                    col.name,
                                    c.top,
                                    c.imbalance_ratio * 100.0
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    if profile.duplicate_rows > 0 {
        issues.push(QualityIssue {
            kind: QualityKind::DuplicateRows,
            severity: if profile.duplicate_fraction >= 0.1 {
                Severity::Warning
            } else {
                Severity::Info
            },
            column: None,
            message: format!(
                "{} of {} rows are exact duplicates ({:.2}%)",
                profile.duplicate_rows,
                profile.n_rows,
                profile.duplicate_fraction * 100.0
            ),
        });
    }

    // Check relationship findings (HighCorrelation & TargetLeakage)
    if let Some(rels) = &profile.relationships {
        // Pearson high correlation
        if let Some(pearson) = &rels.pearson {
            let p = pearson.labels.len();
            for i in 0..p {
                for j in (i + 1)..p {
                    let r = pearson.values[i][j];
                    let abs_r = r.abs();

                    // Check high correlation between feature pairs
                    if abs_r >= thresholds.high_correlation {
                        issues.push(QualityIssue {
                            kind: QualityKind::HighCorrelation,
                            severity: Severity::Warning,
                            column: Some(pearson.labels[i].clone()),
                            message: format!(
                                "High Pearson correlation between '{}' and '{}' (r = {:.3})",
                                pearson.labels[i], pearson.labels[j], r
                            ),
                        });
                    }

                    // Check target leakage if target_column matches either column
                    if let Some(target) = &profile.target_column {
                        let is_i_target = &pearson.labels[i] == target;
                        let is_j_target = &pearson.labels[j] == target;
                        if (is_i_target || is_j_target)
                            && !(is_i_target && is_j_target)
                            && abs_r >= thresholds.target_leakage
                        {
                            let feature = if is_i_target {
                                &pearson.labels[j]
                            } else {
                                &pearson.labels[i]
                            };
                            issues.push(QualityIssue {
                                kind: QualityKind::TargetLeakage,
                                severity: Severity::Critical,
                                column: Some(feature.clone()),
                                message: format!(
                                    "Suspected target leakage: feature '{}' has strong correlation with target '{}' (r = {:.3})",
                                    feature, target, r
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Cramér's V target leakage & high correlation
        if let Some(cramers) = &rels.cramers_v {
            let p = cramers.labels.len();
            for i in 0..p {
                for j in (i + 1)..p {
                    let v = cramers.values[i][j];

                    if let Some(target) = &profile.target_column {
                        let is_i_target = &cramers.labels[i] == target;
                        let is_j_target = &cramers.labels[j] == target;
                        if (is_i_target || is_j_target)
                            && !(is_i_target && is_j_target)
                            && v >= thresholds.target_leakage
                        {
                            let feature = if is_i_target {
                                &cramers.labels[j]
                            } else {
                                &cramers.labels[i]
                            };
                            issues.push(QualityIssue {
                                kind: QualityKind::TargetLeakage,
                                severity: Severity::Critical,
                                column: Some(feature.clone()),
                                message: format!(
                                    "Suspected target leakage: categorical feature '{}' has high Cramér's V with target '{}' (V = {:.3})",
                                    feature, target, v
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Point-biserial target leakage
        if let Some(target) = &profile.target_column {
            for pb in &rels.point_biserial {
                let abs_r = pb.correlation.abs();
                if abs_r >= thresholds.target_leakage {
                    let is_cat_target = &pb.categorical == target;
                    let is_num_target = &pb.numeric == target;
                    if (is_cat_target || is_num_target) && !(is_cat_target && is_num_target) {
                        let feature = if is_cat_target {
                            &pb.numeric
                        } else {
                            &pb.categorical
                        };
                        issues.push(QualityIssue {
                            kind: QualityKind::TargetLeakage,
                            severity: Severity::Critical,
                            column: Some(feature.clone()),
                            message: format!(
                                "Suspected target leakage: feature '{}' has high point-biserial correlation with target '{}' (r = {:.3})",
                                feature, target, pb.correlation
                            ),
                        });
                    }
                }
            }
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{ColumnProfile, DatasetProfile, FiveNumber, Histogram, NumericStats};
    use crate::types::{ColumnType, Severity};

    fn make_numeric_profile(
        name: &str,
        mean: f64,
        std: f64,
        missing_fraction: f64,
    ) -> ColumnProfile {
        ColumnProfile {
            name: name.to_string(),
            column_type: ColumnType::Numeric,
            count: 100,
            missing_count: (missing_fraction * 100.0) as usize,
            missing_fraction,
            numeric: Some(NumericStats {
                mean,
                std,
                five: FiveNumber {
                    min: mean - 2.0 * std,
                    q1: mean - 0.67 * std,
                    median: mean,
                    q3: mean + 0.67 * std,
                    max: mean + 2.0 * std,
                },
                skewness: 0.0,
                kurtosis: 0.0,
                histogram: Histogram {
                    edges: vec![],
                    counts: vec![],
                },
                outlier_count: 0,
                outlier_fraction: 0.0,
            }),
            categorical: None,
        }
    }

    fn make_categorical_profile(
        name: &str,
        unique: usize,
        imbalance_ratio: f64,
        missing_fraction: f64,
    ) -> ColumnProfile {
        ColumnProfile {
            name: name.to_string(),
            column_type: ColumnType::Categorical,
            count: 100,
            missing_count: (missing_fraction * 100.0) as usize,
            missing_fraction,
            numeric: None,
            categorical: Some(crate::profile::CategoricalStats {
                unique,
                top: "dominant".to_string(),
                freq: (imbalance_ratio * 100.0) as usize,
                imbalance_ratio,
                top_values: vec![],
            }),
        }
    }

    #[test]
    fn run_checks_high_missing_warning() {
        let col = make_numeric_profile("high_miss", 0.0, 1.0, 0.6);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::HighMissing && i.severity == Severity::Warning));
    }

    #[test]
    fn run_checks_high_missing_critical() {
        let col = make_numeric_profile("crit_miss", 0.0, 1.0, 0.95);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::HighMissing && i.severity == Severity::Critical));
    }

    #[test]
    fn run_checks_constant_column() {
        let col = make_numeric_profile("const", 5.0, 1e-13, 0.0);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues.iter().any(|i| i.kind == QualityKind::ConstantColumn));
    }

    #[test]
    fn run_checks_outliers_detected() {
        let mut col = make_numeric_profile("out", 0.0, 1.0, 0.0);
        col.numeric.as_mut().unwrap().outlier_count = 10;
        col.numeric.as_mut().unwrap().outlier_fraction = 0.1;
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues.iter().any(|i| i.kind == QualityKind::Outliers));
    }

    #[test]
    fn run_checks_categorical_imbalance() {
        let col = make_categorical_profile("imb", 2, 0.96, 0.0);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::Imbalance && i.severity == Severity::Critical));
    }

    #[test]
    fn run_checks_near_unique() {
        let col = make_categorical_profile("uid", 99, 0.01, 0.0);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues.iter().any(|i| i.kind == QualityKind::NearUnique));
    }

    #[test]
    fn run_checks_duplicate_rows() {
        let col = make_numeric_profile("x", 0.0, 1.0, 0.0);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 10,
            duplicate_fraction: 0.1,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::DuplicateRows && i.severity == Severity::Warning));
    }

    #[test]
    fn run_checks_high_correlation() {
        use crate::profile::relationships::{CorrelationMatrix, Relationships};
        let col1 = make_numeric_profile("a", 0.0, 1.0, 0.0);
        let col2 = make_numeric_profile("b", 0.0, 1.0, 0.0);
        let pearson = CorrelationMatrix {
            labels: vec!["a".to_string(), "b".to_string()],
            values: vec![vec![1.0, 0.99], vec![0.99, 1.0]],
        };
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 2,
            memory_bytes: 1600,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col1, col2],
            relationships: Some(Relationships {
                pearson: Some(pearson),
                cramers_v: None,
                point_biserial: vec![],
            }),
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::HighCorrelation));
    }

    #[test]
    fn run_checks_target_leakage_pearson() {
        use crate::profile::relationships::{CorrelationMatrix, Relationships};
        let col1 = make_numeric_profile("feature", 0.0, 1.0, 0.0);
        let col2 = make_numeric_profile("target", 0.0, 1.0, 0.0);
        let pearson = CorrelationMatrix {
            labels: vec!["feature".to_string(), "target".to_string()],
            values: vec![vec![1.0, 0.95], vec![0.95, 1.0]],
        };
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 2,
            memory_bytes: 1600,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: Some("target".to_string()),
            columns: vec![col1, col2],
            relationships: Some(Relationships {
                pearson: Some(pearson),
                cramers_v: None,
                point_biserial: vec![],
            }),
        };
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(issues
            .iter()
            .any(|i| i.kind == QualityKind::TargetLeakage && i.severity == Severity::Critical));
    }

    #[test]
    fn run_checks_custom_thresholds() {
        let col = make_numeric_profile("x", 0.0, 1.0, 0.3);
        let profile = DatasetProfile {
            n_rows: 100,
            n_columns: 1,
            memory_bytes: 800,
            duplicate_rows: 0,
            duplicate_fraction: 0.0,
            target_column: None,
            columns: vec![col],
            relationships: None,
        };
        // Default threshold is 0.5, so 0.3 shouldn't trigger
        let issues = run_checks(&profile, &Thresholds::default());
        assert!(!issues.iter().any(|i| i.kind == QualityKind::HighMissing));
        // But with custom threshold of 0.25 it should
        let mut t = Thresholds::default();
        t.missing_fraction = 0.25;
        let issues = run_checks(&profile, &t);
        assert!(issues.iter().any(|i| i.kind == QualityKind::HighMissing));
    }

    #[test]
    fn quality_issue_serialization() {
        let issue = QualityIssue {
            kind: QualityKind::HighMissing,
            severity: Severity::Warning,
            column: Some("test".to_string()),
            message: "test message".to_string(),
        };
        // Just verify it can be constructed
        assert_eq!(issue.column, Some("test".to_string()));
        assert_eq!(issue.severity, Severity::Warning);
    }

    #[test]
    fn thresholds_default_values() {
        let t = Thresholds::default();
        assert_eq!(t.missing_fraction, 0.5);
        assert_eq!(t.near_zero_variance, 1e-12);
        assert_eq!(t.near_unique_ratio, 0.98);
        assert_eq!(t.outlier_fraction, 0.05);
        assert_eq!(t.imbalance_ratio, 0.95);
        assert_eq!(t.high_correlation, 0.95);
        assert_eq!(t.target_leakage, 0.90);
    }
}
