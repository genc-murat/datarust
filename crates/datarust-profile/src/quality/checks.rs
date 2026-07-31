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

