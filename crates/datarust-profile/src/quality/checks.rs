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
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            missing_fraction: 0.5,
            near_zero_variance: 1e-12,
            near_unique_ratio: 0.98,
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

    issues
}
