//! JSON serialization of profile reports.
//!
//! Requires the `serde` feature. The on-disk schema is a thin projection of
//! the in-memory [`DatasetProfile`]: every public field is serialized, with
//! `Option` fields omitted when `None`.

use crate::error::Result;
use crate::profile::DatasetProfile;
use crate::quality::checks::run_checks;
use crate::quality::{QualityIssue, Thresholds};

/// A self-contained, serializable report bundling the profile and any quality
/// findings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct JsonReport<'a> {
    /// The dataset profile.
    pub profile: &'a DatasetProfile,
    /// Data-quality findings.
    pub quality: Vec<QualityIssue>,
}

impl<'a> JsonReport<'a> {
    /// Builds a report by running [`run_checks`] with default thresholds.
    pub fn from_profile(profile: &'a DatasetProfile) -> Self {
        let quality = run_checks(profile, &Thresholds::default());
        JsonReport { profile, quality }
    }
}

/// Serializes `report` to a JSON string.
pub fn to_json(report: &JsonReport<'_>) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

/// Serializes `report` to a JSON string and writes it to `path`.
pub fn to_json_file(report: &JsonReport<'_>, path: &str) -> Result<()> {
    let serialized = to_json(report)?;
    std::fs::write(path, serialized)?;
    Ok(())
}
