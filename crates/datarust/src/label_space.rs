//! Compact vocabulary for non-negative integer-valued class labels.

use crate::error::{DatarustError, Result};

/// Validate and canonicalize a hard class label.
pub(crate) fn canonical_label(value: f64) -> Result<f64> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(DatarustError::InvalidInput(format!(
            "classification labels must be non-negative integers, found {value}"
        )));
    }
    // Keep negative zero from becoming a second sorted representation of zero.
    Ok(if value == 0.0 { 0.0 } else { value })
}

/// A sorted, compact vocabulary for numeric class labels.
///
/// The vocabulary separates external class values from their dense model and
/// matrix indices. For example, fitting on `{10, 20}` stores labels `[10, 20]`
/// and encodes them as indices `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelSpace {
    labels: Vec<f64>,
}

impl LabelSpace {
    /// Build a compact label vocabulary from one label slice.
    pub fn fit(labels: &[f64]) -> Result<Self> {
        if labels.is_empty() {
            return Err(DatarustError::EmptyInput("labels are empty".into()));
        }
        let mut canonical = Vec::with_capacity(labels.len());
        for &label in labels {
            canonical.push(canonical_label(label)?);
        }
        canonical.sort_by(f64::total_cmp);
        canonical.dedup_by(|a, b| *a == *b);
        Ok(Self { labels: canonical })
    }

    pub(crate) fn from_pair(y_true: &[f64], y_pred: &[f64]) -> Result<Self> {
        if y_true.is_empty() {
            return Err(DatarustError::EmptyInput("y_true is empty".into()));
        }
        if y_true.len() != y_pred.len() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} predictions", y_true.len()),
                actual: format!("{} predictions", y_pred.len()),
            });
        }
        let mut labels = Vec::with_capacity(y_true.len() + y_pred.len());
        labels.extend_from_slice(y_true);
        labels.extend_from_slice(y_pred);
        Self::fit(&labels)
    }

    /// Sorted original class labels represented by this vocabulary.
    pub fn labels(&self) -> &[f64] {
        &self.labels
    }

    /// Number of distinct classes.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the vocabulary contains no classes.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Convert an original class label to its compact zero-based index.
    pub fn encode(&self, label: f64) -> Result<usize> {
        let label = canonical_label(label)?;
        self.labels
            .binary_search_by(|candidate| candidate.total_cmp(&label))
            .map_err(|_| DatarustError::UnknownLabel(label.to_string()))
    }

    /// Convert a compact index back to its original class label.
    pub fn decode(&self, index: usize) -> Result<f64> {
        self.labels
            .get(index)
            .copied()
            .ok_or_else(|| DatarustError::UnknownLabel(format!("index {index}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_unique_round_trip() {
        let space = LabelSpace::fit(&[20.0, 10.0, 20.0, -0.0]).unwrap();
        assert_eq!(space.labels(), &[0.0, 10.0, 20.0]);
        assert_eq!(space.encode(10.0).unwrap(), 1);
        assert_eq!(space.decode(2).unwrap(), 20.0);
    }

    #[test]
    fn invalid_labels_are_rejected() {
        for invalid in [-1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(LabelSpace::fit(&[0.0, invalid]).is_err());
        }
    }
}
