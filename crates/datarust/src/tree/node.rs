//! Leaf-value summaries for the tree module.

/// Per-class weighted counts at a node.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClassHistogram {
    /// Weighted count per class, in `LabelSpace` order.
    pub counts: Vec<f64>,
}

/// Leaf / node value summary.
///
/// Classification nodes carry a per-class count histogram (one entry per
/// original label, in `LabelSpace` order); regression nodes carry the target
/// sum ingredients. Values are produced by the splitter's criterion state
/// (`node_value`), which guarantees they match sklearn bit-for-bit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ValueSummary {
    /// Classification: weighted class counts over the node's samples.
    Classification(ClassHistogram),
    /// Regression: sum of targets, sum of squared targets, sample count.
    Regression {
        /// Sum of the node's target values.
        sum: f64,
        /// Sum of squared targets.
        sum_sq: f64,
        /// Number of samples.
        n: f64,
    },
}

impl ValueSummary {
    /// Predicted class: index of the largest count; ties resolve to the
    /// **smallest class index**, matching sklearn's stable scan order.
    pub fn argmax_class(&self) -> Result<usize> {
        match self {
            ValueSummary::Classification(h) => {
                let mut best = 0usize;
                let mut best_count = -1.0f64;
                for (k, &c) in h.counts.iter().enumerate() {
                    if c > best_count {
                        best = k;
                        best_count = c;
                    }
                }
                if best_count <= 0.0 {
                    return Err(DatarustError::InvalidInput(
                        "cannot predict from an empty node".into(),
                    ));
                }
                Ok(best)
            }
            ValueSummary::Regression { .. } => Err(DatarustError::InvalidInput(
                "classification prediction on a regression tree".into(),
            )),
        }
    }

    /// Predicted regression value: the node's target mean.
    pub fn regression_mean(&self) -> Result<f64> {
        match self {
            ValueSummary::Regression { sum, n, .. } => {
                if *n <= 0.0 {
                    return Err(DatarustError::InvalidInput(
                        "cannot predict from an empty node".into(),
                    ));
                }
                Ok(sum / *n)
            }
            ValueSummary::Classification(_) => Err(DatarustError::InvalidInput(
                "regression prediction on a classification tree".into(),
            )),
        }
    }

    /// Per-class probability vector (classification only).
    pub fn class_probabilities(&self) -> Result<Vec<f64>> {
        match self {
            ValueSummary::Classification(h) => {
                let n: f64 = h.counts.iter().sum();
                if n <= 0.0 {
                    return Err(DatarustError::InvalidInput(
                        "cannot produce probabilities from an empty node".into(),
                    ));
                }
                Ok(h.counts.iter().map(|&c| c / n).collect())
            }
            ValueSummary::Regression { .. } => Err(DatarustError::InvalidInput(
                "predict_proba on a regression tree".into(),
            )),
        }
    }
}

use crate::error::{DatarustError, Result};
