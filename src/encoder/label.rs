use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::traits::LabelTransformer;

/// How to handle unknown labels in [`LabelEncoder::transform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LabelHandleUnknown {
    /// Return an error when an unknown label is encountered (default).
    #[default]
    Error,
    /// Map unknown labels to [`usize::MAX`].  Callers must check for this
    /// sentinel value before passing to [`inverse_transform`](LabelEncoder::inverse_transform).
    Ignore,
}

/// Encode target labels with values between `0` and `n_classes - 1`, mirroring
/// `sklearn.preprocessing.LabelEncoder`. Classes are sorted (sklearn default).
///
/// Operates on `Vec<String>` (1-D) input.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LabelEncoder {
    classes: Vec<String>,
    indices: HashMap<String, usize>,
    handle_unknown: LabelHandleUnknown,
    fitted: bool,
}

impl LabelEncoder {
    /// Creates a new label encoder.
    pub fn new() -> Self {
        Self {
            classes: vec![],
            indices: HashMap::new(),
            handle_unknown: LabelHandleUnknown::Error,
            fitted: false,
        }
    }

    /// Set how unknown labels are handled during transform.
    pub fn handle_unknown(mut self, val: LabelHandleUnknown) -> Self {
        self.handle_unknown = val;
        self
    }

    /// Returns the sorted class labels.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Learns the sorted class labels from the input.
    pub fn fit<I, S>(&mut self, labels: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for s in labels {
            seen.insert(s.into());
        }
        if seen.is_empty() {
            return Err(DatarustError::EmptyInput("no labels to fit".into()));
        }
        self.classes = seen.into_iter().collect();
        self.indices = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        self.fitted = true;
        Ok(())
    }

    /// Maps each label to its class index.
    pub fn transform<I>(&self, labels: I) -> Result<Vec<usize>>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LabelEncoder".into()));
        }
        let mut out = Vec::new();
        for s in labels {
            let key = s.as_ref();
            match self.indices.get(key) {
                Some(&i) => out.push(i),
                None => match self.handle_unknown {
                    LabelHandleUnknown::Error => {
                        return Err(DatarustError::UnknownLabel(key.to_string()))
                    }
                    LabelHandleUnknown::Ignore => out.push(usize::MAX),
                },
            }
        }
        Ok(out)
    }

    /// Fits the encoder and transforms the labels in one step.
    pub fn fit_transform<I, S>(&mut self, labels: I) -> Result<Vec<usize>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let collected: Vec<String> = labels.into_iter().map(|s| s.into()).collect();
        self.fit(collected.iter().cloned())?;
        self.transform(collected.iter())
    }

    /// Maps class indices back to their labels.
    pub fn inverse_transform(&self, indices: &[usize]) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("LabelEncoder".into()));
        }
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            if i == usize::MAX {
                // Sentinel for unknown labels (HandleUnknown::Ignore)
                out.push(String::new());
            } else if i >= self.classes.len() {
                return Err(DatarustError::UnknownLabel(format!("index {}", i)));
            } else {
                out.push(self.classes[i].clone());
            }
        }
        Ok(out)
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelTransformer for LabelEncoder {
    fn name(&self) -> &'static str {
        "LabelEncoder"
    }

    fn fit(&mut self, x: &[String]) -> Result<()> {
        self.fit(x.iter().cloned())
    }

    fn transform(&self, x: &[String]) -> Result<Vec<usize>> {
        self.transform(x.iter())
    }

    fn inverse_transform(&self, x: &[usize]) -> Result<Vec<String>> {
        self.inverse_transform(x)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_fit_transform() {
        let mut le = LabelEncoder::new();
        let out = le.fit_transform(["dog", "cat", "bird"]).unwrap();
        // classes sorted: bird(0), cat(1), dog(2)
        assert_eq!(le.classes(), &["bird", "cat", "dog"]);
        // input dog, cat, bird -> 2, 1, 0
        assert_eq!(out, vec![2, 1, 0]);
    }

    #[test]
    fn duplicate_labels_deduped() {
        let mut le = LabelEncoder::new();
        let out = le.fit_transform(["a", "b", "a", "b", "c"]).unwrap();
        assert_eq!(le.classes(), &["a", "b", "c"]);
        assert_eq!(out, vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn transform_on_new_data() {
        let mut le = LabelEncoder::new();
        le.fit(["a", "b", "c"]).unwrap();
        let out = le.transform(["c", "a", "b"]).unwrap();
        assert_eq!(out, vec![2, 0, 1]);
    }

    #[test]
    fn unknown_label_errors() {
        let mut le = LabelEncoder::new();
        le.fit(["a", "b"]).unwrap();
        let err = le.transform(["a", "z"]).unwrap_err();
        assert!(matches!(err, DatarustError::UnknownLabel(s) if s == "z"));
    }

    #[test]
    fn inverse_round_trip() {
        let original = ["dog", "cat", "bird", "dog"];
        let mut le = LabelEncoder::new();
        let enc = le.fit_transform(original).unwrap();
        let dec = le.inverse_transform(&enc).unwrap();
        assert_eq!(dec, original);
    }

    #[test]
    fn inverse_bad_index() {
        let mut le = LabelEncoder::new();
        le.fit(["a", "b"]).unwrap();
        assert!(le.inverse_transform(&[0, 5]).is_err());
    }

    #[test]
    fn empty_fit_errors() {
        let mut le = LabelEncoder::new();
        let err = le.fit::<_, &str>([]).unwrap_err();
        assert!(matches!(err, DatarustError::EmptyInput(_)));
    }

    #[test]
    fn transform_before_fit_errors() {
        let le = LabelEncoder::new();
        assert!(matches!(
            le.transform(["a"]),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn preserves_lexicographic_order_numeric_strings() {
        let mut le = LabelEncoder::new();
        le.fit(["10", "2", "1"]).unwrap();
        // sorted lexicographically: "1","10","2"
        assert_eq!(le.classes(), &["1", "10", "2"]);
    }

    #[test]
    fn handle_unknown_ignore_returns_max() {
        let mut le = LabelEncoder::new().handle_unknown(LabelHandleUnknown::Ignore);
        le.fit(["a", "b"]).unwrap();
        let out = le.transform(["a", "z", "b"]).unwrap();
        assert_eq!(out, vec![0, usize::MAX, 1]);
    }

    #[test]
    fn handle_unknown_default_is_error() {
        let mut le = LabelEncoder::new();
        le.fit(["a", "b"]).unwrap();
        assert!(matches!(
            le.transform(["z"]),
            Err(DatarustError::UnknownLabel(_))
        ));
    }

    #[test]
    fn inverse_transform_sentinel_decodes_to_empty() {
        let mut le = LabelEncoder::new().handle_unknown(LabelHandleUnknown::Ignore);
        le.fit(["a", "b"]).unwrap();
        let out = le.transform(["a", "z", "b"]).unwrap();
        assert_eq!(out, vec![0, usize::MAX, 1]);
        let decoded = le.inverse_transform(&out).unwrap();
        assert_eq!(decoded, vec!["a", "", "b"]);
    }
}
