use std::error::Error as StdError;
use std::fmt;

/// Errors returned by datarust operations.
///
/// Every fallible public API returns [`Result<T, DatarustError>`]. The variant
/// describes which class of problem occurred.
///
/// [`Result<T, DatarustError>`]: crate::error::Result
#[derive(Debug)]
pub enum DatarustError {
    /// A transformer was used before being fitted.
    NotFitted(String),
    /// The supplied data is malformed or otherwise invalid for the operation.
    InvalidInput(String),
    /// Two operands did not have compatible dimensions.
    ShapeMismatch {
        /// A description of what was expected.
        expected: String,
        /// A description of what was actually provided.
        actual: String,
    },
    /// An empty collection was provided where at least one element is required.
    EmptyInput(String),
    /// A column (or set of columns) consisted entirely of missing values.
    AllMissing(String),
    /// A category unseen at fit time was encountered at transform time.
    UnknownCategory(String),
    /// A label unseen at fit time was encountered at transform time.
    UnknownLabel(String),
    /// A transformer was constructed with an illegal parameter combination.
    InvalidConfig(String),
    /// A numerical operation broke down (e.g. division by zero, singular matrix).
    Singular(String),
    /// An IO failure while reading from or writing to disk.
    Io(std::io::Error),
    /// A (de)serialization failure (e.g. malformed JSON).
    #[cfg(feature = "serde")]
    Serde(serde_json::Error),
}

impl fmt::Display for DatarustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatarustError::NotFitted(s) => write!(f, "transformer not fitted: {}", s),
            DatarustError::InvalidInput(s) => write!(f, "invalid input: {}", s),
            DatarustError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {}, got {}", expected, actual)
            }
            DatarustError::EmptyInput(s) => write!(f, "empty input: {}", s),
            DatarustError::AllMissing(s) => write!(f, "all values missing: {}", s),
            DatarustError::UnknownCategory(s) => write!(f, "unknown category: {}", s),
            DatarustError::UnknownLabel(s) => write!(f, "unknown label: {}", s),
            DatarustError::InvalidConfig(s) => write!(f, "invalid configuration: {}", s),
            DatarustError::Singular(s) => write!(f, "singular/unstable operation: {}", s),
            DatarustError::Io(e) => write!(f, "io error: {}", e),
            #[cfg(feature = "serde")]
            DatarustError::Serde(e) => write!(f, "serialization error: {}", e),
        }
    }
}

impl StdError for DatarustError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            DatarustError::Io(e) => Some(e),
            #[cfg(feature = "serde")]
            DatarustError::Serde(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DatarustError {
    fn from(e: std::io::Error) -> Self {
        DatarustError::Io(e)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for DatarustError {
    fn from(e: serde_json::Error) -> Self {
        DatarustError::Serde(e)
    }
}

/// The canonical `Result` type alias used throughout the crate.
pub type Result<T> = std::result::Result<T, DatarustError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        assert_eq!(
            DatarustError::NotFitted("scaler".into()).to_string(),
            "transformer not fitted: scaler"
        );
    }

    #[test]
    fn shape_mismatch_display() {
        let e = DatarustError::ShapeMismatch {
            expected: "2 columns".into(),
            actual: "3 columns".into(),
        };
        assert_eq!(
            e.to_string(),
            "shape mismatch: expected 2 columns, got 3 columns"
        );
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e: DatarustError = io_err.into();
        assert!(matches!(e, DatarustError::Io(_)));
        assert!(e.source().is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn from_serde_error() {
        let serde_err = serde_json::from_str::<()>("").unwrap_err();
        let e: DatarustError = serde_err.into();
        assert!(matches!(e, DatarustError::Serde(_)));
        assert!(e.source().is_some());
    }
}
