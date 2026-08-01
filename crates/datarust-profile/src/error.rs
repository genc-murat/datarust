//! Error types returned by datarust-profile.

use std::error::Error as StdError;
use std::fmt;

/// Errors returned by profiling operations.
#[derive(Debug)]
pub enum ProfileError {
    /// The supplied data is malformed or otherwise invalid for the operation.
    InvalidInput(String),
    /// An empty dataset (zero rows or zero columns) was provided.
    EmptyInput(String),
    /// A failure propagated from the underlying `datarust` crate.
    Datarust(datarust::error::DatarustError),
    /// An IO failure while reading from or writing to disk.
    Io(std::io::Error),
    /// A (de)serialization failure (e.g. malformed JSON). Only present under the
    /// `serde` feature.
    #[cfg(feature = "serde")]
    Serde(serde_json::Error),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::InvalidInput(s) => write!(f, "invalid input: {}", s),
            ProfileError::EmptyInput(s) => write!(f, "empty input: {}", s),
            ProfileError::Datarust(e) => write!(f, "datarust error: {}", e),
            ProfileError::Io(e) => write!(f, "io error: {}", e),
            #[cfg(feature = "serde")]
            ProfileError::Serde(e) => write!(f, "serialization error: {}", e),
        }
    }
}

impl StdError for ProfileError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ProfileError::Datarust(e) => Some(e),
            ProfileError::Io(e) => Some(e),
            #[cfg(feature = "serde")]
            ProfileError::Serde(e) => Some(e),
            _ => None,
        }
    }
}

impl From<datarust::error::DatarustError> for ProfileError {
    fn from(e: datarust::error::DatarustError) -> Self {
        ProfileError::Datarust(e)
    }
}

impl From<std::io::Error> for ProfileError {
    fn from(e: std::io::Error) -> Self {
        ProfileError::Io(e)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for ProfileError {
    fn from(e: serde_json::Error) -> Self {
        ProfileError::Serde(e)
    }
}

/// The canonical `Result` type alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ProfileError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_input() {
        let e = ProfileError::InvalidInput("bad".to_string());
        assert_eq!(e.to_string(), "invalid input: bad");
    }

    #[test]
    fn display_empty_input() {
        let e = ProfileError::EmptyInput("0 rows".to_string());
        assert_eq!(e.to_string(), "empty input: 0 rows");
    }

    #[test]
    fn display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e = ProfileError::Io(io_err);
        assert_eq!(e.to_string(), "io error: file missing");
    }

    #[test]
    fn source_invalid_input() {
        let e = ProfileError::InvalidInput("x".to_string());
        assert!(e.source().is_none());
    }

    #[test]
    fn source_empty_input() {
        let e = ProfileError::EmptyInput("x".to_string());
        assert!(e.source().is_none());
    }

    #[test]
    fn source_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "oops");
        let e = ProfileError::Io(io_err);
        assert!(e.source().is_some());
    }

    #[test]
    fn from_datarust_error() {
        let dr_err = datarust::error::DatarustError::ShapeMismatch {
            expected: "(2, 2)".to_string(),
            actual: "(3, 3)".to_string(),
        };
        let e: ProfileError = dr_err.into();
        assert!(e.to_string().contains("datarust error"));
        assert!(e.source().is_some());
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: ProfileError = io_err.into();
        assert!(e.to_string().contains("io error"));
    }

    #[test]
    fn debug_is_impl() {
        let e = ProfileError::InvalidInput("test".to_string());
        let debug = format!("{:?}", e);
        assert!(debug.contains("InvalidInput"));
    }
}
