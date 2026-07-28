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
