use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DatarustError {
    NotFitted(String),
    InvalidInput(String),
    ShapeMismatch { expected: String, actual: String },
    EmptyInput(String),
    AllMissing(String),
    UnknownCategory(String),
    UnknownLabel(String),
    InvalidConfig(String),
    Singular(String),
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
        }
    }
}

impl std::error::Error for DatarustError {}

pub type Result<T> = std::result::Result<T, DatarustError>;
