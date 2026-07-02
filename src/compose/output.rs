//! Output container for column-transformer results, preserving the separation
//! between numeric and categorical columns after transformation.

use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};

/// The result of applying a [`ColumnTransformer`](super::ColumnTransformer),
/// keeping numeric and categorical columns in separate matrices.
///
/// Unlike the input [`Table`](super::Table), an `Output` may have a different
/// number or arrangement of columns.  Categorical passthrough columns are
/// preserved as strings rather than being one-hot encoded.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Output {
    /// Numeric columns after transformation.
    pub numeric: Matrix,
    /// Categorical (string) columns after transformation.
    pub categorical: StrMatrix,
}

impl Output {
    /// Creates an `Output` from separate numeric and categorical matrices.
    ///
    /// Returns an error if the row counts do not match.
    pub fn new(numeric: Matrix, categorical: StrMatrix) -> Result<Self> {
        if numeric.nrows() != categorical.nrows() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} rows", numeric.nrows()),
                actual: format!("{} rows", categorical.nrows()),
            });
        }
        Ok(Self {
            numeric,
            categorical,
        })
    }

    /// Returns the number of rows (must be equal in both matrices).
    pub fn nrows(&self) -> usize {
        self.numeric.nrows()
    }

    /// Returns the total number of columns (numeric + categorical).
    pub fn ncols(&self) -> usize {
        self.numeric.ncols() + self.categorical.ncols()
    }
}
