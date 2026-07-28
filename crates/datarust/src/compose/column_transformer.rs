use crate::categorical_kind::CategoricalTransformerKind;
use crate::encoder::OneHotEncoder;
use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};
use crate::target_kind::TargetTransformerKind;
use crate::traits::{default_input_names, CategoricalTransformer, FeatureNames, TargetTransformer};
use crate::transformer_kind::TransformerKind;
use crate::Transformer;

/// What to do with columns not explicitly selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Remainder {
    /// Drop unselected columns (default, like sklearn `remainder='drop'`).
    #[default]
    Drop,
    /// Pass through columns that were not selected, appended at the end of the
    /// output in original order. Numeric passthrough keeps the raw values;
    /// categorical passthrough is one-hot encoded with the fitted remainder
    /// encoders.
    Passthrough,
}

/// A specification of one block within a [`ColumnTransformer`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColumnSpec {
    /// A block applying a numeric transformer to selected columns.
    Numeric {
        /// Name of the block.
        name: String,
        /// Indices of the numeric columns to transform.
        columns: Vec<usize>,
        /// Transformer applied to the selected numeric columns.
        transformer: TransformerKind,
    },
    /// A block applying a categorical encoder to selected string columns.
    Categorical {
        /// Name of the block.
        name: String,
        /// Indices of the categorical columns to encode.
        columns: Vec<usize>,
        /// Encoder applied to the selected categorical columns.
        encoder: CategoricalTransformerKind,
    },
    /// A block applying a supervised categorical encoder (e.g. TargetEncoder)
    /// that requires target values during fit.
    Target {
        /// Name of the block.
        name: String,
        /// Indices of the categorical columns to encode.
        columns: Vec<usize>,
        /// Supervised encoder applied to the selected categorical columns.
        encoder: TargetTransformerKind,
    },
}

/// Combined input carrying numeric and categorical columns of equal row count.
#[derive(Debug, Clone)]
pub struct Table {
    /// The numeric feature matrix.
    pub numeric: Matrix,
    /// The categorical (string) feature matrix.
    pub categorical: StrMatrix,
}

impl Table {
    /// Creates a new table from numeric and categorical matrices with matching row counts.
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

    /// Build a numeric-only table with no categorical columns. Categorical
    /// selection in specs is an error at fit time.
    pub fn from_numeric(numeric: Matrix) -> Self {
        let cat = StrMatrix {
            data: (0..numeric.nrows()).map(|_| vec![]).collect(),
        };
        Table {
            numeric,
            categorical: cat,
        }
    }

    /// Returns the number of rows in the table.
    pub fn nrows(&self) -> usize {
        self.numeric.nrows()
    }
}

/// Apply different transformers to different columns of a dataset, mirroring
/// `sklearn.compose.ColumnTransformer`.
///
/// Output columns are ordered as: each spec in insertion order (categorical
/// specs expand to their one-hot columns), followed by passthrough numeric
/// columns (when `remainder = Passthrough`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColumnTransformer {
    specs: Vec<ColumnSpec>,
    remainder: Remainder,
    /// Fitted: the set of numeric column indices consumed by specs (to compute remainder).
    #[cfg_attr(feature = "serde", serde(default))]
    consumed_numeric: Vec<usize>,
    /// Fitted: total number of numeric columns in the input Table.
    #[cfg_attr(feature = "serde", serde(default))]
    total_numeric_cols: usize,
    /// Fitted: the set of categorical column indices consumed by specs.
    #[cfg_attr(feature = "serde", serde(default))]
    consumed_categorical: Vec<usize>,
    /// Fitted: total number of categorical columns in the input Table.
    #[cfg_attr(feature = "serde", serde(default))]
    total_categorical_cols: usize,
    /// Fitted: maximum column index referenced by any numeric or categorical spec.
    /// Used by `feature_names_out(None)` to size the synthetic names array.
    #[cfg_attr(feature = "serde", serde(default))]
    max_col_index: usize,
    /// Fitted: categorical encoders for unused categorical columns (remainder passthrough).
    #[cfg_attr(feature = "serde", serde(default))]
    remainder_cat_encoders: Vec<CategoricalTransformerKind>,
    /// Fitted: ordered list of unused categorical column indices.
    #[cfg_attr(feature = "serde", serde(default))]
    remainder_cat_cols: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    fitted: bool,
}

impl ColumnTransformer {
    /// Creates a new empty column transformer.
    pub fn new() -> Self {
        Self {
            specs: vec![],
            remainder: Remainder::Drop,
            consumed_numeric: vec![],
            total_numeric_cols: 0,
            consumed_categorical: vec![],
            total_categorical_cols: 0,
            max_col_index: 0,
            remainder_cat_encoders: vec![],
            remainder_cat_cols: vec![],
            fitted: false,
        }
    }

    /// Sets how unselected columns are handled.
    pub fn remainder(mut self, r: Remainder) -> Self {
        self.remainder = r;
        self
    }

    /// Adds a numeric block applying `transformer` to `columns`.
    pub fn add_numeric<S>(
        mut self,
        name: S,
        columns: Vec<usize>,
        transformer: TransformerKind,
    ) -> Self
    where
        S: Into<String>,
    {
        self.specs.push(ColumnSpec::Numeric {
            name: name.into(),
            columns,
            transformer,
        });
        self
    }

    /// Adds a categorical block encoding `columns` with a categorical
    /// [`CategoricalTransformerKind`] wrapper.
    pub fn add_categorical<S>(
        mut self,
        name: S,
        columns: Vec<usize>,
        encoder: CategoricalTransformerKind,
    ) -> Self
    where
        S: Into<String>,
    {
        self.specs.push(ColumnSpec::Categorical {
            name: name.into(),
            columns,
            encoder,
        });
        self
    }

    /// Adds a target (supervised) category encoding block, using `encoder` on
    /// `columns`.  This spec requires target values during fit; call
    /// [`fit_with_target`](Self::fit_with_target) instead of [`fit`](Self::fit).
    pub fn add_target<S>(
        mut self,
        name: S,
        columns: Vec<usize>,
        encoder: TargetTransformerKind,
    ) -> Self
    where
        S: Into<String>,
    {
        self.specs.push(ColumnSpec::Target {
            name: name.into(),
            columns,
            encoder,
        });
        self
    }

    fn validate_columns(table: &Table, cols: &[usize], categorical: bool) -> Result<()> {
        let max = if categorical {
            table.categorical.ncols()
        } else {
            table.numeric.ncols()
        };
        for &c in cols {
            if c >= max {
                return Err(DatarustError::InvalidInput(format!(
                    "column index {} out of range (max {})",
                    c, max
                )));
            }
        }
        Ok(())
    }

    fn extract_numeric_cols(table: &Table, cols: &[usize]) -> Result<Matrix> {
        Self::validate_columns(table, cols, false)?;
        let mut data = Vec::with_capacity(table.nrows());
        for i in 0..table.nrows() {
            let row: Vec<f64> = cols.iter().map(|&c| table.numeric.get(i, c)).collect();
            data.push(row);
        }
        Matrix::new(data)
    }

    fn extract_categorical_cols(table: &Table, cols: &[usize]) -> Result<StrMatrix> {
        Self::validate_columns(table, cols, true)?;
        let mut data = Vec::with_capacity(table.nrows());
        for i in 0..table.nrows() {
            let row: Vec<String> = cols
                .iter()
                .map(|&c| table.categorical.get(i, c).to_string())
                .collect();
            data.push(row);
        }
        StrMatrix::new(data)
    }

    /// Fits the column specs and remainder encoders to the table.
    ///
    /// Returns an error if any [`ColumnSpec::Target`] specs are present,
    /// because they require target values.  Use
    /// [`fit_with_target`](Self::fit_with_target) instead.
    pub fn fit(&mut self, table: &Table) -> Result<()> {
        if self
            .specs
            .iter()
            .any(|s| matches!(s, ColumnSpec::Target { .. }))
        {
            return Err(DatarustError::InvalidInput(
                "ColumnTransformer contains Target specs; use fit_with_target() instead".into(),
            ));
        }
        self.fit_inner(table)
    }

    /// Internal fit that handles only Numeric and Categorical specs.
    fn fit_inner(&mut self, table: &Table) -> Result<()> {
        if self.specs.is_empty() {
            return Err(DatarustError::InvalidInput("no column specs".into()));
        }
        let mut consumed_num_set = std::collections::HashSet::new();
        let mut consumed_cat_set = std::collections::HashSet::new();
        let mut consumed_num = Vec::new();
        let mut consumed_cat = Vec::new();
        for spec in self.specs.iter_mut() {
            match spec {
                ColumnSpec::Numeric {
                    name,
                    columns,
                    transformer,
                } => {
                    // Check for duplicate column indices within this spec
                    let mut seen = std::collections::HashSet::new();
                    for &c in columns.iter() {
                        if !seen.insert(c) {
                            return Err(DatarustError::InvalidInput(format!(
                                "duplicate column index {} in numeric spec '{}'",
                                c, name
                            )));
                        }
                    }
                    // Check for overlap with previously consumed numeric columns
                    for &c in columns.iter() {
                        if consumed_num_set.contains(&c) {
                            return Err(DatarustError::InvalidInput(format!(
                                "column index {} is already consumed by another numeric spec",
                                c
                            )));
                        }
                    }
                    let sub = Self::extract_numeric_cols(table, columns)?;
                    transformer.fit(&sub)?;
                    for &c in columns.iter() {
                        consumed_num_set.insert(c);
                    }
                    consumed_num.extend_from_slice(columns);
                }
                ColumnSpec::Categorical {
                    name,
                    columns,
                    encoder,
                } => {
                    // Check for duplicate column indices within this spec
                    let mut seen = std::collections::HashSet::new();
                    for &c in columns.iter() {
                        if !seen.insert(c) {
                            return Err(DatarustError::InvalidInput(format!(
                                "duplicate column index {} in categorical spec '{}'",
                                c, name
                            )));
                        }
                    }
                    // Check for overlap with previously consumed categorical columns
                    for &c in columns.iter() {
                        if consumed_cat_set.contains(&c) {
                            return Err(DatarustError::InvalidInput(format!(
                                "column index {} is already consumed by another categorical spec",
                                c
                            )));
                        }
                    }
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    encoder.fit(&sub)?;
                    for &c in columns.iter() {
                        consumed_cat_set.insert(c);
                    }
                    consumed_cat.extend_from_slice(columns);
                }
                ColumnSpec::Target { .. } => {}
            }
        }
        consumed_num.sort_unstable();
        consumed_num.dedup();
        consumed_cat.sort_unstable();
        consumed_cat.dedup();
        self.consumed_numeric = consumed_num;
        self.total_numeric_cols = table.numeric.ncols();
        self.consumed_categorical = consumed_cat;
        self.total_categorical_cols = table.categorical.ncols();
        self.max_col_index = self.total_numeric_cols.max(self.total_categorical_cols);

        // Fit categorical encoders for unused categorical columns (remainder passthrough).
        self.remainder_cat_encoders.clear();
        self.remainder_cat_cols.clear();
        if matches!(self.remainder, Remainder::Passthrough) {
            let unused_cat: Vec<usize> = (0..table.categorical.ncols())
                .filter(|c| self.consumed_categorical.binary_search(c).is_err())
                .collect();
            for &c in &unused_cat {
                let mut enc = OneHotEncoder::new();
                let sub = Self::extract_categorical_cols(table, &[c])?;
                enc.fit(&sub)?;
                self.remainder_cat_encoders
                    .push(CategoricalTransformerKind::OneHotEncoder(enc));
            }
            self.remainder_cat_cols = unused_cat;
        }

        self.fitted = true;
        Ok(())
    }

    /// Fits the column specs **and** `Target` specs using the provided target
    /// values, then returns the transformed result.
    ///
    /// Non-target specs are fitted via [`fit`](Self::fit) (or re-fitted if
    /// already fitted).  Target specs are fitted with `y`.
    pub fn fit_transform_with_target(&mut self, table: &Table, y: &[f64]) -> Result<Matrix> {
        self.fit_with_target(table, y)?;
        self.transform(table)
    }

    /// Fits the column specs **and** `Target` specs using the provided target
    /// values.
    ///
    /// Non-target specs are fitted via [`fit`](Self::fit).  Target specs are
    /// additionally fitted with the given target values.
    pub fn fit_with_target(&mut self, table: &Table, y: &[f64]) -> Result<()> {
        self.fit_inner(table)?;
        for spec in self.specs.iter_mut() {
            if let ColumnSpec::Target {
                columns, encoder, ..
            } = spec
            {
                let sub = Self::extract_categorical_cols(table, columns)?;
                encoder.fit(&sub, y)?;
            }
        }
        Ok(())
    }

    /// Transforms the table by applying each fitted spec and concatenating results.
    pub fn transform(&self, table: &Table) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("ColumnTransformer".into()));
        }
        let mut blocks: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut block_cols: Vec<usize> = Vec::new();
        for spec in &self.specs {
            match spec {
                ColumnSpec::Numeric {
                    columns,
                    transformer,
                    ..
                } => {
                    let sub = Self::extract_numeric_cols(table, columns)?;
                    let t = transformer.transform(&sub)?;
                    block_cols.push(t.ncols());
                    blocks.push(t.into_rows());
                }
                ColumnSpec::Categorical {
                    columns, encoder, ..
                } => {
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    let t = encoder.transform(&sub)?;
                    block_cols.push(t.ncols());
                    blocks.push(t.into_rows());
                }
                ColumnSpec::Target {
                    columns, encoder, ..
                } => {
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    let t = encoder.transform(&sub)?;
                    block_cols.push(t.ncols());
                    blocks.push(t.into_rows());
                }
            }
        }
        // Remainder: passthrough numeric columns not consumed.
        if matches!(self.remainder, Remainder::Passthrough) {
            let nrows = table.nrows();
            let remainder_cols: Vec<usize> = (0..table.numeric.ncols())
                .filter(|c| self.consumed_numeric.binary_search(c).is_err())
                .collect();
            if !remainder_cols.is_empty() {
                let mut rem = vec![vec![0.0; remainder_cols.len()]; nrows];
                for (i, rem_row) in rem.iter_mut().enumerate() {
                    let num_row = table.numeric.row(i);
                    for (k, &c) in remainder_cols.iter().enumerate() {
                        rem_row[k] = num_row[c];
                    }
                }
                block_cols.push(remainder_cols.len());
                blocks.push(rem);
            }
            // Passthrough unused categorical columns (one-hot encoded).
            for (enc, &c) in self
                .remainder_cat_encoders
                .iter()
                .zip(self.remainder_cat_cols.iter())
            {
                let sub = Self::extract_categorical_cols(table, &[c])?;
                let t = enc.transform(&sub)?;
                block_cols.push(t.ncols());
                blocks.push(t.into_rows());
            }
        }
        // Concatenate horizontally
        let nrows = table.nrows();
        let total_cols: usize = block_cols.iter().sum();
        if total_cols == 0 {
            return Err(DatarustError::InvalidInput(
                "transform produced zero columns".into(),
            ));
        }
        let mut out = vec![vec![0.0; total_cols]; nrows];
        for i in 0..nrows {
            let mut offset = 0;
            for (block, &cols) in blocks.iter().zip(block_cols.iter()) {
                for k in 0..cols {
                    out[i][offset + k] = block[i][k];
                }
                offset += cols;
            }
        }
        Matrix::new(out)
    }

    /// Fits the transformer to the table and returns the transformed result.
    pub fn fit_transform(&mut self, table: &Table) -> Result<Matrix> {
        self.fit(table)?;
        self.transform(table)
    }

    /// Transforms the table and returns an [`crate::Output`] that preserves the
    /// separation between numeric and categorical columns.
    ///
    /// Unlike [`transform`](Self::transform), remainder categorical columns are
    /// **passed through as strings** instead of being one-hot encoded.  This
    /// is useful when you want to chain further categorical processing.
    pub fn transform_to_table(&self, table: &Table) -> Result<crate::compose::Output> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("ColumnTransformer".into()));
        }
        let mut num_blocks: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut num_block_cols: Vec<usize> = Vec::new();
        let mut cat_blocks: Vec<Vec<Vec<String>>> = Vec::new();
        let mut cat_block_cols: Vec<usize> = Vec::new();
        let nrows = table.nrows();

        for spec in &self.specs {
            match spec {
                ColumnSpec::Numeric {
                    columns,
                    transformer,
                    ..
                } => {
                    let sub = Self::extract_numeric_cols(table, columns)?;
                    let t = transformer.transform(&sub)?;
                    num_block_cols.push(t.ncols());
                    num_blocks.push(t.into_rows());
                }
                ColumnSpec::Categorical {
                    columns, encoder, ..
                } => {
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    let t = encoder.transform(&sub)?;
                    num_block_cols.push(t.ncols());
                    num_blocks.push(t.into_rows());
                }
                ColumnSpec::Target {
                    columns, encoder, ..
                } => {
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    let t = encoder.transform(&sub)?;
                    num_block_cols.push(t.ncols());
                    num_blocks.push(t.into_rows());
                }
            }
        }

        // Remainder passthrough.
        if matches!(self.remainder, Remainder::Passthrough) {
            // Numeric remainder columns (passthrough to numeric).
            let remainder_cols: Vec<usize> = (0..table.numeric.ncols())
                .filter(|c| self.consumed_numeric.binary_search(c).is_err())
                .collect();
            if !remainder_cols.is_empty() {
                let mut rem = vec![vec![0.0; remainder_cols.len()]; nrows];
                for (i, rem_row) in rem.iter_mut().enumerate() {
                    let num_row = table.numeric.row(i);
                    for (k, &c) in remainder_cols.iter().enumerate() {
                        rem_row[k] = num_row[c];
                    }
                }
                num_block_cols.push(remainder_cols.len());
                num_blocks.push(rem);
            }

            // Categorical remainder columns: pass through as strings (not one-hot encoded).
            for &c in &self.remainder_cat_cols {
                let mut col_data: Vec<String> = Vec::with_capacity(nrows);
                for i in 0..nrows {
                    col_data.push(table.categorical.get(i, c).to_string());
                }
                cat_block_cols.push(1);
                cat_blocks.push(col_data.into_iter().map(|s| vec![s]).collect());
            }
        }

        // Compute output column counts and guard against zero total.
        let total_num_cols: usize = num_block_cols.iter().sum();
        let total_cat_cols: usize = cat_block_cols.iter().sum();

        if total_num_cols == 0 && total_cat_cols == 0 {
            return Err(DatarustError::InvalidInput(
                "transform_to_table produced zero columns".into(),
            ));
        }

        // Build the numeric matrix.
        let numeric = if total_num_cols > 0 {
            let mut out = vec![vec![0.0; total_num_cols]; nrows];
            for i in 0..nrows {
                let mut offset = 0;
                for (block, &cols) in num_blocks.iter().zip(num_block_cols.iter()) {
                    for k in 0..cols {
                        out[i][offset + k] = block[i][k];
                    }
                    offset += cols;
                }
            }
            Matrix::new(out)?
        } else {
            // No numeric columns: build a dummy nrows×1 matrix so the row-count
            // invariant of `Table` holds; its single (unused) column is ignored
            // by callers that consume only the categorical side.
            Matrix::zeros(nrows, 1)?
        };

        // Build the categorical (string) matrix.
        let categorical = if total_cat_cols > 0 {
            let mut out = vec![vec![String::new(); total_cat_cols]; nrows];
            for i in 0..nrows {
                let mut offset = 0;
                for (block, &cols) in cat_blocks.iter().zip(cat_block_cols.iter()) {
                    for k in 0..cols {
                        out[i][offset + k] = block[i][k].clone();
                    }
                    offset += cols;
                }
            }
            StrMatrix::new(out)?
        } else {
            StrMatrix {
                data: vec![vec![]; nrows],
            }
        };

        crate::compose::Output::new(numeric, categorical)
    }

    /// Fits then transforms the table, returning an [`crate::Output`] that preserves
    /// numeric / categorical column separation.
    pub fn fit_transform_to_table(&mut self, table: &Table) -> Result<crate::compose::Output> {
        self.fit(table)?;
        self.transform_to_table(table)
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureNames for ColumnTransformer {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.max_col_index),
        };
        let mut out: Vec<String> = Vec::new();
        for spec in &self.specs {
            match spec {
                ColumnSpec::Numeric { columns, .. } => {
                    for &c in columns {
                        out.push(names.get(c).cloned().unwrap_or_else(|| format!("x{}", c)));
                    }
                }
                ColumnSpec::Categorical {
                    columns, encoder, ..
                } => {
                    let sub: Vec<String> = columns
                        .iter()
                        .map(|&c| names.get(c).cloned().unwrap_or_else(|| format!("cat{}", c)))
                        .collect();
                    out.extend(encoder.feature_names_out(Some(&sub)));
                }
                ColumnSpec::Target {
                    columns, encoder, ..
                } => {
                    let sub: Vec<String> = columns
                        .iter()
                        .map(|&c| names.get(c).cloned().unwrap_or_else(|| format!("cat{}", c)))
                        .collect();
                    out.extend(encoder.feature_names_out(Some(&sub)));
                }
            }
        }
        if matches!(self.remainder, Remainder::Passthrough) {
            // Guard against the caller supplying fewer names than numeric
            // columns: never index past the end of `names`.
            let limit = names.len().min(self.total_numeric_cols);
            for (c, name) in names[..limit].iter().enumerate() {
                if self.consumed_numeric.binary_search(&c).is_err() {
                    out.push(name.clone());
                }
            }
            for (enc, &c) in self
                .remainder_cat_encoders
                .iter()
                .zip(self.remainder_cat_cols.iter())
            {
                let col_name = names.get(c).cloned().unwrap_or_else(|| format!("cat{}", c));
                let sub = vec![col_name];
                out.extend(enc.feature_names_out(Some(&sub)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::TargetEncoder;
    use crate::scaler::{MinMaxScaler, StandardScaler};

    fn sample_table() -> Table {
        // numeric: age, salary ; categorical: city
        let numeric = Matrix::new(vec![
            vec![10.0, 1000.0],
            vec![20.0, 2000.0],
            vec![30.0, 3000.0],
            vec![40.0, 4000.0],
        ])
        .unwrap();
        let categorical = StrMatrix::from_strings(vec![
            vec!["Istanbul"],
            vec!["Ankara"],
            vec!["Izmir"],
            vec!["Istanbul"],
        ])
        .unwrap();
        Table::new(numeric, categorical).unwrap()
    }

    #[test]
    fn numeric_scale_plus_onehot() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0, 1],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        let out = ct.fit_transform(&table).unwrap();
        assert_eq!(out.ncols(), 5);
        assert_eq!(
            [out.get(0, 2), out.get(0, 3), out.get(0, 4)],
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            [out.get(1, 2), out.get(1, 3), out.get(1, 4)],
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            [out.get(2, 2), out.get(2, 3), out.get(2, 4)],
            [0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn remainder_drop_default() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new().add_numeric(
            "num0",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        let out = ct.fit_transform(&table).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn remainder_passthrough_appends_unused_numeric() {
        // Numeric-only table: passthrough only appends numeric.
        let table = Table::from_numeric(
            Matrix::new(vec![
                vec![10.0, 1000.0],
                vec![20.0, 2000.0],
                vec![30.0, 3000.0],
                vec![40.0, 4000.0],
            ])
            .unwrap(),
        );
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        let out = ct.fit_transform(&table).unwrap();
        assert_eq!(out.ncols(), 2);
        for i in 0..4 {
            assert!((out.get(i, 1) - (1000.0 * (i as f64 + 1.0))).abs() < 1e-9);
        }
    }

    #[test]
    fn remainder_passthrough_includes_categorical() {
        // Mixed table with unused categorical column gets one-hot encoded.
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        let out = ct.fit_transform(&table).unwrap();
        // 1 (scaled) + 1 (passthrough numeric col1) + 3 (one-hot city) = 5
        assert_eq!(out.ncols(), 5);
    }

    #[test]
    fn output_column_order_specs_then_remainder() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            )
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        let out = ct.fit_transform(&table).unwrap();
        assert_eq!(out.ncols(), 5);
    }

    #[test]
    fn transform_before_fit_errors() {
        let table = sample_table();
        let ct = ColumnTransformer::new().add_numeric(
            "n",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(matches!(
            ct.transform(&table),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn empty_specs_errors() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new();
        assert!(matches!(
            ct.fit(&table),
            Err(DatarustError::InvalidInput(_))
        ));
    }

    #[test]
    fn bad_column_index_errors() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new().add_numeric(
            "n",
            vec![5],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(ct.fit(&table).is_err());
    }

    #[test]
    fn categorical_on_numeric_only_table_errors() {
        let table = Table::from_numeric(Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap());
        let mut ct = ColumnTransformer::new().add_categorical(
            "c",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        );
        assert!(ct.fit(&table).is_err());
    }

    #[test]
    fn transform_new_data_consistent_shape() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0, 1],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        ct.fit(&table).unwrap();
        let new = sample_table();
        let out = ct.transform(&new).unwrap();
        assert_eq!(out.ncols(), 5);
        assert_eq!(out.nrows(), 4);
    }

    #[test]
    fn row_count_mismatch_in_table_rejected() {
        let numeric = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let categorical = StrMatrix::from_column(["a"]).unwrap();
        assert!(Table::new(numeric, categorical).is_err());
    }

    #[test]
    fn feature_names_default() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0, 1],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(None);
        assert_eq!(
            names,
            vec!["x0", "x1", "x0_Ankara", "x0_Istanbul", "x0_Izmir",]
        );
    }

    #[test]
    fn feature_names_with_input() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(Some(&["age".into(), "salary".into()]));
        assert_eq!(
            names,
            vec!["age", "age_Ankara", "age_Istanbul", "age_Izmir",]
        );
    }

    #[test]
    fn feature_names_remainder_passthrough() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(Some(&["age".into(), "salary".into()]));
        // age (scaled) + salary (passthrough) + age_Ankara, age_Istanbul, age_Izmir (cat passthrough)
        assert_eq!(
            names,
            vec!["age", "salary", "age_Ankara", "age_Istanbul", "age_Izmir",]
        );
    }

    #[test]
    fn feature_names_remainder_passthrough_numeric_only() {
        let table = Table::from_numeric(
            Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap(),
        );
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(Some(&["a".into(), "b".into(), "c".into()]));
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn feature_names_default_categorical_outnumbers_numeric() {
        // 1 numeric column, 3 categorical columns
        let numeric = Matrix::new(vec![vec![10.0], vec![20.0]]).unwrap();
        let categorical =
            StrMatrix::from_strings(vec![vec!["a", "x", "low"], vec!["b", "y", "high"]]).unwrap();
        let table = Table::new(numeric, categorical).unwrap();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "cat",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(None);
        // scaled x0 + one-hot x0_a, x0_b
        assert_eq!(names, vec!["x0", "x0_a", "x0_b"]);
    }

    #[test]
    fn feature_names_default_numeric_outnumbers_categorical() {
        // 3 numeric columns, 1 categorical column
        let numeric = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let categorical = StrMatrix::from_strings(vec![vec!["red"], vec!["blue"]]).unwrap();
        let table = Table::new(numeric, categorical).unwrap();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0, 1],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_categorical(
                "cat",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            );
        ct.fit(&table).unwrap();
        let names = ct.feature_names_out(None);
        // scaled x0, x1 + one-hot x0_blue, x0_red
        assert_eq!(names, vec!["x0", "x1", "x0_blue", "x0_red"]);
    }

    #[test]
    fn target_encoder_with_target_values() {
        let numeric = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let categorical = StrMatrix::from_column(["Istanbul", "Ankara", "Istanbul"]).unwrap();
        let table = Table::new(numeric, categorical).unwrap();
        let y = vec![1.0, 0.0, 1.0];

        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_target(
                "city_target",
                vec![0],
                TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap()),
            );
        let out = ct.fit_transform_with_target(&table, &y).unwrap();
        // 1 scaled feature + 1 target-encoded column = 2
        assert_eq!(out.ncols(), 2);
        // Istanbul mean = (1+1)/2 = 1.0
        assert!((out.get(0, 1) - 1.0).abs() < 1e-9);
        assert!((out.get(1, 1) - 0.0).abs() < 1e-9);
        assert!((out.get(2, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn target_encoder_requires_fit_with_target() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new().add_target(
            "t",
            vec![0],
            TargetTransformerKind::TargetEncoder(TargetEncoder::new(0.0).unwrap()),
        );
        // fit() without y should error when Target specs are present
        assert!(matches!(
            ct.fit(&table),
            Err(DatarustError::InvalidInput(_))
        ));
        // fit_with_target() should succeed
        ct.fit_with_target(&table, &[1.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(ct.transform(&table).is_ok());
    }

    #[test]
    fn transform_to_table_preserves_cat_remainder_as_strings() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "num0",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        let out = ct.fit_transform_to_table(&table).unwrap();
        // 1 scaled + 1 passthrough numeric (col1) = 2 numeric cols
        assert_eq!(out.numeric.ncols(), 2);
        // 1 categorical passthrough (city) as string
        assert_eq!(out.categorical.ncols(), 1);
        // City strings should match original
        for i in 0..4 {
            assert_eq!(out.categorical.get(i, 0), table.categorical.get(i, 0));
        }
    }

    #[test]
    fn transform_to_table_onehot_numeric_block() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_categorical(
                "city",
                vec![0],
                CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
            )
            .add_numeric(
                "num",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        let out = ct.fit_transform_to_table(&table).unwrap();
        // 3 one-hot + 1 scaled = 4 numeric, 0 categorical
        assert_eq!(out.numeric.ncols(), 4);
        assert_eq!(out.categorical.ncols(), 0);
    }

    #[test]
    fn transform_to_table_requires_fitted() {
        let table = sample_table();
        let ct = ColumnTransformer::new().add_numeric(
            "n",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(matches!(
            ct.transform_to_table(&table),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn transform_to_table_remainder_only_categorical() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .remainder(Remainder::Passthrough)
            .add_numeric(
                "n",
                vec![0],
                TransformerKind::StandardScaler(StandardScaler::new()),
            );
        ct.fit(&table).unwrap();
        let out = ct.transform_to_table(&table).unwrap();
        // 1 scaled + 1 passthrough numeric col1 = 2 numeric
        assert_eq!(out.numeric.ncols(), 2);
        // 1 categorical passthrough (city) as string
        assert_eq!(out.categorical.ncols(), 1);
        for i in 0..4 {
            assert_eq!(out.categorical.get(i, 0), table.categorical.get(i, 0));
        }
    }

    #[test]
    fn duplicate_column_across_specs_errors() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new()
            .add_numeric(
                "num",
                vec![0, 1],
                TransformerKind::StandardScaler(StandardScaler::new()),
            )
            .add_numeric(
                "num2",
                vec![0], // duplicate!
                TransformerKind::MinMaxScaler(MinMaxScaler::new()),
            );
        assert!(matches!(
            ct.fit(&table),
            Err(DatarustError::InvalidInput(msg))
                if msg.contains("already consumed")
        ));
    }

    #[test]
    fn duplicate_column_within_spec_errors() {
        let table = sample_table();
        let mut ct = ColumnTransformer::new().add_numeric(
            "num",
            vec![0, 0], // duplicate within same spec
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
        assert!(matches!(
            ct.fit(&table),
            Err(DatarustError::InvalidInput(msg))
                if msg.contains("duplicate column")
        ));
    }
}
