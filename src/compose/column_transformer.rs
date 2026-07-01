use crate::encoder::OneHotEncoder;
use crate::error::{DatarustError, Result};
use crate::matrix::{Matrix, StrMatrix};
use crate::traits::{default_input_names, FeatureNames};
use crate::transformer_kind::TransformerKind;
use crate::Transformer;

/// What to do with columns not explicitly selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Remainder {
    /// Drop unselected columns (default, like sklearn `remainder='drop'`).
    #[default]
    Drop,
    /// Pass through numeric columns that were not selected, appended at the
    /// end of the output in original order. (Categorical passthrough is not
    /// supported in this iteration.)
    Passthrough,
}

/// A specification of one block within a [`ColumnTransformer`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColumnSpec {
    Numeric {
        name: String,
        columns: Vec<usize>,
        transformer: TransformerKind,
    },
    Categorical {
        name: String,
        columns: Vec<usize>,
        encoder: OneHotEncoder,
    },
}

/// Combined input carrying numeric and categorical columns of equal row count.
#[derive(Debug, Clone)]
pub struct Table {
    pub numeric: Matrix,
    pub categorical: StrMatrix,
}

impl Table {
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
    /// Fitted: one-hot encoders for unused categorical columns (remainder passthrough).
    #[cfg_attr(feature = "serde", serde(default))]
    remainder_cat_encoders: Vec<OneHotEncoder>,
    /// Fitted: ordered list of unused categorical column indices.
    #[cfg_attr(feature = "serde", serde(default))]
    remainder_cat_cols: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    fitted: bool,
}

impl ColumnTransformer {
    pub fn new() -> Self {
        Self {
            specs: vec![],
            remainder: Remainder::Drop,
            consumed_numeric: vec![],
            total_numeric_cols: 0,
            consumed_categorical: vec![],
            total_categorical_cols: 0,
            remainder_cat_encoders: vec![],
            remainder_cat_cols: vec![],
            fitted: false,
        }
    }

    pub fn remainder(mut self, r: Remainder) -> Self {
        self.remainder = r;
        self
    }

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

    pub fn add_categorical<S>(
        mut self,
        name: S,
        columns: Vec<usize>,
        encoder: OneHotEncoder,
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

    pub fn fit(&mut self, table: &Table) -> Result<()> {
        if self.specs.is_empty() {
            return Err(DatarustError::InvalidInput("no column specs".into()));
        }
        let mut consumed_num = Vec::new();
        let mut consumed_cat = Vec::new();
        for spec in self.specs.iter_mut() {
            match spec {
                ColumnSpec::Numeric {
                    columns,
                    transformer,
                    ..
                } => {
                    let sub = Self::extract_numeric_cols(table, columns)?;
                    transformer.fit(&sub)?;
                    consumed_num.extend_from_slice(columns);
                }
                ColumnSpec::Categorical {
                    columns, encoder, ..
                } => {
                    let sub = Self::extract_categorical_cols(table, columns)?;
                    encoder.fit(&sub)?;
                    consumed_cat.extend_from_slice(columns);
                }
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

        // Fit one-hot encoders for unused categorical columns (remainder passthrough).
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
                self.remainder_cat_encoders.push(enc);
            }
            self.remainder_cat_cols = unused_cat;
        }

        self.fitted = true;
        Ok(())
    }

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

    pub fn fit_transform(&mut self, table: &Table) -> Result<Matrix> {
        self.fit(table)?;
        self.transform(table)
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
            None => default_input_names(self.total_numeric_cols.max(self.total_categorical_cols)),
        };
        let mut out: Vec<String> = Vec::new();
        for spec in &self.specs {
            match spec {
                ColumnSpec::Numeric { columns, .. } => {
                    for &c in columns {
                        out.push(names[c].clone());
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
            }
        }
        if matches!(self.remainder, Remainder::Passthrough) {
            for (c, name) in names[..self.total_numeric_cols].iter().enumerate() {
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
    use crate::scaler::StandardScaler;

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
            .add_categorical("city", vec![0], OneHotEncoder::new());
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
            .add_categorical("city", vec![0], OneHotEncoder::new())
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
        let mut ct = ColumnTransformer::new().add_categorical("c", vec![0], OneHotEncoder::new());
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
            .add_categorical("city", vec![0], OneHotEncoder::new());
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
            .add_categorical("city", vec![0], OneHotEncoder::new());
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
            .add_categorical("city", vec![0], OneHotEncoder::new());
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
}
