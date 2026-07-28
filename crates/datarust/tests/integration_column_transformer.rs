//! Integration tests for `ColumnTransformer` and realistic mixed-type workflows.

use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::{DropStrategy, HandleUnknown, OneHotEncoder};
use datarust::imputer::{ImputeStrategy, SimpleImputer};
use datarust::scaler::{MinMaxScaler, StandardScaler};
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::{Matrix, StrMatrix};

#[test]
fn mixed_numeric_categorical_workflow() {
    let numeric = Matrix::new(vec![
        vec![25.0, 50000.0],
        vec![f64::NAN, 60000.0],
        vec![40.0, 80000.0],
        vec![35.0, 70000.0],
    ])
    .unwrap();
    let categorical = StrMatrix::from_strings(vec![
        vec!["Istanbul"],
        vec!["Ankara"],
        vec!["Izmir"],
        vec!["Istanbul"],
    ])
    .unwrap();
    let table = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .add_numeric(
            "num",
            vec![0, 1],
            TransformerKind::SimpleImputer(SimpleImputer::new(ImputeStrategy::Mean)),
        )
        .add_categorical(
            "city",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        );

    let out = ct.fit_transform(&table).unwrap();
    assert_eq!(out.ncols(), 5);
    assert_eq!(out.nrows(), 4);
    for i in 0..4 {
        for j in 0..5 {
            assert!(out.get(i, j).is_finite(), "NaN at {},{}", i, j);
        }
    }
}

#[test]
fn column_transformer_with_drop_first_and_passthrough() {
    let numeric = Matrix::new(vec![
        vec![1.0, 100.0, 7.0],
        vec![2.0, 200.0, 8.0],
        vec![3.0, 300.0, 9.0],
    ])
    .unwrap();
    let categorical = StrMatrix::from_column(["x", "y", "x"]).unwrap();
    let table = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Passthrough)
        .add_categorical(
            "cat",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(
                OneHotEncoder::new().drop(DropStrategy::First),
            ),
        )
        .add_numeric(
            "num",
            vec![0],
            TransformerKind::MinMaxScaler(MinMaxScaler::new()),
        );

    let out = ct.fit_transform(&table).unwrap();
    assert_eq!(out.ncols(), 4);
    assert_eq!(out.nrows(), 3);
}

#[test]
fn train_then_inference_with_new_categories() {
    let numeric = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let categorical = StrMatrix::from_column(["a", "b", "c"]).unwrap();
    let train = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .add_categorical(
            "cat",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(
                OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore),
            ),
        )
        .add_numeric(
            "num",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );

    ct.fit(&train).unwrap();

    let numeric2 = Matrix::new(vec![vec![5.0]]).unwrap();
    let categorical2 = StrMatrix::from_column(["z"]).unwrap();
    let test = Table::new(numeric2, categorical2).unwrap();

    let out = ct.transform(&test).unwrap();
    assert_eq!(out.ncols(), 4);
    assert_eq!(
        [out.get(0, 0), out.get(0, 1), out.get(0, 2)],
        [0.0, 0.0, 0.0]
    );
}

#[test]
fn column_transformer_unknown_category_errors_by_default() {
    let numeric = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let categorical = StrMatrix::from_column(["a", "b"]).unwrap();
    let train = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .add_categorical(
            "cat",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        )
        .add_numeric(
            "num",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
    ct.fit(&train).unwrap();

    let numeric2 = Matrix::new(vec![vec![5.0]]).unwrap();
    let categorical2 = StrMatrix::from_column(["z"]).unwrap();
    let test = Table::new(numeric2, categorical2).unwrap();
    assert!(ct.transform(&test).is_err());
}

#[test]
fn numeric_only_column_transformer() {
    let numeric = Matrix::new(vec![
        vec![1.0, 10.0, 100.0],
        vec![2.0, 20.0, 200.0],
        vec![3.0, 30.0, 300.0],
    ])
    .unwrap();
    let table = Table::from_numeric(numeric);

    let mut ct = ColumnTransformer::new()
        .add_numeric(
            "first",
            vec![0, 1],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_numeric(
            "second",
            vec![2],
            TransformerKind::MinMaxScaler(MinMaxScaler::new()),
        );
    let out = ct.fit_transform(&table).unwrap();
    assert_eq!(out.ncols(), 3);
    assert_eq!(out.nrows(), 3);
}

#[test]
fn passthrough_preserves_unscaled_values() {
    let numeric = Matrix::new(vec![vec![1.0, 99.0], vec![2.0, 88.0], vec![3.0, 77.0]]).unwrap();
    let categorical = StrMatrix::from_column(["a", "a", "a"]).unwrap();
    let table = Table::new(numeric, categorical).unwrap();

    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Passthrough)
        .add_numeric(
            "scaled",
            vec![0],
            TransformerKind::StandardScaler(StandardScaler::new()),
        );
    let out = ct.fit_transform(&table).unwrap();
    assert!((out.get(0, 1) - 99.0).abs() < 1e-12);
    assert!((out.get(1, 1) - 88.0).abs() < 1e-12);
    assert!((out.get(2, 1) - 77.0).abs() < 1e-12);
}
