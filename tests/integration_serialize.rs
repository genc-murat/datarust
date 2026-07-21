#![cfg(feature = "serde")]
//! Round-trip serialization tests for fitted transformers (serde feature).

use datarust::decomposition::{PCAComponents, TruncatedSVD, PCA};
use datarust::encoder::{DropStrategy, HandleUnknown, OneHotEncoder};
use datarust::imputer::{ImputeStrategy, SimpleImputer};
use datarust::scaler::{
    BinStrategy, Binarizer, KBinsDiscretizer, KBinsEncode, MinMaxScaler, Norm, Normalizer,
    OutputDistribution, PowerMethod, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler,
};
use datarust::serialize::{from_json, load_json, save_json, to_json};
use datarust::CategoricalTransformerKind;
use datarust::Transformer;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn tmp_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "datarust_serde_{}_{}.json",
        name,
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}

#[test]
fn standard_scaler_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
    ])
    .unwrap();
    let mut scaler = StandardScaler::new();
    let original = scaler.fit_transform(&x).unwrap();

    let json = to_json(&scaler).unwrap();
    let restored: StandardScaler = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn minmax_scaler_file_round_trip() {
    let x = datarust::Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
    let mut scaler = MinMaxScaler::new().feature_range(-1.0, 1.0);
    let original = scaler.fit_transform(&x).unwrap();

    let path = tmp_path("minmax");
    save_json(&scaler, &path).unwrap();
    let restored: MinMaxScaler = load_json(&path).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn robust_scaler_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![100.0],
    ])
    .unwrap();
    let mut scaler = RobustScaler::new();
    let original = scaler.fit_transform(&x).unwrap();
    let json = to_json(&scaler).unwrap();
    let restored: RobustScaler = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        assert!(approx(out.get(i, 0), original.get(i, 0), 1e-12));
    }
}

#[test]
fn normalizer_round_trip() {
    let x = datarust::Matrix::new(vec![vec![3.0, 4.0], vec![1.0, 2.0]]).unwrap();
    let mut n = Normalizer::new(Norm::L1);
    let original = n.fit_transform(&x).unwrap();
    let json = to_json(&n).unwrap();
    let restored: Normalizer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn simple_imputer_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![1.0, f64::NAN],
        vec![2.0, 5.0],
        vec![f64::NAN, 7.0],
    ])
    .unwrap();
    let mut imp = SimpleImputer::new(ImputeStrategy::Mean);
    let original = imp.fit_transform(&x).unwrap();
    let json = to_json(&imp).unwrap();
    let restored: SimpleImputer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn pca_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![2.5, 2.4],
        vec![0.5, 0.7],
        vec![2.2, 2.9],
        vec![1.9, 2.2],
        vec![3.1, 3.0],
    ])
    .unwrap();
    let mut pca = PCA::new(PCAComponents::Count(1));
    let original = pca.fit_transform(&x).unwrap();
    let json = to_json(&pca).unwrap();
    let restored: PCA = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        assert!(approx(out.get(i, 0), original.get(i, 0), 1e-9));
    }
}

#[test]
fn truncated_svd_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ])
    .unwrap();
    let mut svd = TruncatedSVD::new(2).unwrap();
    let original = svd.fit_transform(&x).unwrap();
    let json = to_json(&svd).unwrap();
    let restored: TruncatedSVD = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..2 {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-9));
        }
    }
}

#[test]
fn onehot_round_trip() {
    use datarust::StrMatrix;
    let s = StrMatrix::from_strings(vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]]).unwrap();
    let mut ohe = OneHotEncoder::new()
        .drop(DropStrategy::First)
        .handle_unknown(HandleUnknown::Ignore);
    let original = ohe.fit_transform(&s).unwrap();
    let json = to_json(&ohe).unwrap();
    let restored: OneHotEncoder = from_json(&json).unwrap();
    let out = restored.transform(&s).unwrap();
    for i in 0..s.nrows() {
        for j in 0..out.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn restore_without_refit_is_fitted() {
    let x = datarust::Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let mut scaler = StandardScaler::new();
    scaler.fit(&x).unwrap();
    let json = to_json(&scaler).unwrap();
    let restored: StandardScaler = from_json(&json).unwrap();
    assert!(restored.is_fitted());
    // transform should succeed without refit
    assert!(restored.transform(&x).is_ok());
}

#[test]
fn pipeline_round_trip() {
    use datarust::pipeline::Pipeline;
    use datarust::transformer_kind::TransformerKind;

    let x = datarust::Matrix::new(vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
    ])
    .unwrap();

    let mut pipe = Pipeline::new()
        .push(
            "std",
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .push("minmax", TransformerKind::MinMaxScaler(MinMaxScaler::new()));
    let original = pipe.fit_transform(&x).unwrap();

    let json = to_json(&pipe).unwrap();
    let restored: Pipeline = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
    assert!(restored.is_fitted());
}

#[test]
fn column_transformer_round_trip() {
    use datarust::compose::{ColumnTransformer, Remainder, Table};
    use datarust::transformer_kind::TransformerKind;

    let numeric = datarust::Matrix::new(vec![
        vec![10.0, 1000.0],
        vec![20.0, 2000.0],
        vec![30.0, 3000.0],
        vec![40.0, 4000.0],
    ])
    .unwrap();
    let categorical = datarust::StrMatrix::from_strings(vec![
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
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_categorical(
            "city",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(OneHotEncoder::new()),
        )
        .remainder(Remainder::Passthrough);
    let original = ct.fit_transform(&table).unwrap();

    let json = to_json(&ct).unwrap();
    let restored: ColumnTransformer = from_json(&json).unwrap();
    let out = restored.transform(&table).unwrap();

    assert_eq!(out.ncols(), original.ncols());
    for i in 0..out.nrows() {
        for j in 0..out.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn binarizer_round_trip() {
    let x = datarust::Matrix::new(vec![vec![-1.0, 0.5, 3.0], vec![0.0, 1.5, -2.0]]).unwrap();
    let mut b = Binarizer::new().threshold(0.5);
    let original = b.fit_transform(&x).unwrap();
    let json = to_json(&b).unwrap();
    let restored: Binarizer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn kbins_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![0.0, 10.0],
        vec![1.0, 20.0],
        vec![2.0, 30.0],
        vec![3.0, 40.0],
        vec![4.0, 50.0],
    ])
    .unwrap();
    let mut kb = KBinsDiscretizer::new(3)
        .unwrap()
        .strategy(BinStrategy::Uniform)
        .encode(KBinsEncode::OneHotDense);
    let original = kb.fit_transform(&x).unwrap();
    let json = to_json(&kb).unwrap();
    let restored: KBinsDiscretizer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..original.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-12));
        }
    }
}

#[test]
fn quantile_transformer_round_trip() {
    let x = datarust::Matrix::new(vec![
        vec![0.0, 100.0],
        vec![1.0, 200.0],
        vec![2.0, 300.0],
        vec![3.0, 400.0],
        vec![4.0, 500.0],
    ])
    .unwrap();
    let mut qt = QuantileTransformer::new(5)
        .unwrap()
        .output_distribution(OutputDistribution::Normal);
    let original = qt.fit_transform(&x).unwrap();
    let json = to_json(&qt).unwrap();
    let restored: QuantileTransformer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(approx(out.get(i, j), original.get(i, j), 1e-9));
        }
    }
}

#[test]
fn power_transformer_round_trip() {
    let x =
        datarust::Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]).unwrap();
    let mut pt = PowerTransformer::new().method(PowerMethod::BoxCox);
    let original = pt.fit_transform(&x).unwrap();
    let json = to_json(&pt).unwrap();
    let restored: PowerTransformer = from_json(&json).unwrap();
    let out = restored.transform(&x).unwrap();
    for i in 0..x.nrows() {
        assert!(approx(out.get(i, 0), original.get(i, 0), 1e-9));
    }
}

#[test]
fn linear_regression_round_trip() {
    use datarust::linear_model::LinearRegression;
    use datarust::traits::Predictor;

    // Non-collinear features so the Cholesky path is exercised.
    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 7.0).ln(), (i * 0.3).exp()]
        })
        .collect();
    let x = datarust::Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..20).map(|i| i as f64).collect();

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let json = to_json(&model).unwrap();
    let restored: LinearRegression = from_json(&json).unwrap();
    let out = restored.predict(&x).unwrap();
    for i in 0..y.len() {
        assert!(approx(out[i], original[i], 1e-12));
    }
    // Compare coefficients with tolerance: the GEMM-backed matmul (when the
    // `matrixmultiply` feature is on) may reorder floating-point additions,
    // producing tiny ulp-level differences that are not semantically meaningful.
    for (a, b) in restored.coef().iter().zip(model.coef().iter()) {
        assert!(approx(*a, *b, 1e-12));
    }
    assert!(approx(restored.intercept(), model.intercept(), 1e-12));
    assert_eq!(restored.n_features_in(), model.n_features_in());
}

#[test]
fn linear_regression_svd_round_trip() {
    use datarust::linear_model::{LinearRegression, LinearSolver};
    use datarust::traits::Predictor;

    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 7.0).ln(), (i * 0.3).exp()]
        })
        .collect();
    let x = datarust::Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5).collect();

    let mut model = LinearRegression::new().with_solver(LinearSolver::Svd);
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let path = tmp_path("linreg_svd");
    save_json(&model, &path).unwrap();
    let restored: LinearRegression = load_json(&path).unwrap();
    let out = restored.predict(&x).unwrap();
    for i in 0..y.len() {
        assert!(approx(out[i], original[i], 1e-12));
    }
}

#[test]
fn ridge_round_trip() {
    use datarust::linear_model::{Ridge, RidgeSolver};
    use datarust::traits::Predictor;

    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 7.0).ln(), (i * 0.3).exp()]
        })
        .collect();
    let x = datarust::Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..20).map(|i| i as f64).collect();

    let mut model = Ridge::new().with_alpha(2.5).with_solver(RidgeSolver::Svd);
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let json = to_json(&model).unwrap();
    let restored: Ridge = from_json(&json).unwrap();
    let out = restored.predict(&x).unwrap();
    for (o, orig) in out.iter().zip(original.iter()) {
        assert!(approx(*o, *orig, 1e-12));
    }
    for (a, b) in restored.coef().iter().zip(model.coef().iter()) {
        assert!(approx(*a, *b, 1e-12));
    }
    assert!(approx(restored.intercept(), model.intercept(), 1e-12));
    assert_eq!(restored.n_features_in(), model.n_features_in());
}

#[test]
fn lasso_round_trip() {
    use datarust::linear_model::Lasso;
    use datarust::traits::Predictor;

    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 7.0).ln(), (i * 0.3).exp()]
        })
        .collect();
    let x = datarust::Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..20).map(|i| i as f64).collect();

    let mut model = Lasso::new().with_alpha(0.5).with_max_iter(500);
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let path = tmp_path("lasso");
    save_json(&model, &path).unwrap();
    let restored: Lasso = load_json(&path).unwrap();
    let out = restored.predict(&x).unwrap();
    for (o, orig) in out.iter().zip(original.iter()) {
        assert!(approx(*o, *orig, 1e-12));
    }
    for (a, b) in restored.coef().iter().zip(model.coef().iter()) {
        assert!(approx(*a, *b, 1e-12));
    }
    assert!(approx(restored.intercept(), model.intercept(), 1e-12));
    assert_eq!(restored.n_iter(), model.n_iter());
}

#[test]
fn logistic_regression_round_trip() {
    use datarust::linear_model::LogisticRegression;
    use datarust::traits::Predictor;

    // Overlapping (non-separable) data so the MLE is finite.
    let rows: Vec<Vec<f64>> = vec![
        vec![-2.0, 0.5],
        vec![-1.0, -0.2],
        vec![-0.5, 0.1],
        vec![0.5, -0.1],
        vec![1.0, 0.2],
        vec![-0.8, 0.6],
        vec![0.8, -0.4],
        vec![1.2, 0.5],
    ];
    let x = datarust::Matrix::new(rows).unwrap();
    let y: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];

    let mut model = LogisticRegression::new().with_max_iter(100);
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let json = to_json(&model).unwrap();
    let restored: LogisticRegression = from_json(&json).unwrap();
    let out = restored.predict(&x).unwrap();
    for (o, orig) in out.iter().zip(original.iter()) {
        assert!(approx(*o, *orig, 1e-12));
    }
    for (a, b) in restored.coef().iter().zip(model.coef().iter()) {
        assert!(approx(*a, *b, 1e-12));
    }
    assert!(approx(restored.intercept(), model.intercept(), 1e-12));
    assert_eq!(restored.n_iter(), model.n_iter());
    assert_eq!(restored.n_features_in(), model.n_features_in());
}

#[test]
fn supervised_pipeline_round_trip() {
    use datarust::linear_model::{LogisticRegression, LogisticSolver};
    use datarust::pipeline::{Pipeline, SupervisedPipeline};
    use datarust::selection::{ScoreFunc, SelectKBest};
    use datarust::transformer_kind::TransformerKind;
    use datarust::Predictor;

    let x = datarust::Matrix::new(vec![
        vec![-3.0, 0.2],
        vec![-2.0, -0.3],
        vec![-1.0, 0.5],
        vec![1.0, -0.4],
        vec![2.0, 0.1],
        vec![3.0, 0.6],
    ])
    .unwrap();
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let selector = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
    let mut model = Pipeline::new()
        .push("select", TransformerKind::SelectKBest(selector))
        .with_estimator(LogisticRegression::new().with_solver(LogisticSolver::Svd));
    model.fit(&x, &y).unwrap();
    let original = model.predict(&x).unwrap();

    let json = to_json(&model).unwrap();
    let restored: SupervisedPipeline<LogisticRegression> = from_json(&json).unwrap();
    assert!(restored.is_fitted());
    assert_eq!(restored.predict(&x).unwrap(), original);
}
