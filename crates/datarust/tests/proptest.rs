use proptest::collection::vec as prop_vec;
use proptest::prelude::*;

fn arb_matrix() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (2..10usize, 2..5usize).prop_flat_map(|(rows, cols)| {
        prop_vec(
            prop_vec(
                prop::num::f64::POSITIVE | prop::num::f64::NEGATIVE | prop::num::f64::ZERO,
                cols,
            ),
            rows,
        )
    })
}

proptest! {
    #[test]
    fn standard_scaler_round_trip(data in arb_matrix()) {
        use datarust::scaler::StandardScaler;
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data).unwrap();
        let mut s = StandardScaler::new();
        let transformed = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                prop_assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn standard_scaler_properties(data in arb_matrix()) {
        use datarust::scaler::StandardScaler;
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data).unwrap();
        let mut s = StandardScaler::new();
        let transformed = s.fit_transform(&x).unwrap();
        for j in 0..transformed.ncols() {
            let col: Vec<f64> = (0..transformed.nrows()).map(|i| transformed.get(i, j)).collect();
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            prop_assert!(mean.abs() < 1e-9);
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
            // Constant columns: var is 0 after scaling (not 1). Skip assertion.
            if var.abs() > 1e-12 {
                prop_assert!((var - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn minmax_scaler_round_trip(data in arb_matrix()) {
        use datarust::scaler::MinMaxScaler;
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data).unwrap();
        let mut s = MinMaxScaler::new();
        let transformed = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                prop_assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn robust_scaler_round_trip(data in arb_matrix()) {
        use datarust::scaler::RobustScaler;
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data).unwrap();
        let mut s = RobustScaler::new();
        let transformed = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                prop_assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn maxabs_scaler_round_trip(data in arb_matrix()) {
        use datarust::scaler::MaxAbsScaler;
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data).unwrap();
        let mut s = MaxAbsScaler::new();
        let transformed = s.fit_transform(&x).unwrap();
        let recovered = s.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                prop_assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn linear_regression_recovers_linear_signal(data in arb_matrix()) {
        use datarust::linear_model::LinearRegression;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        // Build a deterministic y from the first column so the signal is truly
        // linear in at least one feature. We then fit and check that the
        // in-sample fit is good (low MSE relative to the target scale).
        let y: Vec<f64> = data.iter().map(|r| 3.0 * r[0] + 1.0).collect();
        let mut model = LinearRegression::new();
        // Some random inputs may be rank-deficient; skip those (SVD would
        // handle them, but Cholesky may return Singular). We only assert the
        // property when the fit succeeds.
        if model.fit(&x, &y).is_ok() {
            let pred = model.predict(&x).unwrap();
            let mse: f64 = pred.iter().zip(y.iter())
                .map(|(p, &t)| (p - t).powi(2))
                .sum::<f64>() / y.len() as f64;
            let y_scale: f64 = y.iter().map(|v| (v - y.iter().sum::<f64>() / y.len() as f64).powi(2)).sum::<f64>() / y.len() as f64;
            // Relative error should be tiny for a noise-free linear signal.
            if y_scale > 1e-12 {
                prop_assert!(mse / y_scale < 1e-12, "mse={} y_scale={}", mse, y_scale);
            }
        }
    }

    #[test]
    fn linear_regression_score_in_unit_interval_on_data_with_signal(data in arb_matrix()) {
        use datarust::linear_model::LinearRegression;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data
            .iter()
            .map(|r| 2.0 * r[0] - r.get(1).copied().unwrap_or(0.0))
            .collect();
        let mut model = LinearRegression::new();
        if model.fit(&x, &y).is_ok() {
            let r2 = model.score(&x, &y).unwrap();
            // For a noise-free linear signal, R² should be ~1.0 (allow tiny slack).
            prop_assert!(r2 > 1.0 - 1e-6, "r2 too low: {}", r2);
        }
    }

    #[test]
    fn matmul_matches_naive(data in arb_matrix()) {
        use datarust::Matrix;
        let a = Matrix::new(data.clone()).unwrap();
        let b = Matrix::new(data.clone()).unwrap();
        // A is rows×cols; transpose to make B cols×rows so inner dims match
        // (A·B only defined when A.ncols == B.nrows; here A==B so we compute A·Aᵀ).
        let bt = b.transpose();
        let c = a.matmul(&bt).unwrap();
        // Naive O(n³) reference.
        let n = a.nrows();
        let m = a.ncols();
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..m {
                    s += a.get(i, k) * bt.get(k, j);
                }
                prop_assert!((c.get(i, j) - s).abs() < 1e-6, "i={} j={} got={} want={}", i, j, c.get(i, j), s);
            }
        }
    }

    #[test]
    fn matrix_aat_is_symmetric(data in arb_matrix()) {
        use datarust::Matrix;
        let a = Matrix::new(data.clone()).unwrap();
        let at = a.transpose();
        let aat = a.matmul(&at).unwrap();
        for i in 0..aat.nrows() {
            for j in 0..aat.ncols() {
                prop_assert!((aat.get(i, j) - aat.get(j, i)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ridge_recovers_linear_signal(data in arb_matrix()) {
        use datarust::linear_model::Ridge;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data.iter().map(|r| 4.0 * r[0] - 2.0).collect();
        let mut model = Ridge::new().with_alpha(0.001);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let mse: f64 = pred.iter().zip(y.iter())
            .map(|(p, &t)| (p - t).powi(2))
            .sum::<f64>() / y.len() as f64;
        let y_scale: f64 = y.iter().map(|v| (v - y.iter().sum::<f64>() / y.len() as f64).powi(2)).sum::<f64>() / y.len() as f64;
        if y_scale > 1e-12 {
            prop_assert!(mse / y_scale < 1e-6, "mse={} y_scale={}", mse, y_scale);
        }
    }

    #[test]
    fn ridge_alpha_zero_matches_linear_regression(data in arb_matrix()) {
        use datarust::linear_model::{LinearRegression, Ridge};
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data.iter().map(|r| 3.0 * r[0] + 1.0).collect();
        let mut ols = LinearRegression::new();
        let mut ridge = Ridge::new().with_alpha(0.0);
        // Only assert when the OLS fit succeeds (non-degenerate inputs).
        if ols.fit(&x, &y).is_ok() && ridge.fit(&x, &y).is_ok() {
            for (a, b) in ols.coef().iter().zip(ridge.coef().iter()) {
                prop_assert!((a - b).abs() < 1e-6, "ols={} ridge={}", a, b);
            }
            prop_assert!((ols.intercept() - ridge.intercept()).abs() < 1e-6);
        }
    }

    #[test]
    fn lasso_zero_alpha_recovers_signal(data in arb_matrix()) {
        use datarust::linear_model::Lasso;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data.iter().map(|r| 2.5 * r[0] + 0.5).collect();
        let mut model = Lasso::new().with_alpha(1e-8).with_max_iter(5000);
        model.fit(&x, &y).unwrap();
        let pred = model.predict(&x).unwrap();
        let mse: f64 = pred.iter().zip(y.iter())
            .map(|(p, &t)| (p - t).powi(2))
            .sum::<f64>() / y.len() as f64;
        let y_scale: f64 = y.iter().map(|v| (v - y.iter().sum::<f64>() / y.len() as f64).powi(2)).sum::<f64>() / y.len() as f64;
        if y_scale > 1e-12 {
            prop_assert!(mse / y_scale < 1e-6, "mse={} y_scale={}", mse, y_scale);
        }
    }

    #[test]
    fn logistic_predict_proba_in_unit_interval(data in arb_matrix()) {
        use datarust::linear_model::LogisticRegression;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        // Binary labels derived from the sign of the first feature.
        let y: Vec<f64> = data.iter().map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 }).collect();
        let mut model = LogisticRegression::new();
        if model.fit(&x, &y).is_ok() {
            let proba = model.predict_proba(&x).unwrap();
            for &p in proba.as_slice() {
                prop_assert!((0.0..=1.0).contains(&p), "proba out of range: {}", p);
            }
        }
    }

    #[test]
    fn pca_reconstruction_all_components(data in arb_matrix()) {
        use datarust::decomposition::{PCA, PCAComponents};
        use datarust::traits::Transformer;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let mut pca = PCA::new(PCAComponents::All);
        let transformed = pca.fit_transform(&x).unwrap();
        let recovered = pca.inverse_transform(&transformed).unwrap();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                prop_assert!((recovered.get(i, j) - x.get(i, j)).abs() < 1e-6, "i={} j={}", i, j);
            }
        }
    }

    #[test]
    fn covariance_matrix_symmetric(data in arb_matrix()) {
        use datarust::stats::covariance_matrix;
        let cov = covariance_matrix(&data, 0);
        for (i, row) in cov.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                prop_assert!((v - cov[j][i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn correlation_diagonal_is_one(data in arb_matrix()) {
        use datarust::stats::correlation_matrix;
        // Skip constant columns: a constant column produces NaN correlation on
        // the diagonal (0/0). Detect and skip those columns.
        let cols = if data.is_empty() { 0 } else { data[0].len() };
        let non_constant: Vec<Vec<f64>> = (0..cols).filter_map(|j| {
            let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (hi - lo).abs() > 1e-12 { Some(col) } else { None }
        }).collect();
        if non_constant.is_empty() || non_constant[0].len() < 2 {
            return Ok(()); // Nothing to check.
        }
        let corr = correlation_matrix(&non_constant);
        for (i, row) in corr.iter().enumerate() {
            prop_assert!((row[i] - 1.0).abs() < 1e-9, "diag[{}] = {} expected 1.0", i, row[i]);
        }
    }

    #[test]
    fn train_test_split_preserves_all_samples(data in arb_matrix()) {
        use datarust::Matrix;
        use datarust::model_selection::TrainTestSplit;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = (0..x.nrows()).map(|i| i as f64).collect();
        let split = TrainTestSplit::new().with_shuffle(false).with_test_size(0.3);
        let (x_train, x_test, _, _) = split.split(&x, &y).unwrap();
        prop_assert_eq!(x_train.nrows() + x_test.nrows(), x.nrows());
    }

    #[test]
    fn kfold_each_sample_tested_exactly_once(n_samples in 5usize..30, n_splits in 2usize..6) {
        use datarust::model_selection::KFold;
        if n_splits > n_samples {
            return Ok(());
        }
        let kf = KFold::new().with_n_splits(n_splits).with_shuffle(false);
        let mut test_count = vec![0usize; n_samples];
        for (_, test_idx) in kf.split(n_samples).unwrap() {
            for &i in &test_idx {
                test_count[i] += 1;
            }
        }
        for (i, &c) in test_count.iter().enumerate() {
            prop_assert_eq!(c, 1, "sample {} tested {} times", i, c);
        }
    }

    #[test]
    fn tree_classifier_fits_separable_targets_exactly(data in arb_matrix()) {
        use datarust::traits::Predictor;
        use datarust::tree::DecisionTreeClassifier;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        // Sign of the first feature is always separable by an unrestricted
        // CART tree, so training predictions must reproduce the labels.
        let y: Vec<f64> = data.iter().map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 }).collect();
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let pred = clf.predict(&x).unwrap();
        for (i, (p, t)) in pred.iter().zip(y.iter()).enumerate() {
            prop_assert_eq!(p, t, "row {} predicted {} expected {}", i, p, t);
        }
    }

    #[test]
    fn tree_regressor_depth1_leaves_are_exact_group_means(data in arb_matrix(), y in prop_vec(-100.0..100.0, 4..10)) {
        use datarust::traits::Predictor;
        use datarust::tree::DecisionTreeRegressor;
        use datarust::Matrix;
        let n = y.len();
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| data[i % data.len()].clone())
            .collect();
        let x = Matrix::new(rows).unwrap();
        let mut reg = DecisionTreeRegressor::new().with_max_depth(Some(1));
        reg.fit(&x, &y).unwrap();
        let tree = reg.tree().unwrap();
        let pred = reg.predict(&x).unwrap();
        if tree.n_nodes() == 1 {
            // Degenerate input (constant column or pure y): the root is a
            // leaf and every prediction is the global mean.
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            for (i, p) in pred.iter().enumerate() {
                prop_assert!((p - mean).abs() < 1e-9,
                    "single-leaf tree: pred[{}] = {} but global mean is {}", i, p, mean);
            }
        } else {
            prop_assert_eq!(tree.get_depth(), 1);
            prop_assert_eq!(tree.n_leaves(), 2);
        }

        // Every prediction must equal the exact mean of the training targets
        // routed to the same leaf.
        let mut per_leaf: std::collections::HashMap<usize, (f64, usize)> =
            std::collections::HashMap::new();
        for (i, &yi) in y.iter().enumerate() {
            let e = per_leaf
                .entry(tree.leaf_for(x.row(i)).unwrap())
                .or_insert((0.0, 0));
            e.0 += yi;
            e.1 += 1;
        }
        for (i, p) in pred.iter().enumerate() {
            let (sum, cnt) = per_leaf[&tree.leaf_for(x.row(i)).unwrap()];
            let mean = sum / cnt as f64;
            prop_assert!((p - mean).abs() < 1e-9,
                "pred[{}] = {} but its leaf mean is {}", i, p, mean);
        }
    }

    #[test]
    fn tree_proba_rows_are_distributions(data in arb_matrix()) {
        use datarust::traits::{PredictProba, Predictor};
        use datarust::tree::DecisionTreeClassifier;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data.iter().map(|r| if r[0] > 0.0 { 1.0 } else { 0.0 }).collect();
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let proba = clf.predict_proba(&x).unwrap();
        let pred = clf.predict(&x).unwrap();
        for (i, p) in pred.iter().enumerate() {
            let mut sum = 0.0;
            for k in 0..proba.ncols() {
                let p = proba.get(i, k);
                prop_assert!((0.0..=1.0).contains(&p), "proba[{},{}] = {} out of range", i, k, p);
                sum += p;
            }
            prop_assert!((sum - 1.0).abs() < 1e-9, "row {} sums to {}", i, sum);
            // argmax class must agree with predict
            let argmax = (0..proba.ncols())
                .max_by(|a, b| proba.get(i, *a).total_cmp(&proba.get(i, *b)))
                .unwrap();
            prop_assert_eq!(argmax as f64, *p, "argmax disagrees with predict at row {}", i);
        }
    }

    #[test]
    fn tree_max_depth_is_respected(data in arb_matrix(), depth in 1usize..4) {
        use datarust::tree::DecisionTreeRegressor;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = (0..x.nrows()).map(|i| (i % 3) as f64 * 7.0).collect();
        let mut reg = DecisionTreeRegressor::new().with_max_depth(Some(depth));
        reg.fit(&x, &y).unwrap();
        prop_assert!(reg.tree().unwrap().get_depth() <= depth);
    }

    #[test]
    fn tree_is_proper_binary_tree(data in arb_matrix()) {
        use datarust::tree::DecisionTreeClassifier;
        use datarust::traits::Predictor;
        use datarust::Matrix;
        let x = Matrix::new(data.clone()).unwrap();
        let y: Vec<f64> = data.iter().map(|r| (r[0] * r[1] > 0.0) as i32 as f64).collect();
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let tree = clf.tree().unwrap();
        // A proper binary tree always has leaves = internal nodes + 1.
        prop_assert_eq!(tree.n_leaves() * 2 - 1, tree.n_nodes());
    }
}
