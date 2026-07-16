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
        use datarust::traits::Regressor;
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
        use datarust::traits::Regressor;
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
}
