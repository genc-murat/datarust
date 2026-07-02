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
}
