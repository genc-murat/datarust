//! Integration tests for `linear_model::LinearRegression` and `metrics`.

use datarust::linear_model::{LinearRegression, LinearSolver};
use datarust::metrics::regression::{
    explained_variance_score, max_error, mean_absolute_error, mean_squared_error, r2_score,
};
use datarust::traits::Predictor;
use datarust::Matrix;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Build a noise-free dataset y = Xβ + b for known β, b and recover them.
fn make_dataset(n: usize, beta: &[f64], intercept: f64) -> (Matrix, Vec<f64>) {
    // Deterministic, non-collinear features derived from the row index.
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let i = i as f64;
            // p distinct basis functions — guarantees full column rank.
            let mut row = Vec::with_capacity(beta.len());
            for (j, &bj) in beta.iter().enumerate() {
                let _ = bj;
                let v = match j {
                    0 => i.sin(),
                    1 => (i + 7.0).ln(),
                    2 => (i * 0.3 + 1.0).exp() * 0.01,
                    _ => (i * (j as f64)).cos(),
                };
                row.push(v);
            }
            row
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| {
            r.iter()
                .zip(beta.iter())
                .map(|(xj, &bj)| xj * bj)
                .sum::<f64>()
                + intercept
        })
        .collect();
    (Matrix::new(rows).unwrap(), y)
}

#[test]
fn recover_known_coefficients_cholesky() {
    let beta = vec![2.0, -3.5, 5.0];
    let (x, y) = make_dataset(40, &beta, 7.0);
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    for (got, expected) in model.coef().iter().zip(beta.iter()) {
        assert!(
            approx(*got, *expected, 1e-6),
            "coef: got {got} expected {expected}"
        );
    }
    assert!(approx(model.intercept(), 7.0, 1e-6));
}

#[test]
fn recover_known_coefficients_svd() {
    let beta = vec![2.0, -3.5, 5.0];
    let (x, y) = make_dataset(40, &beta, 7.0);
    let mut model = LinearRegression::new().with_solver(LinearSolver::Svd);
    model.fit(&x, &y).unwrap();
    for (got, expected) in model.coef().iter().zip(beta.iter()) {
        assert!(
            approx(*got, *expected, 1e-6),
            "coef: got {got} expected {expected}"
        );
    }
    assert!(approx(model.intercept(), 7.0, 1e-6));
}

#[test]
fn solvers_agree_on_full_rank_data() {
    let beta = vec![1.5, -2.0, 0.3];
    let (x, y) = make_dataset(30, &beta, 4.0);
    let mut chol = LinearRegression::new();
    chol.fit(&x, &y).unwrap();
    let mut svd = LinearRegression::new().with_solver(LinearSolver::Svd);
    svd.fit(&x, &y).unwrap();
    for (c, s) in chol.coef().iter().zip(svd.coef().iter()) {
        assert!(
            approx(*c, *s, 1e-6),
            "solver disagreement: chol={c} svd={s}"
        );
    }
    assert!(approx(chol.intercept(), svd.intercept(), 1e-6));
}

#[test]
fn perfect_fit_yields_zero_residual() {
    let beta = vec![1.0, 2.0, -1.0];
    let (x, y) = make_dataset(25, &beta, 3.0);
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    let mse = mean_squared_error(&y, &pred, true).unwrap();
    assert!(mse < 1e-18, "mse too large: {mse}");
    let r2 = model.score(&x, &y).unwrap();
    assert!(approx(r2, 1.0, 1e-9));
}

#[test]
fn predict_on_held_out_data() {
    // y = 2x, fit_intercept=false.
    let train = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
    let y_train = vec![2.0, 4.0, 6.0, 8.0];
    let mut model = LinearRegression::new().with_fit_intercept(false);
    model.fit(&train, &y_train).unwrap();
    let held_out = Matrix::new(vec![vec![100.0]]).unwrap();
    let pred = model.predict(&held_out).unwrap();
    assert!(approx(pred[0], 200.0, 1e-6));
}

#[test]
fn score_returns_r2() {
    let beta = vec![1.0, 0.5];
    let (x, y) = make_dataset(30, &beta, 2.0);
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let score = model.score(&x, &y).unwrap();
    let pred = model.predict(&x).unwrap();
    let direct = r2_score(&y, &pred).unwrap();
    assert!(approx(score, direct, 1e-12));
}

#[test]
fn metrics_sklearn_parity() {
    let y_true = vec![3.0, -0.5, 2.0, 7.0];
    let y_pred = vec![2.5, 0.0, 2.0, 8.0];
    assert!(approx(
        mean_squared_error(&y_true, &y_pred, true).unwrap(),
        0.375,
        1e-12
    ));
    assert!(approx(
        mean_squared_error(&y_true, &y_pred, false).unwrap(),
        0.375_f64.sqrt(),
        1e-12
    ));
    assert!(approx(
        mean_absolute_error(&y_true, &y_pred).unwrap(),
        0.5,
        1e-12
    ));
    assert!(approx(
        r2_score(&y_true, &y_pred).unwrap(),
        0.9486081370449679,
        1e-9
    ));
    assert!(approx(max_error(&y_true, &y_pred).unwrap(), 1.0, 1e-12));
}

#[test]
fn metrics_perfect_predictions() {
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(approx(r2_score(&y, &y).unwrap(), 1.0, 1e-12));
    assert!(approx(
        explained_variance_score(&y, &y).unwrap(),
        1.0,
        1e-12
    ));
    assert!(approx(
        mean_squared_error(&y, &y, true).unwrap(),
        0.0,
        1e-12
    ));
    assert!(approx(max_error(&y, &y).unwrap(), 0.0, 1e-12));
}

#[test]
fn constant_target_recovers_intercept() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![5.0], vec![9.0]]).unwrap();
    let y = vec![7.0, 7.0, 7.0, 7.0];
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    assert!(approx(model.intercept(), 7.0, 1e-9));
    assert!(model.coef()[0].abs() < 1e-9);
}

#[test]
fn predict_before_fit_errors() {
    let model = LinearRegression::new();
    let x = Matrix::new(vec![vec![1.0]]).unwrap();
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::NotFitted(_)));
}

#[test]
fn predict_shape_mismatch_errors() {
    // Use non-collinear features so fit succeeds, then predict with wrong width.
    let rows: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 1.0).ln()]
        })
        .collect();
    let x = Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let bad = Matrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
    let err = model.predict(&bad).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::ShapeMismatch { .. }));
}

#[test]
fn fit_y_length_mismatch_errors() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let mut model = LinearRegression::new();
    let err = model.fit(&x, &[1.0]).unwrap_err();
    assert!(matches!(err, datarust::DatarustError::ShapeMismatch { .. }));
}

#[test]
fn is_fitted_flag() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let mut model = LinearRegression::new();
    assert!(!model.is_fitted());
    model.fit(&x, &[2.0, 4.0]).unwrap();
    assert!(model.is_fitted());
}

#[test]
fn fit_predict_convenience() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let y = vec![2.0, 4.0, 6.0];
    let mut model = LinearRegression::new().with_fit_intercept(false);
    let pred = model.fit_predict(&x, &y).unwrap();
    for (p, &t) in pred.iter().zip(y.iter()) {
        assert!(approx(*p, t, 1e-9));
    }
}
