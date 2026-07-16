//! Integration tests for `linear_model::Ridge` and `linear_model::Lasso`.

use datarust::linear_model::{Lasso, LinearRegression, Ridge, RidgeSolver};
use datarust::traits::Regressor;
use datarust::Matrix;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Non-collinear feature matrix with a known linear signal.
fn sample_xy() -> (Matrix, Vec<f64>) {
    let rows: Vec<Vec<f64>> = (0..50)
        .map(|i| {
            let i = i as f64;
            vec![i.sin(), (i + 7.0).ln(), (i * 0.3 + 1.0).exp() * 0.01]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| 2.0 * r[0] - 1.0 * r[1] + 0.5 * r[2] + 3.0)
        .collect();
    (Matrix::new(rows).unwrap(), y)
}

// ============================== Ridge =====================================

#[test]
fn ridge_alpha_zero_matches_linear_regression() {
    let (x, y) = sample_xy();
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).unwrap();
    let mut ridge = Ridge::new().with_alpha(0.0);
    ridge.fit(&x, &y).unwrap();
    for (r, l) in ridge.coef().iter().zip(lr.coef().iter()) {
        assert!(approx(*r, *l, 1e-6), "ridge={r} lr={l}");
    }
    assert!(approx(ridge.intercept(), lr.intercept(), 1e-6));
}

#[test]
fn ridge_shrinkage_monotonic() {
    let (x, y) = sample_xy();
    let mut norms = Vec::new();
    for &alpha in &[0.01, 1.0, 100.0, 10_000.0] {
        let mut m = Ridge::new().with_alpha(alpha);
        m.fit(&x, &y).unwrap();
        let norm: f64 = m.coef().iter().map(|v| v * v).sum::<f64>().sqrt();
        norms.push(norm);
    }
    // L2 norm should be non-increasing as alpha grows.
    for w in norms.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "norms not decreasing: {norms:?}");
    }
}

#[test]
fn ridge_handles_collinear_features() {
    // Duplicate column → XᵀX singular; LinearRegression (Cholesky) fails, Ridge works.
    let x = Matrix::new(vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
    ])
    .unwrap();
    let y = vec![2.0, 4.0, 6.0, 8.0];

    let mut lr = LinearRegression::new();
    assert!(lr.fit(&x, &y).is_err());

    let mut ridge = Ridge::new().with_alpha(1.0);
    ridge.fit(&x, &y).unwrap();
    let pred = ridge.predict(&x).unwrap();
    assert_eq!(pred.len(), 4);
}

#[test]
fn ridge_solvers_agree() {
    let (x, y) = sample_xy();
    let mut chol = Ridge::new().with_alpha(5.0);
    chol.fit(&x, &y).unwrap();
    let mut svd = Ridge::new().with_alpha(5.0).with_solver(RidgeSolver::Svd);
    svd.fit(&x, &y).unwrap();
    for (c, s) in chol.coef().iter().zip(svd.coef().iter()) {
        assert!(approx(*c, *s, 1e-6), "cholesky={c} svd={s}");
    }
    assert!(approx(chol.intercept(), svd.intercept(), 1e-6));
}

#[test]
fn ridge_score_high_for_clean_signal() {
    let (x, y) = sample_xy();
    let mut m = Ridge::new().with_alpha(0.01);
    m.fit(&x, &y).unwrap();
    let r2 = m.score(&x, &y).unwrap();
    assert!(r2 > 0.99, "r2={r2}");
}

#[test]
fn ridge_predict_new_data() {
    let rows: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let i = i as f64;
            vec![i]
        })
        .collect();
    let x = Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
    let mut m = Ridge::new().with_alpha(0.001);
    m.fit(&x, &y).unwrap();
    let new_x = Matrix::new(vec![vec![100.0]]).unwrap();
    let pred = m.predict(&new_x).unwrap();
    assert!(approx(pred[0], 201.0, 1e-1));
}

#[test]
fn ridge_is_fitted_flag() {
    let (x, y) = sample_xy();
    let mut m = Ridge::new();
    assert!(!m.is_fitted());
    m.fit(&x, &y).unwrap();
    assert!(m.is_fitted());
}

// ============================== Lasso =====================================

#[test]
fn lasso_drives_irrelevant_feature_to_zero() {
    // Feature 1 is pure noise (random-ish), feature 0 carries the signal.
    let rows: Vec<Vec<f64>> = (0..60)
        .map(|i| {
            let i = i as f64;
            // feature 0 = i, feature 1 = unrelated noise.
            vec![i, (i * 17.0).fract()]
        })
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| 3.0 * r[0]).collect();
    let x = Matrix::new(rows).unwrap();

    let mut model = Lasso::new().with_alpha(1.0).with_max_iter(2000);
    model.fit(&x, &y).unwrap();
    assert!(
        model.coef()[1].abs() < 1e-6,
        "irrelevant feature should be zeroed: {}",
        model.coef()[1]
    );
}

#[test]
fn lasso_zero_alpha_fits_signal() {
    let (x, y) = sample_xy();
    let mut m = Lasso::new()
        .with_alpha(0.0)
        .with_max_iter(5000)
        .with_tol(1e-10);
    m.fit(&x, &y).unwrap();
    let r2 = m.score(&x, &y).unwrap();
    assert!(r2 > 0.99, "r2={r2}");
}

#[test]
fn lasso_sparsity_increases_with_alpha() {
    let (x, y) = sample_xy();
    let mut zero_counts = Vec::new();
    for &alpha in &[0.01, 1.0, 10.0, 100.0] {
        let mut m = Lasso::new().with_alpha(alpha).with_max_iter(2000);
        m.fit(&x, &y).unwrap();
        let zeros = m.coef().iter().filter(|c| c.abs() < 1e-10).count();
        zero_counts.push(zeros);
    }
    // Sparsity (zero count) should be non-decreasing with alpha.
    for w in zero_counts.windows(2) {
        assert!(w[1] >= w[0], "sparsity not increasing: {zero_counts:?}");
    }
}

#[test]
fn lasso_n_iter_in_range() {
    let (x, y) = sample_xy();
    let mut m = Lasso::new().with_alpha(0.1).with_max_iter(500);
    m.fit(&x, &y).unwrap();
    assert!(m.n_iter() > 0 && m.n_iter() <= 500);
}

#[test]
fn lasso_predict_before_fit_errors() {
    let model = Lasso::new();
    let x = Matrix::new(vec![vec![1.0]]).unwrap();
    assert!(matches!(
        model.predict(&x).unwrap_err(),
        datarust::DatarustError::NotFitted(_)
    ));
}

#[test]
fn lasso_negative_alpha_rejected() {
    let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let mut m = Lasso::new().with_alpha(-0.5);
    assert!(matches!(
        m.fit(&x, &[1.0, 2.0]).unwrap_err(),
        datarust::DatarustError::InvalidConfig(_)
    ));
}

#[test]
fn lasso_intercept_recovered() {
    let rows: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
    let x = Matrix::new(rows).unwrap();
    let y: Vec<f64> = (0..30).map(|i| 2.0 * i as f64 + 5.0).collect();
    let mut m = Lasso::new()
        .with_alpha(0.001)
        .with_max_iter(5000)
        .with_tol(1e-10);
    m.fit(&x, &y).unwrap();
    assert!(approx(m.coef()[0], 2.0, 1e-2));
    assert!(approx(m.intercept(), 5.0, 1e-1));
}

#[test]
fn ridge_and_lasso_implement_regressor_trait() {
    // Compile-time check that both can be used through the trait.
    fn predict_via_trait<R: Regressor>(m: &R, x: &Matrix) -> usize {
        m.predict(x).map(|v| v.len()).unwrap_or(0)
    }
    let (x, y) = sample_xy();
    let mut ridge = Ridge::new().with_alpha(0.1);
    ridge.fit(&x, &y).unwrap();
    let mut lasso = Lasso::new().with_alpha(0.1);
    lasso.fit(&x, &y).unwrap();
    assert_eq!(predict_via_trait(&ridge, &x), x.nrows());
    assert_eq!(predict_via_trait(&lasso, &x), x.nrows());
}
