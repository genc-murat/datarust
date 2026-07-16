//! Cholesky decomposition for symmetric positive-definite (SPD) matrices.
//!
//! Used by linear-model solvers (and, in future, ridge / least-squares
//! estimators) to solve `A x = b` without forming `A⁻¹` explicitly. `A` must be
//! symmetric positive-definite; the decomposition produces a lower-triangular
//! `L` such that `A = L Lᵀ`, after which `A x = b` is solved by two triangular
//! substitutions: `L y = b` then `Lᵀ x = y`.
//!
//! Both entry points operate on **flat row-major** buffers for cache locality.

use crate::error::{DatarustError, Result};

/// Cholesky–Banachiewicz decomposition of a flat row-major SPD matrix `a`
/// (shape `n × n`, length `n*n`).
///
/// Returns the lower-triangular factor `L` (flat row-major `n × n`) such that
/// `A = L Lᵀ`. The strict upper triangle of the returned buffer is zero.
///
/// Returns [`DatarustError::Singular`] when the matrix is not positive-definite
/// (a non-positive pivot is encountered), which typically indicates rank-deficient
/// or collinear input.
pub(crate) fn cholesky_decompose(a: &[f64], n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(DatarustError::EmptyInput("cholesky of 0×0 matrix".into()));
    }
    if a.len() != n * n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} elements ({}×{})", n * n, n, n),
            actual: format!("{} elements", a.len()),
        });
    }
    let mut l = vec![0.0_f64; n * n];
    // Relative tolerance for detecting a numerically non-positive pivot. The
    // scale is set by the largest diagonal entry of A, mirroring LAPACK's
    // dpotrf which flags a non-PD matrix when a pivot is ≤ 0 (within rounding).
    let diag_max = (0..n)
        .map(|i| a[i * n + i])
        .fold(0.0_f64, f64::max)
        .max(0.0);
    let tol = (diag_max * 1e-14).max(f64::MIN_POSITIVE);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            // Subtract dot product of rows i and j over columns [0, j).
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                // Diagonal pivot. A non-positive (or vanishingly small) pivot
                // means A is not positive-definite — typically rank-deficient
                // or collinear input. Caller should retry with the SVD solver.
                if sum <= tol {
                    return Err(DatarustError::Singular(format!(
                        "matrix is not positive-definite (pivot {:.3e} <= tol {:.3e} at index {})",
                        sum, tol, i
                    )));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let diag = l[j * n + j];
                l[i * n + j] = sum / diag;
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` for an SPD matrix `A` given its Cholesky factor `L` from
/// [`cholesky_decompose`].
///
/// Performs two triangular substitutions: forward `L y = b`, then back `Lᵀ x = y`.
/// `l` is a flat row-major `n × n` lower-triangular buffer (only its lower
/// triangle is read). `b` has length `n`. Returns `x` of length `n`.
pub(crate) fn solve_spd(l: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>> {
    if l.len() != n * n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} elements ({}×{})", n * n, n, n),
            actual: format!("{} elements", l.len()),
        });
    }
    if b.len() != n {
        return Err(DatarustError::ShapeMismatch {
            expected: format!("{} elements", n),
            actual: format!("{} elements", b.len()),
        });
    }
    // Forward substitution: solve L y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        let diag = l[i * n + i];
        if diag == 0.0 {
            return Err(DatarustError::Singular(format!(
                "zero diagonal in Cholesky factor at index {i}"
            )));
        }
        y[i] = sum / diag;
    }
    // Back substitution: solve Lᵀ x = y.
    let mut x = vec![0.0_f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for k in (i + 1)..n {
            // Lᵀ[i,k] = L[k,i].
            sum -= l[k * n + i] * x[k];
        }
        let diag = l[i * n + i];
        x[i] = sum / diag;
    }
    Ok(x)
}

/// Convenience: decompose `A` and solve `A x = b` in one call.
///
/// Equivalent to [`solve_spd`](&[`cholesky_decompose`])`(&cholesky_decompose(a, n)?, n, b)?`.
pub(crate) fn solve_spd_system(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>> {
    let l = cholesky_decompose(a, n)?;
    solve_spd(&l, n, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: reconstruct A from its Cholesky factor L (A = L Lᵀ).
    fn reconstruct(l: &[f64], n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += l[i * n + k] * l[j * n + k];
                }
                a[i * n + j] = s;
            }
        }
        a
    }

    #[test]
    fn cholesky_identity() {
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = cholesky_decompose(&a, n).unwrap();
        let recon = reconstruct(&l, n);
        for x in 0..n * n {
            assert!((recon[x] - a[x]).abs() < 1e-12);
        }
        // L should be identity.
        assert!((l[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_known_pd() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] -> known PD
        let a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let n = 3;
        let l = cholesky_decompose(&a, n).unwrap();
        let recon = reconstruct(&l, n);
        for x in 0..n * n {
            assert!((recon[x] - a[x]).abs() < 1e-9, "mismatch at {x}");
        }
        // Strict upper triangle of L must be zero.
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(l[i * n + j].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_non_pd_returns_singular() {
        // Indefinite matrix: [[1, 2], [2, 1]] has eigenvalues 3, -1.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        let err = cholesky_decompose(&a, 2).unwrap_err();
        assert!(matches!(err, DatarustError::Singular(_)));
    }

    #[test]
    fn cholesky_zero_diagonal_singular() {
        // Zero diagonal entry → not positive-definite.
        let a = vec![0.0, 0.0, 0.0, 1.0];
        let err = cholesky_decompose(&a, 2).unwrap_err();
        assert!(matches!(err, DatarustError::Singular(_)));
    }

    #[test]
    fn solve_spd_basic() {
        // A = [[4, 2], [2, 3]], b = [10, 11].
        // 4x + 2y = 10, 2x + 3y = 11 -> x=1, y=3.
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![10.0, 11.0]; // expected x = [1, 3]
        let l = cholesky_decompose(&a, 2).unwrap();
        let x = solve_spd(&l, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn solve_spd_round_trip() {
        // Build A = MᵀM + nI to guarantee SPD, pick a random x, form b = A x,
        // solve, and check recovery.
        let n = 4;
        let m_flat = vec![
            1.0, 2.0, 0.0, 1.0, 3.0, 1.0, 4.0, 2.0, 0.0, 5.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0,
        ];
        // A = MᵀM.
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += m_flat[k * n + i] * m_flat[k * n + j];
                }
                a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        let true_x = [1.5, -2.0, 0.5, 3.0];
        let b: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * true_x[j]).sum())
            .collect();
        let x = solve_spd_system(&a, n, &b).unwrap();
        for i in 0..n {
            assert!((x[i] - true_x[i]).abs() < 1e-8, "mismatch at {i}");
        }
    }

    #[test]
    fn solve_spd_shape_mismatch() {
        let l = vec![1.0; 4]; // 2×2
        let b = vec![1.0, 2.0, 3.0]; // wrong length
        let err = solve_spd(&l, 2, &b).unwrap_err();
        assert!(matches!(err, DatarustError::ShapeMismatch { .. }));
    }

    #[test]
    fn solve_spd_zero_diagonal() {
        // L with a zero diagonal entry.
        let l = vec![0.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 2.0];
        let err = solve_spd(&l, 2, &b).unwrap_err();
        assert!(matches!(err, DatarustError::Singular(_)));
    }

    #[test]
    fn cholesky_empty_rejected() {
        let err = cholesky_decompose(&[], 0).unwrap_err();
        assert!(matches!(err, DatarustError::EmptyInput(_)));
    }

    #[test]
    fn solve_spd_identity_returns_b() {
        // A = I ⇒ x = b exactly.
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![7.0, -2.0, 5.0];
        let x = solve_spd_system(&a, n, &b).unwrap();
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-12, "i={i}");
        }
    }

    #[test]
    fn solve_spd_3x3_diagonal_system() {
        // Diagonally dominant SPD system with a known solution.
        // A = [[9, 0, 0], [0, 4, 0], [0, 0, 1]], b = [18, -8, 3].
        // Solution: x = [2, -2, 3].
        let a = vec![9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![18.0, -8.0, 3.0];
        let x = solve_spd_system(&a, 3, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - (-2.0)).abs() < 1e-9);
        assert!((x[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn solve_spd_residue_is_zero() {
        // For any SPD A and solution x of A x = b, the residual A x − b is ~0.
        let a = vec![6.0, 2.0, 2.0, 6.0]; // [[6,2],[2,6]] SPD
        let b = vec![10.0, 14.0];
        let x = solve_spd_system(&a, 2, &b).unwrap();
        let r0 = a[0] * x[0] + a[1] * x[1] - b[0];
        let r1 = a[2] * x[0] + a[3] * x[1] - b[1];
        assert!(r0.abs() < 1e-9, "residual row 0 = {r0}");
        assert!(r1.abs() < 1e-9, "residual row 1 = {r1}");
    }

    #[test]
    fn cholesky_well_conditioned_4x4() {
        // A = I + small perturbation stays SPD; check L is lower-triangular
        // with positive diagonal and reconstructs A.
        let n = 4;
        // Hilbert-like matrix scaled up to be well-conditioned and SPD.
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / ((i + j + 1) as f64);
            }
        }
        let l = cholesky_decompose(&a, n).unwrap();
        // Positive diagonal.
        for i in 0..n {
            assert!(l[i * n + i] > 0.0, "non-positive diagonal at {i}");
        }
        // Strict upper triangle is zero.
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(l[i * n + j].abs() < 1e-12);
            }
        }
        // Reconstructs A.
        let recon = reconstruct(&l, n);
        for x in 0..n * n {
            assert!((recon[x] - a[x]).abs() < 1e-9, "mismatch at {x}");
        }
    }
}
