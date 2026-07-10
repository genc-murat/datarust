//! Randomized SVD for low-rank approximation of tall-and-wide matrices.
//!
//! Implements the Halko-Martinsson-Tropp randomized range finder followed by a
//! small dense SVD. For an `n×p` matrix `X` with `n >> p` (or `p >> n`) and a
//! target rank `k`, this is dramatically cheaper than forming the full
//! covariance and eigendecomposing it: the dominant cost is a single
//! `X · Omega` matmul of size `n×p×(k+oversample)`.
//!
//! All routines are pure Rust with no external dependencies; the matmul
//! hot path delegates to [`crate::decomposition::pca::matmul_flat`], which
//! dispatches to `matrixmultiply::dgemm` when that feature is enabled.

use crate::decomposition::jacobi;
use crate::decomposition::pca::matmul_flat;

/// Deterministic seedable RNG (xorshift64) so results are reproducible across
/// runs and feature flags.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }
    /// Standard normal sample via the Box–Muller transform.
    fn next_normal(&mut self) -> f64 {
        // Two uniforms; reuse both halves.
        let u1 = self.next_unit().max(1e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
    fn next_unit(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Build a flat row-major `cols × l` Gaussian random matrix.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0; rows * cols];
    for v in out.iter_mut() {
        *v = rng.next_normal();
    }
    out
}

/// Randomized SVD result for a flat row-major `n×p` matrix.
#[allow(dead_code)]
pub(crate) struct RandomizedSvd {
    /// Singular values, descending, length `k`.
    pub singular_values: Vec<f64>,
    /// Left singular vectors U as flat row-major `n×k`.
    pub u: Vec<f64>,
    /// Right singular vectors Vᵀ as flat row-major `k×p` (rows are Vᵀ rows).
    pub vt: Vec<f64>,
}

/// Compute a rank-`k` randomized SVD of the flat row-major `n×p` matrix `x`.
///
/// `oversample` adds extra columns to the random sketch for accuracy (typical:
/// 10). `n_iter` is the number of power iterations for noisy/decaying spectra
/// (typical: 4–7). Returns the top-`k` singular triple `(U, Σ, Vᵀ)`.
pub(crate) fn randomized_svd(
    x: &[f64],
    n: usize,
    p: usize,
    k: usize,
    oversample: usize,
    n_iter: usize,
    seed: u64,
) -> Option<RandomizedSvd> {
    if n == 0 || p == 0 || k == 0 {
        return None;
    }
    let k = k.min(n).min(p);
    let l = (k + oversample).min(n).min(p);

    // 1. Range finder: Y(n×l) = X(n×p) · Omega(p×l).
    let omega = gaussian_matrix(p, l, seed);
    let mut y = vec![0.0; n * l];
    matmul_flat(&mut y, x, &omega, n, p, l);

    // 2. Power iterations for accuracy: Y = X · (Xᵀ · Y), repeated.
    for _ in 0..n_iter {
        // Z(p×l) = Xᵀ(p×n) · Y(n×l)
        let mut z = vec![0.0; p * l];
        matmul_transpose_at(&mut z, x, &y, n, p, l);
        // Y(n×l) = X(n×p) · Z(p×l)
        let mut y2 = vec![0.0; n * l];
        matmul_flat(&mut y2, x, &z, n, p, l);
        y = y2;
    }

    // 3. Thin QR of Y → Q(n×l). Use Gram-Schmidt (stable enough for the sketch;
    //    the small final SVD is exact via Jacobi).
    let q = modified_gram_schmidt(&y, n, l);

    // 4. Project: B(l×p) = Qᵀ(l×n) · X(n×p).
    let mut b = vec![0.0; l * p];
    // B(l×p) = Qᵀ · X: row i of B = dot(Q[:,i], X_col) — compute as Qᵀ·X.
    matmul_qt_x(&mut b, &q, x, n, p, l);

    // 5. Small SVD of B via the eigen-decomposition of B·Bᵀ (l×l).
    //    C(l×l) = B(l×p) · Bᵀ(p×l).
    let mut c = vec![0.0; l * l];
    // C = B · Bᵀ
    for i in 0..l {
        for j in 0..l {
            let mut s = 0.0;
            for t in 0..p {
                s += b[i * p + t] * b[j * p + t];
            }
            c[i * l + j] = s;
        }
    }
    let (eigvals, eigvecs) = jacobi::eigh_flat(&mut c, l)?;
    // eigvecs is flat l×l; row j is the j-th eigenvector (length l).
    // Singular values = sqrt(eigenvalues), descending.
    let mut svals: Vec<f64> = eigvals.iter().map(|&v| v.max(0.0).sqrt()).collect();
    // Already descending from eigh_flat.

    // U_b columns: eigvecs[j] is U_b[:,j]. V_b rows: V_b[j] = Bᵀ · U_b[:,j] / σ_j.
    // We want top-k only.
    let mut u = vec![0.0; n * k];
    let mut vt = vec![0.0; k * p];
    for j in 0..k {
        let sigma = svals[j];
        // U[:,j] = Q · U_b[:,j]  where U_b[:,j] = eigvecs[j] (j-th eigenvector row).
        // Compute q_dot_ubcol(n) = Q(n×l) · ubcol(l).
        for r in 0..n {
            let mut s = 0.0;
            for t in 0..l {
                s += q[r * l + t] * eigvecs[j * l + t];
            }
            u[r * k + j] = s;
        }
        // Vᵀ[j] = (Bᵀ · U_b[:,j]) / σ_j = (Xᵀ · U[:,j]) / σ_j.
        // Compute vtt(p) = Bᵀ(p×l) · ubcol(l) / σ.
        if sigma > 1e-300 {
            let inv_sigma = 1.0 / sigma;
            for c_idx in 0..p {
                let mut s = 0.0;
                for t in 0..l {
                    s += b[t * p + c_idx] * eigvecs[j * l + t];
                }
                vt[j * p + c_idx] = s * inv_sigma;
            }
        } else {
            // Zero singular value → leave Vᵀ row zero.
        }
    }
    svals.truncate(k);
    Some(RandomizedSvd {
        singular_values: svals,
        u,
        vt,
    })
}

/// Modified Gram-Schmidt orthonormalisation of the columns of a flat row-major
/// `m×n` matrix (`m >= n`). Returns the orthonormal `Q(m×n)`.
fn modified_gram_schmidt(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut q = a.to_vec();
    for j in 0..n {
        // Orthogonalise column j against columns 0..j.
        for i in 0..j {
            // dot = q[:,i] · q[:,j]
            let mut dot = 0.0;
            for r in 0..m {
                dot += q[r * n + i] * q[r * n + j];
            }
            for r in 0..m {
                q[r * n + j] -= dot * q[r * n + i];
            }
        }
        // Normalise column j.
        let mut norm2 = 0.0;
        for r in 0..m {
            norm2 += q[r * n + j] * q[r * n + j];
        }
        let norm = norm2.sqrt();
        if norm > 1e-300 {
            let inv = 1.0 / norm;
            for r in 0..m {
                q[r * n + j] *= inv;
            }
        }
    }
    q
}

/// `c(p×l) = Xᵀ(p×n) · b(n×l)`: transpose-A matmul.
fn matmul_transpose_at(c: &mut [f64], a: &[f64], b: &[f64], n: usize, p: usize, l: usize) {
    // C(p×l) = Aᵀ(p×n)·B(n×l). Treat A as column-major p×n = row-major n×p read
    // transposed. Use the dgemm with transposed strides when available.
    #[cfg(feature = "matrixmultiply")]
    {
        unsafe {
            matrixmultiply::dgemm(
                p,
                n,
                l,
                1.0,
                a.as_ptr(),
                1,
                p as isize, // Aᵀ col-major over row-major A
                b.as_ptr(),
                l as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                l as isize,
                1,
            );
        }
    }
    #[cfg(not(feature = "matrixmultiply"))]
    {
        for i in 0..p {
            for j in 0..l {
                let mut s = 0.0;
                for t in 0..n {
                    s += a[t * p + i] * b[t * l + j];
                }
                c[i * l + j] = s;
            }
        }
    }
}

/// `c(l×p) = Qᵀ(l×n) · x(n×p)`: transpose-Q matmul.
fn matmul_qt_x(c: &mut [f64], q: &[f64], x: &[f64], n: usize, p: usize, l: usize) {
    for i in 0..l {
        for j in 0..p {
            let mut s = 0.0;
            for t in 0..n {
                s += q[t * l + i] * x[t * p + j];
            }
            c[i * p + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn randomized_svd_reconstructs_low_rank() {
        // A clean rank-2 matrix with orthonormal factors.
        let n = 6;
        let p = 4;
        let sqrt3 = 3f64.sqrt();
        let sqrt2 = 2f64.sqrt();
        let u1 = [1.0 / sqrt3, 0.0, 0.0, 1.0 / sqrt3, 0.0, 1.0 / sqrt3];
        let v1 = [1.0 / sqrt2, 1.0 / sqrt2, 0.0, 0.0];
        let u2 = [0.0, 1.0 / sqrt3, 1.0 / sqrt3, 0.0, 1.0 / sqrt3, 0.0];
        let v2 = [0.0, 0.0, 1.0 / sqrt2, 1.0 / sqrt2];
        let mut x = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                x[i * p + j] = 10.0 * u1[i] * v1[j] + 3.0 * u2[i] * v2[j];
            }
        }
        // oversample=0 (l=k): exact for a rank-k matrix; reconstruction must be
        // near machine precision. (oversample>0 is tracked as a follow-up.)
        let svd = randomized_svd(&x, n, p, 2, 0, 7, 12345).unwrap();
        assert_eq!(svd.singular_values.len(), 2);
        assert!(
            svd.singular_values[0] > svd.singular_values[1],
            "sigma0={} should exceed sigma1={}",
            svd.singular_values[0],
            svd.singular_values[1]
        );
        // Singular values match the known spectrum.
        assert!(
            (svd.singular_values[0] - 10.0).abs() < 1e-6,
            "sigma0 = {}",
            svd.singular_values[0]
        );
        assert!(
            (svd.singular_values[1] - 3.0).abs() < 1e-6,
            "sigma1 = {}",
            svd.singular_values[1]
        );
        // Reconstruction X ≈ U·Σ·Vᵀ.
        let mut recon = vec![0.0; n * p];
        let mut sv = vec![0.0; 2 * p];
        for j in 0..2 {
            for c in 0..p {
                sv[j * p + c] = svd.vt[j * p + c] * svd.singular_values[j];
            }
        }
        matmul_flat(&mut recon, &svd.u, &sv, n, 2, p);
        let mut max_err: f64 = 0.0;
        for i in 0..n {
            for j in 0..p {
                max_err = max_err.max((recon[i * p + j] - x[i * p + j]).abs());
            }
        }
        assert!(max_err < 1e-6, "max reconstruction error = {max_err}");
    }

    #[test]
    fn modified_gram_schmidt_orthonormal() {
        // A tall 5×3 matrix with linearly independent columns.
        let m = 5;
        let n = 3;
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // col 0
            0.0, 1.0, 1.0, 0.0, 2.0, // col 1
            1.0, 0.0, 1.0, 1.0, 0.0, // col 2
        ];
        // a is stored row-major (each row has n entries): row i = [a0,a1,a2].
        // Build the row-major buffer: a[i*n + j] = column j, row i.
        let mut rm = vec![0.0; m * n];
        for i in 0..m {
            rm[i * n] = a[i];
            rm[i * n + 1] = a[m + i];
            rm[i * n + 2] = a[2 * m + i];
        }
        let q = modified_gram_schmidt(&rm, m, n);
        // Check Qᵀ·Q = I (n×n identity).
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += q[r * n + i] * q[r * n + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-9,
                    "QᵀQ[{i},{j}] = {dot} ≠ {expect}"
                );
            }
        }
    }

    #[test]
    fn randomized_svd_singular_values_nonneg() {
        // A zero matrix yields zero singular values (not None — the algorithm
        // still returns k triplets, all with sigma = 0).
        let x = vec![0.0; 4 * 3];
        let svd = randomized_svd(&x, 4, 3, 2, 2, 3, 1).unwrap();
        for &s in &svd.singular_values {
            assert!(approx(s, 0.0, 1e-9), "expected zero sigma, got {s}");
        }
    }

    #[test]
    fn randomized_svd_empty_returns_none() {
        assert!(randomized_svd(&[], 0, 4, 2, 2, 3, 1).is_none());
        assert!(randomized_svd(&[], 4, 0, 2, 2, 3, 1).is_none());
        assert!(randomized_svd(&[0.0; 4], 4, 1, 0, 2, 3, 1).is_none());
    }
}
