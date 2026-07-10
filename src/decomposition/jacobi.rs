//! Jacobi eigenvalue algorithm for real symmetric matrices.
//!
//! Computes all eigenvalues and eigenvectors of a symmetric matrix via
//! successive Givens rotations. Returns eigenvalues sorted in descending
//! order with eigenvectors reordered to match.
//!
//! Both entry points operate on **flat row-major** buffers internally for
//! cache locality and auto-vectorisation; the legacy `&[Vec<f64>]` API is
//! kept for backwards compatibility and flattens once before delegating.

const MAX_SWEEPS: usize = 100;
const TOL: f64 = 1e-12;

/// Symmetric eigen-decomposition (legacy nested-vector API).
///
/// Returns `Some((eigenvalues, eigenvectors))` where `eigenvalues[k]` is the
/// k-th largest eigenvalue and `eigenvectors[k]` is the corresponding eigenvector
/// of length `n` (a unit vector). Returns `None` if the input is empty or not
/// square. The caller is responsible for ensuring the matrix is symmetric; only
/// the lower triangle is read.
pub fn eigh(matrix: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return None;
    }
    // Flatten the symmetric matrix once; the hot loops run on contiguous memory.
    let mut flat = Vec::with_capacity(n * n);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    let (vals, vecs_flat) = eigh_flat(&mut flat, n)?;
    // Reshape eigenvectors from flat n×n (row-major) into Vec<Vec<f64>>, one row
    // per eigenvalue rank. vecs_flat[k*n + i] is the i-th component of the k-th
    // eigenvector (the k-th row of the returned buffer).
    let vecs: Vec<Vec<f64>> = (0..n)
        .map(|k| vecs_flat[k * n..(k + 1) * n].to_vec())
        .collect();
    Some((vals, vecs))
}

/// Flat-storage symmetric eigen-decomposition.
///
/// Takes a flat row-major symmetric matrix `a` of shape `n × n` (length `n*n`),
/// mutated in place during the sweep, and returns `(eigenvalues, eigenvectors)`
/// where `eigenvalues[k]` is the k-th largest eigenvalue and the eigenvectors are
/// flat row-major `n × n`: eigenvector `k` occupies indices `[k*n, (k+1)*n)`.
/// Returns `None` if `n == 0`.
pub(crate) fn eigh_flat(a: &mut [f64], n: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    if n == 0 || a.len() != n * n {
        return None;
    }
    if n == 1 {
        return Some((vec![a[0]], vec![1.0]));
    }

    // Eigenvector accumulator starts as the identity.
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..MAX_SWEEPS {
        let off = off_diagonal_norm_flat(a, n);
        if off < TOL {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                rotate_flat(a, &mut v, n, p, q, c, s);
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
            }
        }
    }

    let mut idx: Vec<usize> = (0..n).collect();
    // Sort indices by descending diagonal value (eigenvalue) of `a`.
    idx.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));

    // Reorder eigenvalues and eigenvector rows to match the sorted index.
    let vals: Vec<f64> = idx.iter().map(|&i| a[i * n + i]).collect();
    let mut vecs = vec![0.0; n * n];
    for (k, &i) in idx.iter().enumerate() {
        // Eigenvector for diagonal i is the i-th *column* of v. Copy it into the
        // k-th row of the output so that vecs[k*n + x] = v[x*n + i].
        for x in 0..n {
            vecs[k * n + x] = v[x * n + i];
        }
    }
    Some((vals, vecs))
}

/// Frobenius norm of the strict upper triangle of a flat symmetric matrix.
#[inline]
fn off_diagonal_norm_flat(a: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        let base = i * n;
        for j in (i + 1)..n {
            let v = a[base + j];
            sum += v * v;
        }
    }
    sum.sqrt()
}

/// Apply a Givens rotation on columns/rows `p` and `q` of a flat symmetric
/// matrix `a` (in place) and accumulate the rotation into `v`.
#[allow(clippy::needless_range_loop)]
fn rotate_flat(a: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    // Update columns p, q of a for all rows r != p, q.
    for r in 0..n {
        if r == p || r == q {
            continue;
        }
        let arp = a[r * n + p];
        let arq = a[r * n + q];
        let new_rp = c * arp - s * arq;
        a[r * n + p] = new_rp;
        a[p * n + r] = new_rp;
        let new_rq = s * arp + c * arq;
        a[r * n + q] = new_rq;
        a[q * n + r] = new_rq;
    }
    let app = a[p * n + p];
    let aqq = a[q * n + q];
    let apq = a[p * n + q];
    a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[p * n + q] = 0.0;
    a[q * n + p] = 0.0;
    // Accumulate the rotation into v: new columns = old columns rotated.
    for r in 0..n {
        let vrp = v[r * n + p];
        let vrq = v[r * n + q];
        v[r * n + p] = c * vrp - s * vrq;
        v[r * n + q] = s * vrp + c * vrq;
    }
}

/// Compute the covariance matrix of centered data: `(1/(n-ddof)) * Xcᵀ Xc`.
///
/// This is a thin wrapper over the canonical `covariance_centered`
/// implementation so that PCA, Truncated SVD and the public stats API all share
/// one tested routine. `x_centered` is row-major `n×p`.
pub fn covariance(x_centered: &[Vec<f64>], ddof: usize) -> Vec<Vec<f64>> {
    crate::stats::covariance_centered(x_centered, ddof)
}

/// Compute the top-`k` eigenpairs of a flat symmetric `n×n` matrix via power
/// iteration with deflation.
///
/// Faster than a full [`eigh_flat`] when `k` is small relative to `n`
/// (`O(k·n²·iters)` vs `O(n³·sweeps)`). Returns `(eigenvalues, eigenvectors)`
/// where `eigenvalues[k]` is the k-th largest (descending) and the eigenvectors
/// are flat row-major `k×n`: eigenvector `j` occupies indices `[j*n, (j+1)*n)`.
///
/// Falls back to [`eigh_flat`] (truncating to `k`) when `k >= n` or when the
/// caller requests it. `iters` controls the power-iteration refinement count
/// per eigenpair (a value around 100 is robust for well-separated spectra).
pub(crate) fn eigh_topk_flat(
    matrix: &[f64],
    n: usize,
    k: usize,
    iters: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if n == 0 || matrix.len() != n * n {
        return None;
    }
    let k = k.min(n);
    if k == 0 {
        return Some((vec![], vec![]));
    }
    // When nearly all eigenpairs are wanted, the full Jacobi sweep is simpler
    // and equally fast; defer to it and truncate.
    if k >= n.saturating_sub(1) || n <= 3 {
        let mut buf = matrix.to_vec();
        let (vals, vecs) = eigh_flat(&mut buf, n)?;
        return Some((
            vals.into_iter().take(k).collect(),
            vecs.into_iter().take(k * n).collect(),
        ));
    }

    // Power iteration + deflation on a mutable working copy.
    let mut a = matrix.to_vec();
    let mut out_vals = Vec::with_capacity(k);
    let mut out_vecs = vec![0.0; k * n];
    let mut v = vec![0.0; n];
    let mut w = vec![0.0; n];

    for j in 0..k {
        // Deterministic start vector (avoids a pathological orthogonal start).
        v.iter_mut().enumerate().for_each(|(i, x)| {
            *x = ((i as f64 * 0.5).sin() + 1.0) / n as f64;
        });

        let mut lambda = 0.0;
        for _ in 0..iters {
            // w = A · v
            sym_matvec(&a, n, &v, &mut w);
            // Rayleigh quotient
            lambda = (0..n).map(|i| w[i] * v[i]).sum();
            // Normalize w -> v
            let norm = (0..n).map(|i| w[i] * w[i]).sum::<f64>().sqrt();
            if norm < 1e-300 {
                break;
            }
            let inv = 1.0 / norm;
            v.iter_mut().zip(w.iter()).for_each(|(x, &y)| *x = y * inv);
        }

        out_vals.push(lambda);
        out_vecs[j * n..(j + 1) * n].copy_from_slice(&v);

        // Deflate: A = A - lambda · v · vᵀ (rank-1 update, symmetric).
        for r in 0..n {
            let vr = v[r];
            let base = r * n;
            for c in 0..n {
                a[base + c] -= lambda * vr * v[c];
            }
        }
    }

    // The deflation eigenvalues are already produced in descending order
    // (power iteration converges to the dominant remaining eigenpair), but
    // re-sort defensively in case of clustered spectra.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&i, &j| out_vals[j].total_cmp(&out_vals[i]));
    let mut sorted_vals = vec![0.0; k];
    let mut sorted_vecs = vec![0.0; k * n];
    for (new_k, &old) in idx.iter().enumerate() {
        sorted_vals[new_k] = out_vals[old];
        sorted_vecs[new_k * n..(new_k + 1) * n].copy_from_slice(&out_vecs[old * n..(old + 1) * n]);
    }
    Some((sorted_vals, sorted_vecs))
}

/// Symmetric matrix-vector product: `y = A · x` for a flat row-major symmetric
/// `n×n` matrix.
#[inline]
fn sym_matvec(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for (r, out) in y.iter_mut().enumerate().take(n) {
        let base = r * n;
        let mut s = 0.0;
        for c in 0..n {
            s += a[base + c] * x[c];
        }
        *out = s;
    }
}
