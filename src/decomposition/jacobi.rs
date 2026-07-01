//! Jacobi eigenvalue algorithm for real symmetric matrices.
//!
//! Computes all eigenvalues and eigenvectors of a symmetric matrix via
//! successive Givens rotations. Returns eigenvalues sorted in descending
//! order with eigenvectors reordered to match.

const MAX_SWEEPS: usize = 100;
const TOL: f64 = 1e-12;

/// Symmetric eigen-decomposition.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvalues[k]` is the k-th
/// largest eigenvalue and `eigenvectors[k]` is the corresponding eigenvector
/// of length `n` (a unit vector). The input must be symmetric; only the
/// lower triangle is read but the full matrix is used to verify symmetry
/// is not required (the caller is responsible).
pub fn eigh(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    assert!(n > 0, "eigh: empty matrix");
    assert_eq!(
        matrix[0].len(),
        n,
        "eigh: matrix must be square {}x{}",
        n,
        matrix[0].len()
    );
    if n == 1 {
        return (vec![matrix[0][0]], vec![vec![1.0]]);
    }

    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _ in 0..MAX_SWEEPS {
        let off = off_diagonal_norm(&a);
        if off < TOL {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                rotate(&mut a, &mut v, p, q, c, s);
                a[p][q] = 0.0;
                a[q][p] = 0.0;
            }
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Eigenvectors are the columns of v: column k -> eigenvector for a[k][k].
    // Convert to rows indexed by eigenvalue rank.
    let eigvecs: Vec<Vec<f64>> = (0..n).map(|k| (0..n).map(|i| v[i][k]).collect()).collect();

    // Sort by eigenvalue descending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        eigvals[j]
            .partial_cmp(&eigvals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| eigvals[i]).collect();
    let sorted_vecs: Vec<Vec<f64>> = idx.iter().map(|&i| eigvecs[i].clone()).collect();
    (sorted_vals, sorted_vecs)
}

#[allow(clippy::needless_range_loop)]
fn off_diagonal_norm(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += a[i][j] * a[i][j];
        }
    }
    sum.sqrt()
}

/// Apply a Givens rotation on columns/rows p and q to symmetric matrix `a`
/// (in place) and accumulate the rotation into `v`.
#[allow(clippy::ptr_arg, clippy::needless_range_loop)]
fn rotate(a: &mut Vec<Vec<f64>>, v: &mut Vec<Vec<f64>>, p: usize, q: usize, c: f64, s: f64) {
    let n = a.len();
    // Update columns p, q of a for all rows r != p, q
    for r in 0..n {
        if r == p || r == q {
            continue;
        }
        let arp = a[r][p];
        let arq = a[r][q];
        a[r][p] = c * arp - s * arq;
        a[p][r] = a[r][p];
        a[r][q] = s * arp + c * arq;
        a[q][r] = a[r][q];
    }
    let app = a[p][p];
    let aqq = a[q][q];
    let apq = a[p][q];
    a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;
    // Accumulate rotation into v: new columns = old columns rotated
    for r in 0..n {
        let vrp = v[r][p];
        let vrq = v[r][q];
        v[r][p] = c * vrp - s * vrq;
        v[r][q] = s * vrp + c * vrq;
    }
}

/// Compute the covariance matrix of centered data: (1/(n-1)) * Xc^T Xc.
/// `x_centered` is row-major n×p.
#[allow(clippy::needless_range_loop)]
pub fn covariance(x_centered: &[Vec<f64>], ddof: usize) -> Vec<Vec<f64>> {
    let n = x_centered.len();
    let p = if n > 0 { x_centered[0].len() } else { 0 };
    let mut cov = vec![vec![0.0; p]; p];
    for row in x_centered {
        for i in 0..p {
            let xi = row[i];
            if xi == 0.0 {
                continue;
            }
            for j in 0..p {
                cov[i][j] += xi * row[j];
            }
        }
    }
    let denom = (n - ddof) as f64;
    if denom > 0.0 {
        let inv = 1.0 / denom;
        for i in 0..p {
            for j in 0..p {
                cov[i][j] *= inv;
            }
        }
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn diagonal_matrix() {
        let m = vec![vec![3.0, 0.0], vec![0.0, 7.0]];
        let (vals, vecs) = eigh(&m);
        assert!(approx(vals[0], 7.0, 1e-9));
        assert!(approx(vals[1], 3.0, 1e-9));
        // eigenvector for 7 should be [0,1], for 3 should be [1,0]
        assert!(approx(vecs[0][0], 0.0, 1e-9));
        assert!(approx(vecs[0][1].abs(), 1.0, 1e-9));
        assert!(approx(vecs[1][0].abs(), 1.0, 1e-9));
        assert!(approx(vecs[1][1], 0.0, 1e-9));
    }

    #[test]
    fn known_2x2() {
        // [[2,1],[1,2]] eigenvalues 3 and 1, eigenvectors [1,1]/sqrt2 and [1,-1]/sqrt2
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = eigh(&m);
        assert!(approx(vals[0], 3.0, 1e-9));
        assert!(approx(vals[1], 1.0, 1e-9));
        // eigenvector magnitudes are unit
        for v in &vecs {
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(approx(nrm, 1.0, 1e-9));
        }
        // verify A v = lambda v
        for (k, &lambda) in vals.iter().enumerate() {
            let v = &vecs[k];
            let av0 = m[0][0] * v[0] + m[0][1] * v[1];
            let av1 = m[1][0] * v[0] + m[1][1] * v[1];
            assert!(approx(av0, lambda * v[0], 1e-8));
            assert!(approx(av1, lambda * v[1], 1e-8));
        }
    }

    #[test]
    fn known_3x3() {
        // Symmetric 3x3 with known eigenvalues 3, 6, 9 via a constructed example
        // A = Q D Q^T where Q is a rotation. Use a simple symmetric matrix.
        let m = vec![
            vec![4.0, 1.0, 2.0],
            vec![1.0, 3.0, 0.0],
            vec![2.0, 0.0, 5.0],
        ];
        let (vals, vecs) = eigh(&m);
        // descending order
        assert!(vals[0] >= vals[1]);
        assert!(vals[1] >= vals[2]);
        // check A v = lambda v for each
        for (k, &lambda) in vals.iter().enumerate() {
            let v = &vecs[k];
            for i in 0..3 {
                let avi: f64 = (0..3).map(|j| m[i][j] * v[j]).sum();
                assert!(approx(avi, lambda * v[i], 1e-7));
            }
            // unit norm
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(approx(nrm, 1.0, 1e-9));
        }
        // orthogonality: vecs[0] . vecs[1] ~ 0
        let dot: f64 = vecs[0].iter().zip(vecs[1].iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-8);
    }

    #[test]
    fn covariance_identity() {
        // Centered data with unit variance per column, uncorrelated -> identity
        let xc = vec![vec![-1.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]];
        let cov = covariance(&xc, 1);
        // col0 var = ((-1)^2 + 0 + 1^2)/(3-1) = 1 ; col1 var = 0
        assert!(approx(cov[0][0], 1.0, 1e-12));
        assert!(approx(cov[1][1], 0.0, 1e-12));
        assert!(approx(cov[0][1], 0.0, 1e-12));
    }

    #[test]
    fn covariance_correlated() {
        // x = [1,2,3], y = [2,4,6] (= 2x) centered: [-1,0,1],[-2,0,2]
        // cov = [[1, 2],[2, 4]] with ddof=1
        let xc = vec![vec![-1.0, -2.0], vec![0.0, 0.0], vec![1.0, 2.0]];
        let cov = covariance(&xc, 1);
        assert!(approx(cov[0][0], 1.0, 1e-12));
        assert!(approx(cov[0][1], 2.0, 1e-12));
        assert!(approx(cov[1][1], 4.0, 1e-12));
    }

    #[test]
    fn single_element_matrix() {
        let m = vec![vec![42.0]];
        let (vals, vecs) = eigh(&m);
        assert_eq!(vals, vec![42.0]);
        assert_eq!(vecs, vec![vec![1.0]]);
    }

    #[test]
    fn repeated_eigenvalues() {
        // Identity matrix: all eigenvalues = 1
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (vals, _) = eigh(&m);
        for v in &vals {
            assert!(approx(*v, 1.0, 1e-9));
        }
    }

    #[test]
    fn trace_preserved() {
        let m = vec![vec![6.0, 2.0], vec![2.0, 3.0]]; // trace 9
        let (vals, _) = eigh(&m);
        let sum: f64 = vals.iter().sum();
        assert!(approx(sum, 9.0, 1e-9));
    }
}
