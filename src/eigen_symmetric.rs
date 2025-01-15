use crate::{DMatExt, EigenDecomposition};

use glam::{DMat2, DMat3, DMat4};

/// Computes the eigenvalues of a dense, symmetric 2x2 matrix
// Direct solving of characteristic polynomial
pub(crate) fn eigvals2_symmetric(A: DMat2) -> [f64; 2] {
  let t = A.trace();
  let d = A.determinant();
  let center = 0.5 * t;
  let gap = (0.25 * t * t - d).sqrt();
  [center - gap, center + gap]
}

/// Computes the eigenvalues of a dense, symmetric 3x3 matrix
// TODO: This algorithm is much quicker than the QR algorithm for symmetric 3x3 matrices, but has lower precision
// Uses the algorithm given in https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Symmetric_3%C3%973_matrices (Nov 20, 2024)
pub(crate) fn eigvals3_symmetric(A: DMat3) -> [f64; 3] {
  use std::f64::consts::PI;

  let p1 = A.y_axis.x.powi(2) + A.z_axis.x.powi(2) + A.z_axis.y.powi(2);

  if p1 == 0.0 {
    // A is diagonal
    [A.x_axis.x, A.y_axis.y, A.z_axis.z]
  } else {
    let tr = A.trace();
    let q = tr / 3.0;
    let p2 =
      (A.x_axis.x - q).powi(2) + (A.y_axis.y - q).powi(2) + (A.z_axis.z - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let B = (A - q * DMat3::IDENTITY) / p;
    let r = B.determinant() / 2.0;

    // In exact arithmetic for a symmetric matrix -1 <= r <= 1
    // but computation error can leave it slightly outside this range.
    let phi = r.clamp(-1.0, 1.0).acos() / 3.0;

    // the eigenvalues satisfy c <= b <= a
    let a = q + 2.0 * p * phi.cos();
    let c = q + 2.0 * p * (phi + (2.0 * PI / 3.0)).cos();
    let b = tr - a - c;

    [a, b, c]
  }
}

/// Computes the eigenvalues of a dense, symmetric 4x4 matrix
// TODO: Implement an algorithm that takes advantage of the symmetric structure
pub(crate) fn eigvals4_symmetric(A: DMat4) -> [f64; 4] {
  let [a, b, c, d] = A.eigvals();
  [a.x, b.x, c.x, d.x]
}
