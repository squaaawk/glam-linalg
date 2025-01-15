use crate::utils::*;
use crate::DMatExt;

use glam::{
  dvec2, dvec3, dvec4, DMat2, DMat3, DMat4, DVec2, DVec3, DVec4, Vec3Swizzles, Vec4Swizzles,
};

// These householder functions assume that a.x is nonzero
fn householder2(a: DVec2) -> DMat2 {
  let d = a.x + a.length().copysign(a.x);
  let v = dvec2(1.0, a.y / d).normalize();
  DMat2::IDENTITY - 2.0 * outer_product2(v, v)
}

fn householder3(a: DVec3) -> DMat3 {
  let d = a.x + a.length().copysign(a.x);
  let v = dvec3(1.0, a.y / d, a.z / d).normalize();
  DMat3::IDENTITY - 2.0 * outer_product3(v, v)
}

fn householder4(a: DVec4) -> DMat4 {
  let d = a.x + a.length().copysign(a.x);
  let v = dvec4(1.0, a.y / d, a.z / d, a.w / d).normalize();
  DMat4::IDENTITY - 2.0 * outer_product4(v, v)
}

/// Computes a QR decomposition of a 3x3 hessenberg matrix
// These functions could be made more efficient by taking advantage of the hessenberg form of the inputs matrix
fn qr3(A: DMat3) -> (DMat3, DMat3) {
  let mut A = A;
  let mut Q = DMat3::IDENTITY;

  let H = householder3(A.x_axis);
  Q *= H;
  A = H * A;

  let H = householder2(A.y_axis.yz());
  let H = DMat3::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, H.x_axis.x, H.y_axis.x, 0.0, H.x_axis.y, H.y_axis.y,
  ]);
  Q *= H;
  A = H * A;

  (Q, A)
}

/// Computes a QR decomposition of a 4x4 hessenberg matrix
fn qr4(A: DMat4) -> (DMat4, DMat4) {
  let mut A = A;
  let mut Q = DMat4::IDENTITY;

  let H = householder4(A.x_axis);
  Q *= H;
  A = H * A;

  let H = householder3(A.y_axis.yzw());
  let H = DMat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, H.x_axis.x, H.y_axis.x, H.z_axis.x, 0.0, H.x_axis.y, H.y_axis.y,
    H.z_axis.y, 0.0, H.x_axis.z, H.y_axis.z, H.z_axis.z,
  ]);
  Q *= H;
  A = H * A;

  let H = householder2(A.z_axis.zw());
  let H = DMat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, H.x_axis.x, H.y_axis.x, 0.0, 0.0, H.x_axis.y,
    H.y_axis.y,
  ]);
  Q *= H;
  A = H * A;

  (Q, A)
}

// Computes the hessenberg form of a matrix
pub(crate) fn hessenberg3(A: DMat3) -> DMat3 {
  let mut v = A.x_axis.yz();
  let alpha = -v.length() * v.x.signum();
  v.x -= alpha;
  v = v.normalize_or_zero();

  let Q = DMat2::from_mat3_minor(A, 0, 0);
  let Q = Q - 2.0 * outer_product2(v, lmul2(v, Q));

  let mut A = DMat3::from_cols_array(&[
    A.x_axis.x, alpha, 0.0, A.y_axis.x, Q.x_axis.x, Q.x_axis.y, A.z_axis.x, Q.y_axis.x, Q.y_axis.y,
  ]);

  let V = dvec3(
    A.row(0).yz().dot(v),
    A.row(1).yz().dot(v),
    A.row(2).yz().dot(v),
  );
  A.y_axis -= 2.0 * V * v.x;
  A.z_axis -= 2.0 * V * v.y;

  A
}

pub(crate) fn hessenberg4(A: DMat4) -> DMat4 {
  let mut v = A.x_axis.yzw();
  let alpha = -v.length() * v.x.signum();
  v.x -= alpha;
  v = v.normalize_or_zero();

  let Q = DMat3::from_mat4_minor(A, 0, 0);
  let Q = Q - 2.0 * outer_product3(v, lmul3(v, Q));

  let mut A = DMat4::from_cols_array(&[
    A.x_axis.x, alpha, 0.0, 0.0, A.y_axis.x, Q.x_axis.x, Q.x_axis.y, Q.x_axis.z, A.z_axis.x,
    Q.y_axis.x, Q.y_axis.y, Q.y_axis.z, A.w_axis.x, Q.z_axis.x, Q.z_axis.y, Q.z_axis.z,
  ]);

  let V = dvec4(
    A.row(0).yzw().dot(v),
    A.row(1).yzw().dot(v),
    A.row(2).yzw().dot(v),
    A.row(3).yzw().dot(v),
  );
  A.y_axis -= 2.0 * V * v.x;
  A.z_axis -= 2.0 * V * v.y;
  A.w_axis -= 2.0 * V * v.z;

  let mut v = A.y_axis.zw();
  let alpha = -v.length() * v.x.signum();
  v.x -= alpha;
  v = v.normalize_or_zero();

  let Q = DMat2::from_cols(A.z_axis.zw(), A.w_axis.zw());
  let Q = Q - 2.0 * outer_product2(v, lmul2(v, Q));

  let mut A = DMat4::from_cols_array(&[
    A.x_axis.x, A.x_axis.y, A.x_axis.z, A.x_axis.w, A.y_axis.x, A.y_axis.y, alpha, 0.0, A.z_axis.x,
    A.z_axis.y, Q.x_axis.x, Q.x_axis.y, A.w_axis.x, A.w_axis.y, Q.y_axis.x, Q.y_axis.y,
  ]);

  let V = dvec4(
    A.row(0).zw().dot(v),
    A.row(1).zw().dot(v),
    A.row(2).zw().dot(v),
    A.row(3).zw().dot(v),
  );
  A.z_axis -= 2.0 * V * v.x;
  A.w_axis -= 2.0 * V * v.y;

  A
}

/// Computes the eigenvalues of a dense 2x2 matrix
// Direct solving of characteristic polynomial
pub(crate) fn eigvals2(A: DMat2) -> [DVec2; 2] {
  let t = A.trace();
  let d = A.determinant();
  let center = complex(0.5 * t);
  let gap = csqrt(0.25 * t * t - d);
  [center - gap, center + gap]
}

/// Computes the eigenvalues of a 3x3 hessenberg matrix
// The eigvals*_hessenberg functions use the QR algorithm to determine the eigenvalues
pub(crate) fn eigvals3_hessenberg(mut A: DMat3) -> [DVec2; 3] {
  const CUTOFF: f64 = 1e-14;

  // We shouldn't have more than a couple dozen iterations
  for _ in 0..100 {
    // If some subdiagonal element is small enough, deflate
    if A.x_axis.y.abs() <= CUTOFF * (A.x_axis.x.abs() + A.y_axis.y.abs()) {
      let [a, b] = eigvals2(DMat2::from_mat3_minor(A, 0, 0));
      return [a, b, complex(A.x_axis.x)];
    }
    if A.y_axis.z.abs() <= CUTOFF * (A.y_axis.y.abs() + A.z_axis.z.abs()) {
      let [a, b] = eigvals2(DMat2::from_mat3(A));
      return [a, b, complex(A.z_axis.z)];
    }

    // Double shift QR step
    let s = A.y_axis.y + A.z_axis.z;
    let t = A.y_axis.y * A.z_axis.z - A.y_axis.z * A.z_axis.y;

    let M = A * A - s * A + t * DMat3::IDENTITY;
    let (Z, _) = qr3(M);

    A = Z.transpose() * A * Z;
  }

  unreachable!();
}

/// Computes the eigenvalues of a 4x4 hessenberg matrix
pub(crate) fn eigvals4_hessenberg(mut A: DMat4) -> [DVec2; 4] {
  const CUTOFF: f64 = 1e-14;

  // We shouldn't have more than a couple dozen iterations
  for _ in 0..100 {
    // If some subdiagonal element is small enough, deflate
    if A.x_axis.y.abs() <= CUTOFF * (A.x_axis.x.abs() + A.y_axis.y.abs()) {
      let [a, b, c] = eigvals3_hessenberg(DMat3::from_mat4_minor(A, 0, 0));
      return [complex(A.x_axis.x), a, b, c];
    }
    if A.y_axis.z.abs() <= CUTOFF * (A.y_axis.y.abs() + A.z_axis.z.abs()) {
      let [a, b] = eigvals2(DMat2::from_cols(A.x_axis.xy(), A.y_axis.xy()));
      let [c, d] = eigvals2(DMat2::from_cols(A.z_axis.zw(), A.w_axis.zw()));
      return [a, b, c, d];
    }
    if A.z_axis.w.abs() <= CUTOFF * (A.z_axis.z.abs() + A.w_axis.w.abs()) {
      let [a, b, c] = eigvals3_hessenberg(DMat3::from_mat4(A));
      return [a, b, c, complex(A.w_axis.w)];
    }

    // Double shift QR step
    let s = A.y_axis.z + A.z_axis.w;
    let t = A.y_axis.z * A.z_axis.w - A.y_axis.w * A.z_axis.z;

    let M = A * A - s * A + t * DMat4::IDENTITY;
    let (Z, _) = qr4(M);

    A = Z.transpose() * A * Z;
  }

  unreachable!();
}
