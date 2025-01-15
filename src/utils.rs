use glam::{dvec2, dvec3, DMat2, DMat3, DMat4, DVec2, DVec3, DVec4};

// Mock complex number implementation via a 2d vector
#[inline]
pub(crate) fn complex(x: f64) -> DVec2 {
  dvec2(x, 0.0)
}

#[inline]
pub(crate) fn csqrt(x: f64) -> DVec2 {
  if x.is_sign_positive() {
    dvec2(x.sqrt(), 0.0)
  } else {
    dvec2(0.0, (-x).sqrt())
  }
}

// These left-multiplication functions should be replaced with https://github.com/bitshifter/glam-rs/issues/494
#[inline]
pub(crate) fn lmul2(v: DVec2, M: DMat2) -> DVec2 {
  dvec2(v.dot(M.x_axis), v.dot(M.y_axis))
}

#[inline]
pub(crate) fn lmul3(v: DVec3, M: DMat3) -> DVec3 {
  dvec3(v.dot(M.x_axis), v.dot(M.y_axis), v.dot(M.z_axis))
}

#[inline]
pub(crate) fn outer_product2(a: DVec2, b: DVec2) -> DMat2 {
  DMat2::from_cols(a * b.x, a * b.y)
}

#[inline]
pub(crate) fn outer_product3(a: DVec3, b: DVec3) -> DMat3 {
  DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

#[inline]
pub(crate) fn outer_product4(a: DVec4, b: DVec4) -> DMat4 {
  DMat4::from_cols(a * b.x, a * b.y, a * b.z, a * b.w)
}
