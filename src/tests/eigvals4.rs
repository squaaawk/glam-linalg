use crate::tests::*;
use crate::utils::*;
use crate::*;

use approx::assert_abs_diff_eq;
use glam::{dvec2, DMat4};

#[test]
fn test_eigvals4() {
  let [a, b, c, d] = csorted(
    DMat4::from_cols_array(&[
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ])
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(-2.2093727122985456, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(0.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(0.0, 0.0), EPS);
  assert_dvec2_eq(d, dvec2(36.20937271229853, 0.0), EPS);

  let [a, b, c, d] = csorted(
    DMat4::from_cols_array(&[
      1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ])
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(-0.4476229868548985, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(0.6976613211204787, -0.4951597570817051), EPS);
  assert_dvec2_eq(c, dvec2(0.6976613211204787, 0.4951597570817051), EPS);
  assert_dvec2_eq(d, dvec2(3.0523003446139394, 0.0), EPS);

  // Hilbert matrix
  let [a, b, c, d] = csorted(
    DMat4::from_cols_array(
      &[
        1f64, 2f64, 3f64, 4f64, 2f64, 3f64, 4f64, 5f64, 3f64, 4f64, 5f64, 6f64, 4f64, 5f64, 6f64,
        7f64,
      ]
      .map(|x| x.recip()),
    )
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(9.670230402261436e-5, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(0.006738273605760762, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(0.16914122022145014, 0.0), EPS);
  assert_dvec2_eq(d, dvec2(1.5002142800592426, 0.0), EPS);

  // A matrix with a low condition number, which is a bad case for the base QR algorithm
  let delta = 1e-5;
  let [a, b, c, d] = csorted(
    DMat4::from_cols_array(&[
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -delta, 0.0, 0.0, delta, 1.0,
    ])
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(1.0, -delta), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(d, dvec2(1.0, delta), EPS);
}

#[test]
fn test_eigvals4_symmetric() {
  // These two test cases are the symmetric test cases from test_eigvals4
  // Hilbert matrix
  let [a, b, c, d] = sorted(
    DMat4::from_cols_array(
      &[
        1f64, 2f64, 3f64, 4f64, 2f64, 3f64, 4f64, 5f64, 3f64, 4f64, 5f64, 6f64, 4f64, 5f64, 6f64,
        7f64,
      ]
      .map(|x| x.recip()),
    )
    .eigvals_symmetric(),
  );
  assert_abs_diff_eq!(a, 9.670230402261436e-5, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(b, 0.006738273605760762, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(c, 0.16914122022145014, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(d, 1.5002142800592426, epsilon = SYMMETRIC_EPS);

  // A matrix with a low condition number, which is a bad case for the base QR algorithm
  let delta = 1e-5;
  let [a, b, c, d] = sorted(
    DMat4::from_cols_array(&[
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, delta, 0.0, 0.0, delta, 1.0,
    ])
    .eigvals_symmetric(),
  );
  assert_abs_diff_eq!(a, 1.0 - delta, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(b, 1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(c, 1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(d, 1.0 + delta, epsilon = SYMMETRIC_EPS);
}

test_dmat_rand!(4);
