use crate::tests::*;
use crate::utils::*;
use crate::*;

use approx::assert_abs_diff_eq;
use glam::dvec2;

#[test]
fn test_eigvals2() {
  let [a, b] = csorted(DMat2::from_cols_array(&[1.0, -1.0, -1.0, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(0.0, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(2.0, 0.0), EPS);

  let [a, b] = csorted(DMat2::from_cols_array(&[1.0, 2.0, 3.0, 4.0]).eigvals());
  assert_dvec2_eq(a, dvec2(-0.3722813232690143, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(5.372281323269014, 0.0), EPS);

  let [a, b] = csorted(DMat2::from_cols_array(&[1.0, -1.0, 1.0, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(1.0, -1.0), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 1.0), EPS);
}

#[test]
fn test_eigvals2_symmetric() {
  // These two examples are the symmetric test cases from test_eigvals2
  let [a, b] = sorted(eigvals2_symmetric(DMat2::from_cols_array(&[
    1.0, -1.0, -1.0, 1.0,
  ])));
  assert_abs_diff_eq!(a, 0.0, epsilon = EPS);
  assert_abs_diff_eq!(b, 2.0, epsilon = EPS);

  let [a, b] = sorted(eigvals2_symmetric(DMat2::from_cols_array(&[
    1.0, 2.0, 3.0, 4.0,
  ])));
  assert_abs_diff_eq!(a, -0.3722813232690143, epsilon = EPS);
  assert_abs_diff_eq!(b, 5.372281323269014, epsilon = EPS);
}

test_dmat_rand!(2);
