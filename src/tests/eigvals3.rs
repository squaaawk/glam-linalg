use crate::tests::*;
use crate::utils::*;
use crate::*;

use approx::assert_abs_diff_eq;
use glam::{dvec2, DMat3};

#[test]
fn test_eigvals3() {
  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).eigvals());
  assert_dvec2_eq(a, dvec2(-1.1168439698070436, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(0.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(16.116843969807064, 0.0), EPS);

  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(0.5, -0.8660254037844392), EPS);
  assert_dvec2_eq(b, dvec2(0.5, 0.8660254037844392), EPS);
  assert_dvec2_eq(c, dvec2(2.0, 0.0), EPS);

  // A matrix that the base QR algorithm cannot solve
  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]).eigvals());
  assert_dvec2_eq(a, dvec2(-1.0, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0, 0.0), EPS);

  // A matrix with a low condition number, which is a bad case for the base QR algorithm
  let delta = 1e-5;
  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[1.0, 0.0, delta, 0.0, 1.0, 0.0, delta, 0.0, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(1.0 - delta, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0 + delta, 0.0), EPS);

  // Another matrix with a low condition number, rearranged
  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, delta, 0.0, delta, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(1.0 - delta, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0 + delta, 0.0), EPS);

  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, delta, 0.0, -delta, 1.0]).eigvals());
  assert_dvec2_eq(a, dvec2(1.0, -delta), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0, delta), EPS);

  // A bad case for the non-shifted algorithm
  let [a, b, c] =
    csorted(DMat3::from_cols_array(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).eigvals());
  assert_dvec2_eq(a, dvec2(-1.0, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(1.0, 0.0), EPS);
  assert_dvec2_eq(c, dvec2(1.0, 0.0), EPS);

  // A test case that did non converge with just the Rayleigh Quotient shift
  let [a, b, c] = csorted(
    DMat3::from_cols_array(&[
      -0.3213354775549597,
      0.02169014204590863,
      0.16693006894182016,
      -0.06496747561198643,
      0.1566324195390305,
      -0.21279295908748386,
      0.0,
      0.5943067054739077,
      -0.08187628642878006,
    ])
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(-0.3448497698916063, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(0.0491352127234484, -0.3500020316105234), EPS);
  assert_dvec2_eq(c, dvec2(0.0491352127234484, 0.3500020316105234), EPS);

  // Another test case that did non converge with just the Rayleigh Quotient shift
  let [a, b, c] = csorted(
    DMat3::from_cols_array(&[
      0.16321690858303795,
      0.01680483960771295,
      0.0,
      -0.08822904497924268,
      -0.5184780437091311,
      0.46726403838636654,
      -0.33628128518432093,
      -0.4632840573942136,
      -0.6725939268951061,
    ])
    .eigvals(),
  );
  assert_dvec2_eq(a, dvec2(-0.593049577421523, -0.4563618481950429), EPS);
  assert_dvec2_eq(b, dvec2(-0.593049577421523, 0.4563618481950429), EPS);
  assert_dvec2_eq(c, dvec2(0.15824409282184598, 0.0), EPS);

  let [a, b, c] = csorted(
    DMat3::from_cols_array(&[
      -0.26228172092024793,
      -0.03608976367373878,
      -2.0905308540472976e-17,
      0.03080946367110743,
      -0.06701789957369739,
      -0.5167701017184002,
      -0.5131608320367018,
      0.13828914509415038,
      -0.05187357048737509,
    ])
    .eigvals(),
  );

  assert_dvec2_eq(a, dvec2(-0.3270582792266884, 0.0), EPS);
  assert_dvec2_eq(b, dvec2(-0.027057455877315686, -0.29800047501630117), EPS);
  assert_dvec2_eq(c, dvec2(-0.027057455877315686, 0.29800047501630117), EPS);
}

#[test]
fn test_eigvals3_symmetric() {
  // These three test cases are test_eigvals3
  let [a, b, c] = sorted(
    DMat3::from_cols_array(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]).eigvals_symmetric(),
  );
  assert_abs_diff_eq!(a, -1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(b, 1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(c, 1.0, epsilon = SYMMETRIC_EPS);

  let delta = 1e-5;
  let [a, b, c] = sorted(
    DMat3::from_cols_array(&[1.0, 0.0, delta, 0.0, 1.0, 0.0, delta, 0.0, 1.0]).eigvals_symmetric(),
  );
  assert_abs_diff_eq!(a, 1.0 - delta, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(b, 1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(c, 1.0 + delta, epsilon = SYMMETRIC_EPS);

  let [a, b, c] = sorted(
    DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, delta, 0.0, delta, 1.0]).eigvals_symmetric(),
  );
  assert_abs_diff_eq!(a, 1.0 - delta, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(b, 1.0, epsilon = SYMMETRIC_EPS);
  assert_abs_diff_eq!(c, 1.0 + delta, epsilon = SYMMETRIC_EPS);
}

test_dmat_rand!(3);
