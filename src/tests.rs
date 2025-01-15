mod eigvals2;
mod eigvals3;
mod eigvals4;

use crate::utils::*;

use approx::assert_abs_diff_eq;
use fastrand::Rng;
use glam::DVec2;
use paste::paste;

// TODO: The symmetric cases should not have a lower epsilon. At the moment, this is due to eigvals3_symmetric algorithm
pub(crate) const EPS: f64 = 1e-12;
pub(crate) const SYMMETRIC_EPS: f64 = 1e-8;

pub(crate) fn sorted<const N: usize>(mut x: [f64; N]) -> [f64; N] {
  x.sort_by(|a, b| a.total_cmp(b));
  x
}

pub(crate) fn csorted<const N: usize>(mut x: [DVec2; N]) -> [DVec2; N] {
  x.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
  x
}

pub(crate) fn assert_dvec2_eq(a: DVec2, b: DVec2, eps: f64) {
  assert!(a.abs_diff_eq(b, eps));
}

/// The tests here are from https://math.stackexchange.com/a/894641
pub(crate) fn assert_valid(eigvals: &[DVec2], trace: f64, trace_sq: f64, det: f64, eps: f64) {
  // The sum of the eigenvalues should be equal to the trace
  let sum: DVec2 = eigvals.iter().copied().sum();
  assert_dvec2_eq(sum, complex(trace), eps);

  // The sum of the eigenvalues squared should be equal to the trace of the matrix times itself
  let sum: f64 = eigvals
    .iter()
    .map(|&lambda| lambda.x.powi(2) - lambda.y.powi(2))
    .sum();
  assert_abs_diff_eq!(sum, trace_sq, epsilon = eps);

  // The product of the eigenvalues should be equal to the determinant
  let sign: f64 = eigvals.iter().map(|&lambda| lambda.x.signum()).product();
  let prod: f64 = eigvals.iter().map(|&lambda| lambda.length()).product();
  assert_abs_diff_eq!(sign * prod, det, epsilon = eps);
}

const SEED: u64 = 1234;
const N_TESTS: u64 = 100_000;

macro_rules! test_dmat_rand {
  ($num:expr) => {
    paste! {

      fn [< rand_dmat $num >](rng: &mut Rng) -> [< DMat $num >] {
        let v: Vec<_> = (0..$num * $num).map(|_| rng.f64()).collect();
        [< DMat $num >]::from_cols_slice(&v)
      }

      #[test]
      fn [< test_eigvals $num _rand>]() {
        let mut rng = Rng::with_seed(SEED);

        for _ in 0..N_TESTS {
          let A = [< rand_dmat $num >](&mut rng);
          assert_valid(
            &A.eigvals(),
            A.trace(),
            (A * A).trace(),
            A.determinant(),
            EPS,
          );
        }
      }

      #[test]
      fn [< test_eigvals $num _symmetric_rand>]() {
        let mut rng = Rng::with_seed(SEED);

        for _ in 0..N_TESTS {
          let A = [< rand_dmat $num >](&mut rng);
          let A = A + A.transpose();
          let eigvals: Vec<_> = A.eigvals_symmetric().into_iter().map(complex).collect();
          assert_valid(
            &eigvals,
            A.trace(),
            (A * A).trace(),
            A.determinant(),
            SYMMETRIC_EPS,
          );
        }
      }

    }
  };
}

pub(crate) use test_dmat_rand;
