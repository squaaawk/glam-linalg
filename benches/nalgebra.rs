#![allow(non_snake_case)]

use divan::{bench, black_box, Bencher};
use fastrand::Rng;
use paste::paste;

use nalgebra::{Matrix2, Matrix3, Matrix4};

const N_TESTS: u64 = 1_000;

fn main() {
  divan::main();
}

macro_rules! eigvals_benches {
  ($num:expr) => {
    paste! {

      fn [<rand_nalgebra_matrix $num _vec>]() -> Vec<[<Matrix $num>]<f64>> {
        let mut rng = Rng::with_seed(5678);
        let v: Vec<_> = (0..$num * $num * N_TESTS).map(|_| rng.f64()).collect();
        v.chunks($num * $num).map([<Matrix $num>]::from_column_slice).collect()
      }

      #[bench]
      fn [<nalgebra_eigvals $num>](bencher: Bencher) {
        let matrices = [<rand_nalgebra_matrix $num _vec>]();

        bencher.bench_local(move || {
          for M in matrices.iter() {
            black_box(black_box(M).eigenvalues());
          }
        });
      }

      #[bench]
      fn [<nalgebra_eigvals $num _symmetric>](bencher: Bencher) {
        let matrices = [<rand_nalgebra_matrix $num _vec>]();

        bencher.bench_local(move || {
          for M in matrices.iter() {
            black_box(black_box(M).symmetric_eigenvalues());
          }
        });
      }

    }
  };
}

eigvals_benches!(2);
eigvals_benches!(3);
eigvals_benches!(4);
