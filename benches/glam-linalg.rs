#![allow(non_snake_case)]

use divan::{bench, black_box, Bencher};
use fastrand::Rng;
use paste::paste;

use glam::{DMat2, DMat3, DMat4};
use glam_linalg::EigenDecomposition;

const N_TESTS: u64 = 1_000;

fn main() {
  divan::main();
}

macro_rules! eigvals_benches {
  ($num:expr) => {
    paste! {

      fn [<rand_dmat $num _vec>]() -> Vec<[<DMat $num>]> {
        let mut rng = Rng::with_seed(5678);
        let v: Vec<_> = (0..$num * $num * N_TESTS).map(|_| rng.f64()).collect();
        v.chunks($num * $num).map([<DMat $num>]::from_cols_slice).collect()
      }

      #[bench]
      fn [<eigvals $num>](bencher: Bencher) {
        let matrices = [<rand_dmat $num _vec>]();

        bencher.bench_local(move || {
          for M in matrices.iter() {
            black_box(black_box(M).eigvals());
          }
        });
      }

      #[bench]
      fn [<eigvals $num _symmetric>](bencher: Bencher) {
        let matrices = [<rand_dmat $num _vec>]();

        bencher.bench_local(move || {
          for M in matrices.iter() {
            black_box(black_box(M).eigvals_symmetric());
          }
        });
      }

    }
  };
}

eigvals_benches!(2);
eigvals_benches!(3);
eigvals_benches!(4);
