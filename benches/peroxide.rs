#![allow(non_snake_case)]

use divan::{bench, black_box, Bencher};
use fastrand::Rng;
use paste::paste;

use peroxide::prelude::*;

const N_TESTS: u64 = 1_000;

fn main() {
  divan::main();
}

macro_rules! eigvals_benches {
  ($num:expr) => {
    paste! {

      fn [<rand_peroxide_matrix $num _vec>]() -> Vec<Matrix> {
        let mut rng = Rng::with_seed(5678);
        let v: Vec<_> = (0..$num * $num * N_TESTS).map(|_| rng.f64()).collect();
        v.chunks($num * $num).map(|slice| matrix(slice.to_vec(), $num, $num, Col)).collect()
      }

      #[bench]
      fn [<peroxide_eigvals $num>](bencher: Bencher) {
        let matrices = [<rand_peroxide_matrix $num _vec>]();

        bencher.bench_local(move || {
          for M in matrices.iter() {
            black_box(eigen(black_box(&M)));
          }
        });
      }

    }
  };
}

eigvals_benches!(2);
eigvals_benches!(3);
eigvals_benches!(4);
