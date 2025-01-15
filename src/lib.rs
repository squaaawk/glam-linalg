#![allow(non_snake_case)]

mod eigen;
mod eigen_symmetric;
#[cfg(test)]
mod tests;
mod utils;

use eigen::*;
use eigen_symmetric::*;

use glam::{DMat2, DMat3, DMat4, DVec2};

pub trait DMatExt {
  fn trace(&self) -> f64;
}

impl DMatExt for DMat2 {
  fn trace(&self) -> f64 {
    self.x_axis.x + self.y_axis.y
  }
}

impl DMatExt for DMat3 {
  fn trace(&self) -> f64 {
    self.x_axis.x + self.y_axis.y + self.z_axis.z
  }
}

impl DMatExt for DMat4 {
  fn trace(&self) -> f64 {
    self.x_axis.x + self.y_axis.y + self.z_axis.z + self.w_axis.w
  }
}

pub trait EigenDecomposition<const N: usize> {
  fn eigvals(&self) -> [DVec2; N];

  // The symmetric cases exhibit nice properties, allowing for specialized algorithms
  // Additionally, symmetric matrices always have real eigenvalues, allowing for a simpler return type
  // TODO: Implement glam_assert to assure symmetry
  fn eigvals_symmetric(&self) -> [f64; N];
}

impl EigenDecomposition<2> for DMat2 {
  /// Computes the eigenvalues of a dense 2x2 matrix
  fn eigvals(&self) -> [DVec2; 2] {
    eigvals2(*self)
  }

  /// Computes the eigenvalues of a dense, symmetric 2x2 matrix
  fn eigvals_symmetric(&self) -> [f64; 2] {
    eigvals2_symmetric(*self)
  }
}

impl EigenDecomposition<3> for DMat3 {
  /// Computes the eigenvalues of a dense 3x3 matrix
  fn eigvals(&self) -> [DVec2; 3] {
    eigvals3_hessenberg(hessenberg3(*self))
  }

  /// Computes the eigenvalues of a dense, symmetric 3x3 matrix
  fn eigvals_symmetric(&self) -> [f64; 3] {
    eigvals3_symmetric(*self)
  }
}

impl EigenDecomposition<4> for DMat4 {
  /// Computes the eigenvalues of a dense 4x4 matrix
  fn eigvals(&self) -> [DVec2; 4] {
    eigvals4_hessenberg(hessenberg4(*self))
  }

  /// Computes the eigenvalues of a dense, symmetric 4x4 matrix
  fn eigvals_symmetric(&self) -> [f64; 4] {
    eigvals4_symmetric(*self)
  }
}
