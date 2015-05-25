//! A linear-algebra toolbox.

#![allow(non_snake_case)]

#[cfg(test)]
extern crate assert;

extern crate blas;
extern crate lapack;

mod core;
mod eigen;

pub use core::matrix_product as multiply;
pub use core::scalar_product as dot;
pub use core::scaling as scale;
pub use core::summation as add;

pub use eigen::symmetric as symmetric_eigen;

/// An error.
#[derive(Clone, Copy)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// A result.
pub type Result<T> = std::result::Result<T, Error>;
