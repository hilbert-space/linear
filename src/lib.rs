//! A linear-algebra toolbox.

#![allow(non_snake_case)]

#[cfg(test)]
extern crate assert;

extern crate blas;
extern crate lapack;

use std::{error, fmt};

mod core;
mod decomposition;

pub use core::{add, dot, multiply, scale};
pub use decomposition::symmetric_eigen;

/// An error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// One or more arguments have illegal values.
    InvalidArguments,
    /// The algorithm failed to converge.
    FailedToConverge,
}

/// A result.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        error::Error::description(self).fmt(formatter)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::InvalidArguments => "one or more arguments have illegal values",
            Error::FailedToConverge => "the algorithm failed to converge",
        }
    }
}
