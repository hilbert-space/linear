use blas;

/// Sum up two matrices.
///
/// The formula is as follows:
///
/// ```math
/// Y = α * X + Y.
/// ```
///
/// `X` and `Y` should have the same number of elements.
#[inline]
pub fn add(alpha: f64, X: &[f64], Y: &mut [f64]) {
    debug_assert_eq!(X.len(), Y.len());
    blas::daxpy(X.len(), alpha, X, 1, Y, 1)
}

/// Compute the dot product of two vectors.
///
/// `X` and `Y` should have the same number of elements.
#[inline]
pub fn dot(X: &[f64], Y: &[f64]) -> f64 {
    debug_assert_eq!(X.len(), Y.len());
    blas::ddot(X.len(), X, 1, Y, 1)
}

/// Multiply two matrices.
///
/// The formula is as follows:
///
/// ```math
/// C = α * A * B + β * C.
/// ```
///
/// `A`, `B`, and `C` should have `m × p`, `p × n`, and `m × n` elements, respectively.
#[inline]
pub fn multiply(alpha: f64, A: &[f64], B: &[f64], beta: f64, C: &mut [f64], m: usize) {
    let (p, n) = (A.len() / m, C.len() / m);
    debug_assert_eq!(A.len(), m * p);
    debug_assert_eq!(B.len(), p * n);
    debug_assert_eq!(C.len(), m * n);
    if n == 1 {
        blas::dgemv(blas::Trans::N, m, p, alpha, A, m, B, 1, beta, C, 1);
    } else {
        blas::dgemm(blas::Trans::N, blas::Trans::N, m, n, p, alpha, A, m, B, p, beta, C, m);
    }
}

/// Multiply a matrix by a scalar.
///
/// The formula is as follows:
///
/// ```math
/// X = α * X.
/// ```
#[inline]
pub fn scale(alpha: f64, X: &mut [f64]) {
    blas::dscal(X.len(), alpha, X, 1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn add() {
        let X = vec![1.0, 2.0, 3.0, 4.0];
        let mut Y = vec![-1.0, 2.0, -3.0, 4.0];

        super::add(1.0, &X, &mut Y);

        assert_eq!(&Y, &vec![0.0, 4.0, 0.0, 8.0]);
    }

    #[test]
    fn dot() {
        assert_eq!(super::dot(&[10.0, -4.0], &[5.0, 2.0]), 42.0);
    }

    #[test]
    fn multiply() {
        let A = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let B = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut C = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        super::multiply(1.0, &A, &B, 1.0, &mut C, 2);

        assert_eq!(&C, &vec![23.0, 30.0, 52.0, 68.0, 81.0, 106.0, 110.0, 144.0]);
    }

    #[test]
    fn scale() {
        let mut X = vec![1.0, 2.0, 3.0, 4.0];

        super::scale(2.0, &mut X);

        assert_eq!(&X, &vec![2.0, 4.0, 6.0, 8.0]);
    }
}
