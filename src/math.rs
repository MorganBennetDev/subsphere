//! This module contains a minimal set of linear algebra types and functions used by this crate. It
//! has a very specific grab bag of functionality and is not intended to be publicly exposed for
//! general use.

/// Contains functions related to vectors.
pub(crate) mod vec {
    /// Negates a vector.
    #[inline]
    pub const fn neg<const N: usize>(a: [f64; N]) -> [f64; N] {
        let mut res = [0.0; N];
        let mut i = 0;
        while i < N {
            res[i] = -a[i];
            i += 1;
        }
        res
    }

    /// Adds `a` to `b`.
    #[inline]
    pub const fn add<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
        let mut res = [0.0; N];
        let mut i = 0;
        while i < N {
            res[i] = a[i] + b[i];
            i += 1;
        }
        res
    }

    /// Subtracts `b` from `a`.
    #[inline]
    pub const fn sub<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
        let mut res = [0.0; N];
        let mut i = 0;
        while i < N {
            res[i] = a[i] - b[i];
            i += 1;
        }
        res
    }

    /// Multiplies a vector by a scalar.
    #[inline]
    pub const fn mul<const N: usize>(a: [f64; N], b: f64) -> [f64; N] {
        let mut res = [0.0; N];
        let mut i = 0;
        while i < N {
            res[i] = a[i] * b;
            i += 1;
        }
        res
    }

    /// Divides a vector by a scalar.
    #[inline]
    pub const fn div<const N: usize>(a: [f64; N], b: f64) -> [f64; N] {
        // Since division is more expensive than multiplication, we actually multiply `a` by the
        // reciprocal of `b`. This introduces a very small amount of error, so we may need to
        // re-evaluate if/when we have tighter requirements for numerical error.
        mul(a, 1.0 / b)
    }

    /// Computes the dot product of two vectors.
    #[inline]
    pub const fn dot<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
        let mut res = 0.0;
        let mut i = 0;
        while i < N {
            res += a[i] * b[i];
            i += 1;
        }
        res
    }

    /// Computes the cross product of two vectors.
    #[inline]
    pub const fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    
    /// Computes the magnitude of a vector
    #[inline]
    pub const fn magnitude<const N: usize>(v: [f64; N]) -> f64 {
        let mut res = 0.0;
        let mut i = 0;
        while i < N {
            res += v[i] * v[i];
            i += 1;
        }
        
        res
    }

    /// Normalizes a vector.
    #[inline]
    pub fn normalize<const N: usize>(a: [f64; N]) -> [f64; N] {
        div(a, dot(a, a).sqrt())
    }
}

/// Contains functions related to matrices.
pub(crate) mod mat {
    use super::*;

    /// Multiplies a matrix with a vector.
    #[inline]
    pub const fn apply<const N: usize, const M: usize>(
        mat: [[f64; N]; M],
        vec: [f64; M],
    ) -> [f64; N] {
        let mut res = [0.0; N];
        let mut i = 0;
        while i < M {
            res = vec::add(res, vec::mul(mat[i], vec[i]));
            i += 1;
        }
        res
    }

    /// Computes the determinant of a 3x3 matrix.
    #[inline]
    pub const fn det_3(m: [[f64; 3]; 3]) -> f64 {
        vec::dot(m[0], vec::cross(m[1], m[2]))
    }

    /// Computes the determinant of a 2x2 matrix.
    #[inline]
    pub const fn det_2(m: [[f64; 2]; 2]) -> f64 {
        m[0][0] * m[1][1] - m[0][1] * m[1][0]
    }

    /// Computes the adjoint of a 2x2 matrix.
    #[inline]
    pub const fn adjoint_2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let [[a, b], [c, d]] = m;
        [[d, -b], [-c, a]]
    }
}

pub(crate) mod slerp {
    use std::f64;
    
    /// Map from S^2 to T_q(S^2). Inverse of exp_q(p)
    /// Outputs a vector with magnitude equal to the angle between p and q in the direction from q to p tangent to S^2 at q.
    fn sphere_ln(q: [f64; 3], p: [f64; 3]) -> [f64; 3] {
        let r = super::vec::dot(p, q).acos();
        
        let k = if r == 0.0 {
            1.0
        } else {
            r / r.sin()
        };
        
        super::vec::mul(super::vec::add(p, super::vec::mul(q, -r.cos())), k)
    }
    
    /// Map from T_q(S^2) to S^2. Preserves distance and direction
    fn sphere_exp(q: [f64; 3], dp: [f64; 3]) -> [f64; 3] {
        let r = super::vec::magnitude(dp);
        
        let k = if r == 0.0 {
            1.0
        } else {
            r.sin() / r
        };
        
        super::vec::add(super::vec::mul(q, r.cos()), super::vec::mul(dp, k))
    }
    
    /// Performs weighted spherical linear interpolation on a set of `N` vectors all lying on the same sphere using the
    /// local linear convergence algorithm (A1) described by Buss and Fillmore in [Spherical Averages and Applications
    /// to Spherical Splines and Interpolation](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/spheremean/paper.pdf).
    /// I attempted to implement the quadratic convergence algorithm but was not able to do so in a way that led to
    /// empirically better benchmarks. The authors acknowledge in the paper that the quadratic and linear convergence
    /// algorithms will likely have similar runtimes for single precisions floats. However, if you can manage to improve the
    /// performance of this routine, contributions are always welcome (particularly for this method as it is by far the most
    /// expensive operation in the entire process).
    pub fn slerp_n<const N: usize>(w: [f64; N], p: [[f64; 3]; N]) -> [f64; 3] { 
        let total_weight = w.iter().cloned().reduce(|a, b| a + b);
        debug_assert!(total_weight.is_some(), "Sum of weights must exist.");
        debug_assert!((total_weight.unwrap() - 1.0) <= f64::EPSILON, "Sum of weights must be equal to 1.0.");
        
        let mut q = super::vec::normalize(
            w.iter()
                .zip(p.iter())
                .map(|(w_i, p_i)| super::vec::mul(*p_i, *w_i))
                .reduce(super::vec::add)
                .unwrap()
        );
        
        loop {
            let u = w.iter()
                .zip(p.iter())
                .map(|(w_i, p_i)| super::vec::mul(sphere_ln(q, *p_i), *w_i))
                .reduce(super::vec::add)
                .unwrap();
            
            q = sphere_exp(q, u);
            
            if super::vec::magnitude(u) < 1.0e-6 {
                return q;
            }
        }
    }
    
    /// Shorthand for `slerp_n([w1, w2, w3], [p1, p2, p3])`.
    pub fn slerp_3(w1: f64, p1: [f64; 3], w2: f64, p2: [f64; 3], w3: f64, p3: [f64; 3]) -> [f64; 3] {
        slerp_n([w1, w2, w3], [p1, p2, p3])
    }
}