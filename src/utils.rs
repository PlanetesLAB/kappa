use ndarray::Array1;
use planetes_ext::types::RVector;

pub mod constants {
    //! Defines mathematical expressions commonly used when computing distribution
    //! values as constants
    //!
    //! This module is directly copied from the `statrs` crate.
    //! The original code is licensed under the MIT license.
    //! It was purely done to avoid a dependency on the `statrs` crate.
    //!
    //! The original code can be found here:
    //! <https://github.com/statrs-dev/statrs/blob/master/src/constants.rs>
    #![allow(clippy::excessive_precision)]

    /// Constant value for `ln(pi)`
    pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;

    /// Constant value for `ln(2 * sqrt(e / pi))`
    pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

    /// Constant value for `2 * sqrt(e / pi)`
    pub const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

    /// Default accuracy for `f64`, equivalent to `0.0 * F64_PREC`
    pub const DEFAULT_F64_ACC: f64 = 0.0000000000000011102230246251565;
}
pub mod legendre {
    use anyhow::Result;
    use anyhow::bail;
    use ndarray::Array1;
    use planetes_ext::types::RVector;

    /// Computes the associated Legendre polynomials $P_k^m(x)$ and their derivatives
    /// for degrees $k = 0, \dots, n$.
    ///
    /// # Errors
    /// - `m` must be 0, 1, or 2.
    pub fn lpmns(m: usize, n: usize, x: f64) -> Result<(RVector, RVector)> {
        let mut pm: RVector = Array1::zeros(n + 1);
        let mut pd: RVector = Array1::zeros(n + 1);

        let mf = m as f64;

        if (x.abs() - 1.0).abs() < f64::EPSILON {
            for k in 0..=n {
                let kf = k as f64;
                match m {
                    0 => {
                        pm[k] = 1.0;
                        pd[k] = 0.5 * kf * (kf + 1.0);
                        if x < 0.0 {
                            if k % 2 == 1 {
                                pm[k] = -pm[k];
                            } else {
                                pd[k] = -pd[k];
                            }
                        }
                    }
                    1 => {
                        pd[k] = 1.0e300;
                    }
                    2 => {
                        pd[k] = -0.25 * (kf + 2.0) * (kf + 1.0) * kf * (kf - 1.0);
                        if x < 0.0 && k % 2 == 0 {
                            pd[k] = -pd[k];
                        }
                    }
                    _ => {
                        bail!("m must be 0, 1, or 2");
                    }
                }
            }
            return Ok((pm, pd));
        }

        let x0 = (1.0 - x * x).abs();
        let mut pm0 = 1.0;
        let mut pmk = pm0;
        for k in 1..=m {
            let kf = k as f64;
            pmk = (2.0 * kf - 1.0) * x0.sqrt() * pm0;
            pm0 = pmk;
        }
        let mut pm1 = (2.0 * mf + 1.0) * x * pm0;
        if m <= n {
            pm[m] = pmk;
        }
        if m < n {
            pm[m + 1] = pm1;
        }

        for k in (m + 2)..=n {
            let kf = k as f64;
            let pm2 = ((2.0 * kf - 1.0) * x * pm1 - (kf + mf - 1.0) * pmk) / (kf - mf);
            pm[k] = pm2;
            pmk = pm1;
            pm1 = pm2;
        }

        if n >= 1 {
            pd[0] = ((1.0 - mf) * pm[1] - x * pm[0]) / (x * x - 1.0);
        }

        for k in 1..=n {
            let kf = k as f64;
            pd[k] = (kf * x * pm[k] - (kf + mf) * pm[k - 1]) / (x * x - 1.0);
        }

        Ok((pm, pd))
    }

    /// Computes the Legendre polynomials $P_k(x)$ and their derivatives $P_k'(x)$
    /// for degrees $k = 0, \dots, n$.
    #[must_use]
    pub fn lpn(n: usize, x: f64) -> (RVector, RVector) {
        let mut pn: RVector = Array1::zeros(n + 1);
        let mut pd: RVector = Array1::zeros(n + 1);

        pn[0] = 1.0;
        pd[0] = 0.0;
        if n == 0 {
            return (pn, pd);
        }

        pn[1] = x;
        pd[1] = 1.0;

        let mut p0 = 1.0;
        let mut p1 = x;

        for k in 2..=n {
            let kf = k as f64;
            let pf = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
            pn[k] = pf;
            if (x.abs() - 1.0).abs() < f64::EPSILON {
                pd[k] = 0.5 * x.powi(k as i32 + 1) * kf * (kf + 1.0);
            } else {
                pd[k] = kf * (p1 - x * pf) / (1.0 - x * x);
            }
            p0 = p1;
            p1 = pf;
        }
        (pn, pd)
    }
}

pub mod bessel {
    use anyhow::Result;
    use anyhow::bail;
    use num_complex::{Complex, Complex64};

    use crate::fractal::FractalConfig;
    use crate::fractal::{FractalCutoff, FractalGeometry};

    use super::special::two_point_correlation;

    pub enum BoundaryCondition {
        Jablonski,
        TazakiTanaka,
    }

    impl BoundaryCondition {
        fn coefficients(self) -> ([f64; 4], [f64; 4]) {
            match self {
                BoundaryCondition::Jablonski => {
                    let a = [-3.6693021e-8, -3.1745158e-5, 2.1567720e-2, 9.9123380e-1];
                    let b = [-9.9921351e-8, 8.7822303e-6, 1.0238752e-2, 3.7588265];
                    (a, b)
                }
                BoundaryCondition::TazakiTanaka => {
                    let a = [
                        1.69496268177237e-08,
                        -2.43299782942114e-05,
                        0.0158750501131321,
                        1.00672154148706,
                    ];
                    let b = [
                        2.49355951047228e-08,
                        -2.9387731648675e-05,
                        0.0135005554796179,
                        3.72312019844119,
                    ];
                    (a, b)
                }
            }
        }
    }

    ///  This subroutine performs integration of S_p(kRg) (Equation 31):
    ///  $$
    ///         S_p(k R_g) = \frac{\pi^2}{k^3} \int_0^{\infty} du \, u \, J_{p+1/2}(u) \, H_{p+1/2}^{(1)}(u) \, g(u/k)
    ///  $$
    ///  where g(u) is the two-point correlation function (Equations 18 and 19):
    ///  $$
    ///         g(u) = \frac{1}{4 \pi R_g^3} \left( \frac{u}{R_g} \right)^{d_f - 3} \cdot fc \left( \frac{u}{R_g} \right)
    ///  $$
    ///  where fc is the cut-off function. By substituting g(u) into S_p, we have
    ///  $$
    ///         S_p(k R_g) = \frac{\pi}{4 X_g^{d_f}} \int_{u_{\min}}^{u_{\max}} du \, u^{d_f - 2} J_{p+1/2}(u) H_{p+1/2}^{(1)}(u) fc(u / X_g)
    ///  $$
    ///
    ///  where the integration range is approximated by the range `[u_min,u_max]`.
    ///  By using the spherical Bessel j_p(u) and Hankel functions of 1st kind h_p^(1)(u),
    ///  the Bessel function and the Hankel function are rewritten by
    ///  $$
    ///  J_{p+1/2} (u) = \sqrt{\frac{2u}{\pi}} \, j_p(u)
    ///  $$
    ///  
    ///  $$
    ///  H_{p+1/2}^{(1)}(u) = \sqrt{\frac{2u}{\pi}} \, h_p^{(1)}(u)
    ///  $$
    ///  
    ///  We have:
    ///  
    ///  $$
    ///  S_p(k R_g) = \frac{1}{2 X_g^{d_f}} \int_{u_{\min}}^{u_{\max}} du \, u^{d_f - 1} j_p(u) h_p^{(1)}(u) fc(u / X_g)
    ///  $$
    ///  
    ///  For the unitary condition of the two-point correlation function:
    ///  
    ///  $$
    ///  1 = \int_0^{\infty} dw \, 4\pi w^2 g(w)
    ///  $$
    ///  
    ///  If we take the integration variable as \( w = u/k \), then we obtain:
    ///  
    ///  $$
    ///  1 = \frac{1}{X_g^{d_f}} \int_{u_{\min}}^{u_{\max}} du \, u^{d_f - 1} fc(u / X_g) \quad \text{.... Eq. (*)}
    ///  $$
    ///  
    ///  The integration range `[umin,umax]` is determined as follows.
    ///  The integrand of Equation (*) is
    ///
    /// ```text
    ///         (u/xg)^{df-1}fc(u/xg) du = (u/xg) ^{d_f}fc(u/xg) dlnu
    /// ```
    ///
    ///  u_max is chosen so that fc(u/xg) ~ exp[-eta1].
    ///       For iqcor=1 (Gauss)
    ///               u_max ~ 2 * xg * sqrt(eta1 / d_f)
    ///       For iqcor=1 (Exponential)
    ///               u_max ~ xg * eta1 * sqrt(2.0 /(d_f*(d_f+1))
    ///       For iqcor=1 (FLDIM)
    ///               u_max ~ xg * sqrt( 2.0 * eta1 ) ** (1.0/d_f)
    ///  I adopt eta1 = 25.0.
    ///
    ///  u_min is chosen so that (u_min/xg)^{d_f} ~ exp[-eta2], thus,
    ///
    /// ```text
    ///               umin ~ xg * exp(-eta2/d_f)
    /// ```
    ///
    ///  where eta2 = 40.0.
    ///
    /// # Remarks
    /// [`BoundaryCondition::TazakiTanaka`] is used for the boundary condition and is
    /// recommended; although [`BoundaryCondition::Jablonski`] is also available.
    ///
    /// # Errors
    /// - `Error in sph_bessel (jp)`: An error occurred while computing the spherical Bessel function j_p(u)
    /// - `Error in sph_bessel (yp)`: An error occurred while computing the spherical Hankel function h_p^(1)(u)
    /// - `Error in int_sph_bessel`: The unitary condition of the two-point correlation function is not satisfied within the acceptable error margin
    #[allow(clippy::similar_names)]
    pub fn int_sph_bessel(fracc: &FractalConfig, x_g: f64, p: usize) -> Result<Complex64> {
        let pf = p as f64;

        let floor_val = 1e-30;
        let eta1 = 25.0;
        let eta2 = 40.0;

        let nn = 100_000; // Number of grid points in the numerical integration of S_p(kRg)
        let nnf = f64::from(nn);

        let umax = match fracc.cutoff {
            FractalCutoff::Gaussian => 2.0 * x_g * (eta1 / fracc.df).sqrt(),
            FractalCutoff::Exponential => x_g * eta1 * (2.0 / (fracc.df * (fracc.df + 1.0))).sqrt(),
            FractalCutoff::FractalDimension => x_g * (2.0 * eta1).powf(1.0 / fracc.df),
        };

        let umin = x_g * (-eta2 / fracc.df).exp();
        let ln_umin = umin.ln();
        let d_ln_u = (umax / umin).ln() / (nnf - 1.0);

        let (a, b) = BoundaryCondition::TazakiTanaka.coefficients();
        let lnxa = a[0] * pf.powf(3.0) + a[1] * pf.powf(2.0) + a[2] * pf + a[3];
        let lnxb = b[0] * pf.powf(3.0) + b[1] * pf.powf(2.0) + b[2] * pf + b[3];

        // Reusable downward-recurrence workspace (avoids 10^5 allocations per call).
        let mut k_buf = vec![0.0; p + 101];
        let mut wa = Complex::new(0.0, 0.0);
        let mut unitary = 0.0;
        let mut prev_u = 0.0;
        let mut prev_intg = Complex::new(0.0, 0.0);
        let mut prev_unit = 0.0;

        for n in 0..nn {
            let ln_u = ln_umin + f64::from(n) * d_ln_u;
            let u_val = ln_u.exp();
            let isol = if ln_u < lnxa {
                1
            } else if ln_u > lnxb {
                3
            } else {
                2
            };
            let (jp, yp) = sph_bessel_order(p, u_val, isol, &mut k_buf)?;

            if !jp.is_finite() {
                bail!("Error in sph_bessel (jp)");
            } else if !yp.is_finite() {
                bail!("Error in sph_bessel (yp)");
            }

            let fc = two_point_correlation(fracc, u_val, x_g);
            let u_weight = ((fracc.df - 1.0) * ln_u).exp() * fc;
            let curr_intg = u_weight * jp * Complex::new(jp, yp);

            if n > 0 {
                let du_step = u_val - prev_u;
                wa += 0.5 * (prev_intg + curr_intg) * du_step;
                unitary += f64::midpoint(prev_unit, u_weight) * du_step;
            }
            prev_u = u_val;
            prev_intg = curr_intg;
            prev_unit = u_weight;
        }

        let x_g_df = x_g.powf(fracc.df);
        unitary /= x_g_df;
        let mut sp = 0.5 * wa / x_g_df;
        let error = (1.0 - unitary).abs();

        if sp.re.abs() < floor_val {
            sp = Complex::new(floor_val, sp.im);
        }
        if sp.im.abs() < floor_val {
            sp = Complex::new(sp.re, -floor_val);
        }

        if fracc.geometry != FractalGeometry::Tazaki && error > 1.0e-3 {
            bail!("Error in int_sph_bessel: error = {error}");
        }

        Ok(sp)
    }

    fn sph_bessel_order(m: usize, x: f64, isol: usize, k_buf: &mut [f64]) -> Result<(f64, f64)> {
        let imax = 100; // truncation order of the series expansion
        let nwarmup = 100; // number of warm-up iterations

        let floor_val = 1.0e-70;
        let ceiling_val = 1.0e+70;

        let sin_x = x.sin();
        let cos_x = x.cos();
        let sj0 = sin_x / x;
        let sy0 = -cos_x / x;

        if m == 0 {
            return Ok((sj0, sy0));
        }

        let jp = match isol {
            1 => {
                // Series expansion
                let mut f_n = 1.0;
                let mut val = 0.0;
                for n in 1..=m {
                    let nf = n as f64;
                    f_n *= x / (2.0 * nf + 1.0);
                    let mut xi = 1.0;
                    let mut wa = 0.0;
                    for i in 1..=imax {
                        let ir = f64::from(i);
                        xi = -x * x * xi / (2.0 * ir * (2.0 * ir + 2.0 * nf + 1.0));
                        wa += xi;
                        if (xi / wa).abs() <= floor_val {
                            break;
                        }
                    }
                    val = f_n * (1.0 + wa);
                    if val.abs() <= floor_val {
                        if n < m {
                            val = 0.0;
                        }
                        break;
                    }
                }
                val
            }
            2 => {
                // Downward recursion using caller-provided workspace
                let mut k1 = 0.0;
                let mut k0 = 1.0;
                for n in (0..=m + nwarmup).rev() {
                    let nf = n as f64;
                    let kn = -k1 + (2.0 * nf + 3.0) * k0 / x;
                    k_buf[n] = kn;
                    k1 = k0;
                    k0 = kn;
                }
                let s = sj0 / k_buf[0];
                let mut val = 0.0;
                for &kn in &k_buf[1..=m] {
                    val = s * kn;
                    if val.abs() <= floor_val {
                        break;
                    }
                }
                if (s * k_buf[m]).abs() <= floor_val {
                    0.0
                } else {
                    val
                }
            }
            3 => {
                // Upward recursion
                let mut k0 = sj0;
                let mut k1 = sin_x / (x * x) - cos_x / x;
                for n in 1..m {
                    let nf = n as f64;
                    let k2 = (2.0 * nf + 1.0) * k1 / x - k0;
                    k0 = k1;
                    k1 = k2;
                }
                k1
            }
            _ => {
                bail!("Invalid isol value");
            }
        };

        // Spherical Bessel function of the second kind
        let mut y0 = sy0;
        let mut y1 = -cos_x / (x * x) - sin_x / x;
        let yp = if m == 1 {
            y1
        } else {
            let mut val = y1;
            for n in 2..=m {
                let nf = n as f64;
                val = (2.0 * nf - 1.0) * y1 / x - y0;
                y0 = y1;
                y1 = val;
                if val.abs() >= ceiling_val {
                    if n < m {
                        val = 0.0;
                    }
                    break;
                }
            }
            val
        };

        Ok((jp, yp))
    }
}

pub mod linalg {
    use anyhow::{Result, bail};
    use num_complex::Complex64;
    use planetes_ext::types::{CMatrix, CVector};

    /// Performs in-place Crout LU decomposition with implicit scaled partial pivoting on a
    /// square complex matrix `a` of dimension `n x n`.
    fn complex_lu_decomposition(a: &mut CMatrix, n: usize) -> Result<Vec<usize>> {
        let eps = Complex64::new(1e-20, 0.0);
        let mut row_scale = vec![0.0; n];
        let mut pivot = vec![0usize; n];

        for i in 0..n {
            let mut max_abs = 0.0;
            for j in 0..n {
                let v = a[[i, j]].norm();
                if v > max_abs {
                    max_abs = v;
                }
            }
            if max_abs == 0.0 {
                bail!("Singular matrix in complex_lu_decomposition");
            }
            row_scale[i] = 1.0 / max_abs;
        }

        for j in 0..n {
            for i in 0..j {
                let mut sum = a[[i, j]];
                for k in 0..i {
                    sum -= a[[i, k]] * a[[k, j]];
                }
                a[[i, j]] = sum;
            }

            let mut max_scaled = 0.0;
            let mut imax = j;
            for i in j..n {
                let mut sum = a[[i, j]];
                for k in 0..j {
                    sum -= a[[i, k]] * a[[k, j]];
                }
                a[[i, j]] = sum;
                let scaled = row_scale[i] * sum.norm();
                if scaled >= max_scaled {
                    max_scaled = scaled;
                    imax = i;
                }
            }

            if j != imax {
                for k in 0..n {
                    let tmp = a[[imax, k]];
                    a[[imax, k]] = a[[j, k]];
                    a[[j, k]] = tmp;
                }
                row_scale[imax] = row_scale[j];
            }
            pivot[j] = imax;

            if a[[j, j]] == Complex64::new(0.0, 0.0) {
                a[[j, j]] = eps;
            }
            if j + 1 < n {
                let inv_diag = Complex64::new(1.0, 0.0) / a[[j, j]];
                for i in j + 1..n {
                    a[[i, j]] *= inv_diag;
                }
            }
        }

        Ok(pivot)
    }

    fn complex_lu_solve(lu: &CMatrix, b: &mut CVector, pivot: &[usize], n: usize) {
        let zero = Complex64::new(0.0, 0.0);
        let mut first_nonzero: Option<usize> = None;

        for i in 0..n {
            let ip = pivot[i];
            let mut sum = b[ip];
            b[ip] = b[i];
            if let Some(start) = first_nonzero {
                for j in start..i {
                    sum -= lu[[i, j]] * b[j];
                }
            } else if sum != zero {
                first_nonzero = Some(i);
            }
            b[i] = sum;
        }

        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in i + 1..n {
                sum -= lu[[i, j]] * b[j];
            }
            b[i] = sum / lu[[i, i]];
        }
    }

    /// Solves the complex linear system $T r = p$ for $r$ directly on $\mathbb{C}^{n \times n}$
    /// using LU decomposition with scaled partial pivoting.
    ///
    /// # Errors
    /// Returns an error if the matrix $T$ is singular or has incompatible dimensions.
    pub fn complex_matrix_inverse(
        n: usize,
        _np: usize,
        p: &CVector,
        t: &CMatrix,
    ) -> Result<CVector> {
        if n == 0 || t.nrows() < n || t.ncols() < n || p.len() < n {
            bail!("Invalid dimensions in complex_matrix_inverse");
        }
        let mut lu = t.slice(ndarray::s![0..n, 0..n]).to_owned();
        let mut rhs = p.slice(ndarray::s![0..n]).to_owned();

        let pivot = complex_lu_decomposition(&mut lu, n)?;
        complex_lu_solve(&lu, &mut rhs, &pivot, n);

        Ok(rhs)
    }
}

pub mod gamma {
    //! Gamma function related utilities
    //!
    //! This module is directly copied from the `statrs` crate.
    //! The original code is licensed under the MIT license.
    //! It was purely done to avoid a dependency on the `statrs` crate.
    //!
    //! The original code can be found here:
    //! <https://github.com/statrs-dev/statrs/blob/master/src/function/gamma.rs>

    #![allow(clippy::excessive_precision)]
    use std::f64::consts::EULER_GAMMA;
    use std::f64::consts::{E, PI};

    use anyhow::Result;
    use anyhow::anyhow;
    use approx::ulps_eq;

    use super::constants::{DEFAULT_F64_ACC, LN_2_SQRT_E_OVER_PI, LN_PI, TWO_SQRT_E_OVER_PI};
    use super::special::almost_eq;

    /// Auxiliary variable when evaluating the `gamma_ln` function
    const GAMMA_R: f64 = 10.900511;

    /// Polynomial coefficients for approximating the `gamma_ln` function
    const GAMMA_DK: &[f64] = &[
        2.48574089138753565546e-5,
        1.05142378581721974210,
        -3.45687097222016235469,
        4.51227709466894823700,
        -2.98285225323576655721,
        1.05639711577126713077,
        -1.95428773191645869583e-1,
        1.70970543404441224307e-2,
        -5.71926117404305781283e-4,
        4.63399473359905636708e-6,
        -2.71994908488607703910e-9,
    ];

    /// Computes the logarithm of the gamma function
    /// with an accuracy of 16 floating point digits.
    /// The implementation is derived from
    /// "An Analysis of the Lanczos Gamma Approximation",
    /// Glendon Ralph Pugh, 2004 p. 116
    #[must_use]
    pub fn ln_gamma(x: f64) -> f64 {
        if x < 0.5 {
            let s = GAMMA_DK
                .iter()
                .enumerate()
                .skip(1)
                .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

            LN_PI
                - (PI * x).sin().ln()
                - s.ln()
                - LN_2_SQRT_E_OVER_PI
                - (0.5 - x) * ((0.5 - x + GAMMA_R) / E).ln()
        } else {
            let s = GAMMA_DK
                .iter()
                .enumerate()
                .skip(1)
                .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

            s.ln() + LN_2_SQRT_E_OVER_PI + (x - 0.5) * ((x - 0.5 + GAMMA_R) / E).ln()
        }
    }

    /// Computes the gamma function with an accuracy
    /// of 16 floating point digits. The implementation
    /// is derived from "An Analysis of the Lanczos Gamma Approximation",
    /// Glendon Ralph Pugh, 2004 p. 116
    #[must_use]
    pub fn gamma(x: f64) -> f64 {
        if x < 0.5 {
            let s = GAMMA_DK
                .iter()
                .enumerate()
                .skip(1)
                .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

            PI / ((PI * x).sin() * s * TWO_SQRT_E_OVER_PI * ((0.5 - x + GAMMA_R) / E).powf(0.5 - x))
        } else {
            let s = GAMMA_DK
                .iter()
                .enumerate()
                .skip(1)
                .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

            s * TWO_SQRT_E_OVER_PI * ((x - 0.5 + GAMMA_R) / E).powf(x - 0.5)
        }
    }

    /// Computes the upper incomplete gamma function
    /// `Gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and
    /// `x` is the lower intergral limit.
    ///
    /// # Panics
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    #[must_use]
    pub fn gamma_ui(a: f64, x: f64) -> f64 {
        checked_gamma_ui(a, x).unwrap()
    }

    /// Computes the upper incomplete gamma function
    /// `Gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and
    /// `x` is the lower intergral limit.
    ///
    /// # Errors
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    pub fn checked_gamma_ui(a: f64, x: f64) -> Result<f64> {
        checked_gamma_ur(a, x).map(|x| x * gamma(a))
    }

    /// Computes the lower incomplete gamma function
    /// `gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and `x`
    /// is the upper integral limit.
    ///
    ///
    /// # Panics
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    #[must_use]
    pub fn gamma_li(a: f64, x: f64) -> f64 {
        checked_gamma_li(a, x).unwrap()
    }

    /// Computes the lower incomplete gamma function
    /// `gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and `x`
    /// is the upper integral limit.
    ///
    ///
    /// # Errors
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    pub fn checked_gamma_li(a: f64, x: f64) -> Result<f64> {
        checked_gamma_lr(a, x).map(|x| x * gamma(a))
    }

    /// Computes the upper incomplete regularized gamma function
    /// `Q(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and
    /// `x` is the lower integral limit.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if either argument is `f64::NAN`
    ///
    /// # Panics
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    #[must_use]
    pub fn gamma_ur(a: f64, x: f64) -> f64 {
        checked_gamma_ur(a, x).unwrap()
    }

    /// Computes the upper incomplete regularized gamma function
    /// `Q(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
    /// where `a` is the argument for the gamma function and
    /// `x` is the lower integral limit.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if either argument is `f64::NAN`
    ///
    /// # Errors
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    pub fn checked_gamma_ur(a: f64, x: f64) -> Result<f64> {
        if a.is_nan() || x.is_nan() {
            return Ok(f64::NAN);
        }
        if a <= 0.0 || a == f64::INFINITY {
            return Err(anyhow!("AInvalid"));
        }
        if x <= 0.0 || x == f64::INFINITY {
            return Err(anyhow!("XInvalid"));
        }

        let eps = 0.000000000000001;
        let big = 4503599627370496.0;
        let big_inv = 2.22044604925031308085e-16;

        if x < 1.0 || x <= a {
            return Ok(1.0 - gamma_lr(a, x));
        }

        let mut ax = a * x.ln() - x - ln_gamma(a);
        if ax < -709.78271289338399 {
            return if a < x { Ok(0.0) } else { Ok(1.0) };
        }

        ax = ax.exp();
        let mut y = 1.0 - a;
        let mut z = x + y + 1.0;
        let mut c = 0.0;
        let mut pkm2 = 1.0;
        let mut qkm2 = x;
        let mut pkm1 = x + 1.0;
        let mut qkm1 = z * x;
        let mut ans = pkm1 / qkm1;
        loop {
            y += 1.0;
            z += 2.0;
            c += 1.0;
            let yc = y * c;
            let pk = pkm1 * z - pkm2 * yc;
            let qk = qkm1 * z - qkm2 * yc;

            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if pk.abs() > big {
                pkm2 *= big_inv;
                pkm1 *= big_inv;
                qkm2 *= big_inv;
                qkm1 *= big_inv;
            }

            if qk != 0.0 {
                let r = pk / qk;
                let t = ((ans - r) / r).abs();
                ans = r;

                if t <= eps {
                    break;
                }
            }
        }
        Ok(ans * ax)
    }

    /// Computes the lower incomplete regularized gamma function
    /// `P(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for real a > 0, x > 0`
    /// where `a` is the argument for the gamma function and `x` is the upper
    /// integral limit.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if either argument is `f64::NAN`
    ///
    /// # Panics
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    #[must_use]
    pub fn gamma_lr(a: f64, x: f64) -> f64 {
        checked_gamma_lr(a, x).unwrap()
    }

    /// Computes the lower incomplete regularized gamma function
    /// `P(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for real a > 0, x > 0`
    /// where `a` is the argument for the gamma function and `x` is the upper
    /// integral limit.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if either argument is `f64::NAN`
    ///
    /// # Errors
    ///
    /// if `a` or `x` are not in `(0, +inf)`
    pub fn checked_gamma_lr(a: f64, x: f64) -> Result<f64> {
        if a.is_nan() || x.is_nan() {
            return Ok(f64::NAN);
        }
        if a <= 0.0 || a == f64::INFINITY {
            return Err(anyhow!("AInvalid"));
        }
        if x <= 0.0 || x == f64::INFINITY {
            return Err(anyhow!("XInvalid"));
        }

        let eps = 0.000000000000001;
        let big = 4503599627370496.0;
        let big_inv = 2.22044604925031308085e-16;

        if almost_eq(a, 0.0, DEFAULT_F64_ACC) {
            return Ok(1.0);
        }
        if almost_eq(x, 0.0, DEFAULT_F64_ACC) {
            return Ok(0.0);
        }

        let ax = a * x.ln() - x - ln_gamma(a);
        if ax < -709.78271289338399 {
            if a < x {
                return Ok(1.0);
            }
            return Ok(0.0);
        }
        if x <= 1.0 || x <= a {
            let mut r2 = a;
            let mut c2 = 1.0;
            let mut ans2 = 1.0;
            loop {
                r2 += 1.0;
                c2 *= x / r2;
                ans2 += c2;

                if c2 / ans2 <= eps {
                    break;
                }
            }
            return Ok(ax.exp() * ans2 / a);
        }

        let mut y = 1.0 - a;
        let mut z = x + y + 1.0;
        let mut c = 0;

        let mut p3 = 1.0;
        let mut q3 = x;
        let mut p2 = x + 1.0;
        let mut q2 = z * x;
        let mut ans = p2 / q2;

        loop {
            y += 1.0;
            z += 2.0;
            c += 1;
            let yc = y * f64::from(c);

            let p = p2 * z - p3 * yc;
            let q = q2 * z - q3 * yc;

            p3 = p2;
            p2 = p;
            q3 = q2;
            q2 = q;

            if p.abs() > big {
                p3 *= big_inv;
                p2 *= big_inv;
                q3 *= big_inv;
                q2 *= big_inv;
            }

            if q != 0.0 {
                let nextans = p / q;
                let error = ((ans - nextans) / nextans).abs();
                ans = nextans;

                if error <= eps {
                    break;
                }
            }
        }
        Ok(1.0 - ax.exp() * ans)
    }

    /// Computes the Digamma function which is defined as the derivative of
    /// the log of the gamma function. The implementation is based on
    /// "Algorithm AS 103", Jose Bernardo, Applied Statistics, Volume 25, Number 3
    /// 1976, pages 315 - 317
    #[must_use]
    pub fn digamma(x: f64) -> f64 {
        let c = 12.0;
        let d1 = -EULER_GAMMA;
        let d2 = 1.6449340668482264365;
        let s = 1e-6;
        let s3 = 1.0 / 12.0;
        let s4 = 1.0 / 120.0;
        let s5 = 1.0 / 252.0;
        let s6 = 1.0 / 240.0;
        let s7 = 1.0 / 132.0;

        if x == f64::NEG_INFINITY || x.is_nan() {
            return f64::NAN;
        }
        if x <= 0.0 && ulps_eq!(x.floor(), x) {
            return f64::NEG_INFINITY;
        }
        if x < 0.0 {
            return digamma(1.0 - x) + PI / (-PI * x).tan();
        }
        if x <= s {
            return d1 - 1.0 / x + d2 * x;
        }

        let mut result = 0.0;
        let mut z = x;
        while z < c {
            result -= 1.0 / z;
            z += 1.0;
        }

        if z >= c {
            let mut r = 1.0 / z;
            result += z.ln() - 0.5 * r;
            r *= r;

            result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
        }
        result
    }

    #[must_use]
    pub fn inv_digamma(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x == f64::NEG_INFINITY {
            return 0.0;
        }
        if x == f64::INFINITY {
            return f64::INFINITY;
        }
        let mut y = x.exp();
        let mut i = 1.0;
        while i > 1e-15 {
            y += i * signum(x - digamma(y));
            i /= 2.0;
        }
        y
    }

    // modified signum that returns 0.0 if x == 0.0. Used
    // by inv_digamma, may consider extracting into a public
    // method
    fn signum(x: f64) -> f64 {
        if x == 0.0 { 0.0 } else { x.signum() }
    }
}

pub mod special {
    use std::f64::consts::PI;

    use approx::AbsDiffEq;

    use crate::fractal::{FractalConfig, FractalCutoff};

    use super::gamma::gamma;

    /// Compares if two floats are close via `approx::abs_diff_eq`
    /// using a maximum absolute difference (epsilon) of `acc`.
    ///
    /// This is directly copied from the `statrs` crate.
    /// The original code is licensed under the MIT license.
    /// It was purely done to avoid a dependency on the `statrs` crate.
    ///
    /// The original code can be found here:
    /// <https://github.com/statrs-dev/statrs/blob/master/src/prec.rs>
    #[must_use]
    pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            return a == b;
        }
        a.abs_diff_eq(&b, acc)
    }

    #[must_use]
    pub fn two_point_correlation(fracc: &FractalConfig, u: f64, x: f64) -> f64 {
        match fracc.cutoff {
            FractalCutoff::Gaussian => {
                let c = 0.25 * fracc.df;
                (2.0 * c.powf(0.5 * fracc.df) / gamma(0.5 * fracc.df)) * (-c * u * u / x / x).exp()
            }
            FractalCutoff::Exponential => {
                let c = (0.5 * fracc.df * (fracc.df + 1.0)).sqrt();
                (c.powf(fracc.df) / gamma(fracc.df)) * (-c * u / x).exp()
            }
            FractalCutoff::FractalDimension => {
                let c = 0.5;
                (c * fracc.df) * (-c * (u / x).powf(fracc.df)).exp()
            }
        }
    }

    pub fn output_structure() {
        todo!()
    }

    #[must_use]
    pub fn confluent_hypergeometric(mut a: f64, b: f64, mut x: f64) -> f64 {
        let mut a0 = a;
        let x0 = x;
        let mut y0 = 0.0;
        let mut y1 = 0.0;
        let mut hg = 0.0;
        if b == 0.0 || (b < 0.0 && (b - b.trunc()).abs() < f64::EPSILON) {
            hg = 1e300;
        } else if a == 0.0 || x == 0.0 {
            hg = 1.0;
        } else if (a - -1.0).abs() < f64::EPSILON {
            hg = 1.0 - x / b;
        } else if (a - b).abs() < f64::EPSILON {
            hg = x.exp();
        } else if ((a - b) - 1.0).abs() < f64::EPSILON {
            hg = (1.0 + x / b) * x.exp();
        } else if (a - 1.0).abs() < f64::EPSILON && (b - 2.0).abs() < f64::EPSILON {
            hg = (x.exp() - 1.0) / x;
        } else if a < 0.0 && (a - a.trunc()).abs() < f64::EPSILON {
            let m = (-a.trunc()) as i32;
            let mut r = 1.0;
            hg = 1.0;
            for k in 1..=m {
                let kf = f64::from(k);
                r *= (a + kf - 1.0) / kf / (b + kf - 1.0) * x;
                hg += r;
            }
        }
        if hg != 0.0 {
            return hg;
        }

        if x < 0.0 {
            a = b - a;
            a0 = a;
            x = x.abs();
        }

        let nl = i32::from(a >= 2.0);
        let la = if a >= 2.0 { a as usize } else { 0 };
        let laf = la as f64;
        if a >= 2.0 {
            a = a - laf - 1.0;
        }

        for n in 0..=nl {
            if a0 >= 2.0 {
                a += 1.0;
            }
            if x <= 30.0 + b.abs() || a < 0.0 {
                hg = 1.0;
                let mut rg = 1.0;
                for j in 1..=500 {
                    let jf = f64::from(j);
                    rg *= (a + jf - 1.0) / (jf * (b + jf - 1.0)) * x;
                    hg += rg;
                    if (rg / hg).abs() < 1e-15 {
                        break;
                    }
                }
            } else {
                let ta = gamma(a);
                let tb = gamma(b);
                let xg = b - a;
                let tba = gamma(xg);
                let mut sum1 = 1.0;
                let mut sum2 = 1.0;
                let mut r1 = 1.0;
                let mut r2 = 1.0;
                for i in 1..=8 {
                    let ir = f64::from(i);
                    r1 = -r1 * (a + ir - 1.0) * (a - b + ir) / (x * ir);
                    r2 = -r2 * (b - a + ir - 1.0) * (a - ir) / (x * ir);
                    sum1 += r1;
                    sum2 += r2;
                }
                let hg1 = tb / tba * x.powf(-a) * (PI * a).cos() * sum1;
                let hg2 = tb / ta * x.exp() * x.powf(a - b) * sum2;
                hg = hg1 + hg2;
            }
            if n == 0 {
                y0 = hg;
            } else if n == 1 {
                y1 = hg;
            }
        }
        if a0 >= 2.0 {
            for _ in 1..la {
                hg = ((2.0 * a - b + x) * y1 + (b - a) * y0) / a;
                y0 = y1;
                y1 = hg;
                a += 1.0;
            }
        }
        if x0 < 0.0 {
            hg *= x0.exp();
        }

        hg
    }

    pub fn optic_limit() {
        todo!()
    }
}

#[must_use]
pub fn regrid_lnk_data(
    l0: &[f64],
    n0: &[f64],
    k0: &[f64],
    lam: &RVector,
    loglog: bool,
) -> (RVector, RVector) {
    let n = lam.len();
    let n_actual = l0.len();
    let mut x0 = l0[0];
    let mut y01 = n0[0];
    let mut y02 = k0[0];
    // let wp = (1.0 - y01) / x0.powi(2);
    // let gamma = y02 / x0.powi(3);

    // The first block is space in grid that is before the first specified
    // wavelength. Simple extrapolation: keep same value
    let mut e1: RVector = Array1::zeros(n);
    let mut e2: RVector = Array1::zeros(n);
    let mut i = 0;

    while i < n && x0 >= lam[i] {
        e1[i] = y01;
        e2[i] = y02;
        i += 1;
        if i >= n {
            // All requested lambda are before first data point
            return (e1, e2);
        }
    }
    // Main interpolation loop
    for i0 in 0..n_actual {
        let x1 = l0[i0];
        let y11 = n0[i0];
        let y12 = k0[i0];

        while i < n && lam[i] <= x1 && lam[i] > x0 {
            // log-log interpolation between points
            let lgy11 = y11.log10();
            let lgy12 = y12.log10();
            let lgy01 = y01.log10();
            let lgy02 = y02.log10();
            let lgx0 = x0.log10();
            let lgx1 = x1.log10();
            let lggr = lam[i].log10();

            e1[i] = 10f64.powf(lgy11 + (lggr - lgx1) * (lgy01 - lgy11) / (lgx0 - lgx1));
            e2[i] = 10f64.powf(lgy12 + (lggr - lgx1) * (lgy02 - lgy12) / (lgx0 - lgx1));

            i += 1;
            if i >= n {
                return (e1, e2);
            }
        }

        x0 = x1;
        y01 = y11;
        y02 = y12;
    }

    // Extrapolation to long wavelengths
    if loglog {
        // Use log-log extrapolation
        let slope1 = (n0[n_actual - 1] / n0[n_actual - 2]).log10()
            / (l0[n_actual - 1] / l0[n_actual - 2]).log10();
        let sect1 = n0[n_actual - 2].log10();
        let slope2 = (k0[n_actual - 1] / k0[n_actual - 2]).log10()
            / (l0[n_actual - 1] / l0[n_actual - 2]).log10();
        let sect2 = k0[n_actual - 2].log10();

        for j in i..n {
            let dist = (lam[j] / l0[n_actual - 2]).log10();
            e1[j] = 10f64.powf(slope1 * dist + sect1);
            e2[j] = 10f64.powf(slope2 * dist + sect2);
        }
    } else {
        // Use the dielectric extrapolation, this is the default
        let (base_l, base_n, base_k) = if i == 0 {
            (l0[n_actual - 1], n0[n_actual - 1], k0[n_actual - 1])
        } else {
            (lam[i - 1], e1[i - 1], e2[i - 1])
        };
        for j in i..n {
            e1[j] = base_n;
            e2[j] = base_k * base_l / lam[j];
        }
    }

    (e1, e2)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use crate::fractal::FractalConfig;
    use crate::fractal::{FractalCutoff, FractalGeometry};

    use super::bessel;
    use super::legendre;
    use super::linalg;
    use super::special;

    #[test]
    fn legendre_lpn_interior_and_endpoints() {
        let x: f64 = 0.3;
        let (pn, pd) = legendre::lpn(4, x);
        assert_eq!(pn.len(), 5);
        assert_relative_eq!(pn[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(pn[1], x, epsilon = 1e-14);
        assert_relative_eq!(pn[2], 0.5 * (3.0 * x * x - 1.0), epsilon = 1e-14);
        assert_relative_eq!(pn[3], 0.5 * (5.0 * x.powi(3) - 3.0 * x), epsilon = 1e-14);
        assert_relative_eq!(
            pn[4],
            0.125 * (35.0 * x.powi(4) - 30.0 * x * x + 3.0),
            epsilon = 1e-14
        );
        assert_relative_eq!(pd[2], 3.0 * x, epsilon = 1e-14);
        assert_relative_eq!(pd[3], 1.5 * (5.0 * x * x - 1.0), epsilon = 1e-14);

        let (values_neg, derivs_neg) = legendre::lpn(4, -1.0);
        for k in 0..=4 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert_relative_eq!(values_neg[k], sign, epsilon = 1e-14);
            let kf = k as f64;
            assert_relative_eq!(derivs_neg[k], -sign * 0.5 * kf * (kf + 1.0), epsilon = 1e-14);
        }
    }

    #[test]
    fn legendre_lpmns_interior_and_endpoints() {
        let x: f64 = 0.4;
        let s = (1.0 - x * x).sqrt();
        let (pm, pd) = legendre::lpmns(1, 3, x).unwrap();
        assert_eq!(pm.len(), 4);
        assert_relative_eq!(pm[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(pm[1], s, epsilon = 1e-14);
        assert_relative_eq!(pm[2], 3.0 * x * s, epsilon = 1e-14);
        assert_relative_eq!(pm[3], 1.5 * (5.0 * x * x - 1.0) * s, epsilon = 1e-14);
        assert_relative_eq!(pd[1], -x / s, epsilon = 1e-14);

        // Endpoint |x| = 1 must return cleanly without dividing by zero
        let (pm0, pd0) = legendre::lpmns(0, 3, -1.0).unwrap();
        assert_relative_eq!(pm0[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(pm0[1], -1.0, epsilon = 1e-14);
        assert_relative_eq!(pm0[2], 1.0, epsilon = 1e-14);
        assert_relative_eq!(pm0[3], -1.0, epsilon = 1e-14);
        assert_relative_eq!(pd0[1], 1.0, epsilon = 1e-14);
        assert_relative_eq!(pd0[2], -3.0, epsilon = 1e-14);

        let (pm1, pd1) = legendre::lpmns(1, 3, 1.0).unwrap();
        assert!(pm1.iter().all(|v| v.is_finite() && *v == 0.0));
        assert!(pd1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn confluent_hypergeometric_identities() {
        // M(1, 2, x) = (e^x - 1) / x
        let x: f64 = 1.7;
        assert_relative_eq!(
            special::confluent_hypergeometric(1.0, 2.0, x),
            (x.exp() - 1.0) / x,
            epsilon = 1e-13
        );
        // Negative integer a = -2 polynomial: 1 - 2x/b + x^2/(b(b+1))
        let b = 1.5;
        let expected_poly = 1.0 - 2.0 * x / b + x * x / (b * (b + 1.0));
        assert_relative_eq!(
            special::confluent_hypergeometric(-2.0, b, x),
            expected_poly,
            epsilon = 1e-13
        );
        // Kummer transformation: M(a, b, -x) = e^{-x} M(b - a, b, x)
        let lhs = special::confluent_hypergeometric(0.95, 1.5, -2.5);
        let rhs = (-2.5_f64).exp() * special::confluent_hypergeometric(0.55, 1.5, 2.5);
        assert_relative_eq!(lhs, rhs, max_relative = 1e-12);
        // a >= 2 upward recurrence: M(2.5, 1.5, x) = (1 + 2x/3) e^x
        assert_relative_eq!(
            special::confluent_hypergeometric(2.5, 1.5, 1.2),
            (1.0 + 2.0 * 1.2 / 3.0) * 1.2_f64.exp(),
            max_relative = 1e-12
        );
        // Large argument asymptotic branch: M(0.5, 1.5, -z^2) ~ sqrt(pi)/(2 z)
        let z: f64 = 8.0;
        let asymp = PI.sqrt() / (2.0 * z);
        assert_relative_eq!(
            special::confluent_hypergeometric(0.5, 1.5, -z * z),
            asymp,
            max_relative = 1e-6
        );
    }

    #[test]
    fn complex_linear_solver_with_leading_zero_rhs() {
        let t = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.0, -2.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 2.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 1.5),
                Complex64::new(-2.0, 1.0),
                Complex64::new(4.0, -1.0),
            ],
        )
        .unwrap();
        let p = Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(3.0, 0.5),
        ]);
        let r = linalg::complex_matrix_inverse(3, 6, &p, &t).unwrap();
        assert_eq!(r.len(), 3);
        let residual = t.dot(&r) - &p;
        for val in &residual {
            assert!(val.norm() < 1e-13, "residual too large: {val}");
        }
    }

    #[test]
    fn int_sph_bessel_gaussian_analytic_im_s0() {
        // For df = 2, Gaussian cutoff, p = 0:
        // Im[S_0(x_g)] = -sqrt(2*pi) / (4*x_g) * M(1/2, 3/2, -2*x_g^2)
        let x_g = 1.5;
        let cfg = FractalConfig {
            cutoff: FractalCutoff::Gaussian,
            geometry: FractalGeometry::Circular,
            df: 2.0,
            ..Default::default()
        };
        let sp = bessel::int_sph_bessel(&cfg, x_g, 0).unwrap();
        let analytic_im = -(2.0 * PI).sqrt() / (4.0 * x_g)
            * special::confluent_hypergeometric(0.5, 1.5, -2.0 * x_g * x_g);
        assert_relative_eq!(sp.im, analytic_im, max_relative = 1e-3);

        // Also check FractalDimension and Exponential cutoffs satisfy unitary check
        for cutoff in [FractalCutoff::Exponential, FractalCutoff::FractalDimension] {
            let cfg_c = FractalConfig {
                cutoff,
                geometry: FractalGeometry::Circular,
                df: 1.8,
                ..Default::default()
            };
            let val = bessel::int_sph_bessel(&cfg_c, 2.0, 2).unwrap();
            assert!(val.re.is_finite() && val.im.is_finite());
        }
    }
}
