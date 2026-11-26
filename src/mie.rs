//! Lorenz-Mie scattering calculations for homogeneous spheres.
//!
//! Implements the method of De Rooij & Van der Stap (1984).
#![allow(clippy::inline_always)]

use std::f64::consts::PI;
use std::sync::Once;

use anyhow::Result;
use anyhow::anyhow;
use num_complex::ComplexFloat;
use num_complex::{Complex, Complex64};
use planetes_ext::types::{CVector, RMatrix, RVector};

/// Input parameters for Lorenz-Mie scattering of a homogeneous sphere.
#[derive(Debug)]
pub struct MieConfig {
    pub n_angles: usize,
    pub delta: f64,
    pub theta_min: f64,
    pub theta_max: f64,
    pub step: f64,
    pub wavelength: f64,
    pub refractive_index: Complex64,
    pub radius: f64,
}

/// Output cross sections, efficiencies, and Mueller elements from Lorenz-Mie scattering.
#[derive(Debug)]
pub struct MieResult {
    pub cos_theta: RVector,
    pub theta_deg: RVector,
    pub c_sca: f64,
    pub c_ext: f64,
    pub q_sca: f64,
    pub q_ext: f64,
    pub albedo: f64,
    pub g: f64,
    pub r_eff: f64,
    pub x_eff: f64,
    pub volume: f64,
    pub f11: RVector,
    pub f12: RVector,
    pub f33: RVector,
    pub f34: RVector,
}

/// Computes Lorenz-Mie cross sections and Mueller scattering matrix elements using the method of
/// De Rooij & Van der Stap (1984).
///
/// # Errors
/// Returns an error if the configuration parameters are non-positive or if the number of
/// scattering angles exceeds the internal limit (6000).
pub fn de_rooij_1984(mie_cfg: &MieConfig) -> Result<MieResult> {
    let m = mie_cfg.refractive_index.conj();
    get_scattering_matrix(mie_cfg, m)
}

#[allow(clippy::similar_names)]
#[inline(always)]
fn get_scattering_matrix(mie_cfg: &MieConfig, m: Complex64) -> Result<MieResult> {
    const DEG_TO_RAD: f64 = PI / 180.0;

    if mie_cfg.radius <= 0.0
        || mie_cfg.wavelength <= 0.0
        || mie_cfg.step <= 0.0
        || mie_cfg.n_angles < 2
    {
        return Err(anyhow!("InvalidMieConfig"));
    }

    let n_angles = ((mie_cfg.theta_max - mie_cfg.theta_min) / mie_cfg.step).round() as usize + 1;
    if n_angles > 6000 {
        return Err(anyhow!("ScatteringAnglesOverflow"));
    }

    let mut mie_result = MieResult {
        cos_theta: RVector::zeros(n_angles),
        theta_deg: RVector::zeros(n_angles),
        c_sca: 0.0,
        c_ext: 0.0,
        q_sca: 0.0,
        q_ext: 0.0,
        albedo: 0.0,
        g: 0.0,
        r_eff: 0.0,
        x_eff: 0.0,
        volume: 0.0,
        f11: RVector::zeros(mie_cfg.n_angles),
        f12: RVector::zeros(mie_cfg.n_angles),
        f33: RVector::zeros(mie_cfg.n_angles),
        f34: RVector::zeros(mie_cfg.n_angles),
    };

    let mut fac_90 = 1.0;
    let imag_unit = Complex::new(0.0, 1.0);
    let symmetric = is_symmetric_grid(mie_cfg.theta_min, mie_cfg.theta_max, mie_cfg.step);

    let wave_number = 2.0 * PI / mie_cfg.wavelength;
    let cross_section_scale = mie_cfg.wavelength * mie_cfg.wavelength / (2.0 * PI);
    let radius = mie_cfg.radius;
    let x = wave_number * radius;

    let nmax = (x + 4.05 * x.powf(1.0 / 3.0) + 20.0) as usize;
    let n_psi = nmax + 60;
    let z_abs = x * m.abs();
    // Ensure downward recurrence for D_n(mx) starts well above nmax even when |m| < 1
    // (De Rooij & Van der Stap 1984; Bohren & Huffman 1983, App. A).
    let n_d = ((z_abs + 4.05 * z_abs.powf(1.0 / 3.0) + 70.0) as usize).max(nmax + 70);

    let (psi, chi, d_log) = riccati_bessel_and_log_deriv(m, x, n_psi, nmax, n_d);
    let (an, bn) = mie_coefficients(m, x, &psi, &chi, &d_log, nmax);

    let mut forward_weight = RVector::zeros(nmax);
    let mut backward_weight = RVector::zeros(nmax);
    for n in 0..nmax {
        let nf = n as f64 + 1.0;
        forward_weight[n] = (2.0 * nf + 1.0) / (nf * (nf + 1.0));
        backward_weight[n] = if n % 2 == 0 {
            -forward_weight[n]
        } else {
            forward_weight[n]
        };
    }

    let mut c_ext_sum = 0.0;
    let mut c_sca_sum = 0.0;
    let mut nstop = nmax;

    for n in 0..nmax {
        let nf = n as f64;
        let term = (2.0 * (nf + 1.0) + 1.0) * (an[n].norm_sqr() + bn[n].norm_sqr());
        c_sca_sum += term;
        c_ext_sum += (2.0 * (nf + 1.0) + 1.0) * (an[n] + bn[n]).re;
        if term < mie_cfg.delta {
            nstop = n;
            break;
        }
    }

    let n_terms = nstop + 1;
    if n_terms > nmax {
        static WARNED: Once = Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "Warning: mie: sum not converged at x = {x:.3e}, m = {:.3e} + {:.3e}i; a priori {nmax} terms used (further occurrences not reported)",
                m.re, m.im
            );
        });
    }
    let n_terms = n_terms.min(nmax);

    for iang in 0..n_angles {
        let theta = mie_cfg.theta_min + (iang as f64) * mie_cfg.step;
        mie_result.cos_theta[n_angles - 1 - iang] = (DEG_TO_RAD * theta).cos();
        mie_result.theta_deg[iang] = theta;
    }

    let geom_cross = PI * radius.powi(2);
    mie_result.g = geom_cross;
    mie_result.volume = (4.0 / 3.0) * PI * radius.powi(3);
    mie_result.r_eff = radius;
    mie_result.x_eff = x;

    let mut f_buf = RMatrix::zeros((4, n_angles));
    let pitau_len = n_terms.max(2);
    let mut pi = RVector::zeros(pitau_len);
    let mut tau = RVector::zeros(pitau_len);

    let mut n_half: usize = 0;
    if symmetric {
        if n_angles % 2 == 1 {
            n_half = n_angles.div_ceil(2);
            fac_90 = 0.5;
        } else {
            n_half = n_angles / 2;
        }

        for j in 0..n_half {
            pitau_into(mie_result.cos_theta[j], &mut pi, &mut tau);
            let mut s_plus_f = Complex::new(0.0, 0.0);
            let mut s_min_f = Complex::new(0.0, 0.0);
            let mut s_plus_b = Complex::new(0.0, 0.0);
            let mut s_min_b = Complex::new(0.0, 0.0);

            for n in 0..n_terms {
                s_plus_f += forward_weight[n] * (an[n] + bn[n]) * (pi[n] + tau[n]);
                s_min_f += forward_weight[n] * (an[n] - bn[n]) * (pi[n] - tau[n]);
                s_plus_b += backward_weight[n] * (an[n] + bn[n]) * (pi[n] - tau[n]);
                s_min_b += backward_weight[n] * (an[n] - bn[n]) * (pi[n] + tau[n]);
            }
            let conj_s_plus_f = s_plus_f.conj();
            let conj_s_min_f = s_min_f.conj();
            let conj_s_plus_b = s_plus_b.conj();
            let conj_s_min_b = s_min_b.conj();

            let k = n_angles - 1 - j;

            f_buf[[0, j]] = (s_plus_f * conj_s_plus_f + s_min_f * conj_s_min_f).re;
            f_buf[[1, j]] = -(s_min_f * conj_s_plus_f + s_plus_f * conj_s_min_f).re;
            f_buf[[2, j]] = (s_plus_f * conj_s_plus_f - s_min_f * conj_s_min_f).re;
            f_buf[[3, j]] = (imag_unit * (s_min_f * conj_s_plus_f - s_plus_f * conj_s_min_f)).re;

            f_buf[[0, k]] += (s_plus_b * conj_s_plus_b + s_min_b * conj_s_min_b).re;
            f_buf[[1, k]] -= (s_min_b * conj_s_plus_b + s_plus_b * conj_s_min_b).re;
            f_buf[[2, k]] += (s_plus_b * conj_s_plus_b - s_min_b * conj_s_min_b).re;
            f_buf[[3, k]] += (imag_unit * (s_min_b * conj_s_plus_b - s_plus_b * conj_s_min_b)).re;
        }
    } else {
        for j in 0..n_angles {
            pitau_into(mie_result.cos_theta[j], &mut pi, &mut tau);
            let mut s_plus_f = Complex::new(0.0, 0.0);
            let mut s_min_f = Complex::new(0.0, 0.0);

            for n in 0..n_terms {
                s_plus_f += forward_weight[n] * (an[n] + bn[n]) * (pi[n] + tau[n]);
                s_min_f += forward_weight[n] * (an[n] - bn[n]) * (pi[n] - tau[n]);
            }
            let conj_s_plus_f = s_plus_f.conj();
            let conj_s_min_f = s_min_f.conj();

            f_buf[[0, j]] = (s_plus_f * conj_s_plus_f + s_min_f * conj_s_min_f).re;
            f_buf[[1, j]] = -(s_min_f * conj_s_plus_f + s_plus_f * conj_s_min_f).re;
            f_buf[[2, j]] = (s_plus_f * conj_s_plus_f - s_min_f * conj_s_min_f).re;
            f_buf[[3, j]] = (imag_unit * (s_min_f * conj_s_plus_f - s_plus_f * conj_s_min_f)).re;
        }
    }

    for j in 0..n_angles {
        for k in 0..4 {
            f_buf[[k, j]] /= 2.0 * c_sca_sum;
        }
    }

    if symmetric {
        for k in 0..4 {
            f_buf[[k, n_half - 1]] *= fac_90;
        }
    }

    mie_result.c_sca = c_sca_sum * cross_section_scale;
    mie_result.c_ext = c_ext_sum * cross_section_scale;
    mie_result.q_sca = mie_result.c_sca / geom_cross;
    mie_result.q_ext = mie_result.c_ext / geom_cross;
    mie_result.albedo = mie_result.c_sca / mie_result.c_ext;

    for i in 0..mie_cfg.n_angles {
        let rev = mie_cfg.n_angles - i - 1;
        mie_result.f11[i] = f_buf[[0, rev]];
        mie_result.f12[i] = f_buf[[1, rev]];
        mie_result.f33[i] = f_buf[[2, rev]];
        mie_result.f34[i] = f_buf[[3, rev]];
    }

    Ok(mie_result)
}

fn is_symmetric_grid(theta_min: f64, theta_max: f64, step: f64) -> bool {
    let eps = 1e-6;
    let half_eps = 0.5 * eps;

    ((180.0 - theta_min - theta_max).abs() < eps)
        && ((theta_max - theta_min + half_eps).rem_euclid(step) < eps)
}

#[allow(clippy::similar_names)]
#[inline(always)]
fn riccati_bessel_and_log_deriv(
    m: Complex64,
    x: f64,
    n_psi: usize,
    nmax: usize,
    n_d: usize,
) -> (RVector, RVector, CVector) {
    let z = m * x;
    let inv_z = 1.0 / z;
    let inv_x = 1.0 / x;

    let sin_x = x.sin();
    let cos_x = x.cos();

    let mut psi = RVector::zeros(n_psi + 1);
    let mut chi = RVector::zeros(nmax + 2);
    let mut d_log = CVector::zeros(n_d.max(nmax));

    for n in (0..n_psi).rev() {
        let nf = n as f64;
        psi[n] = 1.0 / ((2.0 * nf + 1.0) / x - psi[n + 1]);
    }

    for n in (0..n_d - 1).rev() {
        let nf = n as f64;
        let zn_1 = (nf + 2.0) * inv_z;
        d_log[n] = zn_1 - 1.0 / (d_log[n + 1] + zn_1);
    }

    psi[0] = sin_x;
    let psi_1 = psi[0] * inv_x - cos_x;
    if psi_1.abs() > 1e-4 {
        psi[1] = psi_1;
        for n in 2..=nmax {
            psi[n] *= psi[n - 1];
        }
    } else {
        for n in 1..=nmax {
            psi[n] *= psi[n - 1];
        }
    }

    chi[0] = cos_x;
    chi[1] = chi[0] * inv_x + sin_x;
    for n in 1..nmax {
        let nf = n as f64;
        chi[n + 1] = (2.0 * nf + 1.0) * chi[n] * inv_x - chi[n - 1];
    }

    (psi, chi, d_log)
}

#[allow(clippy::similar_names)]
#[inline(always)]
fn mie_coefficients(
    m: Complex64,
    x: f64,
    psi: &RVector,
    chi: &RVector,
    d_log: &CVector,
    nmax: usize,
) -> (CVector, CVector) {
    let inv_m = 1.0 / m;
    let inv_x = 1.0 / x;

    let mut an = CVector::zeros(nmax);
    let mut bn = CVector::zeros(nmax);

    for n in 0..nmax {
        let nf = n as f64;
        let zeta_n = Complex::new(psi[n + 1], chi[n + 1]);
        let zeta_nm1 = Complex::new(psi[n], chi[n]);
        let xn = (nf + 1.0) * inv_x;
        let term_a = d_log[n] * inv_m + xn;
        an[n] = (term_a * psi[n + 1] - psi[n]) / (term_a * zeta_n - zeta_nm1);
        let term_b = d_log[n] * m + xn;
        bn[n] = (term_b * psi[n + 1] - psi[n]) / (term_b * zeta_n - zeta_nm1);
    }

    (an, bn)
}

#[inline(always)]
fn pitau_into(cos_theta: f64, pi: &mut RVector, tau: &mut RVector) {
    let nmax = pi.len();
    pi[0] = 1.0;
    pi[1] = 3.0 * cos_theta;
    let mut delta = 3.0 * cos_theta * cos_theta - 1.0;

    tau[0] = cos_theta;
    tau[1] = 2.0 * delta - 1.0;

    for n in 1..(nmax - 1) {
        let nf = n as f64;
        pi[n + 1] = (nf + 2.0) / (nf + 1.0) * delta + cos_theta * pi[n];
        delta = cos_theta * pi[n + 1] - pi[n];
        tau[n + 1] = (nf + 2.0) * delta - pi[n];
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use num_complex::Complex64;

    use super::MieConfig;
    use super::de_rooij_1984;

    fn make_cfg(
        radius: f64,
        wavelength: f64,
        refractive_index: Complex64,
        n_angles: usize,
    ) -> MieConfig {
        let nf = n_angles as f64;
        let theta_min = 180.0 * 0.5 / nf;
        let theta_max = 180.0 * (nf - 0.5) / nf;
        MieConfig {
            n_angles,
            delta: 1e-14,
            theta_min,
            theta_max,
            step: (theta_max - theta_min) / (nf - 1.0),
            wavelength,
            refractive_index,
            radius,
        }
    }

    #[test]
    fn mie_mueller_identity_and_rayleigh_limit() {
        let cfg = make_cfg(1.0, 2.0 * PI / 3.0, Complex64::new(1.5, 0.1), 180);
        let res = de_rooij_1984(&cfg).unwrap();
        assert_relative_eq!(res.q_ext, 3.021_998_248, max_relative = 1e-7);
        assert_relative_eq!(res.q_sca, 2.126_748_708, max_relative = 1e-7);
        for j in 0..cfg.n_angles {
            let lhs = res.f11[j].powi(2);
            let rhs = res.f12[j].powi(2) + res.f33[j].powi(2) + res.f34[j].powi(2);
            assert_relative_eq!(lhs, rhs, max_relative = 1e-12);
        }

        // Deep Rayleigh limit (x = 2*pi*1e-4)
        let ray = make_cfg(1e-4, 1.0, Complex64::new(1.5, 0.01), 180);
        let rayleigh = de_rooij_1984(&ray).unwrap();
        let x = 2.0 * PI * 1e-4;
        let m = ray.refractive_index;
        let alpha_fac = (m * m - 1.0) / (m * m + 2.0);
        let q_sca_ray = 8.0 / 3.0 * x.powi(4) * alpha_fac.norm_sqr();
        let q_abs_ray = 4.0 * x * alpha_fac.im;
        assert_relative_eq!(rayleigh.q_sca, q_sca_ray, max_relative = 1e-4);
        assert_relative_eq!(
            rayleigh.q_ext - rayleigh.q_sca,
            q_abs_ray,
            max_relative = 1e-4
        );
    }

    #[test]
    fn mie_sub_unity_refractive_index_large_x() {
        // Regression test for n_d < nmax when |m| < 1 (e.g. CO2 ice near 4.2 um).
        let x = 200.0;
        let cfg = make_cfg(1.0, 2.0 * PI / x, Complex64::new(0.5, 0.05), 180);
        let res = de_rooij_1984(&cfg).unwrap();
        assert_relative_eq!(res.q_ext, 2.043_146_32, max_relative = 1e-6);
        assert_relative_eq!(res.q_sca, 1.671_128_76, max_relative = 1e-6);
    }
}
