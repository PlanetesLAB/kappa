//! Distribution of Hollow Spheres (DHS) concentric-sphere scattering solver.
//!
//! Implements the stratified-sphere algorithm of Toon & Ackerman (1981).
#![allow(clippy::inline_always)]

use std::f64::consts::PI;

use anyhow::Result;
use anyhow::anyhow;
use num_complex::ComplexFloat;
use num_complex::{Complex, Complex64};
use planetes_ext::types::{CMatrix, CVector, RMatrix, RVector};

/// Input parameters for electromagnetic scattering by a concentric core-shell sphere.
#[derive(Debug)]
pub struct DhsConfig<'a> {
    pub r_core: f64,
    pub r_shell: f64,
    pub wave_number: f64,
    pub m_shell: Complex64,
    pub m_core: Complex64,
    pub cos_theta: &'a RVector,
    pub n_angles: usize,
    pub max_angle: usize,
}

/// Output efficiencies and unnormalized scattering intensities from Toon & Ackerman (1981).
#[derive(Debug)]
pub struct DhsResult {
    pub q_ext: f64,
    pub q_sca: f64,
    pub q_back: f64,
    pub g_q_sca: f64,
    pub i1: RMatrix,
    pub i2: RMatrix,
    pub s21: RMatrix,
    pub d21: RMatrix,
}

/// Computes electromagnetic scattering by a concentric core-shell sphere using Toon & Ackerman
/// (1981), *Appl. Opt.* 20, 3657.
///
/// Refractive indices follow the $e^{+i\omega t}$ convention ($\operatorname{Im}(m) \le 0$).
///
/// # Errors
/// Returns an error if input radii, wavenumber, refractive indices, or angle cosines are invalid,
/// or if the required series order exceeds internal dimension bounds.
#[allow(clippy::similar_names)]
#[inline(always)]
pub fn toon_ackerman_1981(dhs_cfg: &DhsConfig) -> Result<DhsResult> {
    let max_terms = 300_000;
    let max_angles_limit = 1440;
    let tol = 1e-6;
    let imag_unit = Complex::new(0.0, 1.0);

    let x_shell = dhs_cfg.r_shell * dhs_cfg.wave_number;
    let x_core = dhs_cfg.r_core * dhs_cfg.wave_number;

    // Ensure nmx_1/nmx_2 cover the Mie expansion order even when |m_shell| <= 1.
    let max_arg = x_shell.max(x_shell * dhs_cfg.m_shell.abs());
    let nmx_2 = ((max_arg + 4.05 * max_arg.powf(1.0 / 3.0) + 20.0).ceil() as usize).max(135);
    let nmx_1 = ((1.1 * max_arg).ceil() as usize).max(nmx_2 + 15).max(150);

    let mut w = CMatrix::zeros((3, nmx_1 + 1));
    let mut acap = CVector::zeros(nmx_1 + 1);

    if dhs_cfg.wave_number <= 0.0 {
        return Err(anyhow!("InvalidWaveNumber"));
    }
    if dhs_cfg.r_shell <= 0.0 {
        return Err(anyhow!("InvalidShellRadius"));
    }
    if dhs_cfg.r_core <= 0.0 || dhs_cfg.r_core > dhs_cfg.r_shell {
        return Err(anyhow!("InvalidCoreRadius"));
    }
    if dhs_cfg.m_shell.re <= 0.0 || dhs_cfg.m_shell.im > 0.0 {
        return Err(anyhow!("InvalidRefractiveIndex"));
    }
    if dhs_cfg.m_core.re <= 0.0 || dhs_cfg.m_core.im > 0.0 {
        return Err(anyhow!("InvalidRefractiveIndex"));
    }
    if dhs_cfg.n_angles > max_angles_limit
        || dhs_cfg.n_angles > dhs_cfg.max_angle
        || dhs_cfg.cos_theta.len() < dhs_cfg.n_angles
    {
        return Err(anyhow!("`n_angles` is too large."));
    }
    if nmx_1 + 1 > max_terms {
        return Err(anyhow!("`nmx_1` is too large."));
    }

    for j in 0..dhs_cfg.n_angles {
        if dhs_cfg.cos_theta[j] < -tol || dhs_cfg.cos_theta[j] > 1.0 + tol {
            return Err(anyhow!("`cos_theta` is out of bounds."));
        }
    }

    let k1 = dhs_cfg.m_core * dhs_cfg.wave_number;
    let k2 = dhs_cfg.m_shell * dhs_cfg.wave_number;
    let k3 = Complex::new(dhs_cfg.wave_number, 0.0);

    let z = [
        dhs_cfg.m_shell * x_shell,
        Complex::new(x_shell, 0.0),
        dhs_cfg.m_core * x_core,
        dhs_cfg.m_shell * x_core,
    ];

    let x0 = z[0].re;
    let y0 = z[0].im;
    let x3 = z[3].re;
    let y3 = z[3].im;

    let inv_x = 1.0 / x_shell;
    let inv_z0 = 1.0 / (dhs_cfg.m_shell * x_shell);

    for nn in (0..nmx_1).rev() {
        let nnf = nn as f64;
        acap[nn] = ((nnf + 2.0) * inv_z0) - (1.0 / ((nnf + 2.0) * inv_z0 + acap[nn + 1]));
        for m in 0..3 {
            w[[m, nn]] = (nnf + 2.0) / z[m + 1] - (1.0 / ((nnf + 2.0) / z[m + 1] + w[[m, nn + 1]]));
        }
    }

    let mut sin2_theta = RVector::zeros(dhs_cfg.n_angles);
    let mut pi = RMatrix::zeros((dhs_cfg.n_angles, 3));
    let mut tau = RMatrix::zeros((dhs_cfg.n_angles, 3));

    for j in 0..dhs_cfg.n_angles {
        sin2_theta[j] = 1.0 - dhs_cfg.cos_theta[j] * dhs_cfg.cos_theta[j];
        pi[[j, 0]] = 0.0;
        pi[[j, 1]] = 1.0;
        tau[[j, 0]] = 0.0;
        tau[[j, 1]] = dhs_cfg.cos_theta[j];
    }

    let cos_x = x_shell.cos();
    let sin_x = x_shell.sin();

    let mut wm1 = Complex::new(cos_x, -sin_x);
    let mut wfn = [Complex::new(sin_x, cos_x), Complex::new(0.0, 0.0)];
    wfn[1] = inv_x * wfn[0] - wm1;

    let mut dhs_result = DhsResult {
        q_ext: 0.0,
        q_sca: 0.0,
        q_back: 0.0,
        g_q_sca: 0.0,
        i1: RMatrix::zeros((dhs_cfg.n_angles, 2)),
        i2: RMatrix::zeros((dhs_cfg.n_angles, 2)),
        s21: RMatrix::zeros((dhs_cfg.n_angles, 2)),
        d21: RMatrix::zeros((dhs_cfg.n_angles, 2)),
    };

    let mut n = 0;
    let sin_x0 = x0.sin();
    let sin_x3 = x3.sin();
    let cos_x0 = x0.cos();
    let cos_x3 = x3.cos();

    let e_y0 = y0.exp();
    let e2_y0 = e_y0 * e_y0;
    let e_y3 = y3.exp();
    let ey0_my3 = (y0 - y3).exp();
    let ey0_py3 = e_y0 * e_y3;

    let aa = sin_x3 * (ey0_py3 + ey0_my3);
    let bb = cos_x3 * (ey0_py3 - ey0_my3);
    let cc = sin_x0 * (e2_y0 + 1.0);
    let dd = cos_x0 * (e2_y0 - 1.0);

    let denom = 1.0 + e2_y0 * (4.0 * sin_x0 * sin_x0 - 2.0 + e2_y0);
    let mut temp = Complex::new((aa * cc + bb * dd) / denom, (bb * cc - aa * dd) / denom);
    temp *= (acap[n] + 1.0 / z[0]) / (w[[2, n]] + 1.0 / z[3]);
    let mut tempsq = temp * temp;

    let mut p23_h23 = 0.5 + Complex::new(sin_x3 * sin_x3 - 0.5, cos_x3 * sin_x3) * e_y3 * e_y3;
    let mut p23_h20 = 0.5
        * Complex::new(
            sin_x0 * sin_x3 - cos_x0 * cos_x3,
            sin_x0 * cos_x3 + cos_x0 * sin_x3,
        )
        * ey0_py3
        + 0.5
            * Complex::new(
                sin_x0 * sin_x3 + cos_x0 * cos_x3,
                -sin_x0 * cos_x3 + cos_x0 * sin_x3,
            )
            * ey0_my3;

    let mut d_h0 = z[0] / (1.0 + imag_unit * z[0]) - 1.0 / z[0];
    let mut d_h1 = z[1] / (1.0 + imag_unit * z[1]) - 1.0 / z[1];
    let mut d_h3 = z[3] / (1.0 + imag_unit * z[3]) - 1.0 / z[3];

    p23_h23 /= (d_h3 + 1.0 / z[3]) * (w[[2, n]] + 1.0 / z[3]);
    p23_h20 /= (d_h0 + 1.0 / z[0]) * (w[[2, n]] + 1.0 / z[3]);

    let mut u = [Complex::new(0.0, 0.0); 8];

    u[0] = k3 * acap[n] - k2 * w[[0, n]];
    u[1] = k3 * acap[n] - k2 * d_h1;
    u[2] = k2 * acap[n] - k3 * w[[0, n]];
    u[3] = k2 * acap[n] - k3 * d_h1;
    u[4] = k1 * w[[2, n]] - k2 * w[[1, n]];
    u[5] = k2 * w[[2, n]] - k1 * w[[1, n]];
    u[6] = -imag_unit * (temp * p23_h20 - p23_h23);
    u[7] = wfn[1].re / wfn[1];

    let mut acoe = u[7] * (u[0] * u[4] * u[6] + k1 * u[0] - tempsq * k3 * u[4])
        / (u[1] * u[4] * u[6] + k1 * u[1] - tempsq * k3 * u[4]);
    let mut bcoe = u[7] * (u[2] * u[5] * u[6] + k2 * u[2] - tempsq * k2 * u[5])
        / (u[3] * u[5] * u[6] + k2 * u[3] - tempsq * k2 * u[5]);

    let mut prev_acoe = acoe;
    let mut prev_bcoe = bcoe;

    let mut dqext = 3.0 * (acoe.re + bcoe.re);
    let mut dqsca = 3.0 * (acoe.norm_sqr() + bcoe.norm_sqr());
    let mut dgqsc = 0.0;
    let mut sback = 3.0 * (acoe - bcoe);
    let mut alt_sign = 1.0;

    let mut ac = 1.5 * acoe;
    let mut bc = 1.5 * bcoe;

    let mut s1 = CMatrix::zeros((dhs_cfg.n_angles, 2));
    let mut s2 = CMatrix::zeros((dhs_cfg.n_angles, 2));

    for j in 0..dhs_cfg.n_angles {
        s1[[j, 0]] = ac * pi[[j, 1]] + bc * tau[[j, 1]];
        s1[[j, 1]] = ac * pi[[j, 1]] - bc * tau[[j, 1]];
        s2[[j, 0]] = bc * pi[[j, 1]] + ac * tau[[j, 1]];
        s2[[j, 1]] = bc * pi[[j, 1]] - ac * tau[[j, 1]];
    }

    n = 2;
    let mut term_mag = 1.5;

    while term_mag >= 1e-14 {
        let nf = n as f64;
        let two_n_m1 = 2.0 * nf - 1.0;
        let n_m1 = nf - 1.0;

        for j in 0..dhs_cfg.n_angles {
            pi[[j, 2]] = (two_n_m1 * pi[[j, 1]] * dhs_cfg.cos_theta[j] - nf * pi[[j, 0]]) / n_m1;
            tau[[j, 2]] = dhs_cfg.cos_theta[j] * (pi[[j, 2]] - pi[[j, 0]])
                - two_n_m1 * sin2_theta[j] * pi[[j, 1]]
                + tau[[j, 0]];
        }

        wm1 = wfn[0];
        wfn[0] = wfn[1];
        wfn[1] = two_n_m1 * inv_x * wfn[0] - wm1;

        d_h0 = -nf / z[0] + 1.0 / (nf / z[0] - d_h0);
        d_h1 = -nf / z[1] + 1.0 / (nf / z[1] - d_h1);
        d_h3 = -nf / z[3] + 1.0 / (nf / z[3] - d_h3);
        p23_h23 /= (d_h3 + nf / z[3]) * (w[[2, n - 1]] + nf / z[3]);
        p23_h20 /= (d_h0 + nf / z[0]) * (w[[2, n - 1]] + nf / z[3]);
        temp *= (acap[n - 1] + nf / z[0]) / (w[[2, n - 1]] + nf / z[3]);
        tempsq = temp * temp;

        u[0] = k3 * acap[n - 1] - k2 * w[[0, n - 1]];
        u[1] = k3 * acap[n - 1] - k2 * d_h1;
        u[2] = k2 * acap[n - 1] - k3 * w[[0, n - 1]];
        u[3] = k2 * acap[n - 1] - k3 * d_h1;
        u[4] = k1 * w[[2, n - 1]] - k2 * w[[1, n - 1]];
        u[5] = k2 * w[[2, n - 1]] - k1 * w[[1, n - 1]];
        u[6] = -imag_unit * (temp * p23_h20 - p23_h23);
        u[7] = wfn[1].re / wfn[1];

        acoe = u[7] * (u[0] * u[4] * u[6] + k1 * u[0] - tempsq * k3 * u[4])
            / (u[1] * u[4] * u[6] + k1 * u[1] - tempsq * k3 * u[4]);
        bcoe = u[7] * (u[2] * u[5] * u[6] + k2 * u[2] - tempsq * k2 * u[5])
            / (u[3] * u[5] * u[6] + k2 * u[3] - tempsq * k2 * u[5]);

        let c_asym1 = two_n_m1 / (nf * n_m1);
        let c_asym2 = n_m1 * (nf + 1.0) / nf;
        dgqsc += c_asym2
            * (prev_acoe.re * acoe.re
                + prev_acoe.im * acoe.im
                + prev_bcoe.re * bcoe.re
                + prev_bcoe.im * bcoe.im)
            + c_asym1 * (prev_acoe.re * prev_bcoe.re + prev_acoe.im * prev_bcoe.im);

        let two_n_p1 = 2.0 * nf + 1.0;
        dqext += two_n_p1 * (acoe.re + bcoe.re);
        term_mag = acoe.norm_sqr() + bcoe.norm_sqr();
        dqsca += two_n_p1 * term_mag;
        alt_sign = -alt_sign;
        sback += two_n_p1 * alt_sign * (acoe - bcoe);

        let ang_weight = two_n_p1 / (nf * (nf + 1.0));
        ac = ang_weight * acoe;
        bc = ang_weight * bcoe;

        for j in 0..dhs_cfg.n_angles {
            s1[[j, 0]] += ac * pi[[j, 2]] + bc * tau[[j, 2]];
            s2[[j, 0]] += bc * pi[[j, 2]] + ac * tau[[j, 2]];
        }

        if n % 2 == 0 {
            for j in 0..dhs_cfg.n_angles {
                s1[[j, 1]] += bc * tau[[j, 2]] - ac * pi[[j, 2]];
                s2[[j, 1]] += ac * tau[[j, 2]] - bc * pi[[j, 2]];
            }
        } else {
            for j in 0..dhs_cfg.n_angles {
                s1[[j, 1]] += ac * pi[[j, 2]] - bc * tau[[j, 2]];
                s2[[j, 1]] += bc * pi[[j, 2]] - ac * tau[[j, 2]];
            }
        }

        n += 1;
        if n > nmx_2 {
            return Err(anyhow!("InsufficentDimensions"));
        }

        for j in 0..dhs_cfg.n_angles {
            pi[[j, 0]] = pi[[j, 1]];
            pi[[j, 1]] = pi[[j, 2]];
            tau[[j, 0]] = tau[[j, 1]];
            tau[[j, 1]] = tau[[j, 2]];
        }

        prev_acoe = acoe;
        prev_bcoe = bcoe;
    }

    for j in 0..dhs_cfg.n_angles {
        for k in 0..2 {
            dhs_result.i1[[j, k]] = s1[[j, k]].norm_sqr();
            dhs_result.i2[[j, k]] = s2[[j, k]].norm_sqr();
            dhs_result.s21[[j, k]] = s1[[j, k]].re * s2[[j, k]].re + s1[[j, k]].im * s2[[j, k]].im;
            dhs_result.d21[[j, k]] = s2[[j, k]].re * s1[[j, k]].im - s1[[j, k]].re * s2[[j, k]].im;
        }
    }

    let q_scale = 2.0 * inv_x * inv_x;
    dhs_result.q_ext = q_scale * dqext;
    dhs_result.q_sca = q_scale * dqsca;
    dhs_result.g_q_sca = 2.0 * q_scale * dgqsc;
    sback *= 0.5;
    dhs_result.q_back = sback.norm_sqr() / (PI * x_shell * x_shell);

    Ok(dhs_result)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use planetes_ext::types::RVector;

    use crate::mie::MieConfig;
    use crate::mie::de_rooij_1984;

    use super::DhsConfig;
    use super::toon_ackerman_1981;

    #[test]
    fn dhs_homogeneous_limit_matches_mie_exactly() {
        let r_shell = 1.0;
        let x = 3.0;
        let wavelength = 2.0 * PI / x;
        let wave_number = x / r_shell;
        let m_pos = Complex64::new(1.5, 0.1);
        let m_neg = m_pos.conj();
        let n_angles = 180;
        let half = n_angles / 2;

        let cos_theta = RVector::from_iter(
            (0..half).map(|j| ((j as f64 + 0.5) / half as f64 * PI / 2.0).cos()),
        );
        let dhs = toon_ackerman_1981(&DhsConfig {
            r_core: 0.5 * r_shell,
            r_shell,
            wave_number,
            m_shell: m_neg,
            m_core: m_neg,
            cos_theta: &cos_theta,
            n_angles: half,
            max_angle: n_angles,
        })
        .unwrap();

        let nf = n_angles as f64;
        let theta_min = 180.0 * 0.5 / nf;
        let theta_max = 180.0 * (nf - 0.5) / nf;
        let mie = de_rooij_1984(&MieConfig {
            n_angles,
            delta: 1e-14,
            theta_min,
            theta_max,
            step: (theta_max - theta_min) / (nf - 1.0),
            wavelength,
            refractive_index: m_pos,
            radius: r_shell,
        })
        .unwrap();

        assert_relative_eq!(dhs.q_ext, mie.q_ext, max_relative = 1e-10);
        assert_relative_eq!(dhs.q_sca, mie.q_sca, max_relative = 1e-10);

        let c_sca = dhs.q_sca * PI * r_shell * r_shell;
        let factor = 2.0 * PI / (c_sca * wave_number * wave_number);
        for j in 0..half {
            for (k, idx) in [(0, j), (1, n_angles - 1 - j)] {
                let f11 = (dhs.i2[[j, k]] + dhs.i1[[j, k]]) * factor;
                let f12 = (dhs.i2[[j, k]] - dhs.i1[[j, k]]) * factor;
                let f33 = 2.0 * dhs.s21[[j, k]] * factor;
                let f34 = -2.0 * dhs.d21[[j, k]] * factor;
                assert_relative_eq!(f11, mie.f11[idx], max_relative = 1e-9);
                assert_relative_eq!(f12, mie.f12[idx], max_relative = 1e-9);
                assert_relative_eq!(f33, mie.f33[idx], max_relative = 1e-9);
                assert_relative_eq!(f34, mie.f34[idx], max_relative = 1e-8);
            }
        }
    }

    #[test]
    fn dhs_hollow_sphere_satisfies_mueller_identity_and_sub_unity_m() {
        let cos_theta = RVector::from_vec(vec![0.99, 0.5, 0.1]);
        let dhs = toon_ackerman_1981(&DhsConfig {
            r_core: 0.6,
            r_shell: 1.0,
            wave_number: 200.0,
            m_shell: Complex64::new(0.5, -0.05),
            m_core: Complex64::new(1.0, 0.0),
            cos_theta: &cos_theta,
            n_angles: 3,
            max_angle: 6,
        })
        .unwrap();
        assert!(dhs.q_ext > 0.0 && dhs.q_sca > 0.0 && dhs.q_ext >= dhs.q_sca);
        for j in 0..3 {
            for k in 0..2 {
                let s11 = f64::midpoint(dhs.i2[[j, k]], dhs.i1[[j, k]]);
                let s12 = 0.5 * (dhs.i2[[j, k]] - dhs.i1[[j, k]]);
                let s33 = dhs.s21[[j, k]];
                let s34 = -dhs.d21[[j, k]];
                assert_relative_eq!(
                    s11 * s11,
                    s12 * s12 + s33 * s33 + s34 * s34,
                    max_relative = 1e-12
                );
            }
        }
    }
}
