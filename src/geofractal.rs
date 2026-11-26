use std::f64::consts::PI;

use anyhow::Result;
use anyhow::anyhow;
use planetes_ext::types::RVector;

use crate::fractal::FractalCutoff;
use crate::utils::gamma::{gamma, gamma_ur};

/// Method for solving radial and angular integration
/// when computing the mean overlapping efficiency.
#[derive(PartialEq)]
pub enum IntegrationMethod {
    /// Radial (Numerical); Angular (Numerical)
    ///
    /// This is the default method.
    NumNum,
    /// Radial (Numerical); Angular (Analytical)
    NumAn,
    /// Radial (Analytical); Angular (Analytical)
    AnAn,
}

pub enum AFactor {
    Unity,
    TwoRegimes,
}

/// Computing geometrical cross-section of randomly oriented
/// fractal dust aggregates based on a statistical distribution
/// model of monomer particles as discussed in [Tazaki (2021)](
/// https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract).
///
/// # Arguments
///
/// * `iqapp` - Select a method for solving radial and angular integration
///   when computing the mean overlapping efficiency.
///
/// ```text
///         --------------------------------------------------------
///                   |    radial  (x)        |  angular (u)        |
///         ---------------------------------------------------------
///         iqapp = 1 |   numerical           |  numerical          |
///         iqapp = 2 |   numerical  ( D<=2 ) |  analytical         |
///                   |   analytical ( D> 2 ) |                     |
///         --------------------------------------------------------
/// ```
///
///
/// # Errors
/// - `NotEnoughMonomers`: The number of monomers in the aggregate is too small
/// - `ExceedsMaxFractalDimension`: The fractal dimension exceeds the maximum value (3.0).
pub fn get_geometric_cross_section_tazaki(
    method: &IntegrationMethod,
    afac: &AFactor,
    cutoff: &FractalCutoff,
    n_monomers: f64,
    prefactor: f64,
    fractal_dim: f64,
) -> Result<f64> {
    if n_monomers <= 0.9999 {
        return Err(anyhow!("NotEnoughMonomers"));
    }

    if !(0.9999..=3.0001).contains(&fractal_dim) {
        return Err(anyhow!("ExceedsMaxFractalDimension"));
    }

    let a = match afac {
        AFactor::Unity => 1.0,
        AFactor::TwoRegimes => {
            let n_threshold = (11.0 * fractal_dim - 8.5).min(8.0);
            if n_monomers < n_threshold {
                return Ok(minato_fit(n_monomers));
            }
            let sigma_threshold =
                mean_overlap_efficiency(method, cutoff, prefactor, fractal_dim, n_threshold);
            (1.0 + (n_threshold - 1.0) * sigma_threshold) * minato_fit(n_threshold)
        }
    };

    let sigma = mean_overlap_efficiency(method, cutoff, prefactor, fractal_dim, n_monomers);
    Ok(a / (1.0 + (n_monomers - 1.0) * sigma))
}

/// Minato et al. (2006) fitting formula for $N < 16$.
fn minato_fit(n_monomers: f64) -> f64 {
    12.5 * n_monomers.powf(-0.315) * (-2.53 / n_monomers.powf(0.0920)).exp()
}

/// Compute the mean overlapping efficiency
///
/// This subroutine calculations the mean overlapping efficiency: sigma
///
/// For `iqcor = 1` (Gaussian cut-off model)
///     
/// ```text
///                    xmin       /ln(xmax)
///      sigma  = --------------  |   dlnx frad(x)*S(rho),
///               16*Gamma(df/2)  /ln(xmin)
///     
///            xmin = df*(k0/PN)^(2/df); a=(df-2)/2; rho = sqrt(x/xmin)
/// ```
///     
/// For iqcor=2 [Exponential cut-off model]
///     
/// ```text
///                   xmin^2      /ln(xmax)
///      sigma  = --------------  |   dlnx frad(x)*S(rho),
///                16*Gamma(df)   /ln(xmin)
///     
///            xmin = 2*sqrt(df(df+1)/2)*(k0/PN)^(1.0/df); a=df-2; rho = x/xmin
/// ```
///     
/// For iqcor=3 [Fractal dimension cut-off model]
///     
/// ```text
///               xmin^{2/df} /ln(xmax)
///      sigma  = ----------  |  dlnx frad(x)*S(rho),
///                  16      /ln(xmin)
///     
///            xmin = 2^(df-1)*k0/PN; a=(df-2)/df; rho = (x/xmin)^(1/df)
/// ```
///     
/// where df is fractal dimension, R0 is the monomer radius, Rg is
/// the radius of gyration, frad(x) is the radial integrand function,
/// and S(rho) is the angular integral function.
///
/// frad(x)dlnx = x^{a}*exp(-x) dlnx = x^{a-1}exp(-x) dx
///     
/// ```text
///                16*rho^2  / (1/rho)
///      S(rho)  = --------  |  du fang(rho,u),
///                   pi     /0
///     
///     fang(rho,u) = [Arcsin{sqrt(1-rho^2u^2)} - rho*u*sqrt{1-rho^2u^2}]*u/sqrt(1-u^2)
/// ```
///
/// Note that the angular integral function S(rho) ≈1 when rho >> 1.
fn mean_overlap_efficiency(
    method: &IntegrationMethod,
    cutoff: &FractalCutoff,
    prefactor: f64,
    fractal_dim: f64,
    n_monomers: f64,
) -> f64 {
    let xmax = 25.0;
    let nx = 1000;

    let gamma_arg: f64;
    let xmin: f64;

    let mut sigma: f64;

    let factor: f64 = match cutoff {
        FractalCutoff::Gaussian => {
            gamma_arg = 0.5 * (fractal_dim - 2.0);
            xmin = fractal_dim * (prefactor / n_monomers).powf(2.0 / fractal_dim);
            xmin / 16.0 / gamma(0.5 * fractal_dim)
        }
        FractalCutoff::Exponential => {
            gamma_arg = fractal_dim - 2.0;
            xmin = 2.0
                * (0.5 * fractal_dim * (fractal_dim + 1.0)).sqrt()
                * (prefactor / n_monomers).powf(1.0 / fractal_dim);
            xmin * xmin / 16.0 / gamma(fractal_dim)
        }
        FractalCutoff::FractalDimension => {
            gamma_arg = (fractal_dim - 2.0) / fractal_dim;
            xmin = 2.0_f64.powf(fractal_dim - 1.0) * prefactor / n_monomers;
            xmin.powf(2.0 / fractal_dim) / 16.0
        }
    };

    if fractal_dim > 2.0 && *method == IntegrationMethod::AnAn {
        sigma = factor * gamma(gamma_arg) * gamma_ur(gamma_arg, xmin);
    } else {
        let nxf = nx as f64;
        let dlnx = (xmax / xmin).ln() / (nxf - 1.0);
        let mut x = RVector::zeros(nx);
        let mut angular_factor = RVector::ones(nx);
        sigma = 0.0;
        for i in 0..nx {
            x[i] = (xmin.ln() + (i as f64) * dlnx).exp();
            if *method == IntegrationMethod::NumNum {
                angular_factor[i] = angular_integration(cutoff, x[i], xmin, fractal_dim);
            }
        }
        for i in 0..nx - 1 {
            sigma += f64::midpoint(
                radial_integrand(gamma_arg, x[i]) * angular_factor[i],
                radial_integrand(gamma_arg, x[i + 1]) * angular_factor[i + 1],
            );
        }
        sigma *= dlnx * factor;
    }

    sigma
}

/// Radial integrand function in `ln(x)` space
fn radial_integrand(a: f64, x: f64) -> f64 {
    x.powf(a) * (-x).exp()
}

fn angular_integration(cutoff: &FractalCutoff, x: f64, xmin: f64, fractal_dim: f64) -> f64 {
    let nmax_u = 1000;

    let rho: f64 = match cutoff {
        FractalCutoff::Gaussian => (x / xmin).sqrt(),
        FractalCutoff::Exponential => x / xmin,
        FractalCutoff::FractalDimension => (x / xmin).powf(1.0 / fractal_dim),
    };
    let mut u = RVector::zeros(nmax_u);

    let mut angular_sum = 0.0;
    let umin = 0.0;
    let umax = 1.0 / rho;

    let du = (umax - umin) / (nmax_u - 1) as f64;
    for (j, val) in u.iter_mut().enumerate().take(nmax_u) {
        *val = umin + du * (j as f64);
    }
    for j in 0..nmax_u - 1 {
        angular_sum += f64::midpoint(
            angular_integrand(rho, u[j]),
            angular_integrand(rho, u[j + 1]),
        ) * du;
    }
    16.0 * rho * rho * angular_sum / PI
}

/// Angular integrand function in `u` space
fn angular_integrand(rho: f64, u: f64) -> f64 {
    if (1.0 - rho * u).abs() <= 1e-10 {
        0.0
    } else {
        let rad = (1.0 - rho * rho * u * u).max(0.0).sqrt();
        (rad.asin() - rho * u * rad) * u / (1.0 - u * u).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::fractal::FractalCutoff;

    use super::{AFactor, IntegrationMethod};
    use super::get_geometric_cross_section_tazaki;

    #[test]
    fn tazaki_cross_section_single_monomer_is_unity() {
        let g = get_geometric_cross_section_tazaki(
            &IntegrationMethod::AnAn,
            &AFactor::Unity,
            &FractalCutoff::Gaussian,
            1.0,
            1.0,
            2.0,
        )
        .unwrap();
        assert_relative_eq!(g, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn tazaki_cross_section_continuous_across_df_two() {
        // AnAn switches from numerical radial (df <= 2) to analytical Gamma (df > 2)
        let g_below = get_geometric_cross_section_tazaki(
            &IntegrationMethod::AnAn,
            &AFactor::Unity,
            &FractalCutoff::Gaussian,
            125.0,
            1.3,
            1.999,
        )
        .unwrap();
        let g_above = get_geometric_cross_section_tazaki(
            &IntegrationMethod::AnAn,
            &AFactor::Unity,
            &FractalCutoff::Gaussian,
            125.0,
            1.3,
            2.001,
        )
        .unwrap();
        assert!(g_below > 0.0 && g_below < 1.0);
        assert_relative_eq!(g_below, g_above, max_relative = 1e-2);
    }

    #[test]
    fn tazaki_num_and_an_methods_agree_for_large_aggregates() {
        let g_num = get_geometric_cross_section_tazaki(
            &IntegrationMethod::NumNum,
            &AFactor::Unity,
            &FractalCutoff::Gaussian,
            125.0,
            1.3,
            1.8,
        )
        .unwrap();
        let g_an = get_geometric_cross_section_tazaki(
            &IntegrationMethod::AnAn,
            &AFactor::Unity,
            &FractalCutoff::Gaussian,
            125.0,
            1.3,
            1.8,
        )
        .unwrap();
        assert_relative_eq!(g_num, g_an, max_relative = 0.05);
    }
}
