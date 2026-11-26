//! Geometric cross-section models for randomly oriented fractal dust
//! aggregates.
//!
//! Implements the statistical monomer-overlap cross-section model of
//! [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract)
//! together with the small-$N$ empirical fit of
//! [Minato et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..701M/abstract).
//! The entry point [`get_geometric_cross_section_tazaki`] returns the
//! dimensionless normalized cross section $G / (N \pi R_{0}^{2}) \in (0, 1]$,
//! where $N$ is the number of monomers and $R_{0}$ is the monomer radius.

use std::f64::consts::PI;

use anyhow::Result;
use anyhow::anyhow;
use planetes_ext::types::RVector;

use crate::fractal::FractalCutoff;
use crate::utils::gamma::{gamma, gamma_ur};

/// Quadrature strategy for the radial ($x$) and angular ($u$) integrals in the
/// mean monomer overlapping efficiency $\sigma$.
#[derive(PartialEq)]
pub enum IntegrationMethod {
    /// Numerical quadrature for both the radial and angular integrals
    /// (retains the full finite-$\rho$ angular overlap factor $S(\rho)$).
    NumNum,
    /// Numerical radial quadrature combined with the asymptotic analytical
    /// angular limit $S(\rho) = 1$.
    NumAn,
    /// Analytical upper incomplete gamma evaluation for the radial integral
    /// when $d_{f} > 2$ (falling back to numerical radial quadrature for
    /// $d_{f} \le 2$), combined with the analytical angular limit $S(\rho) = 1$.
    AnAn,
}

/// Normalization factor $A$ in the aggregate geometric cross-section relation
/// $G / (N \pi R_{0}^{2}) = A / [1 + (N - 1)\sigma]$.
pub enum AFactor {
    /// Constant normalization $A = 1$ for all monomer counts $N \ge 1$.
    Unity,
    /// Two-regime matching from
    /// [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract):
    /// uses the [Minato et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..701M/abstract)
    /// fitting formula for $N < N_{\mathrm{th}} = \min(11 d_{f} - 8.5, 8)$ and
    /// chooses $A$ so that the two regimes meet continuously at
    /// $N = N_{\mathrm{th}}$.
    TwoRegimes,
}

/// Computes the normalized orientation-averaged geometric cross section
/// $G / (N \pi R_{0}^{2})$ of a randomly oriented fractal dust aggregate
/// following [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract).
///
/// Evaluates
/// $$\frac{G}{N \pi R_{0}^{2}} = \frac{A}{1 + (N - 1)\sigma},$$
/// where $\sigma$ is the mean monomer overlapping efficiency and $A$ is set by
/// `afac` (with small-$N$ switching to the
/// [Minato et al. 2006](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..701M/abstract)
/// fit under [`AFactor::TwoRegimes`]).
///
/// # Arguments
/// - `method`: Radial and angular quadrature method ([`IntegrationMethod`]).
/// - `afac`: Normalization scheme ([`AFactor`]).
/// - `cutoff`: Two-point correlation cutoff model ([`FractalCutoff`]).
/// - `n_monomers`: Number of monomers $N \ge 1$.
/// - `prefactor`: Fractal prefactor $k_{0}$ in $N = k_{0}(R_{g}/R_{0})^{d_{f}}$.
/// - `fractal_dim`: Fractal dimension $d_{f} \in [1, 3]$.
///
/// # Errors
/// Returns an error if `n_monomers <= 0.9999` or if `fractal_dim` lies outside
/// $[0.9999, 3.0001]$.
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

/// Empirical normalized geometric cross section for small aggregates ($N < 16$)
/// from [Minato et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..701M/abstract):
/// $$\frac{G}{N \pi R_{0}^{2}} = 12.5 \thinspace N^{-0.315} \exp\!\left(-2.53 \thinspace N^{-0.0920}\right).$$
fn minato_fit(n_monomers: f64) -> f64 {
    12.5 * n_monomers.powf(-0.315) * (-2.53 / n_monomers.powf(0.0920)).exp()
}

/// Computes the mean monomer overlapping efficiency $\sigma$ of
/// [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract).
///
/// Depending on the two-point correlation cutoff model, $\sigma$ takes the form
/// $$\sigma = C \int_{\ln x_{\min}}^{\ln x_{\max}} f_{\mathrm{rad}}(x) S(\rho) \thinspace \mathrm{d}\ln x,$$
/// where $f_{\mathrm{rad}}(x) = x^{a} e^{-x}$ and:
/// - [`FractalCutoff::Gaussian`]:
///   $C = x_{\min} / [16\thinspace\Gamma(d_{f}/2)]$,
///   $x_{\min} = d_{f}(k_{0}/N)^{2/d_{f}}$, $a = (d_{f} - 2)/2$,
///   $\rho = \sqrt{x / x_{\min}}$.
/// - [`FractalCutoff::Exponential`]:
///   $C = x_{\min}^{2} / [16\thinspace\Gamma(d_{f})]$,
///   $x_{\min} = 2\sqrt{d_{f}(d_{f} + 1)/2}\thinspace (k_{0}/N)^{1/d_{f}}$,
///   $a = d_{f} - 2$, $\rho = x / x_{\min}$.
/// - [`FractalCutoff::FractalDimension`]:
///   $C = x_{\min}^{2/d_{f}} / 16$,
///   $x_{\min} = 2^{d_{f} - 1} k_{0}/N$, $a = (d_{f} - 2)/d_{f}$,
///   $\rho = (x / x_{\min})^{1/d_{f}}$.
///
/// Under [`IntegrationMethod::AnAn`] with $d_{f} > 2$, the radial integral
/// evaluates analytically as $\Gamma(a)\thinspace Q(a, x_{\min})$; otherwise
/// trapezoidal quadrature over $1000$ log-spaced nodes up to $x_{\max} = 25$
/// is performed (with $S(\rho)$ integrated numerically under
/// [`IntegrationMethod::NumNum`] and set to $1$ otherwise).
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

/// Evaluates the radial integrand $f_{\mathrm{rad}}(x) = x^{a} e^{-x}$ in
/// $\ln x$ space, so that
/// $f_{\mathrm{rad}}(x)\thinspace\mathrm{d}\ln x = x^{a - 1} e^{-x}\thinspace\mathrm{d}x$.
fn radial_integrand(a: f64, x: f64) -> f64 {
    x.powf(a) * (-x).exp()
}

/// Evaluates the angular monomer-overlap integral
/// $$S(\rho) = \frac{16 \rho^{2}}{\pi} \int_{0}^{1/\rho} f_{\mathrm{ang}}(\rho, u) \thinspace \mathrm{d}u$$
/// by trapezoidal quadrature over $1000$ uniform nodes in $u \in [0, 1/\rho]$
/// ($S(\rho) \to 1$ in the limit $\rho \gg 1$).
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

/// Evaluates the angular overlap integrand
/// $$f_{\mathrm{ang}}(\rho, u) = \left[\arcsin\!\left(\sqrt{1 - \rho^{2} u^{2}}\right) - \rho u \sqrt{1 - \rho^{2} u^{2}}\right] \frac{u}{\sqrt{1 - u^{2}}},$$
/// returning $0$ within $10^{-10}$ of the upper endpoint $\rho u = 1$.
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
