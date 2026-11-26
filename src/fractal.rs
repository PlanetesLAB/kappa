//! Light scattering properties of randomly oriented fractal dust aggregates.
//!
//! Implements Rayleigh-Gans-Debye, Mean Field, and Modified Mean Field theories
//! following Tazaki & Tanaka (2018).

use std::f64::consts::PI;
use std::sync::Once;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use ndarray::s;
use num_complex::{Complex, Complex64};
use planetes_ext::quadrature::gauss_legendre;
use planetes_ext::types::{CMatrix, CTensor, CVector, RMatrix, RVector};

use crate::geofractal::get_geometric_cross_section_tazaki;
use crate::geofractal::{AFactor, IntegrationMethod};
use crate::utils::bessel;
use crate::utils::gamma::gamma;
use crate::utils::legendre;
use crate::utils::linalg;
use crate::utils::special;

pub struct FractalConfig {
    pub solver: FractalSolver,
    pub cutoff: FractalCutoff,
    pub geometry: FractalGeometry,
    pub n_angles: usize,
    pub n_monomers: f64,
    pub monomer_radius: f64,
    pub fractal_dim: f64,
    pub prefactor: f64,
    pub wavelength: f64,
    pub refractive_index: Complex64,
    pub gauss_leg_grid_size: usize,
    pub truncation_order: usize,
}

impl Default for FractalConfig {
    fn default() -> Self {
        FractalConfig {
            solver: FractalSolver::ModifiedMeanField,
            cutoff: FractalCutoff::Gaussian,
            geometry: FractalGeometry::Circular,
            n_angles: 100,
            n_monomers: 1.0,
            monomer_radius: 0.1,
            fractal_dim: 2.0,
            prefactor: 1.0,
            wavelength: 0.55,
            refractive_index: Complex::new(1.5, 0.0),
            gauss_leg_grid_size: 400,
            truncation_order: 500,
        }
    }
}

impl FractalConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct FractalResult {
    pub c_ext: f64,
    pub c_sca: f64,
    pub c_abs: f64,
    pub g: f64,
    pub phase_shift: f64,
    pub angles: RVector,
    pub scattering_matrix: RMatrix,
    pub phase_function: RVector,
}

pub enum FractalSolver {
    RayleighGansDebye,
    MeanField,
    ModifiedMeanField,
}

pub enum FractalCutoff {
    Gaussian,
    Exponential,
    FractalDimension,
}

#[derive(PartialEq)]
pub enum FractalGeometry {
    Circular,
    Okuzumi,
    Tazaki,
}

/// Computes light scattering properties of randomly oriented
/// fractal dust aggregates by means of the modified mean field theory developed
/// in [Tazaki & Tanaka (2018)](https://iopscience.iop.org/article/10.3847/1538-4357/aac32d/meta). This code is also capable of computing the light
/// scattering solution based on the Rayleigh-Gans-Debye theory and the Mean field
/// theory.
///
/// # Arguments
/// - `iqsca`:  Method switch for the light scatteing solver
///     - `iqsca = 1`: Rayleigh-Gans-Debye theory
///     - `iqsca = 2`: Mean field theory
///     - `iqsca = 3`: Modified mean-field theory
/// - `iqcor`: Switch for the two-point correction function
///     - `iqcor = 1`: Gaussian cut-off
///     - `iqcor = 2`: Exponential cut-off
///     - `iqcor = 3`: Fractal dimension cut-off
/// - `iqgeo`: Switch for the geometric cross-section
///     - `iqgeo = 1`: `π * rc^2`, where `rc` is the characteristic radius.
///     - `iqgeo = 2`: [Okuzumi et al. (2009)](https://iopscience.iop.org/article/10.1088/0004-637X/698/2/1122/meta)
///     - `iqgeo = 3`: [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract)
/// - `nang`: Number of angles `(0, π/2)` for the scattering phase function.
/// - `pn`: Number of monomers in the aggregate.
/// - `r0`: Monomer radius.
/// - `df`: Fractal dimension.
/// - `k`: Prefactor for the fractal dimension.
/// - `lmd`: Wavelength of the incident light.
/// - `refrel`: Complex Refractive index of the monomer.
///
/// # Notes
/// - For `iqsca = 1`:
///   all outputs would be physically reasonable for the phase shift `< ~1`.
///
/// - For `iqsca = 2`:
///   The extinction cross section would be calculated without limitation.
///   However, the other outputs would be reliable for the phase shift < ~1.
///
/// - For `iqsca = 3`:
///   The extinction cross section would be calculated without limitation.
///   Scattering and absorption cross sections could be calculated
///   for the phase shift > 1, however too large phase shift may
///   cause some problem.
///   The asymmetry parameter and the sattering matrix elements would be
///   reliable for the phase shift < ~1.
///
/// - For the two-point correlation function, it is suggested to use `iqcor = 1` because
///   it is numerically stable and is likely to reproduce optical properties
///   of fractal aggregates.
///
/// # References
/// - `iqsca = 1`; `iqcor = 1`: [Tazaki et al. (2016), ApJ, 823, 70](https://iopscience.iop.org/article/10.3847/0004-637X/823/2/70/meta)
/// - `iqsca = 2`: [Botet et al. (1997), ApOpt, 36, 8791](https://opg.optica.org/abstract.cfm?uri=ao-36-33-8791)
/// - `iqsca = 3`: [Tazaki & Tanaka (2018), ApJ, 860, 79](https://iopscience.iop.org/article/10.3847/1538-4357/aac32d/meta)
/// - `iqcor = 2`: [Berry & Percival (1986), AcOpt, 33, 577](https://www.tandfonline.com/doi/abs/10.1080/713821987)
/// - `iqcor = 3`: [Botet et al. (1995), JPhA, 28, 297](https://iopscience.iop.org/article/10.1088/0305-4470/28/2/008)
/// - `iqgeo = 2`: [Okuzumi et al. (2009), ApJ, 707, 1247](https://iopscience.iop.org/article/10.1088/0004-637X/698/2/1122/meta)
/// - `iqgeo = 3`: [Tazaki (2021), MNRAS, 504, 2811](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract)
///
/// # Errors
/// - `LowScatteringAngleResolution`: The number of angles for the scattering phase function is too small (<=1).
/// - `NotEnoughMonomers`: The number of monomers in the aggregate is too small (<1).
/// - `ExceedsMaxFractalDimension`: The fractal dimension exceeds the maximum value (3.0).
/// - `IllegalMatrixElement`: The scattering matrix element is negative, which is unphysical.
/// - `UnnormalizedPhaseFunction`: The phase function is not normalized, which is unphysical.
/// - `MieOverflow`: The truncation order for the Lorenz-Mie coefficients exceeds the maximum value (150,000).
#[allow(clippy::similar_names)]
pub fn mean_scattering(fracc: &FractalConfig) -> Result<FractalResult> {
    let k = 2.0 * PI / fracc.wavelength;
    let gyration_radius =
        fracc.monomer_radius * (fracc.n_monomers / fracc.prefactor).powf(1.0 / fracc.fractal_dim);
    let char_radius = (5.0_f64 / 3.0_f64).sqrt() * gyration_radius;
    let x_g = k * gyration_radius;
    let x0 = k * fracc.monomer_radius;

    let eta = 25.0;
    let q_r_g_critical = 26.0;

    let xmax = x0 + 4.0 * x0.powf(1.0 / 3.0) + 2.0;
    let nmax = xmax.round() as usize;

    if fracc.n_angles <= 1 {
        return Err(anyhow!("LowScatteringAngleResolution"));
    }

    if fracc.n_monomers < 1.0 {
        return Err(anyhow!("NotEnoughMonomers"));
    }

    if fracc.fractal_dim > 3.0 {
        return Err(anyhow!("ExceedsMaxFractalDimension"));
    }

    if (2 * nmax) >= 500 {
        static WARNED: Once = Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "Warning: mmf: monomer truncation order {} >= 500; spherical Bessel functions may fail (further occurrences not reported)",
                2 * nmax
            );
        });
    }

    // Phase shift criterion (Tazaki & Tanaka 2018, eq. 9)
    let filling = fracc.n_monomers * (fracc.monomer_radius / char_radius).powi(3);
    let m_eff = maxwell_garnett_mixing(fracc.refractive_index, filling);
    let dphic = 2.0 * k * char_radius * (m_eff - 1.0).norm();
    let dphi0 = 2.0 * x0 * (fracc.refractive_index - 1.0).norm();
    let phase_shift = dphic.max(dphi0);

    if phase_shift >= 1.0 {
        static WARNED: Once = Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "Warning: mmf: aggregate phase shift {phase_shift:.3e} >= 1; scattering matrix elements not guaranteed (further occurrences not reported)"
            );
        });
    }

    let (an, bn) = lorenz_mie(x0, fracc.refractive_index, nmax)?;

    let mut ad = CMatrix::zeros((nmax, 2));
    let mut dd = CMatrix::zeros((nmax, 2));

    ad.slice_mut(s![.., 0]).assign(&an);
    ad.slice_mut(s![.., 1]).assign(&bn);

    match fracc.solver {
        FractalSolver::RayleighGansDebye => {
            dd.slice_mut(s![.., 0]).assign(&ad.slice(s![.., 0]));
            dd.slice_mut(s![.., 1]).assign(&ad.slice(s![.., 1]));
        }
        FractalSolver::MeanField | FractalSolver::ModifiedMeanField => {
            let r = mean_field(fracc, &ad, x_g, nmax)?;
            for n in 0..nmax {
                dd[[n, 0]] = r[2 * n];
                dd[[n, 1]] = r[2 * n + 1];
            }
        }
    }

    let dang = PI / 2.0 / (fracc.n_angles - 1) as f64;
    let (s1, s2) = renormalize(fracc, &dd, dang, nmax);

    let n_ang_full = 2 * fracc.n_angles - 1;
    let mut scattering_matrix = RMatrix::zeros((n_ang_full, 4));
    let angles = (0..n_ang_full)
        .map(|j| j as f64 * dang)
        .collect::<RVector>();

    for j in 0..n_ang_full {
        let q = 2.0 * k * (0.5 * angles[j]).sin();
        let s11 = f64::midpoint(s2[j].norm_sqr(), s1[j].norm_sqr());
        let s12 = 0.5 * (s2[j].norm_sqr() - s1[j].norm_sqr());
        let s33 = (s1[j] * s2[j].conj()).re;
        // Hovenier (2004) sign convention: Im(S1 S2^*) = -Im(S2 S1^*) for e^{-i\omega t}.
        let s34 = (s1[j] * s2[j].conj()).im;
        let sq = match fracc.cutoff {
            FractalCutoff::Gaussian => {
                if (fracc.fractal_dim - 3.0).abs() < f64::EPSILON {
                    if q * q * gyration_radius * gyration_radius / 3.0 >= eta {
                        0.0
                    } else {
                        (-q * q * gyration_radius * gyration_radius / 3.0).exp()
                    }
                } else {
                    let al = 0.5 * fracc.fractal_dim;
                    let bb = 1.5;
                    let xx = -q * q * gyration_radius * gyration_radius / fracc.fractal_dim;
                    if q * gyration_radius < q_r_g_critical {
                        special::confluent_hypergeometric(al, bb, xx)
                    } else {
                        let cc = PI.sqrt() * fracc.fractal_dim.powf(0.5 * fracc.fractal_dim)
                            / (2.0 * gamma(0.5 * (3.0 - fracc.fractal_dim)));
                        cc * (q * gyration_radius).powf(-fracc.fractal_dim)
                    }
                }
            }
            FractalCutoff::Exponential => {
                let xi = (2.0 / (fracc.fractal_dim * (fracc.fractal_dim + 1.0))).sqrt()
                    * gyration_radius;
                if j == 0 {
                    1.0
                } else if (fracc.fractal_dim - 1.0).abs() < 1e-12 {
                    (q * xi).atan() / (q * xi)
                } else {
                    ((fracc.fractal_dim - 1.0) * (q * xi).atan()).sin()
                        / (q * xi
                            * (fracc.fractal_dim - 1.0)
                            * (1.0 + (q * xi).powi(2)).powf((fracc.fractal_dim - 1.0) / 2.0))
                }
            }
            FractalCutoff::FractalDimension => {
                structure_factor_integration(fracc.fractal_dim, q, gyration_radius)
            }
        };

        let coherent = fracc.n_monomers * (1.0 + (fracc.n_monomers - 1.0) * sq);
        scattering_matrix[[j, 0]] = coherent * s11;
        scattering_matrix[[j, 1]] = coherent * s12;
        scattering_matrix[[j, 2]] = coherent * s33;
        scattering_matrix[[j, 3]] = coherent * s34;

        if scattering_matrix[[j, 0]] < 0.0 {
            return Err(anyhow!("IllegalMatrixElement"));
        }
    }

    let mut c_sca: f64 = (0..(fracc.n_angles - 1))
        .map(|j| {
            let i0 = 2 * j;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            dang * (scattering_matrix[[i0, 0]] * angles[i0].sin()
                + 4.0 * scattering_matrix[[i1, 0]] * angles[i1].sin()
                + scattering_matrix[[i2, 0]] * angles[i2].sin())
                / 3.0
        })
        .sum::<f64>()
        * 2.0
        * PI
        / (k * k);

    let phase_function = scattering_matrix.slice(s![.., 0]).to_owned() / c_sca / (k * k);
    let g = (0..(fracc.n_angles - 1))
        .map(|j| {
            let i0 = 2 * j;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            dang * (phase_function[i0] * angles[i0].sin() * angles[i0].cos()
                + 4.0 * phase_function[i1] * angles[i1].sin() * angles[i1].cos()
                + phase_function[i2] * angles[i2].sin() * angles[i2].cos())
                / 3.0
        })
        .sum::<f64>()
        * 2.0
        * PI;
    let norm = (0..(fracc.n_angles - 1))
        .map(|j| {
            let i0 = 2 * j;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            dang * (phase_function[i0] * angles[i0].sin()
                + 4.0 * phase_function[i1] * angles[i1].sin()
                + phase_function[i2] * angles[i2].sin())
                / 3.0
        })
        .sum::<f64>()
        * 2.0
        * PI;

    if (norm - 1.0).abs() > 1e-3 {
        return Err(anyhow!("UnnormalizedPhaseFunction"));
    }
    let c_abs: f64;
    let c_ext: f64;
    (c_sca, c_abs, c_ext) = match fracc.solver {
        FractalSolver::RayleighGansDebye => {
            let c_abs = (0..nmax)
                .map(|j| {
                    let jf = j as f64;
                    (2.0 * jf + 3.0)
                        * ((ad[[j, 0]].re - ad[[j, 0]].norm_sqr())
                            + (ad[[j, 1]].re - ad[[j, 1]].norm_sqr()))
                })
                .sum::<f64>()
                * 2.0
                * PI
                * fracc.n_monomers
                / (k * k);
            let c_ext = c_abs + c_sca;
            (c_sca, c_abs, c_ext)
        }
        FractalSolver::MeanField => {
            let c_ext = (0..nmax)
                .map(|j| {
                    let jf = j as f64;
                    (2.0 * jf + 3.0) * (dd[[j, 0]] + dd[[j, 1]]).re
                })
                .sum::<f64>()
                * 2.0
                * PI
                * fracc.n_monomers
                / (k * k);
            let c_abs = c_ext - c_sca;
            (c_sca, c_abs, c_ext)
        }
        FractalSolver::ModifiedMeanField => {
            let c_ext = (0..nmax)
                .map(|j| {
                    let jf = j as f64;
                    (2.0 * jf + 3.0) * (dd[[j, 0]] + dd[[j, 1]]).re
                })
                .sum::<f64>()
                * 2.0
                * PI
                * fracc.n_monomers
                / (k * k);
            let mut c_abs = (0..nmax)
                .map(|j| {
                    let jf = j as f64;
                    (2.0 * jf + 3.0)
                        * ((ad[[j, 0]].re - ad[[j, 0]].norm_sqr())
                            + (ad[[j, 1]].re - ad[[j, 1]].norm_sqr()))
                })
                .sum::<f64>()
                * 2.0
                * PI
                * fracc.n_monomers
                / (k * k);

            let gc = match fracc.geometry {
                FractalGeometry::Circular => PI * char_radius * char_radius,
                FractalGeometry::Okuzumi => get_geometric_cross_section_okuzumi(
                    fracc.n_monomers,
                    fracc.monomer_radius,
                    char_radius,
                ),
                FractalGeometry::Tazaki => {
                    get_geometric_cross_section_tazaki(
                        &IntegrationMethod::AnAn,
                        &AFactor::Unity,
                        &fracc.cutoff,
                        fracc.n_monomers,
                        fracc.prefactor,
                        fracc.fractal_dim,
                    )? * fracc.n_monomers
                        * PI
                        * fracc.monomer_radius
                        * fracc.monomer_radius
                }
            };
            let tau = c_abs / gc;
            if tau >= 10.0 {
                c_abs = gc;
            } else {
                c_abs = gc * (1.0 - (-tau).exp());
            }
            c_abs = c_abs.max(c_ext - c_sca);
            c_sca = c_ext - c_abs;
            (c_sca, c_abs, c_ext)
        }
    };

    Ok(FractalResult {
        c_ext,
        c_sca,
        c_abs,
        g,
        phase_shift,
        angles,
        scattering_matrix,
        phase_function,
    })
}

/// This function finds effective refractive index of an aggregates
/// by using Maxwell-Garnett mixing rule.
///
/// # Arguments
/// - `refrel`: Complex refractive index of the monomer.
/// - `f1`: Volume filling factor of the first material.
///
/// # Returns
/// - Effective refractive index of the aggregates.
///
/// # Notes
/// - `eps_1` : Dielectric constant of the first material.
/// - `eps_2` : Dielectric constant of vacuum.
/// - `f1` : Volume filling factor of the first material.
fn maxwell_garnett_mixing(refrel: Complex64, f1: f64) -> Complex64 {
    let eps_1 = refrel * refrel;
    let eps_2 = Complex::new(1.0, 0.0);
    let mg = eps_2 * (2.0 * f1 * (eps_1 - eps_2) + eps_1 + 2.0 * eps_2)
        / (eps_1 + 2.0 * eps_2 - f1 * (eps_1 - eps_2));

    mg.sqrt()
}

/// This subroutine performs integration of the static structure factor by
/// Botet et al. (1995) based on  `df`-dependent cut-off function for two-point
/// correlation function.
/// The statis structure factor `S(q)` is given by
///
///
/// $$
/// S(q) = \frac{c \cdot df}{q R_g} \int_0^{x_{\max}} dx \, x^{df-2} \sin(q R_g x) e^{-c x^{df}}
/// $$
///
/// where `x = u/Rg`; where `Rg` is the radius of gyration and `u` is the
/// distance between two monomers; `q` is the magnitude of scattering
/// vector, `df` is the fractal dimension, `c=0.5` is normalization factor.
///
/// Since we have a steep cut-off function for `x>1`, it would be enough
/// to take `x_max` slightly larger than unity.
/// Here we determine xmax so that `eta=c*x^df` becomes 25.
///
/// We use Simpson's rule for the numerical integration, and then,
/// the integration range `[0,xmax]` will be divided by `2*m` points.
///
/// # Arguments
/// - `d`: Fractal dimension.
/// - `q`: Magnitude of scattering vector.
/// - `r_g`: Radius of gyration.
///
/// # Returns
/// - Value of the static structure factor.
///
/// # Caution
/// For large `df(>2)` and large `qRg`, the numerical integration sometimes
/// returns a negative value of `S(q)`. Since `S(q)` is a power spectrum,
/// it must be positive. I also noticed that the same problem arises with
/// the code written by Pascal Rannou. Because of this ill behavior, I
/// prefer to use Gausian-cut-off model, which is more stable.
///
/// A not-too-bad tentative prescription is simply set `S(q)=0` when `S(q)<0`,
/// which means scattered waves from each monomer is incoherent. This is
/// what I did in if-sentence at the end of this subroutine.
fn structure_factor_integration(d: f64, q: f64, r_g: f64) -> f64 {
    let integration_grid_size = 100_000;
    let normalization_factor = 0.5; // Botet+ 1997

    // Upper limit of integration
    let eta: f64 = 25.0;
    let xmax = (eta / normalization_factor).powf(1.0 / d);

    let mut s_q: f64;
    if q == 0.0 {
        s_q = 1.0;
    } else {
        s_q = 0.0;
        let h = xmax / (2.0 * f64::from(integration_grid_size));
        for i in 0..integration_grid_size {
            let ir = f64::from(i);
            let x0 = 2.0 * ir * h;
            let x1 = (2.0 * ir + 1.0) * h;
            let x2 = (2.0 * ir + 2.0) * h;
            s_q += h
                * (structure_factor_fn(x0, normalization_factor, d, q, r_g)
                    + 4.0 * structure_factor_fn(x1, normalization_factor, d, q, r_g)
                    + structure_factor_fn(x2, normalization_factor, d, q, r_g))
                / 3.0;
        }
        s_q *= normalization_factor * d / q / r_g;
    }
    if s_q < 0.0 {
        s_q = 0.0;
    }

    s_q
}

/// Integrand of the static structure factor.
///
/// # Arguments
/// - `x`: Integration variable.
/// - `c`: Normalization factor.
/// - `d`: Fractal dimension.
/// - `q`: Magnitude of scattering vector.
/// - `r_g`: Radius of gyration.
///
/// # Returns
/// - Value of the integrand.
fn structure_factor_fn(x: f64, c: f64, d: f64, q: f64, r_g: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let val = (q * r_g * x).sin() * (-c * x.powf(d)).exp() * x.powf(d - 2.0);
        if val.abs() <= 1e-30 { 0.0 } else { val }
    }
}

/// Calculate Lorenz-Mie scattering coefficients (an,bn) for a monomer particle.
///
/// Since monomer's size parameter is supposed not to be very large,
/// we use the simple Bohren & Huffman Mie algorithm.
///
/// # Arguments
/// - `x`: Size parameter.
/// - `refrel`: Complex refractive index of the monomer.
/// - `nstop`: Number of terms in the series expansion.
///
/// # Returns
/// - `a`: Scattering coefficient.
/// - `b`: Absorption coefficient.
fn lorenz_mie(x: f64, refrel: Complex64, nstop: usize) -> Result<(CVector, CVector)> {
    let nmxx = 150_000;

    let y = refrel * x;
    let ymod = y.norm();
    let xstop = x + 4.0 * x.powf(1.0 / 3.0) + 2.0;
    let nmx = xstop.max(ymod).round() as usize + 15;
    if nmx > nmxx {
        return Err(anyhow!("MieOverflow"));
    }

    let mut d = CVector::zeros(nmx);

    for k in (1..nmx).rev() {
        let en = (k + 1) as f64;
        d[k - 1] = (en / y) - (1.0 / (d[k] + en / y));
    }

    let mut psi_0 = x.cos();
    let mut psi_1 = x.sin();
    let mut chi_0 = -psi_1;
    let mut chi_1 = psi_0;
    let mut xi_1 = Complex::new(psi_1, -chi_1);

    let mut a = CVector::zeros(nstop);
    let mut b = CVector::zeros(nstop);

    for n in 0..nstop {
        let en = (n + 1) as f64;
        let psi = (2.0 * en - 1.0) * psi_1 / x - psi_0;
        let chi = (2.0 * en - 1.0) * chi_1 / x - chi_0;
        let xi = Complex::new(psi, -chi);
        a[n] = (d[n] / refrel + en / x) * psi - psi_1;
        a[n] /= (d[n] / refrel + en / x) * xi - xi_1;
        b[n] = (refrel * d[n] + en / x) * psi - psi_1;
        b[n] /= (refrel * d[n] + en / x) * xi - xi_1;
        psi_0 = psi_1;
        psi_1 = psi;
        chi_0 = chi_1;
        chi_1 = chi;
        xi_1 = Complex::new(psi_1, -chi_1);
    }
    Ok((a, b))
}

/// Calculate scattering amplitudes $S_1, S_2$ from scattering coefficients `d`.
fn renormalize(fracc: &FractalConfig, d: &CMatrix, dang: f64, nmax: usize) -> (CVector, CVector) {
    let cos_theta = (0..fracc.n_angles)
        .map(|j| (j as f64 * dang).cos())
        .collect::<RVector>();

    let mut pi = RVector::zeros(fracc.n_angles);
    let mut pi0 = RVector::zeros(fracc.n_angles);
    let mut pi1 = RVector::ones(fracc.n_angles);
    let mut tau = RVector::zeros(fracc.n_angles);

    let nn = 2 * fracc.n_angles - 1;
    let mut s1 = CVector::zeros(nn);
    let mut s2 = CVector::zeros(nn);
    let mut p = -1.0;

    for n in 0..nmax {
        let en = (n + 1) as f64;
        let f_n = (2.0 * en + 1.0) / (en * (en + 1.0));
        let an = d[[n, 0]];
        let bn = d[[n, 1]];

        for j in 0..fracc.n_angles {
            pi[j] = pi1[j];
            tau[j] = en * cos_theta[j] * pi[j] - (en + 1.0) * pi0[j];
            s1[j] += f_n * (an * pi[j] + bn * tau[j]);
            s2[j] += f_n * (an * tau[j] + bn * pi[j]);
        }
        p = -p;
        for j in 0..(fracc.n_angles - 1) {
            let jj = nn - 1 - j;
            s1[jj] += f_n * p * (an * pi[j] - bn * tau[j]);
            s2[jj] += f_n * p * (bn * pi[j] - an * tau[j]);
        }

        for j in 0..fracc.n_angles {
            pi1[j] = ((2.0 * en + 1.0) * cos_theta[j] * pi[j] - (en + 1.0) * pi0[j]) / en;
            pi0[j] = pi[j];
        }
    }
    (s1, s2)
}

/// Solves the mean field theory.
///
/// # PART I.
/// Calculate a(nu,n,p) and b(nu,n,p):
///
/// ```text
///                   2p + 1  /+1
///       a(nu,n,p) = -------- | dx P_nu^1(x) * P_n^1(x) * P_p(x),
///                      2     /-1
///     
///                    2p + 1  /+1                         dP_p(x)
///       b(nu,n,p) = -------- | dx P_nu^1(x) * P_n^1(x) * -------,
///                      2     /-1                            dx
/// ```
///
/// where `P_n^m` is the associated Legendre function (`n`: degree, `m`: order),
///  `P_n` is the Legendre polynominal function
///  (see Equations (29, 30) in Tazaki & Tanaka 2018).
///  The integration is performed with the Gauss-Legendre quadrature.
///
/// # PART II.
/// Calculate translation matrix coefficients
///
/// Translation coefficients: A and B (Equation # from Tazaki & Tanaka 2018)
/// T(1,nu,n) : \bar{A}_{1,n}^{1,nu} defined in Equation (14)
/// T(2,nu,n) : \bar{A}_{1,n}^{1,nu} defined in Equation (15)
/// anunp     : a(nu,n,p)            defined in Equation (29)
/// bnunp     : b(nu,n,p)            defined in Equation (30)
///
/// Note about the rule of rum with respect to the index p
/// (RT and Botet, in private communication, 2017).
///
/// Just before Equation (4) in Botet et al. (1997), it is written that
/// p should have the same parity as n+nu. However, this is only true
/// true for a(nu,n,p), and not for b(nu,n,p)
/// Thus, in this code, index p runs ALL integers between |nu-n| and nu+n.
///
/// Tips for efficient computation.
/// The parity of the integrand of a and b leads following properties:
/// - a(nu,n,p) .ne. 0 and b(nu,n,p) = 0 when p has the same parity as n+nu
/// - b(nu,n,p) .ne. 0 and a(nu,n,p) = 0 when p has the opposite parity to n+nu
///
///--------------------------------------------------------------------------------
fn mean_field(fracc: &FractalConfig, ad: &CMatrix, x_g: f64, nmax: usize) -> Result<CVector> {
    let jm = 400; // Number of grid points for Gauss-Legendre quadrature

    // Integration limits
    let x1 = -1.0;
    let x2 = 1.0;

    // Gauss-Legendre quadrature
    let (x, w) = gauss_legendre(x1, x2, jm);

    let order = 1;
    let pmax = 2 * nmax;

    // Store values of Legendre polynomials (P_n^m) and their derivatives,
    // indexed as [degree, quadrature_point] for contiguous dot products.
    let mut al1n = RMatrix::zeros((nmax + 1, jm));
    let mut ln = RMatrix::zeros((pmax + 1, jm));
    let mut dln = RMatrix::zeros((pmax + 1, jm));

    for (j, xj) in x.iter().enumerate() {
        let (pmn, _) =
            legendre::lpmns(order, nmax, *xj).context("IntegrationFailure: In Legendre")?;
        let (lp, dlp) = legendre::lpn(pmax, *xj);
        al1n.slice_mut(s![.., j]).assign(&pmn);
        ln.slice_mut(s![.., j]).assign(&lp);
        dln.slice_mut(s![.., j]).assign(&dlp);
    }

    // Pre-compute the spherical Bessel integrals S_p(k R_g) for p = 0..=2*nmax
    let mut s_p = CVector::zeros(pmax + 1);

    for (p, val) in s_p.iter_mut().enumerate() {
        *val = bessel::int_sph_bessel(fracc, x_g, p).context("IntegrationFailure: In Bessel")?;
    }

    let mut t = CTensor::zeros((2, nmax, nmax));
    for nu in 1..=nmax {
        let nuf = nu as f64;
        for n in 1..=nmax {
            let nf = n as f64;
            let pmin = nu.abs_diff(n);
            let p_hi = n + nu;
            let mut sum_a = Complex::new(0.0, 0.0);
            let mut sum_b = Complex::new(0.0, 0.0);
            for (p, &val) in s_p.iter().enumerate().take(p_hi + 1).skip(pmin) {
                let pf = p as f64;
                let mut anunp = 0.0;
                let mut bnunp = 0.0;
                if p_hi % 2 == p % 2 {
                    anunp = w
                        .iter()
                        .zip(al1n.row(nu))
                        .zip(al1n.row(n))
                        .zip(ln.row(p))
                        .map(|(((w, al1n_nu), al1n_n), ln_p)| w * al1n_nu * al1n_n * ln_p)
                        .sum();
                } else {
                    bnunp = w
                        .iter()
                        .zip(al1n.row(nu))
                        .zip(al1n.row(n))
                        .zip(dln.row(p))
                        .map(|(((w, al1n_nu), al1n_n), dln_p)| w * al1n_nu * al1n_n * dln_p)
                        .sum();
                }
                let factor = f64::midpoint(2.0 * pf, 1.0);
                anunp *= factor;
                bnunp *= factor;

                sum_a += (nf * (nf + 1.0) + nuf * (nuf + 1.0) - pf * (pf + 1.0)) * anunp * val;
                sum_b += bnunp * val;
            }
            let denom = nf * (nf + 1.0) * nuf * (nuf + 1.0);
            t[[0, nu - 1, n - 1]] = sum_a * (2.0 * nuf + 1.0) / denom;
            t[[1, nu - 1, n - 1]] = sum_b * 2.0 * (2.0 * nuf + 1.0) / denom;
        }
    }

    let mut s = CMatrix::zeros((pmax, pmax));
    let mut y = CVector::zeros(pmax);

    for n in 0..nmax {
        for nu in 0..nmax {
            s[[2 * n, 2 * nu]] = (fracc.n_monomers - 1.0) * ad[[n, 0]] * t[[0, nu, n]];
            s[[2 * n, 2 * nu + 1]] = (fracc.n_monomers - 1.0) * ad[[n, 0]] * t[[1, nu, n]];
            s[[2 * n + 1, 2 * nu]] = (fracc.n_monomers - 1.0) * ad[[n, 1]] * t[[1, nu, n]];
            s[[2 * n + 1, 2 * nu + 1]] = (fracc.n_monomers - 1.0) * ad[[n, 1]] * t[[0, nu, n]];
        }
        y[2 * n] = ad[[n, 0]];
        y[2 * n + 1] = ad[[n, 1]];
    }
    for i in 0..pmax {
        s[[i, i]] += Complex::new(1.0, 0.0);
    }

    // Solve S * r = y for mean-field scattering coefficients r
    let r = linalg::complex_matrix_inverse(pmax, &y, &s).context("SingularMatrix")?;

    Ok(r)
}

fn get_geometric_cross_section_okuzumi(n_monomers: f64, monomer_radius: f64, char_radius: f64) -> f64 {
    let k0_bcca = 1.04; // Tazaki+2016; BCCA with oblique collisions
    let df_bcca = 1.90; // Tazaki+2016; BCCA with oblique collisions

    let ac_bcca =
        (5.0_f64 / 3.0_f64).sqrt() * monomer_radius * (n_monomers / k0_bcca).powf(1.0 / df_bcca);
    let gc0 = PI * monomer_radius * monomer_radius;
    let gcc = PI * char_radius * char_radius;

    let gc_bcca2 = PI * ac_bcca * ac_bcca;

    let gc_bcca = if n_monomers <= 16.0 {
        12.5 * n_monomers.powf(0.685) * (-2.53 / n_monomers.powf(0.0920)).exp() * gc0
    } else {
        (0.352 * n_monomers + 0.566 * n_monomers.powf(0.862)) * gc0
    };

    (1.0 / (1.0 / gc_bcca + 1.0 / gcc - 1.0 / gc_bcca2)).min(n_monomers * gc0)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use num_complex::Complex64;

    use super::FractalConfig;
    use super::mean_scattering;
    use super::{FractalCutoff, FractalGeometry, FractalSolver};

    #[test]
    fn mmf_matches_reference_for_df_below_and_above_two() {
        let r_agg: f64 = 0.5;
        let rho = 3.0;
        let mass = 4.0 * PI / 3.0 * r_agg.powi(3) * rho;
        let to_kappa = 1e4 / mass;

        let cfg_18 = FractalConfig {
            solver: FractalSolver::ModifiedMeanField,
            cutoff: FractalCutoff::Gaussian,
            geometry: FractalGeometry::Tazaki,
            n_angles: 91,
            n_monomers: 125.0,
            monomer_radius: 0.1,
            fractal_dim: 1.8,
            prefactor: 1.3,
            wavelength: 10.0,
            refractive_index: Complex64::new(1.6, 0.01),
            ..Default::default()
        };
        let res_18 = mean_scattering(&cfg_18).unwrap();
        assert_relative_eq!(res_18.c_abs * to_kappa, 29.6922, max_relative = 1e-4);
        assert_relative_eq!(res_18.c_sca * to_kappa, 10.8906, max_relative = 1e-4);
        assert_relative_eq!(res_18.c_ext * to_kappa, 40.5828, max_relative = 1e-4);
        assert_relative_eq!(res_18.g, 0.146772, max_relative = 2e-3);

        let cfg_25 = FractalConfig {
            fractal_dim: 2.5,
            ..cfg_18
        };
        let res_25 = mean_scattering(&cfg_25).unwrap();
        assert_relative_eq!(res_25.c_abs * to_kappa, 30.2164, max_relative = 1e-4);
        assert_relative_eq!(res_25.c_sca * to_kappa, 14.3306, max_relative = 1e-4);
        assert_relative_eq!(res_25.c_ext * to_kappa, 44.5470, max_relative = 1e-4);
        assert_relative_eq!(res_25.g, 0.0406998, max_relative = 2e-3);
    }

    #[test]
    fn mmf_single_monomer_limit_matches_mie_including_f34_sign() {
        let monomer_radius: f64 = 0.10001;
        let wavelength: f64 = 1.0;
        let k = 2.0 * PI / wavelength;
        let rho = 3.0;
        let mass = 4.0 * PI / 3.0 * monomer_radius.powi(3) * rho;
        let to_kappa = 1e4 / mass;

        let cfg = FractalConfig {
            solver: FractalSolver::ModifiedMeanField,
            cutoff: FractalCutoff::Gaussian,
            geometry: FractalGeometry::Tazaki,
            n_angles: 91,
            n_monomers: 1.0,
            monomer_radius,
            fractal_dim: 2.0,
            prefactor: 1.0,
            wavelength,
            refractive_index: Complex64::new(1.6, 0.01),
            ..Default::default()
        };
        let res = mean_scattering(&cfg).unwrap();
        assert_relative_eq!(res.c_abs * to_kappa, 363.379, max_relative = 1e-3);
        assert_relative_eq!(res.c_sca * to_kappa, 1260.10, max_relative = 1e-3);
        assert_relative_eq!(res.c_ext * to_kappa, 1623.48, max_relative = 1e-3);
        assert_relative_eq!(res.g, 0.0805527, max_relative = 2e-3);

        let factor = 4.0 * PI / (k * k * res.c_sca);
        let f11 =
            f64::midpoint(res.scattering_matrix[[45, 0]], res.scattering_matrix[[46, 0]]) * factor;
        let f12 =
            f64::midpoint(res.scattering_matrix[[45, 1]], res.scattering_matrix[[46, 1]]) * factor;
        let f33 =
            f64::midpoint(res.scattering_matrix[[45, 2]], res.scattering_matrix[[46, 2]]) * factor;
        let f34 =
            f64::midpoint(res.scattering_matrix[[45, 3]], res.scattering_matrix[[46, 3]]) * factor;
        assert_relative_eq!(f11, 1.28159, max_relative = 1e-3);
        assert_relative_eq!(f12, -0.417852, max_relative = 1e-3);
        assert_relative_eq!(f33, 1.21156, max_relative = 1e-3);
        assert_relative_eq!(f34, 5.68434e-4, max_relative = 2e-3);
    }
}
