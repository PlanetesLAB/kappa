//! Light scattering by randomly oriented fractal dust aggregates.
//!
//! Implements Rayleigh-Gans-Debye (RGD), Mean Field (MF), and Modified Mean
//! Field (MMF) theories following
//! [Tazaki et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...823...70T/abstract),
//! [Botet et al. (1997)](https://ui.adsabs.harvard.edu/abs/1997ApOpt..36.8791B/abstract),
//! and [Tazaki & Tanaka (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract).
//!
//! Refractive indices follow the $e^{-i \omega t}$ harmonic time convention
//! ($m = n + i k$ with $k \ge 0$), and the off-diagonal circular-polarization
//! element $S_{34} = \operatorname{Im}(S_{1} S_{2}^{*})$ matches the
//! [Hovenier et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ASSL..318.....H/abstract)
//! sign convention.

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

/// Input parameters for light scattering by a randomly oriented fractal dust
/// aggregate.
pub struct FractalConfig {
    /// Light-scattering approximation ([`FractalSolver`]).
    pub solver: FractalSolver,
    /// Cutoff function for the two-point monomer correlation function
    /// ([`FractalCutoff`]).
    pub cutoff: FractalCutoff,
    /// Aggregate geometric cross-section model used by MMF absorption
    /// ([`FractalGeometry`]).
    pub geometry: FractalGeometry,
    /// Number of angular nodes on $[0, \pi/2]$ (must be $\ge 2$); the output
    /// scattering grid spans $[0, \pi]$ with $2 N_{\theta} - 1$ uniform nodes.
    pub n_angles: usize,
    /// Number of spherical monomers $N \ge 1$ in the aggregate.
    pub n_monomers: f64,
    /// Radius of an individual monomer $R_{0}$ (same length unit as
    /// `wavelength`, typically $\mu\mathrm{m}$).
    pub monomer_radius: f64,
    /// Fractal dimension $d_{f} \in [1, 3]$ in
    /// $N = k_{0} (R_{g} / R_{0})^{d_{f}}$.
    pub fractal_dim: f64,
    /// Fractal prefactor $k_{0}$ in $N = k_{0} (R_{g} / R_{0})^{d_{f}}$.
    pub prefactor: f64,
    /// Incident wavelength $\lambda$ (same length unit as `monomer_radius`,
    /// typically $\mu\mathrm{m}$).
    pub wavelength: f64,
    /// Complex refractive index of the monomer $m = n + i k$ ($k \ge 0$).
    pub refractive_index: Complex64,
    /// Default Gauss-Legendre quadrature order ($400$; currently informational).
    pub gauss_leg_grid_size: usize,
    /// Default multipole truncation warning threshold ($500$; currently
    /// informational).
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
    /// Creates a [`FractalConfig`] populated with [`Default::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Output cross sections, asymmetry parameter, and scattering matrix elements
/// from [`mean_scattering`].
pub struct FractalResult {
    /// Extinction cross section $C_{\mathrm{ext}}$ (in squared units of
    /// `wavelength`).
    pub c_ext: f64,
    /// Scattering cross section $C_{\mathrm{sca}}$ (in squared units of
    /// `wavelength`).
    pub c_sca: f64,
    /// Absorption cross section $C_{\mathrm{abs}}$ (in squared units of
    /// `wavelength`).
    pub c_abs: f64,
    /// Scattering asymmetry parameter $g = \langle \cos\theta \rangle$.
    pub g: f64,
    /// Aggregate phase-shift diagnostic
    /// $\Delta\phi = \max(\Delta\phi_{c}, \Delta\phi_{0})$
    /// ([Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract),
    /// Eq. 9); scattering matrix accuracy is guaranteed when $\Delta\phi < 1$.
    pub phase_shift: f64,
    /// Scattering angles $\theta \in [0, \pi]$ in radians (length
    /// $2 N_{\theta} - 1$).
    pub angles: RVector,
    /// Unnormalized Mueller matrix elements $[S_{11}, S_{12}, S_{33}, S_{34}]$
    /// of shape `(2 * n_angles - 1, 4)`; normalized
    /// [Hovenier et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004ASSL..318.....H/abstract)
    /// elements are $F_{ij} = \frac{4\pi}{k^{2} C_{\mathrm{sca}}} S_{ij}$.
    pub scattering_matrix: RMatrix,
    /// Phase function $P(\theta) = S_{11}(\theta) / (k^{2} C_{\mathrm{sca}})$
    /// normalized so that
    /// $2\pi \int_{0}^{\pi} P(\theta)\sin\theta\thinspace\mathrm{d}\theta = 1$.
    pub phase_function: RVector,
}

/// Light-scattering approximation for fractal aggregates.
pub enum FractalSolver {
    /// Rayleigh-Gans-Debye theory
    /// ([Tazaki et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...823...70T/abstract));
    /// accurate when the aggregate phase shift $\Delta\phi \lesssim 1$.
    RayleighGansDebye,
    /// Mean Field theory
    /// ([Botet et al. 1997](https://ui.adsabs.harvard.edu/abs/1997ApOpt..36.8791B/abstract));
    /// $C_{\mathrm{ext}}$ is valid at arbitrary $\Delta\phi$, while other
    /// outputs are reliable for $\Delta\phi \lesssim 1$.
    MeanField,
    /// Modified Mean Field theory
    /// ([Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract));
    /// extends $C_{\mathrm{abs}}$ and $C_{\mathrm{sca}}$ beyond
    /// $\Delta\phi \sim 1$ via an optical-depth absorption correction.
    ModifiedMeanField,
}

/// Cutoff function $f_{c}(r / R_{g})$ for the two-point monomer correlation
/// function.
pub enum FractalCutoff {
    /// Gaussian cutoff
    /// ([Tazaki et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...823...70T/abstract));
    /// recommended for numerical stability and accuracy.
    Gaussian,
    /// Exponential cutoff
    /// ([Berry & Percival 1986](https://doi.org/10.1080/713821987)).
    Exponential,
    /// Fractal-dimension-dependent cutoff
    /// ([Botet et al. 1995](https://ui.adsabs.harvard.edu/abs/1995JPhA...28..297B/abstract)).
    FractalDimension,
}

/// Geometric cross-section model $G$ used for the MMF optical-depth absorption
/// correction in [`FractalSolver::ModifiedMeanField`].
#[derive(PartialEq)]
pub enum FractalGeometry {
    /// Characteristic-sphere area $G = \pi R_{c}^{2}$ with
    /// $R_{c} = \sqrt{5/3}\thinspace R_{g}$.
    Circular,
    /// Interpolation formula of
    /// [Okuzumi et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...707.1247O/abstract).
    Okuzumi,
    /// Statistical monomer-overlap model of
    /// [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract)
    /// via [`crate::geofractal::get_geometric_cross_section_tazaki`].
    Tazaki,
}

/// Computes light-scattering cross sections, asymmetry parameter, and Mueller
/// scattering matrix elements of a randomly oriented fractal dust aggregate.
///
/// Evaluates the gyration radius $R_{g} = R_{0}(N/k_{0})^{1/d_{f}}$,
/// characteristic radius $R_{c} = \sqrt{5/3}\thinspace R_{g}$, and phase-shift
/// diagnostic $\Delta\phi = \max(2 k R_{c}|m_{\mathrm{eff}} - 1|, 2 k R_{0}|m - 1|)$
/// ([Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract),
/// Eq. 9). Monomer Lorenz-Mie coefficients are solved directly for
/// [`FractalSolver::RayleighGansDebye`] or coupled through the mean-field
/// linear system for [`FractalSolver::MeanField`] and
/// [`FractalSolver::ModifiedMeanField`], then combined with the coherent
/// aggregate factor $N [1 + (N - 1) S(q)]$ where
/// $q = 2 k \sin(\theta / 2)$. Under [`FractalSolver::ModifiedMeanField`],
/// absorption is corrected for internal optical depth
/// $\tau = C_{\mathrm{abs,RGD}} / G$ via
/// $C_{\mathrm{abs}} = \max[G(1 - e^{-\tau}), C_{\mathrm{ext}} - C_{\mathrm{sca}}]$
/// and $C_{\mathrm{sca}} = C_{\mathrm{ext}} - C_{\mathrm{abs}}$.
///
/// # Errors
/// Returns an error if:
/// - `n_angles <= 1`, `n_monomers < 1`, or `fractal_dim > 3`;
/// - monomer Lorenz-Mie truncation order exceeds $1.5 \times 10^{5}$;
/// - the Legendre/Bessel integration or mean-field linear system fails;
/// - any $S_{11}(\theta) < 0$ or the integrated phase-function normalization
///   deviates from $1$ by more than $10^{-3}$.
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

/// Computes the effective refractive index $m_{\mathrm{eff}} = \sqrt{\epsilon_{\mathrm{eff}}}$
/// of monomer inclusions (refractive index `refrel`, volume filling factor `f1`)
/// embedded in vacuum ($\epsilon_{2} = 1$) via the Maxwell Garnett (1904,
/// *Phil. Trans. R. Soc. A* 203, 385) mixing rule:
/// $$\epsilon_{\mathrm{eff}} = \epsilon_{2} \frac{2 f_{1} (\epsilon_{1} - \epsilon_{2}) + \epsilon_{1} + 2 \epsilon_{2}}{\epsilon_{1} + 2 \epsilon_{2} - f_{1} (\epsilon_{1} - \epsilon_{2})}.$$
fn maxwell_garnett_mixing(refrel: Complex64, f1: f64) -> Complex64 {
    let eps_1 = refrel * refrel;
    let eps_2 = Complex::new(1.0, 0.0);
    let mg = eps_2 * (2.0 * f1 * (eps_1 - eps_2) + eps_1 + 2.0 * eps_2)
        / (eps_1 + 2.0 * eps_2 - f1 * (eps_1 - eps_2));

    mg.sqrt()
}

/// Evaluates the static structure factor $S(q)$ for the
/// [`FractalCutoff::FractalDimension`] cutoff model of
/// [Botet et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995JPhA...28..297B/abstract):
/// $$S(q) = \frac{c \thinspace d_{f}}{q R_{g}} \int_{0}^{x_{\max}} x^{d_{f} - 2} \sin(q R_{g} x) \thinspace e^{-c x^{d_{f}}} \thinspace \mathrm{d}x,$$
/// where $x = r / R_{g}$, $c = 0.5$, and $x_{\max} = (\eta / c)^{1/d_{f}}$ with
/// $\eta = 25$.
///
/// Integrated via composite Simpson's rule over $2 \times 10^{5}$ subintervals
/// ($S(0) = 1$). Because the oscillatory integral can produce small negative
/// artifacts at large $d_{f} > 2$ and large $q R_{g}$, negative values are
/// clamped to $0$ (incoherent limit); [`FractalCutoff::Gaussian`] is preferred
/// when numerical stability at large $q R_{g}$ is critical.
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

/// Evaluates the static structure-factor integrand
/// $x^{d_{f} - 2} \sin(q R_{g} x) \thinspace e^{-c x^{d_{f}}}$ for
/// [`structure_factor_integration`], returning $0$ at $x = 0$ or when the
/// magnitude falls below $10^{-30}$.
fn structure_factor_fn(x: f64, c: f64, d: f64, q: f64, r_g: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let val = (q * r_g * x).sin() * (-c * x.powf(d)).exp() * x.powf(d - 2.0);
        if val.abs() <= 1e-30 { 0.0 } else { val }
    }
}

/// Computes the electric ($a_{n}$) and magnetic ($b_{n}$) Lorenz-Mie
/// scattering coefficients of a single spherical monomer for orders
/// $n = 1, \dots, n_{\mathrm{stop}}$ (0-indexed), following
/// [Bohren & Huffman (1983)](https://ui.adsabs.harvard.edu/abs/1983asls.book.....B/abstract)
/// (App. A).
///
/// Uses downward recurrence for the logarithmic derivative $D_{n}(m x)$ and
/// upward recurrence for the real Riccati-Bessel functions $\psi_{n}(x)$ and
/// $\chi_{n}(x)$ (with $\xi_{n}(x) = \psi_{n}(x) - i \chi_{n}(x)$ in the
/// $e^{-i \omega t}$ convention).
///
/// # Errors
/// Returns an error if the required downward-recurrence order exceeds
/// $1.5 \times 10^{5}$.
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

/// Sums the scattering amplitude functions $S_{1}(\theta)$ and $S_{2}(\theta)$
/// over the full $[0, \pi]$ angular grid ($2 N_{\theta} - 1$ points) from
/// multipole coefficients `d` of shape `(nmax, 2)` (columns `0` and `1`
/// storing $a_{n}$ and $b_{n}$), using forward/backward parity of $\pi_{n}$
/// and $\tau_{n}$.
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

/// Solves the mean-field multiple-scattering linear system for the coupled
/// monomer multipole coefficients
/// ([Botet et al. 1997](https://ui.adsabs.harvard.edu/abs/1997ApOpt..36.8791B/abstract);
/// [Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract)).
///
/// Computes the angular coupling integrals
/// ([Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract),
/// Eqs. 29 and 30) via $400$-point Gauss-Legendre quadrature on $[-1, 1]$:
/// $$a(\nu, n, p) = \frac{2p + 1}{2} \int_{-1}^{1} P_{\nu}^{1}(x) P_{n}^{1}(x) P_{p}(x) \thinspace \mathrm{d}x, \qquad b(\nu, n, p) = \frac{2p + 1}{2} \int_{-1}^{1} P_{\nu}^{1}(x) P_{n}^{1}(x) \frac{\mathrm{d}P_{p}(x)}{\mathrm{d}x} \thinspace \mathrm{d}x.$$
/// By integrand parity, $a(\nu, n, p)$ vanishes unless $p$ shares the parity
/// of $n + \nu$, whereas $b(\nu, n, p)$ vanishes unless $p$ has the opposite
/// parity; thus $p$ spans every integer in $[|\nu - n|, \nu + n]$. Combining
/// $a$ and $b$ with the radial Bessel integrals $S_{p}(X_{g})$ yields the
/// translation matrices $\bar{A}_{1,n}^{1,\nu}$ and $\bar{B}_{1,n}^{1,\nu}$
/// (Eqs. 14 and 15), and the resulting $2 n_{\max} \times 2 n_{\max}$ complex
/// linear system $(I + S) r = y$ is solved via
/// [`linalg::complex_matrix_inverse`].
///
/// # Errors
/// Returns an error if Legendre evaluation, Bessel quadrature, or the complex
/// linear solve fails.
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

/// Computes the aggregate geometric cross section $G$ using the interpolation
/// formula of [Okuzumi et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...707.1247O/abstract)
/// with ballistic cluster-cluster aggregation (BCCA) oblique-collision
/// parameters ($k_{0,\mathrm{BCCA}} = 1.04$, $d_{f,\mathrm{BCCA}} = 1.90$;
/// [Tazaki et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...823...70T/abstract))
/// and the [Minato et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..701M/abstract)
/// fitting formula for $N \le 16$, capped at the non-overlapping monomer sum
/// $N \pi R_{0}^{2}$.
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
