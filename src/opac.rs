//! Core opacity and scattering matrix orchestration for `kappa`.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;

use num_complex::ComplexFloat;
use num_complex::{Complex, Complex64};
use planetes_ext::quadrature::gauss_legendre;
use planetes_ext::types::{BVector, RVector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::components::StaticComponent;
use crate::components::try_get_lnk_data;
use crate::dhs::DHSConfig;
use crate::dhs::toon_ackerman_1981;
use crate::fractal::FractalConfig;
use crate::fractal::mean_scattering;
use crate::fractal::{FractalCutoff, FractalGeometry, FractalSolver};
use crate::io::read_lnk_file;
use crate::mie::MieConfig;
use crate::mie::de_rooij_1984;
use crate::progress::Progress;
use crate::utils::regrid_lnk_data;

/// Maximum number of material components permitted in a single grain mixture.
const MAX_MATERIALS: usize = 20;

/// Structured error type for `kappa` configuration, material lookup, I/O, and scattering solvers.
#[derive(Debug)]
pub enum KappaError {
    /// Invalid or inconsistent model configuration.
    InvalidConfig(String),
    /// Unknown built-in material key.
    UnknownMaterial(String),
    /// Filesystem or file-writing I/O error.
    Io(std::io::Error),
    /// Numerical failure inside effective medium mixing or scattering solvers.
    Numerical(String),
}

impl fmt::Display for KappaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::UnknownMaterial(key) => write!(f, "unknown material key: `{key}`"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Numerical(msg) => write!(f, "numerical error: {msg}"),
        }
    }
}

impl Error for KappaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidConfig(_) | Self::UnknownMaterial(_) | Self::Numerical(_) => None,
        }
    }
}

impl From<std::io::Error> for KappaError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<anyhow::Error> for KappaError {
    fn from(err: anyhow::Error) -> Self {
        Self::Numerical(err.to_string())
    }
}

/// Placement of a material component inside a core-mantle grain.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Core,
    Mantle,
}

/// Optical constants source for a single dust component.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialSource {
    /// Key of a built-in optical constant table (e.g. `"pyr-mg70"`, `"c-z"`).
    Builtin(String),
    /// Path to an external `.lnk` file.
    LnkFile(PathBuf),
    /// Wavelength-independent refractive index $m = n + i k$ and density $\rho$.
    Constant { n: f64, k: f64, density: f64 },
}

impl MaterialSource {
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Builtin(key) => key.clone(),
            Self::LnkFile(path) => path.display().to_string(),
            Self::Constant { n, k, density } => format!("{n}:{k}:{density}"),
        }
    }
}

/// A material component in a dust grain mixture.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub source: MaterialSource,
    pub kind: MaterialKind,
    pub mass_fraction: f64,
}

impl Material {
    #[must_use]
    pub fn builtin(key: impl Into<String>, kind: MaterialKind, mass_fraction: f64) -> Self {
        Self {
            source: MaterialSource::Builtin(key.into()),
            kind,
            mass_fraction,
        }
    }

    #[must_use]
    pub fn lnk_file(path: impl Into<PathBuf>, kind: MaterialKind, mass_fraction: f64) -> Self {
        Self {
            source: MaterialSource::LnkFile(path.into()),
            kind,
            mass_fraction,
        }
    }

    #[must_use]
    pub fn constant(n: f64, k: f64, density: f64, kind: MaterialKind, mass_fraction: f64) -> Self {
        Self {
            source: MaterialSource::Constant { n, k, density },
            kind,
            mass_fraction,
        }
    }
}

/// Grain size distribution specification.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum SizeDistribution {
    PowerLaw {
        min: f64,
        max: f64,
        slope: f64,
        bins: Option<usize>,
    },
    LogNormal {
        min: f64,
        max: f64,
        mean: f64,
        sigma: f64,
        bins: Option<usize>,
    },
    Normal {
        min: f64,
        max: f64,
        mean: f64,
        sigma: f64,
        bins: Option<usize>,
    },
    Monodisperse {
        radius: f64,
    },
    Tabulated {
        radii: Vec<f64>,
        weights: Vec<f64>,
        source: Option<PathBuf>,
    },
}

impl Default for SizeDistribution {
    fn default() -> Self {
        Self::PowerLaw {
            min: 0.05,
            max: 3000.0,
            slope: 3.5,
            bins: None,
        }
    }
}

impl SizeDistribution {
    fn bounds_and_count(&self) -> (f64, f64, usize) {
        match self {
            Self::PowerLaw { min, max, bins, .. }
            | Self::LogNormal { min, max, bins, .. }
            | Self::Normal { min, max, bins, .. } => {
                let (lo, hi) = if min <= max {
                    (*min, *max)
                } else {
                    (*max, *min)
                };
                if (hi - lo).abs() < f64::EPSILON {
                    (lo, hi, 1)
                } else {
                    let count = bins.unwrap_or_else(|| {
                        (((hi.log10() - lo.log10()) * 15.0 + 1.0) as usize).max(5)
                    });
                    (lo, hi, count)
                }
            }
            Self::Monodisperse { radius } => (*radius, *radius, 1),
            Self::Tabulated { radii, .. } => {
                let lo = radii.iter().copied().reduce(f64::min).unwrap_or(0.0);
                let hi = radii.iter().copied().reduce(f64::max).unwrap_or(0.0);
                (lo, hi, radii.len())
            }
        }
    }

    /// Computes $[\langle a \rangle, \langle a^2 \rangle^{1/2}, \langle a^3 \rangle^{1/3}]$.
    #[must_use]
    pub fn moments(&self) -> [f64; 3] {
        let (min, max, _) = self.bounds_and_count();
        self.moments_in_range(min, max)
    }

    fn moments_in_range(&self, amin: f64, amax: f64) -> [f64; 3] {
        match self {
            Self::Monodisperse { radius } => [*radius, *radius, *radius],
            Self::Tabulated { radii, weights, .. } => {
                let mut tot = [0.0; 3];
                let mut totn = 0.0;
                for (&r, &nr) in radii.iter().zip(weights.iter()) {
                    totn += nr;
                    tot[0] += nr * r;
                    tot[1] += nr * r.powi(2);
                    tot[2] += nr * r.powi(3);
                }
                if totn <= 0.0 {
                    return [0.0; 3];
                }
                [
                    tot[0] / totn,
                    (tot[1] / totn).sqrt(),
                    (tot[2] / totn).powf(1.0 / 3.0),
                ]
            }
            Self::PowerLaw { .. } | Self::LogNormal { .. } | Self::Normal { .. } => {
                if ((amax - amin) / amin).abs() < 1e-6 {
                    return [amin, amin, amin];
                }
                let ns: usize = 1000;
                let aminlog = amin.log10();
                let amaxlog = amax.log10();
                let nsf = (ns - 1) as f64;
                let mut tot = [0.0; 3];
                let mut totn = 0.0;
                for is in 0..ns {
                    let r = if is == 0 {
                        amin
                    } else if is == ns - 1 {
                        amax
                    } else {
                        10.0_f64.powf(aminlog + (amaxlog - aminlog) * (is as f64) / nsf)
                    };
                    // On a logarithmic grid (d ln a = const), bin weight is f(a) * a * d ln a.
                    let nr = match self {
                        Self::Normal { mean, sigma, .. } => {
                            let expo = 0.5 * ((r - mean) / sigma).powi(2);
                            if expo > 99.0 { 0.0 } else { r * (-expo).exp() }
                        }
                        Self::LogNormal { mean, sigma, .. } => {
                            let expo = 0.5 * ((r / mean).ln() / sigma).powi(2);
                            if expo > 99.0 { 0.0 } else { (-expo).exp() }
                        }
                        Self::PowerLaw { slope, .. } => r.powf(1.0 - slope),
                        Self::Monodisperse { .. } | Self::Tabulated { .. } => unreachable!(),
                    };
                    totn += nr;
                    tot[0] += nr * r;
                    tot[1] += nr * r.powi(2);
                    tot[2] += nr * r.powi(3);
                }
                if totn == 0.0 {
                    return [0.0; 3];
                }
                [
                    tot[0] / totn,
                    (tot[1] / totn).sqrt(),
                    (tot[2] / totn).powf(1.0 / 3.0),
                ]
            }
        }
    }
}

/// Wavelength grid specification in micrometers.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum WavelengthGrid {
    LogSpaced { min: f64, max: f64, count: usize },
    Single { wavelength: f64 },
    Tabulated { wavelengths: Vec<f64> },
}

impl Default for WavelengthGrid {
    fn default() -> Self {
        Self::LogSpaced {
            min: 0.05,
            max: 1e4,
            count: 300,
        }
    }
}

/// Structural parameterization for MMF fractal aggregates.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggregateStructure {
    /// Fractal dimension $d_f \in (1, 3]$.
    FractalDimension(f64),
    /// Volume filling factor $f \in (0, 1]$.
    FillingFactor(f64),
}

/// Scattering theory and solver parameters.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScatteringMethod {
    /// Lorenz-Mie theory for compact homogeneous spheres.
    Mie { size_limit: f64 },
    /// Distribution of Hollow Spheres (Min et al. 2005).
    Dhs {
        fmax: f64,
        size_limit: f64,
        dhs_size_limit: f64,
    },
    /// Modified Mean Field theory for fractal aggregates (Tazaki & Tanaka 2018).
    Mmf {
        monomer_radius: f64,
        structure: AggregateStructure,
        prefactor: Option<f64>,
        keep_approximate_scattering: bool,
    },
    /// Continuous Distribution of Ellipsoids in the Rayleigh limit.
    Cde,
}

impl Default for ScatteringMethod {
    fn default() -> Self {
        Self::dhs(0.8)
    }
}

impl ScatteringMethod {
    #[must_use]
    pub const fn mie() -> Self {
        Self::Mie { size_limit: 1e8 }
    }

    #[must_use]
    pub const fn dhs(fmax: f64) -> Self {
        Self::Dhs {
            fmax,
            size_limit: 1e8,
            dhs_size_limit: 1e4,
        }
    }

    #[must_use]
    pub const fn mmf(monomer_radius: f64, structure: AggregateStructure) -> Self {
        Self::Mmf {
            monomer_radius,
            structure,
            prefactor: None,
            keep_approximate_scattering: false,
        }
    }
}

/// Declarative dust opacity configuration.
#[allow(clippy::struct_excessive_bools)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct KappaConfig {
    pub materials: Vec<Material>,
    pub core_porosity: f64,
    pub mantle_porosity: f64,
    pub size_distribution: SizeDistribution,
    pub wavelengths: WavelengthGrid,
    pub method: ScatteringMethod,
    pub n_angles: usize,
    pub chop_angle: Option<f64>,
    pub write_scatter: bool,
    pub for_radmc: bool,
    pub sparse_ranges: Vec<(f64, f64)>,
    pub split_subgrains: Option<usize>,
    pub write_grid: bool,
    pub blend_only: bool,
    pub outdir: PathBuf,
    pub label: String,
    pub quiet: bool,
}

impl Default for KappaConfig {
    fn default() -> Self {
        Self {
            materials: Vec::new(),
            core_porosity: 0.0,
            mantle_porosity: 0.0,
            size_distribution: SizeDistribution::default(),
            wavelengths: WavelengthGrid::default(),
            method: ScatteringMethod::default(),
            n_angles: 180,
            chop_angle: None,
            write_scatter: false,
            for_radmc: false,
            sparse_ranges: Vec::new(),
            split_subgrains: None,
            write_grid: false,
            blend_only: false,
            outdir: PathBuf::from("output"),
            label: String::from("kappa_opacity"),
            quiet: false,
        }
    }
}

impl KappaConfig {
    /// DIANA standard dust mixture (87% pyroxene, 13% amorphous carbon, 25% porosity).
    #[must_use]
    pub fn diana() -> Self {
        Self {
            core_porosity: 0.25,
            materials: vec![
                Material::builtin("pyr-mg70", MaterialKind::Core, 0.87),
                Material::builtin("c-z", MaterialKind::Core, 0.13),
            ],
            ..Self::default()
        }
    }

    /// Ice-free DSHARP dust mixture (astrosilicate, organics, troilite).
    #[must_use]
    pub fn dsharp_no_ice() -> Self {
        Self {
            materials: vec![
                Material::builtin("astrosil", MaterialKind::Core, 0.3291),
                Material::builtin("c-org", MaterialKind::Core, 0.3966),
                Material::builtin("fes", MaterialKind::Core, 0.0743),
            ],
            ..Self::default()
        }
    }

    /// DSHARP dust mixture (astrosilicate, organics, troilite, water ice).
    #[must_use]
    pub fn dsharp() -> Self {
        let mut cfg = Self::dsharp_no_ice();
        cfg.materials
            .push(Material::builtin("h2o-w", MaterialKind::Core, 0.2000));
        cfg
    }

    /// Validates this configuration, loads and regrids optical constants, and computes the
    /// effective-medium blended refractive index.
    ///
    /// # Errors
    /// Returns [`KappaError`] if validation, material loading, or effective-medium mixing fails.
    pub fn prepare(&self) -> Result<PreparedModel, KappaError> {
        PreparedModel::from_config(self)
    }

    /// Validates, blends, and computes opacities and Mueller matrices in memory (no disk output).
    ///
    /// # Errors
    /// Returns [`KappaError`] if validation, loading, mixing, or scattering fails.
    pub fn compute(&self) -> Result<Particle, KappaError> {
        self.prepare()?.compute()
    }

    /// Validates, blends, computes, and writes requested output files to `self.outdir`.
    ///
    /// # Errors
    /// Returns [`KappaError`] if validation, computation, or writing fails.
    pub fn run(&self) -> Result<(), KappaError> {
        self.prepare()?.run()
    }
}

#[derive(Clone, Debug)]
struct ResolvedMaterial {
    label: String,
    kind: MaterialKind,
    density: f64,
    mass_fraction: f64,
}

/// Validated, mixed dust model ready for opacity calculation and file output.
#[derive(Debug)]
pub struct PreparedModel {
    config: KappaConfig,
    wavelengths: RVector,
    lmin: f64,
    lmax: f64,
    amin: f64,
    amax: f64,
    na: usize,
    materials: Vec<ResolvedMaterial>,
    rho_av: f64,
    n_blend: RVector,
    k_blend: RVector,
    sparse_indices: HashSet<usize>,
}

impl PreparedModel {
    fn from_config(input: &KappaConfig) -> Result<Self, KappaError> {
        let mut config = input.clone();

        // Materials
        if config.materials.is_empty() {
            return Err(KappaError::InvalidConfig(
                "at least one core material is required".into(),
            ));
        }
        if config.materials.len() > MAX_MATERIALS {
            return Err(KappaError::InvalidConfig(format!(
                "at most {MAX_MATERIALS} materials are supported"
            )));
        }
        let n_core = config
            .materials
            .iter()
            .filter(|m| m.kind == MaterialKind::Core)
            .count();
        if n_core == 0 {
            return Err(KappaError::InvalidConfig(
                "at least one core material is required".into(),
            ));
        }
        if config
            .materials
            .iter()
            .any(|m| m.mass_fraction < 0.0 || !m.mass_fraction.is_finite())
        {
            return Err(KappaError::InvalidConfig(
                "material mass fractions must be finite and non-negative".into(),
            ));
        }

        // Porosity
        if config.core_porosity < 0.0
            || config.core_porosity >= 1.0
            || config.mantle_porosity < 0.0
            || config.mantle_porosity >= 1.0
        {
            return Err(KappaError::InvalidConfig(
                "core and mantle porosities must lie in [0, 1)".into(),
            ));
        }

        // Size distribution validation
        match &mut config.size_distribution {
            SizeDistribution::PowerLaw {
                min,
                max,
                slope,
                bins,
            } => {
                if *min <= 0.0 || *max <= 0.0 || !min.is_finite() || !max.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "grain size bounds must be finite and positive".into(),
                    ));
                }
                if *min > *max {
                    std::mem::swap(min, max);
                }
                if *slope < 0.0 || !slope.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "power-law slope must be non-negative".into(),
                    ));
                }
                if let Some(0) = *bins {
                    return Err(KappaError::InvalidConfig(
                        "grain size bin count must be positive".into(),
                    ));
                }
            }
            SizeDistribution::LogNormal {
                min,
                max,
                mean,
                sigma,
                bins,
            }
            | SizeDistribution::Normal {
                min,
                max,
                mean,
                sigma,
                bins,
            } => {
                if *min <= 0.0 || *max <= 0.0 || !min.is_finite() || !max.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "grain size bounds must be finite and positive".into(),
                    ));
                }
                if *min > *max {
                    std::mem::swap(min, max);
                }
                if *mean <= 0.0 || !mean.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "size distribution mean must be finite and positive".into(),
                    ));
                }
                if *sigma <= 0.0 || !sigma.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "size distribution sigma must be finite and positive".into(),
                    ));
                }
                if let Some(0) = *bins {
                    return Err(KappaError::InvalidConfig(
                        "grain size bin count must be positive".into(),
                    ));
                }
            }
            SizeDistribution::Monodisperse { radius } => {
                if *radius <= 0.0 || !radius.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "monodisperse grain radius must be finite and positive".into(),
                    ));
                }
            }
            SizeDistribution::Tabulated { radii, weights, .. } => {
                if radii.is_empty() || radii.len() != weights.len() {
                    return Err(KappaError::InvalidConfig(
                        "tabulated size distribution must have matching non-empty radii and weights"
                            .into(),
                    ));
                }
                if radii.iter().any(|&r| r <= 0.0 || !r.is_finite())
                    || weights.iter().any(|&w| w < 0.0 || !w.is_finite())
                {
                    return Err(KappaError::InvalidConfig(
                        "tabulated radii must be positive and weights non-negative".into(),
                    ));
                }
            }
        }
        let (amin, amax, na) = config.size_distribution.bounds_and_count();

        // Wavelength grid construction
        let (wavelengths, lmin, lmax) = match &config.wavelengths {
            WavelengthGrid::Single { wavelength } => {
                if *wavelength <= 0.0 || !wavelength.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "wavelength must be finite and positive".into(),
                    ));
                }
                (RVector::from_elem(1, *wavelength), *wavelength, *wavelength)
            }
            WavelengthGrid::LogSpaced { min, max, count } => {
                if *min <= 0.0 || *max <= 0.0 || !min.is_finite() || !max.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "wavelength bounds must be finite and positive".into(),
                    ));
                }
                let (lo, hi) = if min <= max {
                    (*min, *max)
                } else {
                    (*max, *min)
                };
                if *count == 0 || (*count == 1 && (hi - lo).abs() > f64::EPSILON) {
                    return Err(KappaError::InvalidConfig(
                        "multiple wavelength samples required when lmin != lmax".into(),
                    ));
                }
                let grid = if *count == 1 || (hi - lo).abs() < f64::EPSILON {
                    RVector::from_elem(1, lo)
                } else {
                    let mut g = RVector::logspace(10.0, lo.log10(), hi.log10(), *count);
                    g[0] = lo;
                    g[*count - 1] = hi;
                    g
                };
                (grid, lo, hi)
            }
            WavelengthGrid::Tabulated { wavelengths } => {
                if wavelengths.is_empty() || wavelengths.iter().any(|&l| l <= 0.0 || !l.is_finite())
                {
                    return Err(KappaError::InvalidConfig(
                        "tabulated wavelengths must be non-empty, finite, and positive".into(),
                    ));
                }
                let lo = wavelengths.iter().copied().reduce(f64::min).unwrap_or(0.0);
                let hi = wavelengths.iter().copied().reduce(f64::max).unwrap_or(0.0);
                (RVector::from_vec(wavelengths.clone()), lo, hi)
            }
        };
        let nlam = wavelengths.len();

        // Scattering method validation
        match config.method {
            ScatteringMethod::Mie { size_limit } => {
                if size_limit <= 0.0 || !size_limit.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "Mie size_limit must be finite and positive".into(),
                    ));
                }
            }
            ScatteringMethod::Dhs {
                fmax,
                size_limit,
                dhs_size_limit,
            } => {
                if !(0.0..1.0).contains(&fmax) {
                    return Err(KappaError::InvalidConfig(
                        "DHS fmax must lie in [0, 1)".into(),
                    ));
                }
                if size_limit <= 0.0 || dhs_size_limit <= 0.0 {
                    return Err(KappaError::InvalidConfig(
                        "DHS size limits must be positive".into(),
                    ));
                }
            }
            ScatteringMethod::Mmf {
                monomer_radius,
                structure,
                prefactor,
                ..
            } => {
                if monomer_radius <= 0.0 || !monomer_radius.is_finite() {
                    return Err(KappaError::InvalidConfig(
                        "MMF monomer_radius must be finite and positive".into(),
                    ));
                }
                if monomer_radius >= amin {
                    return Err(KappaError::InvalidConfig(
                        "MMF monomer_radius must be smaller than minimum grain radius".into(),
                    ));
                }
                match structure {
                    AggregateStructure::FractalDimension(df) if !(df > 1.0 && df <= 3.0) => {
                        return Err(KappaError::InvalidConfig(
                            "MMF fractal dimension must lie in (1, 3]".into(),
                        ));
                    }
                    AggregateStructure::FillingFactor(fill) if !(fill > 0.0 && fill <= 1.0) => {
                        return Err(KappaError::InvalidConfig(
                            "MMF filling factor must lie in (0, 1]".into(),
                        ));
                    }
                    _ => {}
                }
                if let Some(k0) = prefactor
                    && (k0 <= 0.0 || !k0.is_finite())
                {
                    return Err(KappaError::InvalidConfig(
                        "MMF fractal prefactor must be finite and positive".into(),
                    ));
                }
            }
            ScatteringMethod::Cde => {
                if lmin <= 2.0 * PI * amax && !config.quiet {
                    eprintln!("Warning: CDE requires Rayleigh limit!");
                }
            }
        }

        // Angular grid & chop
        if config.n_angles < 2 || config.n_angles % 2 == 1 {
            return Err(KappaError::InvalidConfig(
                "n_angles must be even and at least 2".into(),
            ));
        }
        if let Some(angle) = config.chop_angle
            && (!(0.0..180.0).contains(&angle) || !angle.is_finite())
        {
            return Err(KappaError::InvalidConfig(
                "chop_angle must lie in [0, 180)".into(),
            ));
        }

        // Split validation
        if config.split_subgrains.is_some() && config.blend_only {
            if !config.quiet {
                eprintln!("Warning: turning off split mode for blend-only output");
            }
            config.split_subgrains = None;
        }
        if let Some(nsub) = config.split_subgrains {
            if nsub == 0 {
                return Err(KappaError::InvalidConfig(
                    "split_subgrains must be at least 1".into(),
                ));
            }
            if !matches!(config.size_distribution, SizeDistribution::PowerLaw { .. }) {
                return Err(KappaError::InvalidConfig(
                    "split mode is only supported for power-law size distributions".into(),
                ));
            }
        }

        // Sort cores before mantles (stable)
        config.materials.sort_by(|a, b| match (a.kind, b.kind) {
            (MaterialKind::Core, MaterialKind::Mantle) => Ordering::Less,
            (MaterialKind::Mantle, MaterialKind::Core) => Ordering::Greater,
            _ => Ordering::Equal,
        });

        // Load optical constants on the wavelength grid
        let mut mat_re = Vec::with_capacity(config.materials.len());
        let mut mat_im = Vec::with_capacity(config.materials.len());
        let mut mat_rho = Vec::with_capacity(config.materials.len());

        for material in &config.materials {
            let (re, im, rho) = match &material.source {
                MaterialSource::Builtin(key) => {
                    let comp = try_get_lnk_data(key)
                        .ok_or_else(|| KappaError::UnknownMaterial(key.clone()))?;
                    let (re, im) = regrid_lnk_data(comp.l0, comp.n0, comp.k0, &wavelengths, true);
                    (re, im, comp.rho)
                }
                MaterialSource::LnkFile(path) => {
                    let path_str = path.to_string_lossy();
                    let comp = read_lnk_file(&path_str, None)?;
                    let l0 = comp.l0.to_vec();
                    let n0 = comp.n0.to_vec();
                    let k0 = comp.k0.to_vec();
                    let (re, im) = regrid_lnk_data(&l0, &n0, &k0, &wavelengths, true);
                    (re, im, comp.rho)
                }
                MaterialSource::Constant { n, k, density } => {
                    if *density <= 0.0 || !density.is_finite() {
                        return Err(KappaError::InvalidConfig(
                            "material density must be finite and positive".into(),
                        ));
                    }
                    (
                        RVector::from_elem(nlam, *n),
                        RVector::from_elem(nlam, *k),
                        *density,
                    )
                }
            };
            if rho <= 0.0 || !rho.is_finite() {
                return Err(KappaError::InvalidConfig(
                    "material density must be finite and positive".into(),
                ));
            }
            mat_re.push(re);
            mat_im.push(im);
            mat_rho.push(rho);
        }

        // Normalize mass fractions and derive volume fractions & densities
        let (total_mfrac, raw_core_mfrac, raw_mantle_mfrac) =
            config
                .materials
                .iter()
                .fold((0.0, 0.0, 0.0), |(tot, c, m), mat| match mat.kind {
                    MaterialKind::Core => (tot + mat.mass_fraction, c + mat.mass_fraction, m),
                    MaterialKind::Mantle => (tot + mat.mass_fraction, c, m + mat.mass_fraction),
                });
        if total_mfrac <= 0.0 || raw_core_mfrac <= 0.0 {
            return Err(KappaError::InvalidConfig(
                "total core mass fraction must be positive".into(),
            ));
        }

        let tot_mfrac_core = raw_core_mfrac / total_mfrac;
        let tot_mfrac_mantle = raw_mantle_mfrac / total_mfrac;
        let mut resolved = Vec::with_capacity(config.materials.len());
        let mut raw_vfracs = Vec::with_capacity(config.materials.len());
        let mut core_vol_sum = 0.0;
        let mut mantle_vol_sum = 0.0;

        for (mat, &rho) in config.materials.iter().zip(&mat_rho) {
            let mfrac = mat.mass_fraction / total_mfrac;
            let vf = mfrac / rho;
            match mat.kind {
                MaterialKind::Core => core_vol_sum += vf,
                MaterialKind::Mantle => mantle_vol_sum += vf,
            }
            raw_vfracs.push(vf);
            resolved.push(ResolvedMaterial {
                label: mat.source.label(),
                kind: mat.kind,
                density: rho,
                mass_fraction: mfrac,
            });
        }

        let mut rho_core = tot_mfrac_core / core_vol_sum;
        let has_mantle = mantle_vol_sum > 0.0;
        let mut rho_mantle = if has_mantle {
            tot_mfrac_mantle / mantle_vol_sum
        } else {
            0.0
        };

        let mut core_vfracs = Vec::with_capacity(n_core);
        let mut mantle_vfracs = Vec::with_capacity(config.materials.len() - n_core);
        for (mat, &vf) in resolved.iter().zip(&raw_vfracs) {
            match mat.kind {
                MaterialKind::Core => {
                    core_vfracs.push((vf / core_vol_sum) * (1.0 - config.core_porosity));
                }
                MaterialKind::Mantle if has_mantle => {
                    mantle_vfracs.push((vf / mantle_vol_sum) * (1.0 - config.mantle_porosity));
                }
                MaterialKind::Mantle => {}
            }
        }
        rho_core *= 1.0 - config.core_porosity;
        if has_mantle {
            rho_mantle *= 1.0 - config.mantle_porosity;
        }

        let (rho_av, mantle_vol_frac) = if has_mantle {
            let rho = rho_core / (1.0 + tot_mfrac_mantle * (rho_core / rho_mantle - 1.0));
            (rho, tot_mfrac_mantle * rho / rho_mantle)
        } else {
            (rho_core, 0.0)
        };

        // Effective medium mixing (Bruggeman within core/mantle, Maxwell-Garnett core+mantle)
        let n_mantle = mantle_vfracs.len();
        let mut n_blend = RVector::zeros(nlam);
        let mut k_blend = RVector::zeros(nlam);
        let mut core_buf = vec![Complex64::new(0.0, 0.0); n_core];
        let mut mantle_buf = vec![Complex64::new(0.0, 0.0); n_mantle];
        let mut bruggeman_warned = false;

        for il in 0..nlam {
            let mut m_eff = if n_core == 1 && config.core_porosity == 0.0 {
                Complex::new(mat_re[0][il], mat_im[0][il])
            } else {
                for im in 0..n_core {
                    core_buf[im] = Complex::new(mat_re[im][il], mat_im[im][il]);
                }
                let (m_out, converged) = bruggeman_blend(&core_vfracs, &core_buf)?;
                if !converged && !bruggeman_warned && !config.quiet {
                    eprintln!(
                        "Warning: bruggeman: blend not converged (further occurrences not reported)"
                    );
                    bruggeman_warned = true;
                }
                m_out
            };

            if has_mantle {
                let m_mantle = if n_mantle == 1 && config.mantle_porosity == 0.0 {
                    Complex::new(mat_re[n_core][il], mat_im[n_core][il])
                } else {
                    for im in 0..n_mantle {
                        mantle_buf[im] =
                            Complex::new(mat_re[im + n_core][il], mat_im[im + n_core][il]);
                    }
                    let (m_out, converged) = bruggeman_blend(&mantle_vfracs, &mantle_buf)?;
                    if !converged && !bruggeman_warned && !config.quiet {
                        eprintln!(
                            "Warning: bruggeman: blend not converged (further occurrences not reported)"
                        );
                        bruggeman_warned = true;
                    }
                    m_out
                };
                let (re_mg, im_mg) = maxwell_garnet_blend(m_eff, m_mantle, mantle_vol_frac);
                m_eff = Complex::new(re_mg, im_mg);
            }

            n_blend[il] = m_eff.re;
            k_blend[il] = m_eff.im;
        }

        // Sparse wavelength indices
        let sparse_indices = compute_sparse_indices(&wavelengths, &config.sparse_ranges);

        Ok(Self {
            config,
            wavelengths,
            lmin,
            lmax,
            amin,
            amax,
            na,
            materials: resolved,
            rho_av,
            n_blend,
            k_blend,
            sparse_indices,
        })
    }

    /// Validated configuration used to build this model.
    #[must_use]
    pub fn config(&self) -> &KappaConfig {
        &self.config
    }

    /// Wavelength grid in micrometers.
    #[must_use]
    pub fn wavelengths(&self) -> &RVector {
        &self.wavelengths
    }

    /// Blended refractive index `(n, k)` on [`Self::wavelengths`].
    #[must_use]
    pub fn blended_refractive_index(&self) -> (&RVector, &RVector) {
        (&self.n_blend, &self.k_blend)
    }

    /// Average grain density in $\mathrm{g\,cm^{-3}}$ including porosity.
    #[must_use]
    pub fn average_density(&self) -> f64 {
        self.rho_av
    }

    /// Size distribution moments $[\langle a \rangle, \langle a^2 \rangle^{1/2}, \langle a^3 \rangle^{1/3}]$.
    #[must_use]
    pub fn size_moments(&self) -> [f64; 3] {
        self.config.size_distribution.moments()
    }

    /// Computes opacities and Mueller matrices in memory (no filesystem writes).
    ///
    /// # Errors
    /// Returns [`KappaError`] if size-grid construction or any scattering solver fails.
    pub fn compute(&self) -> Result<Particle, KappaError> {
        self.compute_range(self.amin, self.amax, self.na, false)
    }

    /// Executes the configured workflow and writes requested files to `config.outdir`.
    ///
    /// # Errors
    /// Returns [`KappaError`] if computation or file I/O fails.
    pub fn run(&self) -> Result<(), KappaError> {
        if self.config.write_grid {
            self.write_wavelength_grid()?;
        }
        if self.config.blend_only {
            self.write_blended_lnk()?;
        }

        if let Some(raw_nsub) = self.config.split_subgrains {
            let nsub = if raw_nsub > 1 && raw_nsub.is_multiple_of(2) {
                raw_nsub + 1
            } else {
                raw_nsub
            };
            let afact = (self.amax / self.amin).powf(1.0 / self.na as f64);
            let afsub = if nsub > 1 {
                afact.powf(1.0 / (nsub - 1) as f64)
            } else {
                1.0
            };
            let half = (nsub / 2) as i32;

            if self.config.write_grid {
                let asplit = self.amin * afact.powf(0.5);
                let (r, nr) = self.build_size_grid(
                    asplit * afsub.powi(-half),
                    asplit * afsub.powi(half),
                    nsub,
                )?;
                self.write_sizedis_file(&r, &nr)?;
                return Ok(());
            }
            if self.config.blend_only {
                return Ok(());
            }

            let progress = Progress::new("kappa", "sizes", self.na);
            (0..self.na).into_par_iter().try_for_each(|ia| {
                let asplit = self.amin * afact.powf(ia as f64 + 0.5);
                let aminsplit = asplit * afsub.powi(-half);
                let amaxsplit = asplit * afsub.powi(half);
                let p = self.compute_range(aminsplit, amaxsplit, nsub, true)?;
                if !p.is_ok && !self.config.quiet {
                    let keep_approx = matches!(
                        self.config.method,
                        ScatteringMethod::Mmf {
                            keep_approximate_scattering: true,
                            ..
                        }
                    );
                    if keep_approx {
                        eprintln!(
                            "Warning: opacities OK, but some F_nn,g_asym may not be accurate (particle {}, a={:.3}, lam<={:.3})",
                            ia + 1,
                            asplit,
                            p.is_ok_lmin
                        );
                    } else {
                        eprintln!(
                            "Warning: opacities OK, but some F_nn,g_asym are set to zero (particle {}, a={:.3}, lam<={:.3})",
                            ia + 1,
                            asplit,
                            p.is_ok_lmin
                        );
                    }
                }
                let fname = format!("{}_{:03}", self.config.label, ia + 1);
                self.write_opacities_with_bounds(&fname, &p, aminsplit, amaxsplit, nsub)?;
                if !self.config.quiet {
                    progress.add(1);
                }
                Ok::<(), KappaError>(())
            })?;
            return Ok(());
        }

        if self.config.write_grid {
            let (r, nr) = self.build_size_grid(self.amin, self.amax, self.na)?;
            self.write_sizedis_file(&r, &nr)?;
            return Ok(());
        }
        if self.config.blend_only {
            return Ok(());
        }

        let p = self.compute()?;
        if !p.is_ok && !self.config.quiet {
            let keep_approx = matches!(
                self.config.method,
                ScatteringMethod::Mmf {
                    keep_approximate_scattering: true,
                    ..
                }
            );
            if keep_approx {
                eprintln!("Warning: opacities OK, but some F_nn,g_asym may not be accurate");
            } else {
                eprintln!("Warning: opacities OK, but some F_nn,g_asym are set to zero");
            }
        }

        self.write_opacities(&self.config.label, &p)
    }

    fn build_size_grid(
        &self,
        amin: f64,
        amax: f64,
        ns: usize,
    ) -> Result<(RVector, RVector), KappaError> {
        match &self.config.size_distribution {
            SizeDistribution::Tabulated { radii, weights, .. } => {
                let r = RVector::from_vec(radii.clone());
                let mut nr = RVector::from_vec(weights.clone());
                let tot: f64 = r
                    .iter()
                    .zip(nr.iter())
                    .map(|(&rv, &nrv)| nrv * rv.powi(3))
                    .sum();
                if tot <= 0.0 {
                    return Err(KappaError::InvalidConfig(
                        "tabulated size distribution has zero total volume".into(),
                    ));
                }
                nr /= tot;
                Ok((r, nr))
            }
            SizeDistribution::Monodisperse { radius } => {
                let r = RVector::from_elem(1, *radius);
                let nr = RVector::from_elem(1, 1.0 / radius.powi(3));
                Ok((r, nr))
            }
            SizeDistribution::PowerLaw { .. }
            | SizeDistribution::LogNormal { .. }
            | SizeDistribution::Normal { .. } => {
                if ns == 1 {
                    let r0 = (amin * amax).sqrt();
                    return Ok((
                        RVector::from_elem(1, r0),
                        RVector::from_elem(1, 1.0 / r0.powi(3)),
                    ));
                }
                let mut r = RVector::zeros(ns);
                let mut nr = RVector::zeros(ns);
                let aminlog = amin.log10();
                let amaxlog = amax.log10();
                let nsf = (ns - 1) as f64;
                let mut tot = 0.0;
                for (is, (r_val, nr_val)) in r.iter_mut().zip(nr.iter_mut()).enumerate() {
                    *r_val = if is == 0 {
                        amin
                    } else if is == ns - 1 {
                        amax
                    } else {
                        10.0_f64.powf(aminlog + (amaxlog - aminlog) * (is as f64) / nsf)
                    };
                    // On a logarithmic grid (d ln a = const), bin weight is f(a) * a * d ln a.
                    *nr_val = match &self.config.size_distribution {
                        SizeDistribution::Normal { mean, sigma, .. } => {
                            let expo = 0.5 * ((*r_val - mean) / sigma).powi(2);
                            if expo > 99.0 {
                                0.0
                            } else {
                                *r_val * (-expo).exp()
                            }
                        }
                        SizeDistribution::LogNormal { mean, sigma, .. } => {
                            let expo = 0.5 * ((*r_val / mean).ln() / sigma).powi(2);
                            if expo > 99.0 { 0.0 } else { (-expo).exp() }
                        }
                        SizeDistribution::PowerLaw { slope, .. } => r_val.powf(1.0 - slope),
                        SizeDistribution::Monodisperse { .. }
                        | SizeDistribution::Tabulated { .. } => unreachable!(),
                    };
                    tot += *nr_val * r_val.powi(3);
                }
                if tot <= 0.0 {
                    return Err(KappaError::InvalidConfig(
                        "size distribution has zero total volume".into(),
                    ));
                }
                nr /= tot;
                Ok((r, nr))
            }
        }
    }

    fn compute_range(
        &self,
        amin: f64,
        amax: f64,
        ns: usize,
        serial_wavelengths: bool,
    ) -> Result<Particle, KappaError> {
        let (r, nr) = self.build_size_grid(amin, amax, ns)?;
        let ns_actual = r.len();

        let (nf, ifmn, f, wf) = match self.config.method {
            ScatteringMethod::Dhs { fmax, .. } if fmax > 0.01 => {
                let (nodes, weights) = gauss_legendre(0.01_f64, fmax, 20);
                (20, 11, nodes, weights)
            }
            ScatteringMethod::Dhs { fmax, .. } => {
                (1, 0, RVector::from_elem(1, fmax), RVector::ones(1))
            }
            _ => (1, 0, RVector::zeros(1), RVector::ones(1)),
        };

        let nang = self.config.n_angles;
        let mut mu = RVector::zeros(nang);
        let half_f = (nang / 2) as f64;
        for (j, val) in mu.iter_mut().take(nang / 2).enumerate() {
            let theta = (j as f64 + 0.5) / half_f * PI / 2.0;
            *val = theta.cos();
        }

        let kps = KappaState {
            ns: ns_actual,
            nf,
            ifmn,
            r: &r,
            nr: &nr,
            f: &f,
            wf: &wf,
            mu: &mu,
        };

        let nlam = self.wavelengths.len();
        let results: Result<Vec<KappaResult>, KappaError> = if serial_wavelengths {
            (0..nlam)
                .map(|ilam| self.over_wavelengths(ilam, &kps))
                .collect()
        } else {
            let progress = Progress::new("kappa", "wavelengths", nlam);
            (0..nlam)
                .into_par_iter()
                .map(|ilam| {
                    let res = self.over_wavelengths(ilam, &kps);
                    if !self.config.quiet {
                        progress.add(1);
                    }
                    res
                })
                .collect()
        };

        let lamrs = results?;
        let rho = lamrs[0].mass / lamrs[0].vol;
        let k_ext = lamrs.iter().map(|r| r.k_ext).collect::<RVector>();
        let k_sca = lamrs.iter().map(|r| r.k_sca).collect::<RVector>();
        let k_abs = lamrs.iter().map(|r| r.k_abs).collect::<RVector>();
        let g = lamrs.iter().map(|r| r.g).collect::<RVector>();
        let trust = lamrs.iter().map(|r| r.trust).collect::<BVector>();
        let mueller = lamrs
            .into_iter()
            .map(|r| r.mueller)
            .collect::<Vec<Mueller>>();

        let mut is_ok = true;
        let mut is_ok_lmin = 0.0;
        for ilam in 0..nlam {
            if !trust[ilam] {
                is_ok = false;
                is_ok_lmin = self.wavelengths[ilam];
            }
        }

        Ok(Particle {
            rv: (amin * amax).sqrt(),
            rvmin: amin,
            rvmax: amax,
            rho,
            k_abs,
            k_ext,
            k_sca,
            g,
            f: mueller,
            trust,
            is_ok,
            is_ok_lmin,
        })
    }

    fn over_wavelengths(&self, ilam: usize, kps: &KappaState) -> Result<KappaResult, KappaError> {
        let qabsdqext_min = 1e-4;
        let wavelength = self.wavelengths[ilam];
        let wvno = 2.0 * PI / wavelength;
        let nang = self.config.n_angles;
        let nangf = nang as f64;

        let mut f11 = RVector::zeros(nang);
        let mut f12 = RVector::zeros(nang);
        let mut f22 = RVector::zeros(nang);
        let mut f33 = RVector::zeros(nang);
        let mut f34 = RVector::zeros(nang);
        let mut f44 = RVector::zeros(nang);
        let mut csca = 0.0;
        let mut cabs = 0.0;
        let mut cext = 0.0;
        let mut smat_nbad = 0;
        let mut mass = 0.0;
        let mut vol = 0.0;

        let mut mie_f11 = RVector::zeros(nang);
        let mut mie_f12 = RVector::zeros(nang);
        let mut mie_f22 = RVector::zeros(nang);
        let mut mie_f33 = RVector::zeros(nang);
        let mut mie_f34 = RVector::zeros(nang);
        let mut mie_f44 = RVector::zeros(nang);
        let thmin = 180.0 * 0.5 / nangf;
        let thmax = 180.0 * (nangf - 0.5) / nangf;
        let step = (thmax - thmin) / (nangf - 1.0);
        let sin_th0 = (PI * 0.5 / nangf).sin();

        let eval_mie = |rad: f64,
                        mie_f11: &mut RVector,
                        mie_f12: &mut RVector,
                        mie_f22: &mut RVector,
                        mie_f33: &mut RVector,
                        mie_f34: &mut RVector,
                        mie_f44: &mut RVector|
         -> Result<(f64, f64), KappaError> {
            let lmie = if rad / wavelength < 5000.0 {
                wavelength
            } else {
                rad / 5000.0
            };
            let miec = MieConfig {
                nangle: nang,
                delta: 1e-12,
                thmin,
                thmax,
                step,
                lam: lmie,
                cmm: Complex::new(self.n_blend[ilam], self.k_blend[ilam]),
                rad,
            };
            let mie_result = de_rooij_1984(&miec)?;
            for j in 0..nang {
                mie_f11[j] = mie_result.f_11[j];
                mie_f12[j] = mie_result.f_12[j];
                mie_f22[j] = mie_result.f_11[j];
                mie_f33[j] = mie_result.f_33[j];
                mie_f34[j] = mie_result.f_34[j];
                mie_f44[j] = mie_result.f_33[j];
            }
            Ok((mie_result.c_sca, mie_result.c_ext))
        };

        let normalize_forward_peak = |mie_f11: &mut RVector, mie_f22: &mut RVector| {
            let mut tot = 0.0;
            let mut tot2 = 0.0;
            for j in 0..nang {
                let sin_th = (PI * (j as f64 + 0.5) / nangf).sin();
                tot += mie_f11[j] * sin_th;
                tot2 += sin_th;
            }
            mie_f11[0] = (mie_f11[0] + (tot2 - tot) / sin_th0).max(0.0);
            mie_f22[0] = mie_f11[0];
        };

        for is in 0..kps.ns {
            let r1 = kps.r[is];
            match self.config.method {
                ScatteringMethod::Mie { .. } => {
                    let (scat_mie, mut ext_mie) = eval_mie(
                        r1,
                        &mut mie_f11,
                        &mut mie_f12,
                        &mut mie_f22,
                        &mut mie_f33,
                        &mut mie_f34,
                        &mut mie_f44,
                    )?;
                    normalize_forward_peak(&mut mie_f11, &mut mie_f22);
                    let weight = kps.nr[is];
                    for j in 0..nang {
                        f11[j] += weight * mie_f11[j] * scat_mie;
                        f12[j] += weight * mie_f12[j] * scat_mie;
                        f22[j] += weight * mie_f22[j] * scat_mie;
                        f33[j] += weight * mie_f33[j] * scat_mie;
                        f34[j] += weight * mie_f34[j] * scat_mie;
                        f44[j] += weight * mie_f44[j] * scat_mie;
                    }
                    let mut camie = ext_mie - scat_mie;
                    if camie < ext_mie * qabsdqext_min {
                        camie = ext_mie * qabsdqext_min;
                        ext_mie = camie + scat_mie;
                    }
                    cext += weight * ext_mie;
                    csca += weight * scat_mie;
                    cabs += weight * camie;
                    let v_grain = 4.0 * PI * r1.powi(3) / 3.0;
                    mass += weight * self.rho_av * v_grain;
                    vol += weight * v_grain;
                }
                ScatteringMethod::Dhs {
                    size_limit,
                    dhs_size_limit,
                    ..
                } => {
                    let m_in = Complex::new(1.0, 0.0);
                    let too_large = kps.f[0] > 0.0 && r1 * wvno > size_limit;
                    if too_large {
                        let rad = r1 / (1.0 - kps.f[kps.ifmn]).powf(1.0 / 3.0);
                        let (scat_mie, mut ext_mie) = eval_mie(
                            rad,
                            &mut mie_f11,
                            &mut mie_f12,
                            &mut mie_f22,
                            &mut mie_f33,
                            &mut mie_f34,
                            &mut mie_f44,
                        )?;
                        normalize_forward_peak(&mut mie_f11, &mut mie_f22);
                        let weight = kps.wf.sum() * kps.nr[is];
                        for j in 0..nang {
                            f11[j] += weight * mie_f11[j] * scat_mie;
                            f12[j] += weight * mie_f12[j] * scat_mie;
                            f22[j] += weight * mie_f22[j] * scat_mie;
                            f33[j] += weight * mie_f33[j] * scat_mie;
                            f34[j] += weight * mie_f34[j] * scat_mie;
                            f44[j] += weight * mie_f44[j] * scat_mie;
                        }
                        let mut camie = ext_mie - scat_mie;
                        if camie < ext_mie * qabsdqext_min {
                            camie = ext_mie * qabsdqext_min;
                            ext_mie = camie + scat_mie;
                        }
                        cext += weight * ext_mie;
                        csca += weight * scat_mie;
                        cabs += weight * camie;
                        let v_grain = 4.0 * PI * r1.powi(3) / 3.0;
                        mass += weight * self.rho_av * v_grain;
                        vol += weight * v_grain;
                        continue;
                    }

                    for ifn in 0..kps.nf {
                        let mut spheres = kps.f[ifn] == 0.0;
                        let mut scat_mie: f64 = 0.0;
                        let mut ext_mie: f64 = 0.0;

                        if !spheres {
                            let rad = r1 / (1.0 - kps.f[ifn]).powf(1.0 / 3.0);
                            let rcore = rad * kps.f[ifn].powf(1.0 / 3.0);
                            let mconj = Complex::new(self.n_blend[ilam], -self.k_blend[ilam]);
                            let wvno_1 = wvno.min(dhs_size_limit / rad);
                            let dhs_cfg = DHSConfig {
                                r_core: rcore,
                                r_shell: rad,
                                wave_number: wvno_1,
                                r_indsh: mconj,
                                r_indco: m_in,
                                mu: kps.mu,
                                numang: nang / 2,
                                max_angle: nang,
                            };

                            if let Ok(dhs_result) = toon_ackerman_1981(&dhs_cfg) {
                                let ext_mie_factor = dhs_result.q_ext * PI * rad.powi(2);
                                let scat_mie_factor = dhs_result.q_sca * PI * rad.powi(2);
                                // factor absorbs the 1/2 in F11, F12; s10 and d10 need 2*factor.
                                let factor = if scat_mie_factor > 0.0 {
                                    2.0 * PI / (scat_mie_factor * wvno_1.powi(2))
                                } else {
                                    0.0
                                };
                                let factor2 = 2.0 * factor;
                                for j in 0..(nang / 2) {
                                    let back = nang - j - 1;
                                    mie_f11[j] =
                                        (dhs_result.m1[[j, 0]] + dhs_result.m0[[j, 0]]) * factor;
                                    mie_f12[j] =
                                        (dhs_result.m1[[j, 0]] - dhs_result.m0[[j, 0]]) * factor;
                                    mie_f22[j] = mie_f11[j];
                                    mie_f33[j] = dhs_result.s10[[j, 0]] * factor2;
                                    mie_f34[j] = -dhs_result.d10[[j, 0]] * factor2;
                                    mie_f44[j] = mie_f33[j];

                                    mie_f11[back] =
                                        (dhs_result.m1[[j, 1]] + dhs_result.m0[[j, 1]]) * factor;
                                    mie_f12[back] =
                                        (dhs_result.m1[[j, 1]] - dhs_result.m0[[j, 1]]) * factor;
                                    mie_f22[back] = mie_f11[back];
                                    mie_f33[back] = dhs_result.s10[[j, 1]] * factor2;
                                    mie_f34[back] = -dhs_result.d10[[j, 1]] * factor2;
                                    mie_f44[back] = mie_f33[back];
                                }
                                scat_mie = scat_mie_factor;
                                ext_mie = ext_mie_factor;
                            } else {
                                spheres = true;
                            }
                        }

                        if spheres {
                            (scat_mie, ext_mie) = eval_mie(
                                r1,
                                &mut mie_f11,
                                &mut mie_f12,
                                &mut mie_f22,
                                &mut mie_f33,
                                &mut mie_f34,
                                &mut mie_f44,
                            )?;
                        }

                        normalize_forward_peak(&mut mie_f11, &mut mie_f22);
                        let weight = kps.wf[ifn] * kps.nr[is];
                        for j in 0..nang {
                            f11[j] += weight * mie_f11[j] * scat_mie;
                            f12[j] += weight * mie_f12[j] * scat_mie;
                            f22[j] += weight * mie_f22[j] * scat_mie;
                            f33[j] += weight * mie_f33[j] * scat_mie;
                            f34[j] += weight * mie_f34[j] * scat_mie;
                            f44[j] += weight * mie_f44[j] * scat_mie;
                        }
                        let mut camie = ext_mie - scat_mie;
                        if camie < ext_mie * qabsdqext_min {
                            camie = ext_mie * qabsdqext_min;
                            ext_mie = camie + scat_mie;
                        }
                        cext += weight * ext_mie;
                        csca += weight * scat_mie;
                        cabs += weight * camie;
                        let v_grain = 4.0 * PI * r1.powi(3) / 3.0;
                        mass += weight * self.rho_av * v_grain;
                        vol += weight * v_grain;
                    }
                }
                ScatteringMethod::Mmf {
                    monomer_radius,
                    structure,
                    prefactor,
                    ..
                } => {
                    let m_mono = 4.0 * PI / 3.0 * monomer_radius.powi(3) * self.rho_av;
                    let v_agg = 4.0 * PI / 3.0 * r1.powi(3);
                    let m_agg = v_agg * self.rho_av;
                    let nmono = m_agg / m_mono;
                    let dfrac = match structure {
                        AggregateStructure::FractalDimension(df) => df,
                        AggregateStructure::FillingFactor(fill) => {
                            3.0 * nmono.ln() / (nmono / fill).ln()
                        }
                    };
                    let kfrac = prefactor.unwrap_or_else(|| (5.0 / 3.0).powf(dfrac / 2.0));
                    let fracc = FractalConfig {
                        solver: FractalSolver::ModifiedMeanField,
                        cutoff: FractalCutoff::Gaussian,
                        geometry: FractalGeometry::Tazaki,
                        nang: nang / 2 + 1,
                        pn: nmono,
                        r0: monomer_radius,
                        df: dfrac,
                        k0: kfrac,
                        lmd: wavelength,
                        refrel: Complex::new(self.n_blend[ilam], self.k_blend[ilam]),
                        ..Default::default()
                    };
                    let frac_result = mean_scattering(&fracc)?;
                    if frac_result.dphi > 1.0 {
                        smat_nbad += 1;
                    }
                    let factor = 4.0 * PI / wvno.powi(2);
                    for j in 0..nang {
                        let s11 =
                            f64::midpoint(frac_result.smat[[j, 0]], frac_result.smat[[j + 1, 0]])
                                * factor;
                        let s12 =
                            f64::midpoint(frac_result.smat[[j, 1]], frac_result.smat[[j + 1, 1]])
                                * factor;
                        let s33 =
                            f64::midpoint(frac_result.smat[[j, 2]], frac_result.smat[[j + 1, 2]])
                                * factor;
                        let s34 =
                            f64::midpoint(frac_result.smat[[j, 3]], frac_result.smat[[j + 1, 3]])
                                * factor;
                        f11[j] += kps.nr[is] * s11;
                        f12[j] += kps.nr[is] * s12;
                        f22[j] += kps.nr[is] * s11;
                        f33[j] += kps.nr[is] * s33;
                        f34[j] += kps.nr[is] * s34;
                        f44[j] += kps.nr[is] * s33;
                    }
                    cext += kps.nr[is] * frac_result.c_ext;
                    csca += kps.nr[is] * frac_result.c_sca;
                    cabs += kps.nr[is] * frac_result.c_abs;
                    mass += kps.nr[is] * m_agg;
                    vol += kps.nr[is] * v_agg;
                }
                ScatteringMethod::Cde => {
                    let v = 4.0 * PI * r1.powi(3) / 3.0;
                    vol += kps.nr[is] * v;
                    mass += kps.nr[is] * self.rho_av * v;
                    let m = Complex::new(self.n_blend[ilam], self.k_blend[ilam]);
                    let m2 = m.powi(2);
                    let dcabs = 2.0 * wvno * v * (m2 / (m2 - 1.0) * m2.ln()).im;
                    let dcsca = if (self.k_blend[ilam] / self.n_blend[ilam]).abs() < 1e-6 {
                        let mre2 = m.re.powi(2);
                        wvno.powi(4) * v.powi(2) / (3.0 * PI) * (mre2 - 1.0 - mre2.ln())
                    } else {
                        wvno.powi(4) * v.powi(2) * (m2 - 1.0).norm_sqr() / (3.0 * PI * m2.im)
                            * (m2 / (m2 - 1.0) * m2.ln()).im
                    };
                    cabs += kps.nr[is] * dcabs;
                    csca += kps.nr[is] * dcsca;
                    cext += kps.nr[is] * (dcabs + dcsca);
                    if is == kps.ns - 1 {
                        // Exact Rayleigh scattering matrix under Hovenier normalization
                        for j in 0..nang {
                            let theta = PI * (j as f64 + 0.5) / nangf;
                            let cos_t = theta.cos();
                            let sin_t = theta.sin();
                            let f11_r = 0.75 * (1.0 + cos_t * cos_t);
                            let f12_r = -0.75 * sin_t * sin_t;
                            let f33_r = 1.5 * cos_t;
                            f11[j] = f11_r * csca;
                            f12[j] = f12_r * csca;
                            f22[j] = f11_r * csca;
                            f33[j] = f33_r * csca;
                            f34[j] = 0.0;
                            f44[j] = f33_r * csca;
                        }
                    }
                }
            }
        }

        let mut k_ext = 1e4 * cext / mass;
        let mut k_sca = 1e4 * csca / mass;
        let k_abs = 1e4 * cabs / mass;
        if csca > 0.0 {
            f11 /= csca;
            f12 /= csca;
            f22 /= csca;
            f33 /= csca;
            f34 /= csca;
            f44 /= csca;
        }

        if let Some(chop_angle) = self.config.chop_angle
            && chop_angle > 0.0
        {
            let ichop = (chop_angle / (180.0 / nangf)) as usize;
            if ichop > 0 && ichop < nang {
                for i in 0..ichop {
                    f11[i] = f11[ichop];
                    f12[i] = f12[ichop];
                    f22[i] = f22[ichop];
                    f33[i] = f33[ichop];
                    f34[i] = f34[ichop];
                    f44[i] = f44[ichop];
                }
            }
            let mut tot = 0.0;
            let mut tot2 = 0.0;
            for j in 0..nang {
                let sin_th = (PI * (j as f64 + 0.5) / nangf).sin();
                tot += f11[j] * sin_th;
                tot2 += sin_th;
            }
            if tot > 0.0 {
                let scale = tot2 / tot;
                f11 *= scale;
                f12 *= scale;
                f22 *= scale;
                f33 *= scale;
                f34 *= scale;
                f44 *= scale;
                k_sca *= tot / tot2;
                k_ext = k_sca + k_abs;
            }
        }

        let mut tot = 0.0;
        let mut g = 0.0;
        let trust = smat_nbad == 0;
        for i in 0..nang {
            let theta = PI * (i as f64 + 0.5) / nangf;
            let sin_th = theta.sin();
            g += f11[i] * theta.cos() * sin_th;
            tot += f11[i] * sin_th;
        }
        if tot > 0.0 {
            g /= tot;
        }

        let keep_approx = matches!(
            self.config.method,
            ScatteringMethod::Mmf {
                keep_approximate_scattering: true,
                ..
            }
        );
        if !trust && !keep_approx {
            g = 0.0;
            f11.fill(0.0);
            f12.fill(0.0);
            f22.fill(0.0);
            f33.fill(0.0);
            f34.fill(0.0);
            f44.fill(0.0);
        }

        Ok(KappaResult {
            mueller: Mueller {
                f11,
                f12,
                f22,
                f33,
                f34,
                f44,
            },
            mass,
            vol,
            k_ext,
            k_sca,
            k_abs,
            g,
            trust,
        })
    }

    fn create_output_file(&self, filename: &str) -> Result<File, KappaError> {
        fs::create_dir_all(&self.config.outdir)?;
        let filepath = self.config.outdir.join(filename);
        if filepath.exists() {
            fs::remove_file(&filepath)?;
        }
        Ok(File::create(filepath)?)
    }

    /// Writes the wavelength grid to `kappa_lam.dat` inside `config.outdir`.
    ///
    /// # Errors
    /// Returns [`KappaError::Io`] if file creation or writing fails.
    pub fn write_wavelength_grid(&self) -> Result<(), KappaError> {
        let file = self.create_output_file("kappa_lam.dat")?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "# Wavelength grid written by kappa, can be read back in with --lam-file kappa_lam.dat"
        )?;
        writeln!(writer, "# First line: number of wavelengths")?;
        writeln!(writer, "# Then one lambda per line, in micrometer")?;
        writeln!(writer, "{}", self.wavelengths.len())?;
        for &lam in &self.wavelengths {
            writeln!(writer, "{lam:18.5e}")?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Writes the effective-medium blended refractive index table to `kappa_mix.lnk`.
    ///
    /// # Errors
    /// Returns [`KappaError::Io`] if file creation or writing fails.
    pub fn write_blended_lnk(&self) -> Result<(), KappaError> {
        let file = self.create_output_file("kappa_mix.lnk")?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "{:5} {:5.2}", self.wavelengths.len(), self.rho_av)?;
        for ilam in 0..self.wavelengths.len() {
            writeln!(
                writer,
                "{:15.5e}{:15.5e}{:15.5e}",
                self.wavelengths[ilam], self.n_blend[ilam], self.k_blend[ilam]
            )?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Writes a grain size distribution table to `kappa_sd.dat`.
    ///
    /// # Errors
    /// Returns [`KappaError::Io`] if file creation or writing fails.
    pub fn write_sizedis_file(&self, r: &RVector, nr: &RVector) -> Result<(), KappaError> {
        let file = self.create_output_file("kappa_sd.dat")?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "# Size distribution written by kappa, can be read in with --size-file kappa_sd.dat"
        )?;
        if self.config.split_subgrains.is_some() {
            writeln!(
                writer,
                "# This is only the first subparticle because of --split"
            )?;
        }
        writeln!(writer, "# First line: Number of grain size bins NA")?;
        writeln!(writer, "# Then NA lines with:  agrain[um]  n(a)")?;
        writeln!(writer, "#   n(a) is the number of grains in the bin.")?;
        writeln!(
            writer,
            "#   In a linear grid      (da   =const), this would be n(a) = f(a)*da"
        )?;
        writeln!(
            writer,
            "#   In a logarithmic grid (dloga=const), this would be n(a) = f(a)*a*dloga."
        )?;
        writeln!(
            writer,
            "#   In an arbitrary grid,  just give the number of grains in the bin."
        )?;
        writeln!(
            writer,
            "# No normalization is necessary, it is done automatically."
        )?;
        writeln!(writer, "{}", r.len())?;

        for (&rv, &nrv) in r.iter().zip(nr.iter()) {
            writeln!(writer, "{rv:18.5e} {nrv:18.5e}")?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Writes computed opacities (and optionally the scattering matrix) to `{fname}.{ext}`.
    ///
    /// # Errors
    /// Returns [`KappaError::Io`] if file creation or writing fails.
    pub fn write_opacities(&self, fname: &str, p: &Particle) -> Result<(), KappaError> {
        self.write_opacities_with_bounds(fname, p, self.amin, self.amax, self.na)
    }

    fn write_opacities_with_bounds(
        &self,
        fname: &str,
        p: &Particle,
        amin: f64,
        amax: f64,
        na: usize,
    ) -> Result<(), KappaError> {
        let (ext, ml) = if self.config.for_radmc {
            ("inp", "Z11 Z12 Z22 Z33 Z34 Z44")
        } else {
            ("dat", "F11 F12 F22 F33 F34 F44")
        };
        let file = self.create_output_file(&format!("{fname}.{ext}"))?;
        let mut writer = BufWriter::new(file);

        let cc = "#";
        let ameans = self.config.size_distribution.moments_in_range(amin, amax);
        let nlam = self.wavelengths.len();
        let nang = self.config.n_angles;
        let chop_angle = self.config.chop_angle.unwrap_or(0.0);
        let fmax_header = match self.config.method {
            ScatteringMethod::Dhs { fmax, .. } => fmax,
            _ => 0.0,
        };

        writeln!(
            writer,
            "{cc}============================================================================"
        )?;
        writeln!(
            writer,
            "{} Opacities computed by kappa        <a^n>= {:11.4e} {:11.4e} {:11.4e}",
            cc, ameans[0], ameans[1], ameans[2]
        )?;
        match self.config.method {
            ScatteringMethod::Mmf {
                monomer_radius,
                structure,
                ..
            } => {
                let (val, label) = match structure {
                    AggregateStructure::FractalDimension(df) => (df, "(fractal dimension)"),
                    AggregateStructure::FillingFactor(filling) => (filling, "(filling factor)"),
                };
                writeln!(
                    writer,
                    "{cc} Method:   MMF  a0={monomer_radius:7.3}  Struct={val:7.3}{label}"
                )?;
            }
            ScatteringMethod::Cde => {
                writeln!(writer, "{cc} Method:   CDE")?;
            }
            ScatteringMethod::Mie { .. } => {
                writeln!(writer, "{cc} Method:   Mie  fmax=  0.000")?;
            }
            ScatteringMethod::Dhs { fmax, .. } => {
                writeln!(writer, "{cc} Method:   DHS  fmax={fmax:7.3}")?;
            }
        }
        writeln!(writer, "{cc} Parameters:")?;
        match &self.config.size_distribution {
            SizeDistribution::Tabulated { source, .. } => {
                let label = source
                    .as_ref()
                    .map_or_else(|| "<tabulated>".to_string(), |s| s.display().to_string());
                writeln!(
                    writer,
                    "{cc}   amin [um]={amin:11.3} amax [um]={amax:11.3}  na  ={na:5}    file={label}"
                )?;
            }
            SizeDistribution::LogNormal { mean, sigma, .. } => {
                writeln!(
                    writer,
                    "{cc}   amin [um]={amin:11.3} amax [um]={amax:11.3}  na  ={na:5}    LogNormal={mean:.4}:{sigma:.4}"
                )?;
            }
            SizeDistribution::Normal { mean, sigma, .. } => {
                writeln!(
                    writer,
                    "{cc}   amin [um]={amin:11.3} amax [um]={amax:11.3}  na  ={na:5}    Normal={mean:.4}:{sigma:.4}"
                )?;
            }
            SizeDistribution::PowerLaw { slope, .. } => {
                writeln!(
                    writer,
                    "{cc}   amin [um]={amin:11.3} amax [um]={amax:11.3}  na  ={na:5}     apow={slope:10.2}"
                )?;
            }
            SizeDistribution::Monodisperse { radius } => {
                writeln!(
                    writer,
                    "{cc}   amin [um]={radius:11.3} amax [um]={radius:11.3}  na  =    1     monodisperse"
                )?;
            }
        }
        writeln!(
            writer,
            "{cc}   lmin [um]={:11.3} lmax [um]={:11.3}  nlam={nlam:5}     nang={nang:6}",
            self.lmin, self.lmax
        )?;
        writeln!(
            writer,
            "{cc}   porosity ={:11.3} p_mantle ={:11.3}  fmax={fmax_header:9.2} chop=  {chop_angle:4.1}",
            self.config.core_porosity, self.config.mantle_porosity
        )?;

        writeln!(writer, "{cc} Composition:")?;
        writeln!(writer, "{cc}  Where   mfrac  rho   Material")?;
        writeln!(
            writer,
            "{cc}  -----   -----  ----  -----------------------------------------------------"
        )?;
        for mat in &self.materials {
            writeln!(
                writer,
                "{cc}  {:?} {:7.3} {:6.2}  {}",
                mat.kind, mat.mass_fraction, mat.density, mat.label
            )?;
        }
        if self.rho_av > 0.0 {
            writeln!(
                writer,
                "{cc}  - - -   - - -  -  -  - - - - - - - - - - - - - - - - - - - - - - - - - - -"
            )?;
            let label = if (self.config.core_porosity + self.config.mantle_porosity) > 0.0 {
                "materials and vacuum"
            } else {
                "materials"
            };
            writeln!(
                writer,
                "{cc}  {:<6} {:7.3} {:6.2}  mixture of {:3} {label}",
                "grain",
                1.0,
                self.rho_av,
                self.materials.len()
            )?;
        }
        writeln!(
            writer,
            "{cc}----------------------------------------------------------------------------"
        )?;
        writeln!(
            writer,
            "{cc}============================================================================"
        )?;

        let is_sparse = !self.config.sparse_ranges.is_empty();
        if self.config.write_scatter {
            if self.config.for_radmc {
                writeln!(
                    writer,
                    "# Output file formatted for RADMC-3D, dustkapscatmat, RADMC normalization"
                )?;
            } else {
                writeln!(
                    writer,
                    "# Standard output file, scattering matrix normalization: <F11>=1/sr"
                )?;
            }
            writeln!(
                writer,
                "#    iformat                                     ! format number"
            )?;
            writeln!(
                writer,
                "#    nlam                                        ! number of wavelengths"
            )?;
            writeln!(
                writer,
                "#    nang                                        ! number of angles 0-180"
            )?;
            writeln!(writer, "#")?;
            if is_sparse {
                writeln!(
                    writer,
                    "#    lam(1) kabs(1) ksca(1) g(1) imat(1)         ! um, 2x cm^2/g, none, flg"
                )?;
            } else {
                writeln!(
                    writer,
                    "#    lam(1)    kabs(1)    ksca(1)    g(1)        ! um, cm^2/g, cm^2/g, none"
                )?;
            }
            writeln!(writer, "#    ...")?;
            writeln!(writer, "#")?;
            writeln!(
                writer,
                "#    {ml}                     ! (ilam=   1,iang=   1)"
            )?;
            writeln!(
                writer,
                "#============================================================================"
            )?;

            let iformat = match (self.config.for_radmc, is_sparse) {
                (true, false) => 1,
                (true, true) => 101,
                (false, false) => 0,
                (false, true) => 100,
            };
            writeln!(writer, "{iformat}")?;
            writeln!(writer, "{nlam}")?;
            writeln!(
                writer,
                "{}",
                if self.config.for_radmc {
                    nang + 1
                } else {
                    nang
                }
            )?;
            writeln!(writer)?;

            for ilam in 0..nlam {
                if is_sparse {
                    let flag = usize::from(self.sparse_indices.contains(&ilam));
                    writeln!(
                        writer,
                        "{:15.6e} {:15.6e} {:15.6e} {:15.6e}{flag:5}",
                        self.wavelengths[ilam], p.k_abs[ilam], p.k_sca[ilam], p.g[ilam]
                    )?;
                } else {
                    writeln!(
                        writer,
                        "{:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                        self.wavelengths[ilam], p.k_abs[ilam], p.k_sca[ilam], p.g[ilam]
                    )?;
                }
            }

            writeln!(writer)?;
            let nangf = nang as f64;
            if self.config.for_radmc {
                for iang in 0..=nang {
                    writeln!(writer, "{:11.5}", (iang as f64) / nangf * 180.0)?;
                }
            } else {
                for iang in 0..nang {
                    writeln!(writer, "{:11.5}", (iang as f64 + 0.5) / nangf * 180.0)?;
                }
            }

            writeln!(writer)?;
            for ilam in 0..nlam {
                if !is_sparse || self.sparse_indices.contains(&ilam) {
                    let m = &p.f[ilam];
                    if self.config.for_radmc {
                        let mut z11 = vec![0.0; nang + 1];
                        let mut z12 = vec![0.0; nang + 1];
                        let mut z22 = vec![0.0; nang + 1];
                        let mut z33 = vec![0.0; nang + 1];
                        let mut z34 = vec![0.0; nang + 1];
                        let mut z44 = vec![0.0; nang + 1];
                        for iang in 0..=nang {
                            let i1 = iang.saturating_sub(1);
                            let i2 = iang.min(nang - 1);
                            z11[iang] = f64::midpoint(m.f11[i1], m.f11[i2]);
                            z12[iang] = f64::midpoint(m.f12[i1], m.f12[i2]);
                            z22[iang] = f64::midpoint(m.f22[i1], m.f22[i2]);
                            z33[iang] = f64::midpoint(m.f33[i1], m.f33[i2]);
                            z34[iang] = f64::midpoint(m.f34[i1], m.f34[i2]);
                            z44[iang] = f64::midpoint(m.f44[i1], m.f44[i2]);
                        }
                        let mut tot = 0.0;
                        for iang in 1..=nang {
                            let theta1 = ((iang - 1) as f64) / nangf * PI;
                            let theta2 = (iang as f64) / nangf * PI;
                            let dmu = theta1.cos() - theta2.cos();
                            tot += f64::midpoint(z11[iang - 1], z11[iang]) * dmu;
                        }
                        tot *= 2.0 * PI;
                        let scale = if tot > 0.0 { p.k_sca[ilam] / tot } else { 0.0 };
                        for iang in 0..=nang {
                            writeln!(
                                writer,
                                "{:15.6e} {:15.6e} {:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                                scale * z11[iang],
                                scale * z12[iang],
                                scale * z22[iang],
                                scale * z33[iang],
                                scale * z34[iang],
                                scale * z44[iang],
                            )?;
                        }
                    } else {
                        for i in 0..nang {
                            writeln!(
                                writer,
                                "{:15.6e} {:15.6e} {:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                                m.f11[i], m.f12[i], m.f22[i], m.f33[i], m.f34[i], m.f44[i]
                            )?;
                        }
                    }
                }
            }
        } else {
            let header_line = if self.config.for_radmc {
                "# Output file formatted for RADMC-3D, dustkappa, no scattering matrix"
            } else {
                "# Standard output file, no scattering matrix"
            };
            writeln!(writer, "{header_line}")?;
            writeln!(writer, "#    iformat")?;
            writeln!(writer, "#    nlambda")?;
            writeln!(
                writer,
                "#    lambda[um]  kabs [cm^2/g]  ksca [cm^2/g]    g_asymmetry"
            )?;
            writeln!(
                writer,
                "#============================================================================"
            )?;
            writeln!(writer, "{}", 3)?;
            writeln!(writer, "{nlam}")?;

            for i in 0..nlam {
                writeln!(
                    writer,
                    "{:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                    self.wavelengths[i], p.k_abs[i], p.k_sca[i], p.g[i]
                )?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

fn compute_sparse_indices(wavelengths: &RVector, ranges: &[(f64, f64)]) -> HashSet<usize> {
    let mut indices = HashSet::new();
    let nlam = wavelengths.len();
    if ranges.is_empty() || nlam == 0 {
        return indices;
    }

    let mut adjusted: Vec<(f64, f64)> = ranges
        .iter()
        .map(|&(a, b)| if a <= b { (a, b) } else { (b, a) })
        .collect();

    if nlam >= 2 {
        let f = wavelengths[nlam - 1] / wavelengths[nlam - 2];
        for (lo, hi) in &mut adjusted {
            if *lo > 0.0 && *hi / *lo < f {
                *lo /= f.sqrt() / 1.0001;
                *hi *= f.sqrt() / 1.0001;
            }
        }
    }

    for (il, &lam) in wavelengths.iter().enumerate() {
        if adjusted.iter().any(|&(lo, hi)| lam >= lo && lam <= hi) {
            indices.insert(il);
            if il > 0 {
                indices.insert(il - 1);
            }
            if il + 1 < nlam {
                indices.insert(il + 1);
            }
        }
    }
    indices
}

#[derive(Debug)]
struct KappaState<'a> {
    ns: usize,
    nf: usize,
    ifmn: usize,
    r: &'a RVector,
    nr: &'a RVector,
    f: &'a RVector,
    wf: &'a RVector,
    mu: &'a RVector,
}

/// Six independent elements of the Mueller scattering matrix for randomly oriented particles.
pub struct Mueller {
    pub f11: RVector,
    pub f12: RVector,
    pub f22: RVector,
    pub f33: RVector,
    pub f34: RVector,
    pub f44: RVector,
}

/// Computed opacities, asymmetry factor, and Mueller matrices for a dust population.
#[derive(Default)]
pub struct Particle {
    pub rv: f64,
    pub rvmin: f64,
    pub rvmax: f64,
    pub rho: f64,
    pub k_abs: RVector,
    pub k_sca: RVector,
    pub k_ext: RVector,
    pub g: RVector,
    pub f: Vec<Mueller>,
    pub trust: BVector,
    pub is_ok: bool,
    pub is_ok_lmin: f64,
}

struct KappaResult {
    mueller: Mueller,
    mass: f64,
    vol: f64,
    k_ext: f64,
    k_sca: f64,
    k_abs: f64,
    g: f64,
    trust: bool,
}

/// Runtime-loaded material optical constants from a `.lnk` file.
#[derive(Debug)]
pub struct Component {
    /// Name of the component.
    pub name: String,
    /// Class of the component.
    pub class: String,
    /// State of the component.
    pub state: String,
    /// Density of the component in $\mathrm{g\,cm^{-3}}$.
    pub rho: f64,
    /// Number of wavelength points.
    pub size: usize,
    /// Wavelengths in $\mu\mathrm{m}$.
    pub l0: RVector,
    /// Real part of the refractive index.
    pub n0: RVector,
    /// Imaginary part of the refractive index.
    pub k0: RVector,
}

impl From<StaticComponent> for Component {
    fn from(sc: StaticComponent) -> Self {
        Self::from(&sc)
    }
}

impl From<&StaticComponent> for Component {
    fn from(sc: &StaticComponent) -> Self {
        Component {
            name: sc.name.to_string(),
            class: sc.class.to_string(),
            state: sc.state.to_string(),
            rho: sc.rho,
            size: sc.size,
            l0: RVector::from(sc.l0.to_vec()),
            n0: RVector::from(sc.n0.to_vec()),
            k0: RVector::from(sc.k0.to_vec()),
        }
    }
}

fn bruggeman_blend(abun: &[f64], e_in: &[Complex64]) -> Result<(Complex64, bool), KappaError> {
    let mut abunvac = 1.0 - abun.iter().sum::<f64>();
    let mvac = Complex::new(1.0, 0.0);
    if abunvac < 0.0 {
        if abunvac.abs() < 1e-5 {
            abunvac = 0.0;
        } else {
            return Err(KappaError::Numerical(
                "unnormalized volume fractions in Bruggeman mixing".into(),
            ));
        }
    }

    let mut mm = mvac;
    let mut tot = Complex::new(0.0, 0.0);
    let mut me = Complex::new(0.0, 0.0);
    for _ in 0..100 {
        tot = ((mvac.powi(2) - mm.powi(2)) / (mvac.powi(2) + 2.0 * mm.powi(2))) * abunvac;
        tot += e_in
            .iter()
            .zip(abun.iter())
            .map(|(&e_in_j, &abun_j)| {
                ((e_in_j.powi(2) - mm.powi(2)) / (e_in_j.powi(2) + 2.0 * mm.powi(2))) * abun_j
            })
            .sum::<Complex64>();

        me = mm * ((2.0 * tot + 1.0) / (1.0 - tot)).sqrt();
        mm = me;
    }
    let converged = tot.abs() / mm.abs() <= 1e-6;
    Ok((me, converged))
}

fn maxwell_garnet_blend(m1: Complex64, m2: Complex64, vf_m: f64) -> (f64, f64) {
    let vf_c = 1.0 - vf_m;
    let me = m2.powi(2)
        * ((2.0 * m2.powi(2) + m1.powi(2) - 2.0 * vf_c * (m2.powi(2) - m1.powi(2)))
            / (2.0 * m2.powi(2) + m1.powi(2) + vf_c * (m2.powi(2) - m1.powi(2))));

    (me.sqrt().re, me.sqrt().im)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use approx::assert_relative_eq;

    use super::{
        AggregateStructure, KappaError, MaterialKind, ScatteringMethod, SizeDistribution,
        WavelengthGrid,
    };
    use super::{KappaConfig, Material};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        PathBuf::from(format!("/tmp/kappa_opac_test_{label}_{nanos}"))
    }

    #[test]
    fn diana_dhs_and_mueller_consistency() {
        let mut cfg = KappaConfig::diana();
        cfg.size_distribution = SizeDistribution::Monodisperse { radius: 1.0 };
        cfg.wavelengths = WavelengthGrid::Single { wavelength: 1.0 };
        cfg.method = ScatteringMethod::dhs(0.8);
        let p = cfg.compute().unwrap();

        assert_relative_eq!(p.k_abs[0], 5403.01, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 11232.9, max_relative = 1e-3);
        assert_relative_eq!(p.k_ext[0], 16635.9, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.878_882, max_relative = 2e-3);

        let m = &p.f[0];
        for j in 0..cfg.n_angles {
            assert_relative_eq!(m.f22[j], m.f11[j], max_relative = 1e-12);
            assert_relative_eq!(m.f44[j], m.f33[j], max_relative = 1e-12);
        }
        assert!(
            m.f33[0] / m.f11[0] > 0.95,
            "DHS F33(0)/F11(0) = {}",
            m.f33[0] / m.f11[0]
        );

        // 0 < fmax <= 0.01 single hollow sphere
        cfg.method = ScatteringMethod::dhs(0.005);
        let p_tiny = cfg.compute().unwrap();
        assert!(p_tiny.k_ext[0].is_finite() && p_tiny.k_ext[0] > 0.0);

        // Capped DHS wvno_1 normalization
        cfg.size_distribution = SizeDistribution::Monodisperse { radius: 5.0 };
        cfg.method = ScatteringMethod::Dhs {
            fmax: 0.8,
            size_limit: 1e8,
            dhs_size_limit: 10.0,
        };
        let p_cap = cfg.compute().unwrap();
        let m_cap = &p_cap.f[0];
        assert_relative_eq!(m_cap.f22[0], m_cap.f11[0], max_relative = 1e-12);
        assert!(m_cap.f33[0] / m_cap.f11[0] > 0.90);
    }

    #[test]
    fn dsharp_mie_and_large_grain_fallback() {
        let mut cfg = KappaConfig::dsharp();
        cfg.size_distribution = SizeDistribution::PowerLaw {
            min: 0.1,
            max: 100.0,
            slope: 3.5,
            bins: Some(15),
        };
        cfg.wavelengths = WavelengthGrid::Single { wavelength: 10.0 };
        cfg.method = ScatteringMethod::mie();
        let p = cfg.compute().unwrap();

        assert_relative_eq!(p.k_abs[0], 455.568, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 293.273, max_relative = 1e-3);
        assert_relative_eq!(p.k_ext[0], 748.842, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.779_307, max_relative = 1e-3);

        // Large-grain Mie fallback (rad / lam = 10_000 >= 5000)
        let big = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 1e4 },
            wavelengths: WavelengthGrid::Single { wavelength: 1.0 },
            method: ScatteringMethod::Dhs {
                fmax: 0.8,
                size_limit: 1e4,
                dhs_size_limit: 1e4,
            },
            materials: vec![Material::builtin("pyr", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        let prepared = big.prepare().unwrap();
        let p_big = prepared.compute().unwrap();
        let expected_geo = 1.5e4 / (prepared.average_density() * 1e4);
        assert!(p_big.k_ext[0] > 0.5 * expected_geo && p_big.k_ext[0] < 3.0 * expected_geo);
    }

    #[test]
    fn core_mantle_porosity_and_constant_material() {
        let cfg = KappaConfig {
            size_distribution: SizeDistribution::PowerLaw {
                min: 0.1,
                max: 10.0,
                slope: 3.5,
                bins: Some(10),
            },
            wavelengths: WavelengthGrid::Single { wavelength: 3.0 },
            method: ScatteringMethod::mie(),
            core_porosity: 0.25,
            mantle_porosity: 0.1,
            materials: vec![
                Material::builtin("pyr", MaterialKind::Core, 0.7),
                Material::builtin("c-z", MaterialKind::Core, 0.3),
                Material::builtin("h2o-w", MaterialKind::Mantle, 0.2),
            ],
            ..KappaConfig::default()
        };
        let p = cfg.compute().unwrap();
        assert_relative_eq!(p.k_abs[0], 2991.23, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 1890.09, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.730_994, max_relative = 1e-3);

        // 1 core + 2 mantles (nmant > ncore)
        let cfg2 = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 1.0 },
            wavelengths: WavelengthGrid::Single { wavelength: 3.0 },
            method: ScatteringMethod::mie(),
            core_porosity: 0.2,
            mantle_porosity: 0.1,
            materials: vec![
                Material::builtin("pyr", MaterialKind::Core, 0.7),
                Material::builtin("h2o-w", MaterialKind::Mantle, 0.2),
                Material::builtin("c-z", MaterialKind::Mantle, 0.1),
            ],
            ..KappaConfig::default()
        };
        let p2 = cfg2.compute().unwrap();
        assert_relative_eq!(p2.k_abs[0], 4747.15, max_relative = 1e-3);
        assert_relative_eq!(p2.k_sca[0], 3297.32, max_relative = 1e-3);
        assert_relative_eq!(p2.g[0], 0.699_504, max_relative = 1e-3);

        // Constant optical constants with nlam = 5
        let cmd_cfg = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 0.5 },
            wavelengths: WavelengthGrid::LogSpaced {
                min: 1.0,
                max: 10.0,
                count: 5,
            },
            method: ScatteringMethod::mie(),
            materials: vec![Material::constant(1.6, 0.05, 2.5, MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        let p_cmd = cmd_cfg.compute().unwrap();
        assert_eq!(p_cmd.k_ext.len(), 5);
        assert_relative_eq!(p_cmd.rho, 2.5, max_relative = 1e-12);
    }

    #[test]
    fn mmf_fractal_and_filling_factor() {
        let mut cfg = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 1.0 },
            wavelengths: WavelengthGrid::Single { wavelength: 10.0 },
            method: ScatteringMethod::mmf(0.1, AggregateStructure::FractalDimension(2.0)),
            materials: vec![Material::builtin("pyr", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        let p = cfg.compute().unwrap();
        assert_relative_eq!(p.k_abs[0], 2744.68, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 180.933, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.399_211, max_relative = 2e-3);
        assert!(p.f[0].f11[0] > 1.0);

        cfg.method = ScatteringMethod::mmf(0.1, AggregateStructure::FillingFactor(0.2));
        let p_ff = cfg.compute().unwrap();
        assert_relative_eq!(p_ff.k_abs[0], 2850.34, max_relative = 1e-3);
        assert_relative_eq!(p_ff.k_sca[0], 331.925, max_relative = 1e-3);
        assert_relative_eq!(p_ff.g[0], 0.173_558, max_relative = 2e-3);
    }

    #[test]
    fn cde_exact_rayleigh_and_chop() {
        let mut cde = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 0.1 },
            wavelengths: WavelengthGrid::Single { wavelength: 10.0 },
            method: ScatteringMethod::Cde,
            materials: vec![Material::builtin("astrosil", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        let p = cde.compute().unwrap();
        assert_relative_eq!(p.k_abs[0], 3071.49, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 0.435_909, max_relative = 1e-3);
        assert!(p.g[0].abs() < 1e-13);

        // Exact [0.05, 0.1] endpoint retention
        cde.size_distribution = SizeDistribution::PowerLaw {
            min: 0.05,
            max: 0.1,
            slope: 3.5,
            bins: Some(5),
        };
        let p_grid = cde.compute().unwrap();
        assert_relative_eq!(p_grid.k_abs[0], 3071.49, max_relative = 1e-3);
        assert_relative_eq!(p_grid.k_sca[0], 0.215_381, max_relative = 1e-3);

        let m = &p.f[0];
        for j in 0..cde.n_angles {
            let theta = PI * (j as f64 + 0.5) / cde.n_angles as f64;
            assert_relative_eq!(
                m.f11[j],
                0.75 * (1.0 + theta.cos().powi(2)),
                max_relative = 1e-12
            );
            assert_relative_eq!(m.f12[j], -0.75 * theta.sin().powi(2), max_relative = 1e-12);
            assert_relative_eq!(m.f33[j], 1.5 * theta.cos(), max_relative = 1e-12);
            assert_relative_eq!(m.f34[j], 0.0, epsilon = 1e-15);
            assert_relative_eq!(
                m.f11[j].powi(2),
                m.f12[j].powi(2) + m.f33[j].powi(2),
                max_relative = 1e-12
            );
        }

        let chop = KappaConfig {
            size_distribution: SizeDistribution::Monodisperse { radius: 10.0 },
            wavelengths: WavelengthGrid::Single { wavelength: 1.0 },
            method: ScatteringMethod::mie(),
            chop_angle: Some(5.0),
            materials: vec![Material::builtin("pyr", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        let pc = chop.compute().unwrap();
        assert_relative_eq!(pc.k_abs[0], 117.230, max_relative = 1e-3);
        assert_relative_eq!(pc.k_sca[0], 191.453, max_relative = 1e-3);
        assert_relative_eq!(pc.g[0], 0.671_673, max_relative = 1e-3);
        for i in 0..5 {
            assert_relative_eq!(pc.f[0].f11[i], pc.f[0].f11[5], max_relative = 1e-12);
        }
    }

    #[test]
    fn size_distribution_moments_outputs_and_validation() {
        let norm_sd = SizeDistribution::Normal {
            min: 1.0,
            max: 100.0,
            mean: 10.0,
            sigma: 1.0,
            bins: None,
        };
        let norm_m = norm_sd.moments();
        assert_relative_eq!(norm_m[0], 10.0, max_relative = 1e-3);
        assert_relative_eq!(norm_m[1], 101.0_f64.sqrt(), max_relative = 1e-3);

        let lgnm_sd = SizeDistribution::LogNormal {
            min: 0.01,
            max: 1000.0,
            mean: 1.0,
            sigma: 0.5,
            bins: None,
        };
        let lgnm_m = lgnm_sd.moments();
        assert_relative_eq!(lgnm_m[0], (0.5 * 0.25_f64).exp(), max_relative = 1e-3);
        assert_relative_eq!(lgnm_m[1], 0.25_f64.exp(), max_relative = 1e-3);
        assert_relative_eq!(lgnm_m[2], (1.5 * 0.25_f64).exp(), max_relative = 1e-3);

        // Structured validation errors
        assert!(matches!(
            KappaConfig::default().prepare(),
            Err(KappaError::InvalidConfig(_))
        ));
        let bad_mat = KappaConfig {
            materials: vec![Material::builtin("not-a-material", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        assert!(matches!(
            bad_mat.prepare(),
            Err(KappaError::UnknownMaterial(_))
        ));

        // Tabulated size distribution + RADMC-3D scattering matrix + blend_only + split
        let out = temp_dir("outputs");
        let file_cfg = KappaConfig {
            size_distribution: SizeDistribution::Tabulated {
                radii: vec![0.1, 1.0],
                weights: vec![100.0, 1.0],
                source: Some(PathBuf::from("synthetic.dat")),
            },
            wavelengths: WavelengthGrid::LogSpaced {
                min: 1.0,
                max: 10.0,
                count: 2,
            },
            method: ScatteringMethod::mie(),
            write_scatter: true,
            for_radmc: true,
            outdir: out.clone(),
            materials: vec![Material::builtin("pyr", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        file_cfg.run().unwrap();
        let radmc_text = fs::read_to_string(out.join("kappa_opacity.inp")).unwrap();
        assert!(radmc_text.contains("file=synthetic.dat"));
        assert!(radmc_text.contains("RADMC-3D"));

        let mut blend_cfg = KappaConfig::diana();
        blend_cfg.wavelengths = WavelengthGrid::LogSpaced {
            min: 1.0,
            max: 10.0,
            count: 3,
        };
        blend_cfg.blend_only = true;
        blend_cfg.outdir.clone_from(&out);
        blend_cfg.run().unwrap();
        let mix_text = fs::read_to_string(out.join("kappa_mix.lnk")).unwrap();
        assert_eq!(mix_text.lines().count(), 4);

        let split_cfg = KappaConfig {
            size_distribution: SizeDistribution::PowerLaw {
                min: 0.1,
                max: 1.0,
                slope: 3.5,
                bins: Some(3),
            },
            wavelengths: WavelengthGrid::Single { wavelength: 1.0 },
            method: ScatteringMethod::mie(),
            split_subgrains: Some(3),
            outdir: out.clone(),
            materials: vec![Material::builtin("pyr", MaterialKind::Core, 1.0)],
            ..KappaConfig::default()
        };
        split_cfg.run().unwrap();
        for idx in 1..=3 {
            assert!(out.join(format!("kappa_opacity_{idx:03}.dat")).exists());
        }

        let _ = fs::remove_dir_all(&out);
    }
}
