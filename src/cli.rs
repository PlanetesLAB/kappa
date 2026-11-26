//! Command-line interface for `kappa`.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;

use crate::components::MATERIAL_KEYS;
use crate::components::try_get_lnk_data;
use crate::io::{read_sizedis_file, read_wavelength_grid};
use crate::opac::{
    AggregateStructure, KappaError, MaterialKind, ScatteringMethod, SizeDistribution,
    WavelengthGrid,
};
use crate::opac::{KappaConfig, Material};

/// Result of parsing CLI arguments.
#[derive(Debug)]
pub enum CliAction {
    /// Execute an opacity calculation with the constructed configuration.
    Run(Box<KappaConfig>),
    /// The command completed directly (e.g., `kappa materials`).
    Done,
}

/// Built-in dust composition presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Preset {
    /// DIANA standard composition (87% pyroxene, 13% amorphous carbon, 25% porosity).
    Diana,
    /// DSHARP composition (astrosilicate, organics, troilite, water ice).
    Dsharp,
    /// Ice-free DSHARP composition (astrosilicate, organics, troilite).
    DsharpNoIce,
}

/// Scattering calculation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum MethodArg {
    /// Distribution of Hollow Spheres (Min et al. 2005).
    Dhs,
    /// Lorenz-Mie theory for compact spheres.
    Mie,
    /// Modified Mean Field theory for fractal aggregates (Tazaki & Tanaka 2018).
    Mmf,
    /// Continuous Distribution of Ellipsoids in the Rayleigh limit.
    Cde,
}

/// Subcommands for `kappa`.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// List all built-in materials, or inspect metadata for a single material key.
    Materials {
        /// Optional material key to inspect in detail (e.g. `pyr-mg70`, `h2o-w`).
        key: Option<String>,
    },
}

/// Dust opacity and scattering matrix calculator.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(
    name = "kappa",
    version,
    about = "Dust opacity and scattering matrix calculator",
    long_about = "Computes dust opacities and Mueller scattering matrices using DHS, Mie, MMF, or CDE.\n\
                  Without composition flags, defaults to the DIANA standard mixture."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    // ---- Composition ----
    /// Start from a built-in grain composition preset.
    #[arg(long, value_enum, help_heading = "Composition")]
    pub preset: Option<Preset>,

    /// Core material specification `<key|file.lnk|n:k:rho>[=mfrac]` (repeatable).
    #[arg(
        short = 'c',
        long = "core",
        value_name = "SPEC",
        help_heading = "Composition"
    )]
    pub core: Vec<String>,

    /// Mantle material specification `<key|file.lnk|n:k:rho>[=mfrac]` (repeatable).
    #[arg(
        short = 'm',
        long = "mantle",
        value_name = "SPEC",
        help_heading = "Composition"
    )]
    pub mantle: Vec<String>,

    /// Core vacuum porosity in [0, 1). Defaults to 0.25 for DIANA preset, 0.0 otherwise.
    #[arg(
        short = 'p',
        long = "porosity",
        value_name = "FRAC",
        help_heading = "Composition"
    )]
    pub porosity: Option<f64>,

    /// Mantle vacuum porosity in [0, 1). Defaults to the core porosity if omitted.
    #[arg(
        long = "mantle-porosity",
        value_name = "FRAC",
        help_heading = "Composition"
    )]
    pub mantle_porosity: Option<f64>,

    // ---- Size Distribution ----
    /// Minimum grain radius in micrometers.
    #[arg(
        long,
        default_value_t = 0.05,
        conflicts_with_all = ["size", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub amin: f64,

    /// Maximum grain radius in micrometers.
    #[arg(
        long,
        default_value_t = 3000.0,
        conflicts_with_all = ["size", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub amax: f64,

    /// Monodisperse grain radius in micrometers (sets amin = amax = SIZE, na = 1).
    #[arg(
        long,
        value_name = "UM",
        conflicts_with_all = ["amin", "amax", "na", "apow", "log_normal", "normal", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub size: Option<f64>,

    /// Number of grain-size bins (auto-computed if omitted).
    #[arg(
        long,
        conflicts_with_all = ["size", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub na: Option<usize>,

    /// Power-law size distribution slope p in dn/da ~ a^(-p).
    #[arg(
        long,
        default_value_t = 3.5,
        conflicts_with_all = ["log_normal", "normal", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub apow: f64,

    /// Log-normal size distribution `MEAN:SIGMA` (centroid in um, dimensionless log width).
    #[arg(
        long = "log-normal",
        value_name = "MEAN:SIGMA",
        conflicts_with_all = ["normal", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub log_normal: Option<String>,

    /// Linear normal size distribution `MEAN:SIGMA` (centroid and width in um).
    #[arg(
        long = "normal",
        value_name = "MEAN:SIGMA",
        conflicts_with_all = ["log_normal", "size_file"],
        help_heading = "Size Distribution"
    )]
    pub normal: Option<String>,

    /// Read a tabulated grain size distribution from file.
    #[arg(
        long = "size-file",
        value_name = "PATH",
        help_heading = "Size Distribution"
    )]
    pub size_file: Option<PathBuf>,

    // ---- Wavelength Grid ----
    /// Minimum wavelength in micrometers.
    #[arg(
        long,
        default_value_t = 0.05,
        conflicts_with_all = ["lam", "lam_file"],
        help_heading = "Wavelength Grid"
    )]
    pub lmin: f64,

    /// Maximum wavelength in micrometers.
    #[arg(
        long,
        default_value_t = 1e4,
        conflicts_with_all = ["lam", "lam_file"],
        help_heading = "Wavelength Grid"
    )]
    pub lmax: f64,

    /// Number of logarithmically spaced wavelength points.
    #[arg(
        long,
        default_value_t = 300,
        conflicts_with_all = ["lam", "lam_file"],
        help_heading = "Wavelength Grid"
    )]
    pub nlam: usize,

    /// Single wavelength in micrometers (sets lmin = lmax = LAM, nlam = 1).
    #[arg(
        long,
        value_name = "UM",
        conflicts_with_all = ["lmin", "lmax", "nlam", "lam_file"],
        help_heading = "Wavelength Grid"
    )]
    pub lam: Option<f64>,

    /// Read the wavelength grid from a file (single-column or `.lnk` file).
    #[arg(
        long = "lam-file",
        value_name = "PATH",
        help_heading = "Wavelength Grid"
    )]
    pub lam_file: Option<PathBuf>,

    // ---- Scattering Method ----
    /// Scattering theory to use.
    #[arg(
        long,
        value_enum,
        default_value_t = MethodArg::Dhs,
        help_heading = "Scattering Method"
    )]
    pub method: MethodArg,

    /// Maximum hollow-sphere volume fraction for DHS in [0, 1).
    #[arg(long, default_value_t = 0.8, help_heading = "Scattering Method")]
    pub fmax: f64,

    /// Size parameter limit above which DHS falls back to Mie theory.
    #[arg(long, default_value_t = 1e8, help_heading = "Scattering Method")]
    pub xlim: f64,

    /// Size parameter cap passed into the DHS concentric-sphere solver.
    #[arg(
        long = "xlim-dhs",
        default_value_t = 1e4,
        help_heading = "Scattering Method"
    )]
    pub xlim_dhs: f64,

    /// Monomer radius in micrometers for MMF aggregates.
    #[arg(
        long = "mmf-a0",
        default_value_t = 0.1,
        help_heading = "Scattering Method"
    )]
    pub mmf_a0: f64,

    /// Fractal dimension in (1, 3] for MMF aggregates.
    #[arg(
        long = "mmf-df",
        value_name = "DF",
        conflicts_with = "mmf_fill",
        help_heading = "Scattering Method"
    )]
    pub mmf_df: Option<f64>,

    /// Volume filling factor in (0, 1] for MMF aggregates (default 0.2 when `--mmf-df` is unset).
    #[arg(
        long = "mmf-fill",
        value_name = "FILL",
        conflicts_with = "mmf_df",
        help_heading = "Scattering Method"
    )]
    pub mmf_fill: Option<f64>,

    /// Fractal prefactor k0 for MMF aggregates (defaults to (5/3)^(df/2) if omitted).
    #[arg(long = "mmf-kf", value_name = "K0", help_heading = "Scattering Method")]
    pub mmf_kf: Option<f64>,

    /// Keep approximate MMF scattering matrix when phase-shift criterion fails.
    #[arg(long = "mmf-ss", help_heading = "Scattering Method")]
    pub mmf_ss: bool,

    // ---- Output ----
    /// Number of angular bins (must be even and >= 2).
    #[arg(long, default_value_t = 180, help_heading = "Output")]
    pub nang: usize,

    /// Forward-scattering chopping angle in degrees.
    #[arg(long, value_name = "DEG", help_heading = "Output")]
    pub chop: Option<f64>,

    /// Write the full 6-element Mueller scattering matrix.
    #[arg(short = 's', long = "scatter", help_heading = "Output")]
    pub scatter: bool,

    /// Write scattering matrices only at wavelengths in `LMIN[:LMAX]` (repeatable, implies `--scatter`).
    #[arg(long = "sparse", value_name = "LMIN[:LMAX]", help_heading = "Output")]
    pub sparse: Vec<String>,

    /// Format output files for RADMC-3D (`.inp` extension and Z-matrix normalization).
    #[arg(long = "radmc", help_heading = "Output")]
    pub radmc: bool,

    /// Split output into one file per grain size bin, averaging over `NSUB` sub-grains (default 5).
    #[arg(
        long = "split",
        value_name = "NSUB",
        num_args = 0..=1,
        default_missing_value = "5",
        help_heading = "Output"
    )]
    pub split: Option<usize>,

    /// Write the wavelength and grain-size grids and exit without computing opacities.
    #[arg(long = "write-grid", help_heading = "Output")]
    pub write_grid: bool,

    /// Write only the blended refractive index `.lnk` table and exit.
    #[arg(long = "blend-only", help_heading = "Output")]
    pub blend_only: bool,

    /// Output directory.
    #[arg(
        short = 'o',
        long = "outdir",
        default_value = "output",
        value_name = "DIR",
        help_heading = "Output"
    )]
    pub outdir: PathBuf,

    /// Base filename prefix for opacity output files.
    #[arg(
        long = "label",
        default_value = "kappa_opacity",
        value_name = "NAME",
        help_heading = "Output"
    )]
    pub label: String,

    /// Suppress progress heartbeat and non-fatal warning output.
    #[arg(short = 'q', long = "quiet", help_heading = "Output")]
    pub quiet: bool,
}

/// Parses command-line arguments from `std::env::args` and builds a [`CliAction`].
///
/// # Errors
/// Returns a [`KappaError`] if argument parsing, material parsing, or file loading fails.
pub fn launch() -> Result<CliAction, KappaError> {
    let cli = Cli::parse();
    cli.into_action()
}

impl Cli {
    /// Converts parsed CLI arguments into a [`CliAction`].
    ///
    /// # Errors
    /// Returns a [`KappaError`] if any material specification, distribution parameter, or input
    /// file is invalid.
    pub fn into_action(self) -> Result<CliAction, KappaError> {
        if let Some(Command::Materials { key }) = self.command {
            match key {
                Some(k) => inspect_material(&k)?,
                None => list_materials(),
            }
            return Ok(CliAction::Done);
        }

        let config = self.into_config()?;
        Ok(CliAction::Run(Box::new(config)))
    }

    /// Converts parsed CLI options into a [`KappaConfig`].
    ///
    /// # Errors
    /// Returns a [`KappaError`] if any material specification, distribution parameter, sparse range,
    /// or referenced input file is invalid.
    pub fn into_config(self) -> Result<KappaConfig, KappaError> {
        let mut kpc = match self.preset {
            Some(Preset::Diana) => KappaConfig::diana(),
            Some(Preset::Dsharp) => KappaConfig::dsharp(),
            Some(Preset::DsharpNoIce) => KappaConfig::dsharp_no_ice(),
            None if self.core.is_empty() && self.mantle.is_empty() => KappaConfig::diana(),
            None => KappaConfig::default(),
        };

        for spec in &self.core {
            if let Some(mat) = parse_material_spec(spec, MaterialKind::Core)? {
                kpc.materials.push(mat);
            }
        }
        for spec in &self.mantle {
            if let Some(mat) = parse_material_spec(spec, MaterialKind::Mantle)? {
                kpc.materials.push(mat);
            }
        }

        if let Some(pcore) = self.porosity {
            kpc.core_porosity = pcore;
        }
        kpc.mantle_porosity = self.mantle_porosity.unwrap_or(kpc.core_porosity);

        // Size distribution
        kpc.size_distribution = if let Some(path) = self.size_file {
            let path_str = path.to_string_lossy();
            let (r, nr, _) = read_sizedis_file(&path_str)?;
            SizeDistribution::Tabulated {
                radii: r.to_vec(),
                weights: nr.to_vec(),
                source: Some(path),
            }
        } else if let Some(radius) = self.size {
            SizeDistribution::Monodisperse { radius }
        } else if let Some(spec) = self.log_normal {
            let (mean, sigma) = parse_pair(&spec, "--log-normal")?;
            SizeDistribution::LogNormal {
                min: self.amin,
                max: self.amax,
                mean,
                sigma,
                bins: self.na,
            }
        } else if let Some(spec) = self.normal {
            let (mean, sigma) = parse_pair(&spec, "--normal")?;
            SizeDistribution::Normal {
                min: self.amin,
                max: self.amax,
                mean,
                sigma,
                bins: self.na,
            }
        } else {
            SizeDistribution::PowerLaw {
                min: self.amin,
                max: self.amax,
                slope: self.apow,
                bins: self.na,
            }
        };

        // Wavelength grid
        kpc.wavelengths = if let Some(path) = self.lam_file {
            let path_str = path.to_string_lossy();
            let (_, lam) = read_wavelength_grid(&path_str)?;
            WavelengthGrid::Tabulated {
                wavelengths: lam.to_vec(),
            }
        } else if let Some(wavelength) = self.lam {
            WavelengthGrid::Single { wavelength }
        } else {
            WavelengthGrid::LogSpaced {
                min: self.lmin,
                max: self.lmax,
                count: self.nlam,
            }
        };

        // Scattering method
        kpc.method = match self.method {
            MethodArg::Dhs => ScatteringMethod::Dhs {
                fmax: self.fmax,
                size_limit: self.xlim,
                dhs_size_limit: self.xlim_dhs,
            },
            MethodArg::Mie => ScatteringMethod::Mie {
                size_limit: self.xlim,
            },
            MethodArg::Mmf => {
                let structure = if let Some(df) = self.mmf_df {
                    AggregateStructure::FractalDimension(df)
                } else {
                    AggregateStructure::FillingFactor(self.mmf_fill.unwrap_or(0.2))
                };
                ScatteringMethod::Mmf {
                    monomer_radius: self.mmf_a0,
                    structure,
                    prefactor: self.mmf_kf,
                    keep_approximate_scattering: self.mmf_ss,
                }
            }
            MethodArg::Cde => ScatteringMethod::Cde,
        };

        // Angular grid & output
        kpc.n_angles = self.nang;
        kpc.chop_angle = self.chop;
        kpc.write_scatter = self.scatter || !self.sparse.is_empty();
        kpc.for_radmc = self.radmc;
        kpc.split_subgrains = self.split;
        kpc.write_grid = self.write_grid;
        kpc.blend_only = self.blend_only;
        kpc.outdir = self.outdir;
        kpc.label = self.label;
        kpc.quiet = self.quiet;

        for range_spec in &self.sparse {
            let (l1, l2) = if let Some((a, b)) = range_spec.split_once(':') {
                let v1 = a.parse::<f64>().map_err(|_| {
                    KappaError::InvalidConfig(format!("invalid `--sparse` start wavelength: `{a}`"))
                })?;
                let v2 = b.parse::<f64>().map_err(|_| {
                    KappaError::InvalidConfig(format!("invalid `--sparse` end wavelength: `{b}`"))
                })?;
                if v1 <= v2 { (v1, v2) } else { (v2, v1) }
            } else {
                let v = range_spec.parse::<f64>().map_err(|_| {
                    KappaError::InvalidConfig(format!(
                        "invalid `--sparse` wavelength: `{range_spec}`"
                    ))
                })?;
                (v, v)
            };
            if l1 <= 0.0 || l2 <= 0.0 {
                return Err(KappaError::InvalidConfig(
                    "`--sparse` wavelengths must be positive".into(),
                ));
            }
            kpc.sparse_ranges.push((l1, l2));
        }

        Ok(kpc)
    }
}

/// Parses `<key|file.lnk|n:k:rho>[=mfrac]`. Returns `Ok(None)` if `mfrac == 0.0`.
fn parse_material_spec(spec: &str, kind: MaterialKind) -> Result<Option<Material>, KappaError> {
    let (target, mfrac) = if let Some((lhs, rhs)) = spec.rsplit_once('=') {
        let frac = rhs.parse::<f64>().map_err(|_| {
            KappaError::InvalidConfig(format!("invalid mass fraction `{rhs}` in `{spec}`"))
        })?;
        if frac < 0.0 || !frac.is_finite() {
            return Err(KappaError::InvalidConfig(format!(
                "mass fraction must be non-negative in `{spec}`"
            )));
        }
        if frac == 0.0 {
            return Ok(None);
        }
        (lhs, frac)
    } else {
        (spec, 1.0)
    };

    if try_get_lnk_data(target).is_some() {
        Ok(Some(Material::builtin(target, kind, mfrac)))
    } else if Path::new(target)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lnk"))
    {
        Ok(Some(Material::lnk_file(target, kind, mfrac)))
    } else if target.contains(':') {
        let parts: Vec<&str> = target.split(':').collect();
        if parts.len() != 3 {
            return Err(KappaError::InvalidConfig(format!(
                "expected `n:k:rho` for constant optical constants, got `{target}`"
            )));
        }
        let n = parts[0].parse::<f64>().map_err(|_| {
            KappaError::InvalidConfig(format!("invalid refractive index `n` in `{target}`"))
        })?;
        let k = parts[1].parse::<f64>().map_err(|_| {
            KappaError::InvalidConfig(format!("invalid extinction coefficient `k` in `{target}`"))
        })?;
        let density = parts[2].parse::<f64>().map_err(|_| {
            KappaError::InvalidConfig(format!("invalid density `rho` in `{target}`"))
        })?;
        Ok(Some(Material::constant(n, k, density, kind, mfrac)))
    } else {
        Err(KappaError::UnknownMaterial(target.to_string()))
    }
}

fn parse_pair(spec: &str, flag: &str) -> Result<(f64, f64), KappaError> {
    let (lhs, rhs) = spec.split_once(':').ok_or_else(|| {
        KappaError::InvalidConfig(format!("{flag} expects `MEAN:SIGMA`, got `{spec}`"))
    })?;
    let a = lhs
        .parse::<f64>()
        .map_err(|_| KappaError::InvalidConfig(format!("invalid mean in {flag}: `{lhs}`")))?;
    let b = rhs
        .parse::<f64>()
        .map_err(|_| KappaError::InvalidConfig(format!("invalid sigma in {flag}: `{rhs}`")))?;
    Ok((a, b))
}

fn inspect_material(key: &str) -> Result<(), KappaError> {
    let comp =
        try_get_lnk_data(key).ok_or_else(|| KappaError::UnknownMaterial(key.to_string()))?;
    println!("{}: {}", "Key".bold(), key.green());
    println!("{}: {}", "Name".bold(), comp.name);
    println!("{}: {}", "Class".bold(), comp.class);
    println!("{}: {}", "State".bold(), comp.state);
    println!("{}: {:.3} g/cm^3", "Density".bold(), comp.rho);
    println!(
        "{}: {:.4e} .. {:.4e} um ({} points)",
        "Wavelength".bold(),
        comp.wavelengths[0],
        comp.wavelengths[comp.size - 1],
        comp.size
    );
    Ok(())
}

fn list_materials() {
    println!(
        "{:<14} {:<10} {:<12} {:>6}  {:>23}  {}",
        "Key".bold(),
        "Class".bold(),
        "State".bold(),
        "rho".bold(),
        "lambda [um]".bold(),
        "Name".bold()
    );
    println!("{}", "-".repeat(82));
    for &key in MATERIAL_KEYS {
        if let Some(comp) = try_get_lnk_data(key) {
            println!(
                "{:<14} {:<10} {:<12} {:>6.2}  {:>10.3e} .. {:<10.3e}  {}",
                key.green(),
                comp.class,
                comp.state,
                comp.rho,
                comp.wavelengths[0],
                comp.wavelengths[comp.size - 1],
                comp.name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use clap::Parser;

    use crate::opac::{
        AggregateStructure, MaterialKind, MaterialSource, ScatteringMethod, SizeDistribution,
        WavelengthGrid,
    };

    use super::Cli;
    use super::CliAction;

    #[test]
    fn default_cli_uses_diana_and_dhs() {
        let cli = Cli::try_parse_from(["kappa"]).unwrap();
        let cfg = cli.into_config().unwrap();
        assert_eq!(cfg.materials.len(), 2);
        assert_relative_eq!(cfg.core_porosity, 0.25);
        assert!(matches!(
            cfg.method,
            ScatteringMethod::Dhs { fmax, .. } if (fmax - 0.8).abs() < 1e-12
        ));
    }

    #[test]
    fn preset_does_not_wipe_other_flags_and_supports_mantles() {
        let cli = Cli::try_parse_from([
            "kappa",
            "--lam",
            "10.0",
            "--method",
            "mie",
            "--preset",
            "diana",
            "-p",
            "0.1",
            "--mantle-porosity",
            "0.05",
            "-m",
            "h2o-w=0.2",
            "-s",
            "--radmc",
            "--label",
            "custom",
            "-q",
        ])
        .unwrap();
        let cfg = cli.into_config().unwrap();
        assert_eq!(cfg.wavelengths, WavelengthGrid::Single { wavelength: 10.0 });
        assert!(matches!(cfg.method, ScatteringMethod::Mie { .. }));
        assert_relative_eq!(cfg.core_porosity, 0.1);
        assert_relative_eq!(cfg.mantle_porosity, 0.05);
        assert_eq!(cfg.materials.len(), 3);
        assert_eq!(cfg.materials[2].kind, MaterialKind::Mantle);
        assert_relative_eq!(cfg.materials[2].mass_fraction, 0.2);
        assert!(cfg.write_scatter);
        assert!(cfg.for_radmc);
        assert_eq!(cfg.label, "custom");
        assert!(cfg.quiet);
    }

    #[test]
    fn constant_optical_constants_and_distributions_and_mmf() {
        let cli = Cli::try_parse_from([
            "kappa",
            "-c",
            "1.5:0.01:2.5=0.7",
            "-c",
            "pyr=0.3",
            "--amin",
            "0.5",
            "--amax",
            "10.0",
            "--log-normal",
            "2.0:0.4",
            "--lmin",
            "1.0",
            "--lmax",
            "100.0",
            "--nlam",
            "12",
            "--method",
            "mmf",
            "--mmf-a0",
            "0.1",
            "--mmf-df",
            "2.2",
            "--sparse",
            "5.0",
            "--sparse",
            "20.0:40.0",
        ])
        .unwrap();
        let cfg = cli.into_config().unwrap();
        assert_eq!(
            cfg.materials[0].source,
            MaterialSource::Constant {
                n: 1.5,
                k: 0.01,
                density: 2.5
            }
        );
        assert!(matches!(
            cfg.size_distribution,
            SizeDistribution::LogNormal { .. }
        ));
        assert!(matches!(
            cfg.method,
            ScatteringMethod::Mmf {
                structure: AggregateStructure::FractalDimension(df),
                ..
            } if (df - 2.2).abs() < 1e-12
        ));
        assert_eq!(cfg.sparse_ranges.len(), 2);
        assert!(cfg.write_scatter);

        let prepared = cfg.prepare().unwrap();
        assert_eq!(prepared.wavelengths().len(), 12);
    }

    #[test]
    fn split_optional_value_and_conflicts_and_subcommand() {
        let c1 = Cli::try_parse_from(["kappa", "--split"]).unwrap();
        let cfg1 = c1.into_config().unwrap();
        assert_eq!(cfg1.split_subgrains, Some(5));

        let c2 = Cli::try_parse_from(["kappa", "--split", "9"]).unwrap();
        let cfg2 = c2.into_config().unwrap();
        assert_eq!(cfg2.split_subgrains, Some(9));

        assert!(Cli::try_parse_from(["kappa", "--mmf-df", "2.0", "--mmf-fill", "0.2"]).is_err());
        assert!(
            Cli::try_parse_from(["kappa", "--log-normal", "1:0.2", "--normal", "1:0.2"]).is_err()
        );
        assert!(Cli::try_parse_from(["kappa", "--unknown-flag"]).is_err());

        let sub = Cli::try_parse_from(["kappa", "materials", "pyr"]).unwrap();
        assert!(matches!(sub.into_action().unwrap(), CliAction::Done));
    }
}
