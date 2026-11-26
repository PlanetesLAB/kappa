//! The heart of `kappa`.
//! This module contains the main routines to compute opacities
//! and scattering matrices.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::mem::swap;
use std::path::Path;
use std::sync::Once;

use anyhow::Result;
use anyhow::anyhow;
use ndarray::s;
use num_complex::ComplexFloat;
use num_complex::{Complex, Complex64};
use planetes_ext::quadrature::gauss_legendre;
use planetes_ext::types::{BVector, RMatrix, RVector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::components::get_lnk_data;
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

#[derive(Debug)]
pub struct Material {
    pub key: String,
    pub kind: MaterialKind,
    pub n: f64,
    pub k: f64,
    pub rho: f64,
    pub mfrac: f64,
    pub vfrac: f64,
    pub re: RVector,
    pub im: RVector,
    pub cmd: RefractiveIndexKind,
}

impl Default for Material {
    fn default() -> Self {
        let nlam = 300;
        Material {
            key: String::new(),
            kind: MaterialKind::Core,
            n: 0.0,
            k: 0.0,
            rho: 0.0,
            mfrac: 1.0,
            vfrac: 0.0,
            re: RVector::zeros(nlam),
            im: RVector::zeros(nlam),
            cmd: RefractiveIndexKind::Other,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, Clone))]
#[derive(Debug, PartialEq)]
pub enum MaterialKind {
    Core,
    Mantle,
}

#[derive(Debug, PartialEq)]
pub enum RefractiveIndexKind {
    File,
    CmdLine,
    Other,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct KappaConfig {
    pub amin: f64,
    pub amax: f64,
    pub apow: f64,
    pub amean: f64,
    pub asigma: f64,
    pub na: usize,
    pub ameans_file: [f64; 3],
    pub sizedis: SizeDistribution,
    pub sizedis_file: Option<String>,
    pub r_file: RVector,
    pub nr_file: RVector,
    pub wavelength_kind: WavelengthKind,
    pub lmin: f64,
    pub lmax: f64,
    pub nlam: usize,
    pub lam: RVector,
    pub sparse_indices: HashSet<usize>,
    pub nang: usize,
    pub chop_angle: f64,
    pub pcore: f64,
    pub pmantle: f64,
    pub rho_core: f64,
    pub rho_mantle: f64,
    pub rho_av: f64,
    pub total_mfrac: f64,
    pub tot_mfrac_core: f64,
    pub tot_vfrac_core: f64,
    pub tot_mfrac_mantle: f64,
    pub tot_vfrac_mantle: f64,
    pub method: KappaMethod,
    pub fmax: f64,
    pub write_fits: bool,
    pub write_scatter: bool,
    pub write_grid: bool,
    pub for_radmc: bool,
    pub nsparse: usize,
    pub scatlammin: RVector,
    pub scatlammax: RVector,
    pub nsubgrains: usize,
    pub mmf_struct: f64,
    pub mmf_a0: f64,
    pub mmf_kf: f64,
    pub mmf_ss: bool,
    pub split: bool,
    pub blend_only: bool,
    pub xlim: f64,
    pub xlim_dhs: f64,
    pub materials: Vec<Material>,
    pub outdir: String,
    pub label: String,
    pub quiet: bool,
}

impl Default for KappaConfig {
    fn default() -> Self {
        let nlam = 300;
        KappaConfig {
            amin: 0.05,
            amax: 3000.0,
            apow: 3.5,
            amean: 0.0,
            asigma: 0.0,
            na: 0,
            ameans_file: [0.0; 3],
            sizedis: SizeDistribution::PowerLaw,
            sizedis_file: None,
            r_file: RVector::zeros(0),
            nr_file: RVector::zeros(0),
            wavelength_kind: WavelengthKind::Other,
            lmin: 0.05,
            lmax: 1e4,
            nlam,
            lam: RVector::zeros(nlam),
            sparse_indices: HashSet::new(),
            nang: 180,
            chop_angle: 0.0,
            pcore: 0.0,
            pmantle: 0.0,
            rho_core: 0.0,
            rho_mantle: 0.0,
            rho_av: 0.0,
            total_mfrac: 0.0,
            tot_mfrac_core: 0.0,
            tot_vfrac_core: 0.0,
            tot_mfrac_mantle: 0.0,
            tot_vfrac_mantle: 0.0,
            method: KappaMethod::DHS,
            fmax: 0.8,
            write_fits: false,
            write_scatter: false,
            write_grid: false,
            for_radmc: false,
            nsparse: 0,
            scatlammin: RVector::zeros(30),
            scatlammax: RVector::zeros(30),
            nsubgrains: 5,
            mmf_struct: 0.0,
            mmf_a0: 0.0,
            mmf_kf: 0.0,
            mmf_ss: false,
            split: false,
            blend_only: false,
            xlim: 1e8,
            xlim_dhs: 1e4,
            materials: Vec::with_capacity(20),
            outdir: String::from("output"),
            label: String::from("kappa_opacity"),
            quiet: false,
        }
    }
}

impl KappaConfig {
    #[must_use]
    pub fn ncore(&self) -> usize {
        self.materials
            .iter()
            .filter(|m| matches!(m.kind, MaterialKind::Core))
            .count()
    }

    #[must_use]
    pub fn nmant(&self) -> usize {
        self.materials
            .iter()
            .filter(|m| matches!(m.kind, MaterialKind::Mantle))
            .count()
    }

    /// Pre-process the inputs collected from the command-line
    ///
    /// # Errors
    /// Returns a [`KappaError`] if there is an issue with parsing the arguments,
    /// such as missing required arguments, invalid argument formats, or unknown options.
    pub fn prepare_inputs(&mut self) -> Result<()> {
        // Materials
        if self.materials.is_empty() {
            return Err(anyhow!("NoMaterials"));
        }
        if self.materials.len() > 20 {
            return Err(anyhow!("TooManyMaterials"));
        }
        if self.ncore() == 0 {
            return Err(anyhow!("NoCoreMaterial"));
        }

        // Porosity
        if self.pcore < 0.0 || self.pcore >= 1.0 || self.pmantle < 0.0 || self.pmantle >= 1.0 {
            return Err(anyhow!("InvalidPorosity"));
        }

        // Grain size distribution
        if self.sizedis == SizeDistribution::File {
            if self.r_file.is_empty() || self.r_file.len() != self.nr_file.len() {
                return Err(anyhow!("InvalidSizeInput"));
            }
            self.na = self.r_file.len();
            if let (Some(&min_r), Some(&max_r)) = (
                self.r_file.iter().reduce(|a, b| if a < b { a } else { b }),
                self.r_file.iter().reduce(|a, b| if a > b { a } else { b }),
            ) {
                self.amin = min_r;
                self.amax = max_r;
            }
        } else {
            if self.amin <= 0.0 || self.amax <= 0.0 {
                return Err(anyhow!("InvalidSizeInput"));
            }
            if self.amin > self.amax {
                swap(&mut self.amin, &mut self.amax);
            }
            if self.na == 0 {
                self.na =
                    (((self.amax.log10() - self.amin.log10()) * 15.0 + 1.0) as usize).max(5);
            }
            if (self.amin - self.amax).abs() < f64::EPSILON && self.na != 1 {
                self.na = 1;
            }
        }
        match self.validate_size_distribution() {
            Ok(()) => {}
            Err(KappaError::ForceLogNormal) => {
                self.sizedis = SizeDistribution::LogNormal;
            }
            Err(_) => return Err(anyhow!("InvalidSizeParam")),
        }

        // Wavelength grid
        if self.lmin <= 0.0 || self.lmax <= 0.0 {
            return Err(anyhow!("InvalidWavelengthInput"));
        }
        if self.lmin > self.lmax {
            swap(&mut self.lmin, &mut self.lmax);
        }
        if self.nlam <= 1 && (self.lmin - self.lmax).abs() > f64::EPSILON {
            return Err(anyhow!("SamplingRequired"));
        }
        if (self.lmin - self.lmax).abs() < f64::EPSILON && self.nlam != 1 {
            self.nlam = 1;
        }

        // Method parameters
        match self.method {
            KappaMethod::DHS => {
                if self.fmax < 0.0 || self.fmax >= 1.0 {
                    return Err(anyhow!("InvalidArgument: fmax"));
                }
            }
            KappaMethod::MMF => {
                if self.mmf_struct > 3.0 {
                    return Err(anyhow!("`mmf_struct` must be between 1 and 3"));
                }
                if self.mmf_struct <= 0.0 {
                    return Err(anyhow!("`mmf_struct` must be positive"));
                }
                if self.mmf_a0 <= 0.0 {
                    return Err(anyhow!("`mmf_a0` must be positive"));
                }
                if self.mmf_a0 >= self.amin {
                    return Err(anyhow!("`mmf_a0` must be smaller than `amin`"));
                }
            }
            KappaMethod::CDE => {
                if self.lmin <= 2.0 * PI * self.amax {
                    eprintln!("Warning: CDE requires Rayleigh limit!");
                }
            }
        }

        // Angular grid
        if self.nang < 2 || self.nang % 2 == 1 {
            return Err(anyhow!("`nang` must be even and at least 2"));
        }
        if self.chop_angle < 0.0 || self.chop_angle >= 180.0 {
            return Err(anyhow!("`chop_angle` must be in [0, 180)"));
        }

        // Other
        if self.split && self.blend_only {
            eprintln!("Warning: Turning off `-s` for `-blend_only`");
            self.split = false;
        }
        if self.split && self.sizedis != SizeDistribution::PowerLaw {
            return Err(anyhow!(
                "`split` is only supported for powerlaw size distribution"
            ));
        }
        if self.split && self.nsubgrains == 0 {
            return Err(anyhow!("`nsubgrains` must be at least 1"));
        }

        // Sort by `kind = MaterialKind::Core`
        self.materials.sort_by(|a, b| match (&a.kind, &b.kind) {
            (MaterialKind::Core, MaterialKind::Core) => Ordering::Equal,
            (MaterialKind::Core, _) => Ordering::Less,
            (_, MaterialKind::Core) => Ordering::Greater,
            _ => Ordering::Equal,
        });

        // Make logarithmic wavelength grid with exact endpoints
        match self.wavelength_kind {
            WavelengthKind::Other => {
                if self.nlam == 1 {
                    self.lam = RVector::from_elem(1, self.lmin);
                } else {
                    self.lam =
                        RVector::logspace(10.0, self.lmin.log10(), self.lmax.log10(), self.nlam);
                    self.lam[0] = self.lmin;
                    self.lam[self.nlam - 1] = self.lmax;
                }
            }
            WavelengthKind::File => {
                // already read from file
            }
        }

        // Prepare sparse scattering file
        if self.nsparse > 0 {
            self.prepare_sparse();
        }
        // Writing wavelength grid file
        if self.write_grid {
            self.write_wavelength_grid()?;
        }

        Ok(())
    }

    /// Initialize the `KappaConfig` struct.
    ///
    /// # Errors
    /// Returns a [`KappaError`] if there is an issue with parsing the arguments,
    /// such as missing required arguments, invalid argument formats, or unknown options.
    pub fn initialize(&mut self) -> Result<(), KappaError> {
        if self.materials.is_empty() {
            return Err(anyhow!("NoMaterials").into());
        }
        if self.ncore() == 0 {
            return Err(anyhow!("NoCoreMaterial").into());
        }

        for material in &mut self.materials {
            match material.cmd {
                RefractiveIndexKind::CmdLine => {
                    material.re = RVector::from_elem(self.nlam, material.n);
                    material.im = RVector::from_elem(self.nlam, material.k);
                }
                RefractiveIndexKind::File => {
                    let component = read_lnk_file(&material.key, None)?;
                    let l0_slice: &[f64] = &component.l0.to_vec();
                    let n0_slice: &[f64] = &component.n0.to_vec();
                    let k0_slice: &[f64] = &component.k0.to_vec();
                    (material.re, material.im) =
                        regrid_lnk_data(l0_slice, n0_slice, k0_slice, &self.lam, true);
                    material.rho = component.rho;
                }
                RefractiveIndexKind::Other => {
                    let component = get_lnk_data(&material.key);
                    (material.re, material.im) =
                        regrid_lnk_data(component.l0, component.n0, component.k0, &self.lam, true);
                    material.rho = component.rho;
                }
            }
            if material.rho <= 0.0 {
                return Err(anyhow!("Material density must be positive").into());
            }
        }

        // Normalize the mass fractions
        (self.total_mfrac, self.tot_mfrac_core, self.tot_mfrac_mantle) = self
            .materials
            .iter()
            .fold((0.0, 0.0, 0.0), |(tot, tot_core, tot_mantle), m| {
                match m.kind {
                    MaterialKind::Core => (tot + m.mfrac, tot_core + m.mfrac, tot_mantle),
                    MaterialKind::Mantle => (tot + m.mfrac, tot_core, tot_mantle + m.mfrac),
                }
            });

        if self.total_mfrac <= 0.0 || self.tot_mfrac_core <= 0.0 {
            return Err(anyhow!("Total mass fraction must be positive").into());
        }

        self.tot_mfrac_core /= self.total_mfrac;
        self.tot_mfrac_mantle /= self.total_mfrac;
        self.materials.iter_mut().for_each(|m| {
            m.mfrac /= self.total_mfrac;
            m.vfrac = m.mfrac / m.rho;
        });
        (self.tot_vfrac_core, self.tot_vfrac_mantle) =
            self.materials
                .iter()
                .fold((0.0, 0.0), |(tot_core, tot_mantle), m| match m.kind {
                    MaterialKind::Core => (tot_core + m.vfrac, tot_mantle),
                    MaterialKind::Mantle => (tot_core, tot_mantle + m.vfrac),
                });

        // Turn mass fractions into volume fractions
        self.rho_core = self.tot_mfrac_core / self.tot_vfrac_core;
        let has_mantle = self.nmant() > 0 && self.tot_vfrac_mantle > 0.0;
        self.rho_mantle = if has_mantle {
            self.tot_mfrac_mantle / self.tot_vfrac_mantle
        } else {
            0.0
        };

        // Normalize volume fractions
        self.materials.iter_mut().for_each(|m| match m.kind {
            MaterialKind::Core => m.vfrac /= self.tot_vfrac_core,
            MaterialKind::Mantle if has_mantle => m.vfrac /= self.tot_vfrac_mantle,
            MaterialKind::Mantle => {}
        });

        if self.pcore > 0.0 {
            self.materials.iter_mut().for_each(|m| {
                if m.kind == MaterialKind::Core {
                    m.vfrac *= 1.0 - self.pcore;
                }
            });
            self.rho_core *= 1.0 - self.pcore;
        }

        if has_mantle && self.pmantle > 0.0 {
            self.materials.iter_mut().for_each(|m| {
                if m.kind == MaterialKind::Mantle {
                    m.vfrac *= 1.0 - self.pmantle;
                }
            });
            self.rho_mantle *= 1.0 - self.pmantle;
        }

        // Calculate average density of the whole grain
        if has_mantle {
            self.rho_av = self.rho_core
                / (1.0 + self.tot_mfrac_mantle * (self.rho_core / self.rho_mantle - 1.0));
            self.tot_vfrac_mantle = self.tot_mfrac_mantle * self.rho_av / self.rho_mantle;
        } else {
            self.rho_av = self.rho_core;
            self.tot_vfrac_mantle = 0.0;
        }

        Ok(())
    }

    fn validate_size_distribution(&self) -> Result<(), KappaError> {
        match self.sizedis {
            SizeDistribution::PowerLaw if self.apow < 0.0 => Err(anyhow!("UnexpectedApow").into()),
            SizeDistribution::Normal => {
                if self.amean <= 0.0 {
                    return Err(
                        anyhow!("`amean` must be positive for (log-)normal distribution").into(),
                    );
                }
                if self.asigma == 0.0 {
                    return Err(
                        anyhow!("`asigma` cannot be zero for (log-)normal distribution").into(),
                    );
                }
                if self.asigma > 0.0 {
                    return Err(KappaError::ForceLogNormal);
                }
                Ok(())
            }
            SizeDistribution::LogNormal => {
                if self.amean <= 0.0 {
                    return Err(
                        anyhow!("`amean` must be positive for log-normal distribution").into(),
                    );
                }
                if self.asigma == 0.0 {
                    return Err(
                        anyhow!("`asigma` cannot be zero for log-normal distribution").into(),
                    );
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn prepare_sparse(&mut self) {
        if self.nsparse == 0 || self.nlam == 0 {
            return;
        }
        if self.nlam >= 2 {
            let f = self.lam[self.nlam - 1] / self.lam[self.nlam - 2];
            for (valmin, valmax) in self
                .scatlammin
                .iter_mut()
                .zip(self.scatlammax.iter_mut())
                .take(self.nsparse)
            {
                if *valmin > 0.0 && *valmax / *valmin < f {
                    *valmin /= f.sqrt() / 1.0001;
                    *valmax *= f.sqrt() / 1.0001;
                }
            }
        }

        for (il, &lam_val) in self.lam.iter().enumerate() {
            if self
                .scatlammin
                .iter()
                .zip(self.scatlammax.iter())
                .take(self.nsparse)
                .any(|(&min_val, &max_val)| lam_val >= min_val && lam_val <= max_val)
            {
                self.sparse_indices.insert(il);
                if il > 0 {
                    self.sparse_indices.insert(il - 1);
                }
                if il + 1 < self.nlam {
                    self.sparse_indices.insert(il + 1);
                }
            }
        }
    }

    /// Compute the moments of the size distribution
    ///
    /// The results are returned in `ameans`, an array of length 3:
    /// $$
    ///                [\langle a \rangle,\quad \langle a^2 \rangle^{1/2},\quad \langle a^3 \rangle^{1/3}]
    /// $$
    #[must_use]
    pub fn get_sizedis_moments(&self) -> [f64; 3] {
        self.get_sizedis_moments_for(self.amin, self.amax)
    }

    fn get_sizedis_moments_for(&self, amin: f64, amax: f64) -> [f64; 3] {
        if self.sizedis == SizeDistribution::File {
            if self.ameans_file.iter().any(|&v| v > 0.0) {
                return self.ameans_file;
            }
            let mut tot = [0.0; 3];
            let mut totn = 0.0;
            for (&r, &nr) in self.r_file.iter().zip(self.nr_file.iter()) {
                totn += nr;
                tot[0] += nr * r;
                tot[1] += nr * r.powi(2);
                tot[2] += nr * r.powi(3);
            }
            if totn <= 0.0 {
                return [0.0; 3];
            }
            return [
                tot[0] / totn,
                (tot[1] / totn).sqrt(),
                (tot[2] / totn).powf(1.0 / 3.0),
            ];
        }

        if ((amax - amin) / amin).abs() < 1e-6 {
            return [amin, amin, amin];
        }

        let ns: usize = 1000;
        let aminlog = amin.log10();
        let amaxlog = amax.log10();
        let nsf = (ns - 1) as f64;
        let pow = -self.apow;

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
            let nr = match self.sizedis {
                SizeDistribution::Normal => {
                    let expo = 0.5 * ((r - self.amean) / self.asigma).powi(2);
                    if expo > 99.0 { 0.0 } else { r * (-expo).exp() }
                }
                SizeDistribution::LogNormal => {
                    let expo = 0.5 * ((r / self.amean).ln() / self.asigma).powi(2);
                    if expo > 99.0 { 0.0 } else { (-expo).exp() }
                }
                SizeDistribution::PowerLaw => r.powf(pow + 1.0),
                SizeDistribution::File => unreachable!(),
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

    /// Write the wavelength grid to a file in the specified format.
    ///
    /// # Errors
    /// Returns an error if the output file cannot be created or written.
    pub fn write_wavelength_grid(&self) -> Result<()> {
        let file = self.create_output_file("kappa_lam.dat")?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "# Wavelength grid written by kappa, can be read back in with -l kappa_lam.dat"
        )?;
        writeln!(writer, "# First line: number of wavelengths")?;
        writeln!(writer, "# Then one lambda per line, in micrometer")?;
        writeln!(writer, "{}", self.nlam)?;
        for i in 0..self.nlam {
            writeln!(writer, "{:18.5e}", self.lam[i])?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Run the simulation.
    ///
    /// # Panics
    /// - If the number of materials exceeds 20.
    /// - If the porosity values are not in the range [0, 1).
    /// - If the grain size distribution parameters are invalid.
    /// - If the wavelength grid parameters are invalid.
    /// - If the DHS `fmax` parameter is not in the range [0, 1).
    /// # Errors
    /// - `TooManyMaterials`: The number of materials exceeds 20.
    /// - `InvalidPorosity`: The porosity values are not in the range [0, 1).
    /// - `InvalidSizeInput`: The grain size distribution parameters are invalid.
    /// - `InvalidWavelengthInput`: The wavelength grid parameters are invalid.
    /// - `SamplingRequired`: More than one wavelength is required when `lmin` and `lmax` are different.
    pub fn run(&self) -> Result<(), KappaError> {
        // Loop for splitting the output into files by grain size
        if self.split {
            let mut nsub = self.nsubgrains;
            if nsub > 1 && nsub.is_multiple_of(2) {
                nsub += 1;
            }
            let afact = (self.amax / self.amin).powf(1.0 / self.na as f64);
            let afsub = if nsub > 1 {
                afact.powf(1.0 / (nsub - 1) as f64)
            } else {
                1.0
            };
            let half = (nsub / 2) as i32;

            if self.write_grid || self.blend_only {
                let asplit = self.amin * afact.powf(0.5);
                let aminsplit = asplit * afsub.powi(-half);
                let amaxsplit = asplit * afsub.powi(half);
                let _ = self.compute_kappa_with_ns(0, aminsplit, amaxsplit, nsub)?;
                return Ok(());
            }

            let progress = Progress::new("kappa", "sizes", self.na);

            (0..self.na).into_par_iter().try_for_each(|ia| {
                let iaf = ia as f64;
                let asplit = self.amin * afact.powf(iaf + 0.5);
                let aminsplit = asplit * afsub.powi(-half);
                let amaxsplit = asplit * afsub.powi(half);
                let p = self.compute_kappa_with_ns(ia, aminsplit, amaxsplit, nsub)?;
                if !p.is_ok && !self.quiet {
                    if self.mmf_ss {
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
                let fname = format!("{}_{:03}", self.label, ia + 1);
                self.write_opacities_with_bounds(&fname, &p, aminsplit, amaxsplit, nsub)?;
                if !self.quiet {
                    progress.add(1);
                }
                Ok::<(), KappaError>(())
            })?;
            return Ok(());
        }

        let p = self.compute_kappa(0, self.amin, self.amax)?;
        if self.write_grid || self.blend_only {
            return Ok(());
        }
        if !p.is_ok && !self.quiet {
            if self.mmf_ss {
                eprintln!("Warning: opacities OK, but some F_nn,g_asym may not be accurate");
            } else {
                eprintln!("Warning: opacities OK, but some F_nn,g_asym are set to zero");
            }
        }

        // TODO: FITS writer will be implemented later

        self.write_opacities(&self.label, &p)?;

        Ok(())
    }

    /// Calculate the opacities for a grain-size range.
    ///
    /// # Errors
    /// Returns a [`KappaError`] if size distribution, mixing, or scattering computation fails.
    pub fn compute_kappa(&self, ia: usize, amin: f64, amax: f64) -> Result<Particle, KappaError> {
        self.compute_kappa_with_ns(ia, amin, amax, self.na)
    }

    fn compute_kappa_with_ns(
        &self,
        ia: usize,
        amin: f64,
        amax: f64,
        ns: usize,
    ) -> Result<Particle, KappaError> {
        let (nf, ifmn) = if self.fmax <= 0.01 { (1, 0) } else { (20, 11) };

        let mut e1mantle = RVector::zeros(self.nlam);
        let mut e2mantle = RVector::zeros(self.nlam);

        let (ns, r, nr) = if self.sizedis == SizeDistribution::File {
            if self.r_file.is_empty() || self.r_file.len() != self.nr_file.len() {
                return Err(anyhow!("InvalidSizeInput").into());
            }
            let r = self.r_file.clone();
            let mut nr = self.nr_file.clone();
            let tot: f64 = r
                .iter()
                .zip(nr.iter())
                .map(|(&rv, &nrv)| nrv * rv.powi(3))
                .sum();
            if tot <= 0.0 {
                return Err(anyhow!("Size distribution has zero total volume").into());
            }
            nr /= tot;
            (r.len(), r, nr)
        } else {
            let mut r = RVector::zeros(ns);
            let mut nr = RVector::zeros(ns);
            let aminlog = amin.log10();
            let amaxlog = amax.log10();
            let pow = -self.apow;

            if ns == 1 {
                r[0] = (amin * amax).sqrt();
                nr[0] = 1.0 / r[0].powi(3);
            } else {
                let mut tot = 0.0;
                let nsf = (ns - 1) as f64;
                for (is, (r_val, nr_val)) in r.iter_mut().zip(nr.iter_mut()).enumerate() {
                    let isf = is as f64;
                    *r_val = if is == 0 {
                        amin
                    } else if is == ns - 1 {
                        amax
                    } else {
                        10.0_f64.powf(aminlog + (amaxlog - aminlog) * isf / nsf)
                    };
                    // On a logarithmic grid (d ln a = const), bin weight is f(a) * a * d ln a.
                    *nr_val = match self.sizedis {
                        SizeDistribution::Normal => {
                            let expo = 0.5 * ((*r_val - self.amean) / self.asigma).powi(2);
                            if expo > 99.0 {
                                0.0
                            } else {
                                *r_val * (-expo).exp()
                            }
                        }
                        SizeDistribution::LogNormal => {
                            let expo = 0.5 * ((*r_val / self.amean).ln() / self.asigma).powi(2);
                            if expo > 99.0 { 0.0 } else { (-expo).exp() }
                        }
                        SizeDistribution::PowerLaw => r_val.powf(pow + 1.0),
                        SizeDistribution::File => unreachable!(),
                    };
                    tot += *nr_val * r_val.powi(3);
                }
                if tot <= 0.0 {
                    return Err(anyhow!("Size distribution has zero total volume").into());
                }
                nr /= tot;
            }
            (ns, r, nr)
        };

        if self.write_grid && ia == 0 {
            self.write_sizedis_file(ns, &r, &nr)?;
        }

        // Copy the refractive index data for all materials into local arrays
        let mut e1 = RMatrix::zeros((self.nlam, self.materials.len()));
        let mut e2 = RMatrix::zeros((self.nlam, self.materials.len()));
        for (im, material) in self.materials.iter().enumerate() {
            e1.slice_mut(s![.., im]).assign(&material.re);
            e2.slice_mut(s![.., im]).assign(&material.im);
        }

        let ncore = self.ncore();
        let nmant = self.nmant();
        let core_vfracs: Vec<f64> = self
            .materials
            .iter()
            .filter(|m| matches!(m.kind, MaterialKind::Core))
            .map(|m| m.vfrac)
            .collect();
        let mantle_vfracs: Vec<f64> = self
            .materials
            .iter()
            .filter(|m| matches!(m.kind, MaterialKind::Mantle))
            .map(|m| m.vfrac)
            .collect();

        let mut e1_blend = RVector::zeros(self.nlam);
        let mut e2_blend = RVector::zeros(self.nlam);
        let mut core_buf = vec![Complex64::new(0.0, 0.0); ncore];
        let mut mantle_buf = vec![Complex64::new(0.0, 0.0); nmant];

        // Mixing, for all wavelengths
        for il in 0..self.nlam {
            // Core
            if ncore == 1 && self.pcore == 0.0 {
                e1_blend[il] = e1[[il, 0]];
                e2_blend[il] = e2[[il, 0]];
            } else {
                for im in 0..ncore {
                    core_buf[im] = Complex::new(e1[[il, im]], e2[[il, im]]);
                }
                let e_out = bruggeman_blend(&core_vfracs, &core_buf)?;
                e1_blend[il] = e_out.re;
                e2_blend[il] = e_out.im;
            }
            // Mantle
            if nmant > 0 {
                if nmant == 1 && self.pmantle == 0.0 {
                    e1mantle[il] = e1[[il, ncore]];
                    e2mantle[il] = e2[[il, ncore]];
                } else {
                    for im in 0..nmant {
                        mantle_buf[im] =
                            Complex::new(e1[[il, im + ncore]], e2[[il, im + ncore]]);
                    }
                    let e_out = bruggeman_blend(&mantle_vfracs, &mantle_buf)?;
                    e1mantle[il] = e_out.re;
                    e2mantle[il] = e_out.im;
                }
                let (e1mg, e2mg) = maxwell_garnet_blend(
                    Complex::new(e1_blend[il], e2_blend[il]), // Core
                    Complex::new(e1mantle[il], e2mantle[il]), // Mantle
                    self.tot_vfrac_mantle,
                );
                e1_blend[il] = e1mg;
                e2_blend[il] = e2mg;
            }
        }

        if self.blend_only {
            let file = self.create_output_file("kappa_mix.lnk")?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{:5} {:5.2}", self.nlam, self.rho_av)
                .map_err(|e| anyhow!(e))?;
            for ilam in 0..self.nlam {
                writeln!(
                    writer,
                    "{:15.5e}{:15.5e}{:15.5e}",
                    self.lam[ilam], e1_blend[ilam], e2_blend[ilam]
                )
                .map_err(|e| anyhow!(e))?;
            }
            writer.flush().map_err(|e| anyhow!(e))?;
        }

        if self.write_grid || self.blend_only {
            return Ok(Particle::default());
        }

        // Check how we are going to average over hollow sphere components
        let (f, wf) = if nf > 1 {
            // Get the weights for Gauss-Legendre integration
            gauss_legendre(0.01_f64, self.fmax, nf)
        } else {
            // Single sphere (fmax == 0) or single hollow sphere (0 < fmax <= 0.01)
            (RVector::from_elem(1, self.fmax), RVector::ones(1))
        };

        // Initialize mu
        let mut mu = RVector::zeros(self.nang);
        let nangby2f = (self.nang / 2) as f64;
        for (j, val) in mu.iter_mut().take(self.nang / 2).enumerate() {
            let jf = j as f64;
            let theta = (jf + 0.5) / nangby2f * PI / 2.0;
            *val = theta.cos();
        }

        // Create shared memory for the variables
        let kps = KappaState {
            ns,
            nf,
            ifmn,
            r: &r,
            nr: &nr,
            f: &f,
            wf: &wf,
            e1_blend: &e1_blend,
            e2_blend: &e2_blend,
            mu: &mu,
        };

        let p = {
            let results: Result<Vec<KappaResult>, _> = if self.split {
                (0..self.nlam)
                    .map(|ilam| self.over_wavelengths(ilam, &kps))
                    .collect()
            } else {
                let progress = Progress::new("kappa", "wavelengths", self.nlam);
                (0..self.nlam)
                    .into_par_iter()
                    .map(|ilam| {
                        let r = self.over_wavelengths(ilam, &kps);
                        if !self.quiet {
                            progress.add(1);
                        }
                        r
                    })
                    .collect()
            };

            results.map(|lamrs| {
                let rho = lamrs[0].mass / lamrs[0].vol;
                let k_ext = lamrs.iter().map(|lamr| lamr.k_ext).collect::<RVector>();
                let k_sca = lamrs.iter().map(|lamr| lamr.k_sca).collect::<RVector>();
                let k_abs = lamrs.iter().map(|lamr| lamr.k_abs).collect::<RVector>();
                let g = lamrs.iter().map(|lamr| lamr.g).collect::<RVector>();
                let trust = lamrs.iter().map(|lamr| lamr.trust).collect::<BVector>();
                let mueller = lamrs
                    .into_iter()
                    .map(|lamr| lamr.mueller)
                    .collect::<Vec<Mueller>>();

                let mut is_ok = true;
                let mut is_ok_lmin = 0.0;

                for ilam in 0..self.nlam {
                    if !trust[ilam] {
                        is_ok = false;
                        is_ok_lmin = self.lam[ilam];
                    }
                }

                Particle {
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
                }
            })?
        };

        Ok(p)
    }

    /// Calculate the opacities for a wavelength value.
    fn over_wavelengths(&self, ilam: usize, kps: &KappaState) -> Result<KappaResult> {
        let qabsdqext_min = 1e-4;
        let wvno = 2.0 * PI / self.lam[ilam];
        let mut f11 = RVector::zeros(self.nang);
        let mut f12 = RVector::zeros(self.nang);
        let mut f22 = RVector::zeros(self.nang);
        let mut f33 = RVector::zeros(self.nang);
        let mut f34 = RVector::zeros(self.nang);
        let mut f44 = RVector::zeros(self.nang);
        let mut csca = 0.0;
        let mut cabs = 0.0;
        let mut cext = 0.0;
        let mut smat_nbad = 0;
        let mut mass = 0.0;
        let mut vol = 0.0;
        let nangf = self.nang as f64;
        let mut mie_f11 = RVector::zeros(self.nang);
        let mut mie_f12 = RVector::zeros(self.nang);
        let mut mie_f22 = RVector::zeros(self.nang);
        let mut mie_f33 = RVector::zeros(self.nang);
        let mut mie_f34 = RVector::zeros(self.nang);
        let mut mie_f44 = RVector::zeros(self.nang);
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
         -> Result<(f64, f64)> {
            let lmie = if rad / self.lam[ilam] < 5000.0 {
                self.lam[ilam]
            } else {
                rad / 5000.0
            };
            let miec = MieConfig {
                nangle: self.nang,
                delta: 1e-12,
                thmin,
                thmax,
                step,
                lam: lmie,
                cmm: Complex::new(kps.e1_blend[ilam], kps.e2_blend[ilam]),
                rad,
            };
            let mie_result = de_rooij_1984(&miec)?;
            for j in 0..self.nang {
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
            for j in 0..self.nang {
                let sin_th = (PI * (j as f64 + 0.5) / nangf).sin();
                tot += mie_f11[j] * sin_th;
                tot2 += sin_th;
            }
            mie_f11[0] = (mie_f11[0] + (tot2 - tot) / sin_th0).max(0.0);
            mie_f22[0] = mie_f11[0];
        };

        for is in 0..kps.ns {
            let r1 = kps.r[is];
            match self.method {
                KappaMethod::DHS => {
                    let m_in = Complex::new(1.0, 0.0);
                    let too_large = kps.f[0] > 0.0 && r1 * wvno > self.xlim;
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
                        for j in 0..self.nang {
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
                            let mconj = Complex::new(kps.e1_blend[ilam], -kps.e2_blend[ilam]);
                            let wvno_1 = wvno.min(self.xlim_dhs / rad);
                            let dhs_cfg = DHSConfig {
                                r_core: rcore,
                                r_shell: rad,
                                wave_number: wvno_1,
                                r_indsh: mconj,
                                r_indco: m_in,
                                mu: kps.mu,
                                numang: self.nang / 2,
                                max_angle: self.nang,
                            };

                            match toon_ackerman_1981(&dhs_cfg) {
                                Ok(dhs_result) => {
                                    let ext_mie_factor = dhs_result.q_ext * PI * rad.powi(2);
                                    let scat_mie_factor = dhs_result.q_sca * PI * rad.powi(2);
                                    // factor absorbs the 1/2 in F11, F12; s10 and d10 need 2*factor.
                                    let factor = if scat_mie_factor > 0.0 {
                                        2.0 * PI / (scat_mie_factor * wvno_1.powi(2))
                                    } else {
                                        0.0
                                    };
                                    let factor2 = 2.0 * factor;
                                    for j in 0..(self.nang / 2) {
                                        let back = self.nang - j - 1;
                                        mie_f11[j] = (dhs_result.m1[[j, 0]]
                                            + dhs_result.m0[[j, 0]])
                                            * factor;
                                        mie_f12[j] = (dhs_result.m1[[j, 0]]
                                            - dhs_result.m0[[j, 0]])
                                            * factor;
                                        mie_f22[j] = mie_f11[j];
                                        mie_f33[j] = dhs_result.s10[[j, 0]] * factor2;
                                        mie_f34[j] = -dhs_result.d10[[j, 0]] * factor2;
                                        mie_f44[j] = mie_f33[j];

                                        mie_f11[back] = (dhs_result.m1[[j, 1]]
                                            + dhs_result.m0[[j, 1]])
                                            * factor;
                                        mie_f12[back] = (dhs_result.m1[[j, 1]]
                                            - dhs_result.m0[[j, 1]])
                                            * factor;
                                        mie_f22[back] = mie_f11[back];
                                        mie_f33[back] = dhs_result.s10[[j, 1]] * factor2;
                                        mie_f34[back] = -dhs_result.d10[[j, 1]] * factor2;
                                        mie_f44[back] = mie_f33[back];
                                    }
                                    scat_mie = scat_mie_factor;
                                    ext_mie = ext_mie_factor;
                                }
                                Err(_) => {
                                    spheres = true;
                                }
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
                        for j in 0..self.nang {
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
                KappaMethod::MMF => {
                    let m_mono = 4.0 * PI / 3.0 * self.mmf_a0.powi(3) * self.rho_av;
                    let v_agg = 4.0 * PI / 3.0 * r1.powi(3);
                    let m_agg = v_agg * self.rho_av;
                    let nmono = m_agg / m_mono;
                    let dfrac = if self.mmf_struct > 1.0 {
                        self.mmf_struct
                    } else {
                        3.0 * nmono.ln() / (nmono / self.mmf_struct).ln()
                    };
                    let kfrac = if self.mmf_kf > 0.0 {
                        self.mmf_kf
                    } else {
                        (5.0 / 3.0).powf(dfrac / 2.0)
                    };
                    let fracc = FractalConfig {
                        solver: FractalSolver::ModifiedMeanField,
                        cutoff: FractalCutoff::Gaussian,
                        geometry: FractalGeometry::Tazaki,
                        nang: self.nang / 2 + 1,
                        pn: nmono,
                        r0: self.mmf_a0,
                        df: dfrac,
                        k0: kfrac,
                        lmd: self.lam[ilam],
                        refrel: Complex::new(kps.e1_blend[ilam], kps.e2_blend[ilam]),
                        ..Default::default()
                    };
                    let frac_result = mean_scattering(&fracc)?;
                    if frac_result.dphi > 1.0 {
                        smat_nbad += 1;
                    }
                    let factor = 4.0 * PI / wvno.powi(2);
                    for j in 0..self.nang {
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
                KappaMethod::CDE => {
                    let v = 4.0 * PI * r1.powi(3) / 3.0;
                    vol += kps.nr[is] * v;
                    mass += kps.nr[is] * self.rho_av * v;
                    let m = Complex::new(kps.e1_blend[ilam], kps.e2_blend[ilam]);
                    let m2 = m.powi(2);
                    let dcabs = 2.0 * wvno * v * (m2 / (m2 - 1.0) * m2.ln()).im;
                    let dcsca = if (kps.e2_blend[ilam] / kps.e1_blend[ilam]).abs() < 1e-6 {
                        let mre2 = m.re.powi(2);
                        wvno.powi(4) * v.powi(2) / (3.0 * PI) * (mre2 - 1.0 - mre2.ln())
                    } else {
                        wvno.powi(4) * v.powi(2) * (m2 - 1.0).norm_sqr()
                            / (3.0 * PI * m2.im)
                            * (m2 / (m2 - 1.0) * m2.ln()).im
                    };
                    cabs += kps.nr[is] * dcabs;
                    csca += kps.nr[is] * dcsca;
                    cext += kps.nr[is] * (dcabs + dcsca);
                    if is == kps.ns - 1 {
                        // Exact Rayleigh scattering matrix under Hovenier normalization
                        for j in 0..self.nang {
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

        if self.chop_angle > 0.0 {
            // Flatten the forward-scattering peak
            let ichop = (self.chop_angle / (180.0 / nangf)) as usize;
            if ichop > 0 && ichop < self.nang {
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
            for j in 0..self.nang {
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
        let mut test_scat = true;

        if smat_nbad > 0 {
            test_scat = false;
        }
        for i in 0..self.nang {
            let theta = PI * (i as f64 + 0.5) / nangf;
            let sin_th = theta.sin();
            g += f11[i] * theta.cos() * sin_th;
            tot += f11[i] * sin_th;
        }
        if tot > 0.0 {
            g /= tot;
        }
        if smat_nbad > 0 && !self.mmf_ss {
            g = 0.0;
            f11.fill(0.0);
            f12.fill(0.0);
            f22.fill(0.0);
            f33.fill(0.0);
            f34.fill(0.0);
            f44.fill(0.0);
        }

        let mueller = Mueller {
            f11,
            f12,
            f22,
            f33,
            f34,
            f44,
        };

        let lamr = KappaResult {
            mueller,
            mass,
            vol,
            k_ext,
            k_sca,
            k_abs,
            g,
            trust: test_scat,
        };

        Ok(lamr)
    }

    fn create_output_file(&self, filename: &str) -> Result<File> {
        // Create directory if it doesn't exist
        fs::create_dir_all(&self.outdir)?;

        // Construct full file path
        let filepath = format!("{}/{}", self.outdir, filename);

        // Remove existing file if it exists
        if Path::new(&filepath).exists() {
            fs::remove_file(&filepath)?;
        }

        // Create and return the new file handle
        File::create(&filepath).map_err(|_| anyhow!("Failed to create file!"))
    }

    /// Write the computed opacities to a file in the specified format.
    ///
    /// # Errors
    /// Returns an error if the output file cannot be created or written.
    pub fn write_opacities(&self, fname: &str, p: &Particle) -> Result<()> {
        self.write_opacities_with_bounds(fname, p, self.amin, self.amax, self.na)
    }

    fn write_opacities_with_bounds(
        &self,
        fname: &str,
        p: &Particle,
        amin: f64,
        amax: f64,
        na: usize,
    ) -> Result<()> {
        let (ext, ml) = if self.for_radmc {
            ("inp", "Z11 Z12 Z22 Z33 Z34 Z44")
        } else {
            ("dat", "F11 F12 F22 F33 F34 F44")
        };

        let opacity_filename = format!("{fname}.{ext}");

        let file_opacity = self.create_output_file(&opacity_filename)?;
        let mut writer = BufWriter::new(file_opacity);

        // Write header
        let cc = "#";
        let ameans = self.get_sizedis_moments_for(amin, amax);
        writeln!(
            writer,
            "{cc}============================================================================"
        )?;
        writeln!(
            writer,
            "{} Opacities computed by kappa        <a^n>= {:11.4e} {:11.4e} {:11.4e}",
            cc, ameans[0], ameans[1], ameans[2]
        )?;
        match self.method {
            KappaMethod::MMF => {
                let s_struct = if self.mmf_struct > 1.0 {
                    "(fractal dimension)"
                } else {
                    "(filling factor)"
                };
                writeln!(
                    writer,
                    "{} Method:   {:?}  a0={:7.3}  Struct={:7.3}{}",
                    cc, self.method, self.mmf_a0, self.mmf_struct, s_struct
                )?;
            }
            KappaMethod::CDE => {
                writeln!(writer, "{} Method:   {:?}", cc, self.method)?;
            }
            KappaMethod::DHS => {
                writeln!(
                    writer,
                    "{} Method:   {:?}  fmax={:7.3}",
                    cc, self.method, self.fmax
                )?;
            }
        }
        writeln!(writer, "{cc} Parameters:")?;
        match self.sizedis {
            SizeDistribution::File => {
                let label = self.sizedis_file.as_deref().unwrap_or("<tabulated>");
                writeln!(
                    writer,
                    "{cc}   amin [um]={amin:11.3} amax [um]={amax:11.3}  na  ={na:5}    file={label}"
                )?;
            }
            SizeDistribution::LogNormal | SizeDistribution::Normal => {
                writeln!(
                    writer,
                    "{}   amin [um]={:11.3} amax [um]={:11.3}  na  ={:5}    {:?}={:.4}:{:.4}",
                    cc, amin, amax, na, self.sizedis, self.amean, self.asigma
                )?;
            }
            SizeDistribution::PowerLaw => {
                writeln!(
                    writer,
                    "{}   amin [um]={:11.3} amax [um]={:11.3}  na  ={:5}     apow={:10.2}",
                    cc, amin, amax, na, self.apow
                )?;
            }
        }
        writeln!(
            writer,
            "{}   lmin [um]={:11.3} lmax [um]={:11.3}  nlam={:5}     nang={:6}",
            cc, self.lmin, self.lmax, self.nlam, self.nang
        )?;
        writeln!(
            writer,
            "{}   porosity ={:11.3} p_mantle ={:11.3}  fmax={:9.2} chop=  {:4.1}",
            cc, self.pcore, self.pmantle, self.fmax, self.chop_angle
        )?;

        writeln!(writer, "{cc} Composition:")?;
        writeln!(writer, "{cc}  Where   mfrac  rho   Material")?;
        writeln!(
            writer,
            "{cc}  -----   -----  ----  -----------------------------------------------------"
        )?;

        let total_mfrac: f64 = self.materials.iter().map(|m| m.mfrac).sum();
        for mat in &self.materials {
            writeln!(
                writer,
                "{}  {:?} {:7.3} {:6.2}  {}",
                cc,
                mat.kind,
                mat.mfrac / total_mfrac,
                mat.rho,
                mat.key,
            )?;
        }

        if self.rho_av > 0.0 {
            writeln!(
                writer,
                "{cc}  - - -   - - -  -  -  - - - - - - - - - - - - - - - - - - - - - - - - - - -",
            )?;
            let label = if (self.pcore + self.pmantle) > 0.0 {
                "materials and vacuum"
            } else {
                "materials"
            };
            writeln!(
                writer,
                "{}  {:<6} {:7.3} {:6.2}  mixture of {:3} {}",
                cc,
                "grain",
                1.0,
                self.rho_av,
                self.materials.len(),
                label
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

        if self.write_scatter {
            if self.for_radmc {
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
            if self.nsparse == 0 {
                writeln!(
                    writer,
                    "#    lam(1)    kabs(1)    ksca(1)    g(1)        ! um, cm^2/g, cm^2/g, none"
                )?;
            } else {
                writeln!(
                    writer,
                    "#    lam(1) kabs(1) ksca(1) g(1) imat(1)         ! um, 2x cm^2/g, none, flg"
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

            let iformat = match (self.for_radmc, self.nsparse == 0) {
                (true, true) => 1,
                (true, false) => 101,
                (false, true) => 0,
                (false, false) => 100,
            };
            writeln!(writer, "{iformat}")?;
            writeln!(writer, "{}", self.nlam)?;
            writeln!(
                writer,
                "{}",
                if self.for_radmc {
                    self.nang + 1
                } else {
                    self.nang
                }
            )?;
            writeln!(writer)?;

            for ilam in 0..self.nlam {
                if self.nsparse == 0 {
                    writeln!(
                        writer,
                        "{:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                        self.lam[ilam], p.k_abs[ilam], p.k_sca[ilam], p.g[ilam]
                    )?;
                } else {
                    let flag = usize::from(self.sparse_indices.contains(&ilam));
                    writeln!(
                        writer,
                        "{:15.6e} {:15.6e} {:15.6e} {:15.6e}{:5}",
                        self.lam[ilam], p.k_abs[ilam], p.k_sca[ilam], p.g[ilam], flag
                    )?;
                }
            }

            writeln!(writer)?;
            let nangf = self.nang as f64;
            if self.for_radmc {
                for iang in 0..=self.nang {
                    writeln!(writer, "{:11.5}", (iang as f64) / nangf * 180.0)?;
                }
            } else {
                for iang in 0..self.nang {
                    writeln!(writer, "{:11.5}", (iang as f64 + 0.5) / nangf * 180.0)?;
                }
            }

            writeln!(writer)?;
            for ilam in 0..self.nlam {
                if self.nsparse == 0 || self.sparse_indices.contains(&ilam) {
                    let m = &p.f[ilam];
                    if self.for_radmc {
                        // Interpolate cell centers to cell boundaries 0..=nang and normalize to k_sca.
                        let mut z11 = vec![0.0; self.nang + 1];
                        let mut z12 = vec![0.0; self.nang + 1];
                        let mut z22 = vec![0.0; self.nang + 1];
                        let mut z33 = vec![0.0; self.nang + 1];
                        let mut z34 = vec![0.0; self.nang + 1];
                        let mut z44 = vec![0.0; self.nang + 1];
                        for iang in 0..=self.nang {
                            let i1 = iang.saturating_sub(1);
                            let i2 = iang.min(self.nang - 1);
                            z11[iang] = f64::midpoint(m.f11[i1], m.f11[i2]);
                            z12[iang] = f64::midpoint(m.f12[i1], m.f12[i2]);
                            z22[iang] = f64::midpoint(m.f22[i1], m.f22[i2]);
                            z33[iang] = f64::midpoint(m.f33[i1], m.f33[i2]);
                            z34[iang] = f64::midpoint(m.f34[i1], m.f34[i2]);
                            z44[iang] = f64::midpoint(m.f44[i1], m.f44[i2]);
                        }
                        let mut tot = 0.0;
                        for iang in 1..=self.nang {
                            let theta1 = ((iang - 1) as f64) / nangf * PI;
                            let theta2 = (iang as f64) / nangf * PI;
                            let dmu = theta1.cos() - theta2.cos();
                            tot += f64::midpoint(z11[iang - 1], z11[iang]) * dmu;
                        }
                        tot *= 2.0 * PI;
                        let scale = if tot > 0.0 { p.k_sca[ilam] / tot } else { 0.0 };
                        for iang in 0..=self.nang {
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
                        for i in 0..self.nang {
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
            let header_line = if self.for_radmc {
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
            writeln!(writer, "{}", self.nlam)?;

            for i in 0..self.nlam {
                writeln!(
                    writer,
                    "{:15.6e} {:15.6e} {:15.6e} {:15.6e}",
                    self.lam[i], p.k_abs[i], p.k_sca[i], p.g[i]
                )?;
            }
        }

        writer.flush()?;

        Ok(())
    }

    /// Write the size distribution to a file in the specified format.
    ///
    /// # Errors
    /// Returns an error if the output file cannot be created or written.
    pub fn write_sizedis_file(&self, ns: usize, r: &RVector, nr: &RVector) -> Result<()> {
        let file = self.create_output_file("kappa_sd.dat")?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "# Size distribution written by kappa, can be read in with -a kappa_sd.dat"
        )?;
        if self.split {
            writeln!(
                writer,
                "# This is only the first subparticle because of the -d switch"
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
        writeln!(writer, "{ns}")?;

        for i in 0..ns {
            writeln!(writer, "{:18.5e} {:18.5e}", r[i], nr[i])?;
        }
        writer.flush()?;
        Ok(())
    }
}

pub trait SpecialConfigs {
    fn diana() -> Self;
    fn dsharp() -> Self;
    fn dsharp_no_ice() -> Self;
}

impl SpecialConfigs for KappaConfig {
    fn diana() -> Self {
        let mut kpc = KappaConfig {
            pcore: 0.25,
            ..Default::default()
        };
        kpc.materials.push(Material {
            key: "pyr-mg70".into(),
            kind: MaterialKind::Core,
            mfrac: 0.87,
            ..Default::default()
        });
        kpc.materials.push(Material {
            key: "c-z".into(),
            kind: MaterialKind::Core,
            mfrac: 0.13,
            ..Default::default()
        });
        kpc
    }

    fn dsharp_no_ice() -> Self {
        let mut kpc = KappaConfig {
            ..Default::default()
        };
        kpc.materials.push(Material {
            key: "astrosil".into(),
            kind: MaterialKind::Core,
            mfrac: 0.3291,
            ..Default::default()
        });
        kpc.materials.push(Material {
            key: "c-org".into(),
            kind: MaterialKind::Core,
            mfrac: 0.3966,
            ..Default::default()
        });
        kpc.materials.push(Material {
            key: "fes".into(),
            kind: MaterialKind::Core,
            mfrac: 0.0743,
            ..Default::default()
        });
        kpc
    }

    fn dsharp() -> Self {
        let mut kpc = KappaConfig::dsharp_no_ice();
        kpc.materials.push(Material {
            key: "h2o-w".into(),
            kind: MaterialKind::Core,
            mfrac: 0.2000,
            ..Default::default()
        });
        kpc
    }
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
    e1_blend: &'a RVector,
    e2_blend: &'a RVector,
    mu: &'a RVector,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, Clone))]
#[derive(PartialEq, Debug)]
pub enum SizeDistribution {
    PowerLaw,
    File,
    Normal,
    LogNormal,
}

#[derive(Debug)]
pub enum WavelengthKind {
    Other,
    File,
}

impl SizeDistribution {}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, Clone))]
#[derive(Debug)]
pub enum KappaMethod {
    DHS,
    MMF,
    CDE,
}

#[derive(Debug)]
pub enum KappaError {
    ZeroMassFraction,
    ForceLogNormal,
    Other(String),
}

impl From<anyhow::Error> for KappaError {
    fn from(err: anyhow::Error) -> Self {
        KappaError::Other(err.to_string())
    }
}

/// Mueller matrix structure
pub struct Mueller {
    pub f11: RVector,
    pub f12: RVector,
    pub f22: RVector,
    pub f33: RVector,
    pub f44: RVector,
    pub f34: RVector,
}

#[derive(Default)]
/// Particle
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

/// Defines a material component which is initialized
/// at runtime by consuming a LNK file.
///
/// See [`crate::components::StaticComponent`] for an equivalent
/// version which can is instantiated statically at compile time.
#[derive(Debug)]
pub struct Component {
    /// Name of the component.
    pub name: String,
    /// Class of the component.
    pub class: String,
    /// State of the component.
    pub state: String,
    /// Density of the component.
    pub rho: f64,
    /// Number of wavelengths.
    pub size: usize,
    /// Wavelengths.
    pub l0: RVector,
    /// Refractive index.
    pub n0: RVector,
    /// Extinction coefficient.
    pub k0: RVector,
}

/// Converts a [`crate::components::StaticComponent`] by value into a [`Component`].
///
/// This consumes the input [`crate::components::StaticComponent`], moving its data into the new [`Component`].
impl From<crate::components::StaticComponent> for Component {
    fn from(sc: crate::components::StaticComponent) -> Self {
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

/// Converts a reference to a [`crate::components::StaticComponent`] into a [`Component`].
///
/// This clones data from the borrowed [`crate::components::StaticComponent`], allowing reuse of the original.
impl From<&crate::components::StaticComponent> for Component {
    fn from(sc: &crate::components::StaticComponent) -> Self {
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

fn bruggeman_blend(abun: &[f64], e_in: &[Complex64]) -> Result<Complex64> {
    let mut abunvac = 1.0 - abun.iter().sum::<f64>();
    let mvac = Complex::new(1.0, 0.0);
    if abunvac < 0.0 {
        if abunvac.abs() < 1e-5 {
            abunvac = 0.0;
        } else {
            return Err(anyhow!("UnnormalizedAbundance"));
        }
    }

    // Blend iteratively
    let mut mm = mvac;
    let mut tot: Complex64 = Complex::new(0.0, 0.0);
    let mut me: Complex64 = Complex::new(0.0, 0.0);
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
    if tot.abs() / mm.abs() > 1e-6 {
        static WARNED: Once = Once::new();
        WARNED.call_once(|| {
            eprintln!("Warning: bruggeman: blend not converged (further occurrences not reported)");
        });
    }
    Ok(me)
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use approx::assert_relative_eq;
    use planetes_ext::types::RVector;

    use super::{KappaConfig, Material, SpecialConfigs};
    use super::{KappaMethod, MaterialKind, RefractiveIndexKind, SizeDistribution};

    fn builtin_material(key: &str, kind: MaterialKind, mfrac: f64) -> Material {
        Material {
            key: key.to_string(),
            kind,
            mfrac,
            ..Default::default()
        }
    }

    fn temp_dir(label: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/kappa_opac_test_{label}_{nanos}")
    }

    #[test]
    fn diana_dhs_and_mueller_consistency() {
        let mut cfg = KappaConfig::diana();
        cfg.amin = 1.0;
        cfg.amax = 1.0;
        cfg.na = 1;
        cfg.lmin = 1.0;
        cfg.lmax = 1.0;
        cfg.nlam = 1;
        cfg.method = KappaMethod::DHS;
        cfg.fmax = 0.8;
        cfg.prepare_inputs().unwrap();
        cfg.initialize().unwrap();
        let p = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();

        assert_relative_eq!(p.k_abs[0], 5403.01, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 11232.9, max_relative = 1e-3);
        assert_relative_eq!(p.k_ext[0], 16635.9, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.878_882, max_relative = 2e-3);

        // B1 & B7 checks: F22 == F11 everywhere (including bin 0) and F33(0) == F44(0) ~ F11(0)
        let m = &p.f[0];
        for j in 0..cfg.nang {
            assert_relative_eq!(m.f22[j], m.f11[j], max_relative = 1e-12);
            assert_relative_eq!(m.f44[j], m.f33[j], max_relative = 1e-12);
        }
        assert!(
            m.f33[0] / m.f11[0] > 0.95,
            "DHS F33(0)/F11(0) = {} (missing factor of 2 if ~0.5)",
            m.f33[0] / m.f11[0]
        );

        // B7 check: 0 < fmax <= 0.01 uses a single hollow sphere cleanly
        cfg.fmax = 0.005;
        let p_tiny = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();
        assert!(p_tiny.k_ext[0].is_finite() && p_tiny.k_ext[0] > 0.0);

        // B3 check: capped DHS wvno_1 keeps scattering matrix properly normalized
        cfg.fmax = 0.8;
        cfg.amin = 5.0;
        cfg.amax = 5.0;
        cfg.xlim_dhs = 10.0;
        let p_cap = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();
        let m_cap = &p_cap.f[0];
        assert_relative_eq!(m_cap.f22[0], m_cap.f11[0], max_relative = 1e-12);
        assert!(
            m_cap.f33[0] / m_cap.f11[0] > 0.90,
            "capped DHS F33(0)/F11(0) = {} (wvno mismatch if tiny)",
            m_cap.f33[0] / m_cap.f11[0]
        );
    }

    #[test]
    fn dsharp_mie_and_large_grain_fallback() {
        let mut cfg = KappaConfig::dsharp();
        cfg.amin = 0.1;
        cfg.amax = 100.0;
        cfg.apow = 3.5;
        cfg.na = 15;
        cfg.lmin = 10.0;
        cfg.lmax = 10.0;
        cfg.nlam = 1;
        cfg.method = KappaMethod::DHS;
        cfg.fmax = 0.0;
        cfg.prepare_inputs().unwrap();
        cfg.initialize().unwrap();
        let p = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();

        assert_relative_eq!(p.k_abs[0], 455.568, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 293.273, max_relative = 1e-3);
        assert_relative_eq!(p.k_ext[0], 748.842, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.779_307, max_relative = 1e-3);

        // O1 check: large grain (rad / lam = 10_000 >= 5000) must cap lam, not rad,
        // giving geometric-optics extinction kappa_ext ~ 2 * (3 / (4 * rho * a)) * 1e4 = 1.5e4 / (rho * a).
        let mut big = KappaConfig {
            amin: 1e4,
            amax: 1e4,
            na: 1,
            lmin: 1.0,
            lmax: 1.0,
            nlam: 1,
            fmax: 0.8,
            xlim: 1e4,
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        big.prepare_inputs().unwrap();
        big.initialize().unwrap();
        let p_big = big.compute_kappa(0, big.amin, big.amax).unwrap();
        let expected_geo = 1.5e4 / (big.rho_av * 1e4);
        assert!(
            p_big.k_ext[0] > 0.5 * expected_geo && p_big.k_ext[0] < 3.0 * expected_geo,
            "large-grain k_ext = {}, expected order {expected_geo}",
            p_big.k_ext[0]
        );
    }

    #[test]
    fn core_mantle_porosity_and_cmdline_material() {
        // 2 core + 1 mantle + distinct core/mantle porosities
        let mut cfg = KappaConfig {
            amin: 0.1,
            amax: 10.0,
            apow: 3.5,
            na: 10,
            lmin: 3.0,
            lmax: 3.0,
            nlam: 1,
            fmax: 0.0,
            pcore: 0.25,
            pmantle: 0.1,
            materials: vec![
                builtin_material("pyr", MaterialKind::Core, 0.7),
                builtin_material("c-z", MaterialKind::Core, 0.3),
                builtin_material("h2o-w", MaterialKind::Mantle, 0.2),
            ],
            ..Default::default()
        };
        cfg.prepare_inputs().unwrap();
        cfg.initialize().unwrap();
        let p = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();
        assert_relative_eq!(p.k_abs[0], 2991.23, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 1890.09, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.730_994, max_relative = 1e-3);

        // O4 & O7 check: 1 core + 2 mantles (nmant > ncore) must not panic or corrupt mixing
        let mut cfg2 = KappaConfig {
            amin: 1.0,
            amax: 1.0,
            na: 1,
            lmin: 3.0,
            lmax: 3.0,
            nlam: 1,
            fmax: 0.0,
            pcore: 0.2,
            pmantle: 0.1,
            materials: vec![
                builtin_material("pyr", MaterialKind::Core, 0.7),
                builtin_material("h2o-w", MaterialKind::Mantle, 0.2),
                builtin_material("c-z", MaterialKind::Mantle, 0.1),
            ],
            ..Default::default()
        };
        cfg2.prepare_inputs().unwrap();
        cfg2.initialize().unwrap();
        let p2 = cfg2.compute_kappa(0, cfg2.amin, cfg2.amax).unwrap();
        assert_relative_eq!(p2.k_abs[0], 4747.15, max_relative = 1e-3);
        assert_relative_eq!(p2.k_sca[0], 3297.32, max_relative = 1e-3);
        assert_relative_eq!(p2.g[0], 0.699_504, max_relative = 1e-3);

        // CmdLine (n:k:rho) material with nlam != 300
        let mut cmd_cfg = KappaConfig {
            amin: 0.5,
            amax: 0.5,
            na: 1,
            lmin: 1.0,
            lmax: 10.0,
            nlam: 5,
            fmax: 0.0,
            materials: vec![Material {
                key: "1.6:0.05:2.5".to_string(),
                kind: MaterialKind::Core,
                n: 1.6,
                k: 0.05,
                rho: 2.5,
                mfrac: 1.0,
                cmd: RefractiveIndexKind::CmdLine,
                ..Default::default()
            }],
            ..Default::default()
        };
        cmd_cfg.prepare_inputs().unwrap();
        cmd_cfg.initialize().unwrap();
        let p_cmd = cmd_cfg.compute_kappa(0, cmd_cfg.amin, cmd_cfg.amax).unwrap();
        assert_eq!(p_cmd.k_ext.len(), 5);
        assert_relative_eq!(p_cmd.rho, 2.5, max_relative = 1e-12);
    }

    #[test]
    fn mmf_fractal_and_filling_factor() {
        // Fractal dimension mode (mmf_struct > 1)
        let mut cfg = KappaConfig {
            amin: 1.0,
            amax: 1.0,
            na: 1,
            lmin: 10.0,
            lmax: 10.0,
            nlam: 1,
            method: KappaMethod::MMF,
            mmf_a0: 0.1,
            mmf_struct: 2.0,
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        cfg.prepare_inputs().unwrap();
        cfg.initialize().unwrap();
        let p = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();
        assert_relative_eq!(p.k_abs[0], 2744.68, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 180.933, max_relative = 1e-3);
        assert_relative_eq!(p.g[0], 0.399_211, max_relative = 2e-3);
        assert!(p.f[0].f11[0] > 1.0);

        // Filling factor mode (mmf_struct <= 1)
        cfg.mmf_struct = 0.2;
        let p_ff = cfg.compute_kappa(0, cfg.amin, cfg.amax).unwrap();
        assert_relative_eq!(p_ff.k_abs[0], 2850.34, max_relative = 1e-3);
        assert_relative_eq!(p_ff.k_sca[0], 331.925, max_relative = 1e-3);
        assert_relative_eq!(p_ff.g[0], 0.173_558, max_relative = 2e-3);
    }

    #[test]
    fn cde_exact_rayleigh_and_chop() {
        // Single-grain CDE (exact Rayleigh scattering matrix)
        let mut cde = KappaConfig {
            amin: 0.1,
            amax: 0.1,
            na: 1,
            lmin: 10.0,
            lmax: 10.0,
            nlam: 1,
            method: KappaMethod::CDE,
            materials: vec![builtin_material("astrosil", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        cde.prepare_inputs().unwrap();
        cde.initialize().unwrap();
        let p = cde.compute_kappa(0, cde.amin, cde.amax).unwrap();
        assert_relative_eq!(p.k_abs[0], 3071.49, max_relative = 1e-3);
        assert_relative_eq!(p.k_sca[0], 0.435_909, max_relative = 1e-3);
        assert!(
            p.g[0].abs() < 1e-13,
            "exact Rayleigh g must be 0, got {}",
            p.g[0]
        );

        // B5 regression check: [0.05, 0.1] must retain the 0.05 um endpoint bin
        // (dropping it inflates k_sca from 0.21538 to 0.24762).
        cde.amin = 0.05;
        cde.amax = 0.1;
        cde.apow = 3.5;
        cde.na = 5;
        let p_grid = cde.compute_kappa(0, cde.amin, cde.amax).unwrap();
        assert_relative_eq!(p_grid.k_abs[0], 3071.49, max_relative = 1e-3);
        assert_relative_eq!(p_grid.k_sca[0], 0.215_381, max_relative = 1e-3);

        let m = &p.f[0];
        for j in 0..cde.nang {
            let theta = PI * (j as f64 + 0.5) / cde.nang as f64;
            assert_relative_eq!(m.f11[j], 0.75 * (1.0 + theta.cos().powi(2)), max_relative = 1e-12);
            assert_relative_eq!(m.f12[j], -0.75 * theta.sin().powi(2), max_relative = 1e-12);
            assert_relative_eq!(m.f33[j], 1.5 * theta.cos(), max_relative = 1e-12);
            assert_relative_eq!(m.f34[j], 0.0, epsilon = 1e-15);
            assert_relative_eq!(
                m.f11[j].powi(2),
                m.f12[j].powi(2) + m.f33[j].powi(2),
                max_relative = 1e-12
            );
        }

        // Forward-peak chopping check (including off-by-1 fix at ichop)
        let mut chop = KappaConfig {
            amin: 10.0,
            amax: 10.0,
            na: 1,
            lmin: 1.0,
            lmax: 1.0,
            nlam: 1,
            fmax: 0.0,
            chop_angle: 5.0,
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        chop.prepare_inputs().unwrap();
        chop.initialize().unwrap();
        let pc = chop.compute_kappa(0, chop.amin, chop.amax).unwrap();
        assert_relative_eq!(pc.k_abs[0], 117.230, max_relative = 1e-3);
        assert_relative_eq!(pc.k_sca[0], 191.453, max_relative = 1e-3);
        assert_relative_eq!(pc.g[0], 0.671_673, max_relative = 1e-3);
        for i in 0..5 {
            assert_relative_eq!(pc.f[0].f11[i], pc.f[0].f11[5], max_relative = 1e-12);
        }
    }

    #[test]
    fn size_distribution_moments_and_outputs() {
        // Normal distribution moments with log-grid Jacobian: mean of N(10, 1) on [1, 100] is 10
        let mut norm_cfg = KappaConfig {
            amin: 1.0,
            amax: 100.0,
            amean: 10.0,
            asigma: -1.0,
            sizedis: SizeDistribution::Normal,
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        norm_cfg.prepare_inputs().unwrap();
        assert_eq!(norm_cfg.sizedis, SizeDistribution::Normal);
        let norm_m = norm_cfg.get_sizedis_moments();
        assert_relative_eq!(norm_m[0], 10.0, max_relative = 1e-3);
        assert_relative_eq!(norm_m[1], (100.0_f64 + 1.0).sqrt(), max_relative = 1e-3);

        // Log-normal moments: <a^k>^(1/k) = a0 * exp(k * sigma^2 / 2)
        let mut lgnm_cfg = KappaConfig {
            amin: 0.01,
            amax: 1000.0,
            amean: 1.0,
            asigma: 0.5,
            sizedis: SizeDistribution::Normal,
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        lgnm_cfg.prepare_inputs().unwrap();
        assert_eq!(lgnm_cfg.sizedis, SizeDistribution::LogNormal);
        let lgnm_m = lgnm_cfg.get_sizedis_moments();
        assert_relative_eq!(lgnm_m[0], (0.5 * 0.25_f64).exp(), max_relative = 1e-3);
        assert_relative_eq!(lgnm_m[1], 0.25_f64.exp(), max_relative = 1e-3);
        assert_relative_eq!(lgnm_m[2], (1.5 * 0.25_f64).exp(), max_relative = 1e-3);

        // SizeDistribution::File + RADMC-3D scattering matrix + blend_only + split outputs
        let out = temp_dir("outputs");
        let mut file_cfg = KappaConfig {
            sizedis: SizeDistribution::File,
            sizedis_file: Some("synthetic.dat".to_string()),
            r_file: RVector::from_vec(vec![0.1, 1.0]),
            nr_file: RVector::from_vec(vec![100.0, 1.0]),
            lmin: 1.0,
            lmax: 10.0,
            nlam: 2,
            fmax: 0.0,
            write_scatter: true,
            for_radmc: true,
            outdir: out.clone(),
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        file_cfg.prepare_inputs().unwrap();
        file_cfg.initialize().unwrap();
        file_cfg.run().unwrap();
        let radmc_path = format!("{out}/kappa_opacity.inp");
        let radmc_text = fs::read_to_string(&radmc_path).unwrap();
        assert!(radmc_text.contains("file=synthetic.dat"));
        assert!(radmc_text.contains("RADMC-3D"));

        // Blend-only output (must return Ok without exit(0))
        let mut blend_cfg = KappaConfig::diana();
        blend_cfg.lmin = 1.0;
        blend_cfg.lmax = 10.0;
        blend_cfg.nlam = 3;
        blend_cfg.blend_only = true;
        blend_cfg.outdir = out.clone();
        blend_cfg.prepare_inputs().unwrap();
        blend_cfg.initialize().unwrap();
        blend_cfg.run().unwrap();
        let mix_text = fs::read_to_string(format!("{out}/kappa_mix.lnk")).unwrap();
        assert_eq!(mix_text.lines().count(), 4);

        // Split mode (-d) writes na files without calling exit(0)
        let mut split_cfg = KappaConfig {
            amin: 0.1,
            amax: 1.0,
            na: 3,
            nsubgrains: 3,
            split: true,
            lmin: 1.0,
            lmax: 1.0,
            nlam: 1,
            fmax: 0.0,
            outdir: out.clone(),
            materials: vec![builtin_material("pyr", MaterialKind::Core, 1.0)],
            ..Default::default()
        };
        split_cfg.prepare_inputs().unwrap();
        split_cfg.initialize().unwrap();
        split_cfg.run().unwrap();
        for idx in 1..=3 {
            let split_file = format!("{out}/kappa_opacity_{idx:03}.dat");
            assert!(
                fs::metadata(&split_file).is_ok(),
                "missing split output {split_file}"
            );
        }

        let _ = fs::remove_dir_all(&out);
    }
}
