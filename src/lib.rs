#![allow(clippy::too_many_lines)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_wrap)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/PlanetesLAB/documentation/refs/heads/main/logo.jpg"
)]

//! Dust opacity and Mueller scattering matrix calculator for astrophysics.
//!
//! `kappa` computes absorption, scattering, and extinction mass opacities
//! ($\kappa_{\mathrm{abs}}$, $\kappa_{\mathrm{sca}}$, $\kappa_{\mathrm{ext}}$),
//! the scattering asymmetry parameter $g = \langle \cos\theta \rangle$, and the
//! six independent elements of the Mueller scattering matrix
//! ($F_{11}, F_{12}, F_{22}, F_{33}, F_{34}, F_{44}$) for complex dust grain
//! populations. It ships as both a Rust library and a standalone command-line
//! executable (feature-gated under `cli`).
//!
//! For theoretical background, see the
//! [PlanetesLAB documentation](https://planeteslab.github.io/tools/software/kappa/).
//!
//! # Scattering Theories
//!
//! - **Distribution of Hollow Spheres (DHS):** Models irregular grains and
//!   low-porosity aggregates by averaging concentric hollow spheres over inner
//!   cavity volume fractions $f \in [0, f_{\max}]$
//!   ([Min et al. 2005](https://ui.adsabs.harvard.edu/abs/2005A%26A...432..909M/abstract)),
//!   using the stratified-sphere algorithm of
//!   [Toon & Ackerman (1981)](https://ui.adsabs.harvard.edu/abs/1981ApOpt..20.3657T/abstract).
//! - **Lorenz-Mie theory:** Exact scattering by compact homogeneous spheres
//!   ([De Rooij & Van der Stap 1984](https://ui.adsabs.harvard.edu/abs/1984A%26A...131..237D/abstract)).
//! - **Modified Mean Field theory (MMF):** Light scattering by porous fractal
//!   aggregates of spherical monomers
//!   ([Tazaki & Tanaka 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract)),
//!   with analytical and numerical geometric cross sections from
//!   [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract).
//! - **Continuous Distribution of Ellipsoids (CDE):** Analytical Rayleigh-limit
//!   shape distribution (Bohren & Huffman 1983, Ch. 12) paired with the exact
//!   Rayleigh phase matrix.
//!
//! Multi-component core and mantle mixtures are blended with the Bruggeman
//! effective medium rule (including optional vacuum porosity), and mantles are
//! coated onto cores via the Maxwell Garnett rule. Built-in presets are provided
//! for the DIANA standard opacity model
//! ([Woitke et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...586A.103W/abstract))
//! and the DSHARP opacity model
//! ([Birnstiel et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..45B/abstract)).
//!
//! # Physical Units and Conventions
//!
//! - Grain radii $a$ and wavelengths $\lambda$: $\mu\mathrm{m}$
//! - Material mass densities $\rho$: $\mathrm{g\thinspace cm^{-3}}$
//! - Mass opacities $\kappa_{\mathrm{abs}}, \kappa_{\mathrm{sca}}, \kappa_{\mathrm{ext}}$:
//!   $\mathrm{cm^{2}\thinspace g^{-1}}$
//! - Standard scattering matrix elements $F_{ij}(\theta)$: dimensionless
//!   Hovenier et al. (2004) convention normalized so that
//!   $$\frac{1}{2} \int_{0}^{\pi} F_{11}(\theta) \sin\theta \thinspace \mathrm{d}\theta = 1$$
//!   with consistent sign convention for $F_{34}$ across Mie, DHS, MMF, and CDE.
//! - RADMC-3D output (`for_radmc = true`): $Z_{ij}(\theta)$ in
//!   $\mathrm{cm^{2}\thinspace g^{-1}\thinspace sr^{-1}}$ normalized so that
//!   $2\pi \int_{0}^{\pi} Z_{11}(\theta)\sin\theta\thinspace\mathrm{d}\theta = \kappa_{\mathrm{sca}}$.
//!
//! # Library Quickstart
//!
//! ```no_run
//! use kappa::opac::{KappaConfig, ScatteringMethod, SizeDistribution, WavelengthGrid};
//!
//! let mut config = KappaConfig::diana();
//! config.size_distribution = SizeDistribution::PowerLaw {
//!     min: 0.05,
//!     max: 1000.0,
//!     slope: 3.5,
//!     bins: Some(50),
//! };
//! config.wavelengths = WavelengthGrid::LogSpaced {
//!     min: 0.1,
//!     max: 1e4,
//!     count: 100,
//! };
//! config.method = ScatteringMethod::dhs(0.8);
//!
//! let particle = config.compute()?;
//! println!("k_abs(0) = {:.3e} cm^2/g", particle.k_abs[0]);
//! # Ok::<(), kappa::opac::KappaError>(())
//! ```
//!
//! # Command-Line Interface
//!
//! Build the executable with default features (`cli` enabled):
//!
//! ```text
//! cargo build --release
//! kappa --help
//! kappa materials
//! kappa --preset dsharp --method mie --lam 1300 --scatter --radmc -o out
//! kappa -c pyr=0.87 -c c-z=0.13 -m h2o-w=0.2 -p 0.25 --method dhs --fmax 0.8
//! ```
//!
//! # Crate Features
//!
//! - `cli` (enabled by default): builds the `kappa` binary and exposes [`cli`],
//!   pulling in `clap` and `colored`. Library consumers can set
//!   `default-features = false` in `Cargo.toml` to omit CLI dependencies.
//! - `serde`: enables `Serialize` and `Deserialize` derives on configuration
//!   types in [`opac`].
//!
//! # Module Overview
//!
//! - [`opac`]: High-level configuration ([`opac::KappaConfig`]), validation and
//!   mixing ([`opac::PreparedModel`]), opacity evaluation ([`opac::Particle`]),
//!   and ASCII/RADMC-3D writers.
//! - [`mie`]: Lorenz-Mie solver ([`mie::de_rooij_1984`]).
//! - [`dhs`]: Concentric core-shell sphere solver ([`dhs::toon_ackerman_1981`]).
//! - [`fractal`]: RGD, Mean Field, and Modified Mean Field aggregate scattering
//!   ([`fractal::mean_scattering`]).
//! - [`geofractal`]: Aggregate geometric cross-section models
//!   ([`geofractal::get_geometric_cross_section_tazaki`]).
//! - [`components`]: Built-in optical constant tables and key lookup.
//! - [`io`]: Readers for `.lnk` optical constants, size distributions, and
//!   wavelength grids.
//! - [`utils`]: Special functions (Legendre, spherical Bessel, Gamma, confluent
//!   hypergeometric), complex LU linear solver, and log-log regridding.
//!
//! # Built-in Materials
//!
//! Built-in optical constants are accessed by key via [`opac::MaterialSource::Builtin`]
//! or [`components::try_get_lnk_data`]. Wavelengths outside the tabulated range
//! are extrapolated automatically.
//!
//! | Key (alias) | Composition | State | $\rho\ [\mathrm{g\thinspace cm^{-3}}]$ | $\lambda_{\min}\ [\mu\mathrm{m}]$ | $\lambda_{\max}\ [\mu\mathrm{m}]$ | Reference |
//! | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
//! | `pyr-mg100` | $\mathrm{MgSiO_{3}}$ | Amorphous | $2.71$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg95` | $\mathrm{Mg_{0.95}Fe_{0.05}SiO_{3}}$ | Amorphous | $2.74$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg80` | $\mathrm{Mg_{0.8}Fe_{0.2}SiO_{3}}$ | Amorphous | $2.90$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg70` (`pyr`) | $\mathrm{Mg_{0.7}Fe_{0.3}SiO_{3}}$ | Amorphous | $3.01$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg60` | $\mathrm{Mg_{0.6}Fe_{0.4}SiO_{3}}$ | Amorphous | $3.10$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg50` | $\mathrm{Mg_{0.5}Fe_{0.5}SiO_{3}}$ | Amorphous | $3.20$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-mg40` | $\mathrm{Mg_{0.4}Fe_{0.6}SiO_{3}}$ | Amorphous | $3.30$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `pyr-c-mg96` (`ens`) | $\mathrm{Mg_{0.96}Fe_{0.04}SiO_{3}}$ | Crystalline | $2.80$ | $2.0$ | $98.9$ | [Jaeger et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998A%26A...339..904J/abstract) |
//! | `ol-mg50` (`ol`) | $\mathrm{MgFeSiO_{4}}$ | Amorphous | $3.71$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `ol-mg40` | $\mathrm{Mg_{0.8}Fe_{1.2}SiO_{4}}$ | Amorphous | $3.71$ | $0.2$ | $500$ | [Dorschner et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract) |
//! | `ol-c-mg100` (`for`) | $\mathrm{Mg_{2}SiO_{4}}$ | Crystalline | $3.27$ | $5.0$ | $200$ | [Suto et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.370.1599S/abstract) |
//! | `ol-c-mg95` | $\mathrm{Mg_{1.9}Fe_{0.1}SiO_{4}}$ | Crystalline | $3.33$ | $2.0$ | $8190$ | [Fabian et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001A%26A...378..228F/abstract) |
//! | `ol-c-mg00` (`fay`) | $\mathrm{Fe_{2}SiO_{4}}$ | Crystalline | $4.39$ | $2.0$ | $10^{4}$ | [Fabian et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001A%26A...378..228F/abstract) |
//! | `astrosil` | $\mathrm{MgFeSiO_{4}}$ | Amorphous | $3.30$ | $6.2 \times 10^{-5}$ | $1.24 \times 10^{5}$ | [Draine (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract) |
//! | `c-z` (`c`) | $\mathrm{C}$ | Amorphous | $1.80$ | $0.05$ | $10^{4}$ | [Zubko et al. (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282.1321Z/abstract) |
//! | `c-p` | $\mathrm{C}$ | Amorphous | $1.80$ | $0.105$ | $800$ | [Preibisch et al. (1993)](https://ui.adsabs.harvard.edu/abs/1993A%26A...279..577P/abstract) |
//! | `c-gra` (`gra`) | $\mathrm{C}$ graphite | Crystalline | $2.16$ | $10^{-3}$ | $10^{3}$ | [Draine (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1026D/abstract) |
//! | `c-org` (`org`) | $\mathrm{CHON}$ organics | Amorphous | $1.50$ | $0.1$ | $10^{5}$ | [Henning & Stognienko (1996)](https://ui.adsabs.harvard.edu/abs/1996A%26A...311..291H/abstract) |
//! | `c-nano` (`nano`) | $\mathrm{C}$ nano-diamond | Crystalline | $2.30$ | $0.0202$ | $110$ | [Mutschke et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004A%26A...423..983M/abstract) |
//! | `fe-c` (`iron`) | $\mathrm{Fe}$ | Metallic | $7.87$ | $0.1$ | $10^{5}$ | [Henning & Stognienko (1996)](https://ui.adsabs.harvard.edu/abs/1996A%26A...311..291H/abstract) |
//! | `fes` (`tro`) | $\mathrm{FeS}$ | Crystalline | $4.83$ | $0.1$ | $10^{5}$ | [Henning & Stognienko (1996)](https://ui.adsabs.harvard.edu/abs/1996A%26A...311..291H/abstract) |
//! | `sic` | $\mathrm{SiC}$ | Crystalline | $3.22$ | $10^{-3}$ | $10^{3}$ | [Laor & Draine (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJ...402..441L/abstract) |
//! | `sio2` (`qua`) | $\mathrm{SiO_{2}}$ | Amorphous | $2.65$ | $6.2 \times 10^{-4}$ | $502$ | [Kitamura et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007ApOpt..46.8118K/abstract) |
//! | `cor-c` (`cor`) | $\mathrm{Al_{2}O_{3}}$ | Crystalline | $4.00$ | $0.5$ | $400$ | [Koike et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995Icar..114..203K/abstract) |
//! | `h2o-w` (`h2o`, `ice-w`) | $\mathrm{H_{2}O}$ ice | Crystalline | $0.92$ | $0.0443$ | $2 \times 10^{6}$ | [Warren & Brandt (2008)](https://ui.adsabs.harvard.edu/abs/2008JGRD..11314220W/abstract) |
//! | `h2o-a` | $\mathrm{H_{2}O}$ ice | Amorphous | $0.92$ | $0.0443$ | $2 \times 10^{6}$ | [Hudgins et al. (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJS...86..713H/abstract) |
//! | `co2-w` (`co2`) | $\mathrm{CO_{2}}$ ice | Crystalline | $1.60$ | $0.052$ | $2 \times 10^{5}$ | [Warren (1986)](https://ui.adsabs.harvard.edu/abs/1986ApOpt..25.2650W/abstract) |
//! | `nh3-m` (`nh3`) | $\mathrm{NH_{3}}$ ice | Crystalline | $0.75$ | $0.14$ | $200$ | [Martonchik et al. (1984)](https://ui.adsabs.harvard.edu/abs/1984ApOpt..23..541M/abstract) |
//! | `co-a` (`co`) | $\mathrm{CO}$ ice | Amorphous | $0.81$ | $3.85$ | $6.0$ | [Palumbo et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006PCCP....8..279P/abstract) |
//! | `co2-a` / `co2-c` | $\mathrm{CO_{2}}$ ice | Amorphous / Crystalline | $1.20 / 1.67$ | $2.5$ | $19.9$ | [Gerakines & Hudson (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...901...52G/abstract) |
//! | `ch4-a` (`ch4`) / `ch4-c` | $\mathrm{CH_{4}}$ ice | Amorphous / Crystalline | $0.47$ | $2.0$ | $20.0$ | [Gerakines & Hudson (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...901...52G/abstract) |
//! | `ch3oh-a` (`ch3oh`) / `ch3oh-c` | $\mathrm{CH_{3}OH}$ ice | Amorphous / Crystalline | $0.78 / 1.02$ | $2.0$ | $24.0$ | [Gerakines & Hudson (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...901...52G/abstract) |

#[cfg(feature = "cli")]
pub mod cli;
pub mod components;
pub mod dhs;
pub mod fractal;
pub mod geofractal;
pub mod io;
pub mod mie;
pub mod opac;
pub mod utils;
