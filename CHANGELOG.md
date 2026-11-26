# Changelog

All notable changes to `kappa` are documented in this file.

## [1.0.0] - 2026-11-26

First stable release of `kappa` as both a Rust library and a standalone command-line tool.

### Breaking Changes

- **Idiomatic library API (`opac`):** Replaced flat configuration fields with algebraic types (`SizeDistribution`, `WavelengthGrid`, `ScatteringMethod`, `AggregateStructure`, `MaterialSource`), a two-stage validated `KappaConfig::prepare() -> PreparedModel` pipeline (with one-line `compute()` and `run()` helpers), and structured `KappaError` variants.
- **Feature-gated CLI:** Moved the executable and `cli` module behind the default `cli` Cargo feature (`clap` + `colored`), allowing library consumers to pass `default-features = false`.
- **Modern `clap`-based CLI:** Migrated from ad-hoc argument parsing to `clap` with explicit conflict checking, repeatable `--core`/`--mantle` material specs (`<key|file.lnk|n:k:rho>[=mfrac]`), `--preset`, `--method`, `--radmc`, `--sparse`, `--split`, `--label`, `--quiet`, and a `materials` subcommand.
- **Scientific naming & cleanup:** Standardized public struct and field names across `mie`, `dhs`, `fractal`, `geofractal`, `components`, and `opac`, removed dead `config` and unimplemented `utils::special::optic_limit` stubs, and adopted `planetes-ext` `v1.10.1` `progress::Progress`.

### Fixes

- **Special functions & linear algebra (`utils`):** Repaired Legendre endpoint handling (`lpmns`/`lpn`), confluent hypergeometric negative-integer and recurrence branches, spherical Bessel integration bounds and unitarity checks, and replaced real-expanded inversion with a direct complex LU solver on $\mathbb{C}^{n \times n}$.
- **Mie & DHS solvers (`mie`, `dhs`):** Safeguarded downward logarithmic-derivative truncation when $|m| < 1$, fixed angle-grid rounding and parity handling, and repaired core-shell DHS weighting and fallback behavior.
- **Fractal aggregate scattering (`fractal`, `geofractal`):** Fixed out-of-bounds indexing and overflow guards in Modified Mean Field (MMF) and Tazaki (2021) geometric cross-section calculations, and aligned the MMF $F_{34}$ sign with the Hovenier et al. (2004) convention across all solvers.
- **Opacity orchestration & quadrature (`opac`):** Replaced the Mie workaround in Continuous Distribution of Ellipsoids (CDE) with the exact Rayleigh scattering matrix, adopted composite trapezoidal endpoint weights ($1/2$ at $a_{\min}$ and $a_{\max}$) on analytic logarithmic size grids to eliminate $O(\Delta \ln a)$ endpoint bias, fixed forward-chopping index handling, and completed standard `.dat`, RADMC-3D `.inp`, `--sparse`, `--blend-only`, `--write-grid`, and `--split` writers.
- **Materials & I/O (`components`, `io`):** Fixed Warren (1986) `co2-w` optical constant typo, `pyr-mg80` density (`2.09 -> 2.90`), `ch4-a` and `co2-w` names, missing `"ch4"`/`"ch3oh"` aliases, non-panicking `try_get_lnk_data`, and robust `.lnk`/size-distribution/wavelength-grid readers.

### Performance & Memory

- Streaming zero-allocation spherical Bessel quadrature and direct $\mathbb{C}^{n \times n}$ complex LU cut MMF peak memory by ~10x and achieve 2.0x–4.5x multithreaded speedups over OpenMP `optool` across CDE, DHS, Mie, and MMF on 10 cores.

### Documentation

- Rewrote crate-level and per-module API documentation (`lib`, `utils`, `mie`, `dhs`, `geofractal`, `fractal`, `io`, `opac`, `cli`, `main`) with KaTeX-verified LaTeX equations, verified ADS/DOI literature links, expanded abbreviations, and compiling library quickstart doctests.
