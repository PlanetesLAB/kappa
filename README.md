# kappa

Dust opacity and Mueller scattering matrix calculator for astrophysics, implemented in Rust as both a library crate and a standalone command-line tool.

`kappa` computes absorption, scattering, and extinction mass opacities ($\kappa_{\mathrm{abs}}$, $\kappa_{\mathrm{sca}}$, $\kappa_{\mathrm{ext}}$), the scattering asymmetry parameter $g = \langle \cos\theta \rangle$, and the six independent elements of the Mueller scattering matrix ($F_{11}, F_{12}, F_{22}, F_{33}, F_{34}, F_{44}$) using:

- **Distribution of Hollow Spheres (DHS)** for irregular grains and low-porosity aggregates
- **Lorenz-Mie theory** for compact homogeneous spheres
- **Modified Mean Field (MMF) theory** for porous fractal aggregates
- **Continuous Distribution of Ellipsoids (CDE)** in the Rayleigh limit

Multi-component core and mantle mixtures (with optional vacuum porosity) are blended via the Bruggeman and Maxwell Garnett effective medium rules, with built-in presets for the DiscAnalysis (DIANA) and Disk Substructures at High Angular Resolution Project (DSHARP) opacity models and native RADMC-3D output support.

---

## Attribution & Citation

> [!IMPORTANT]
> `kappa` is a derivative Rust implementation of [`optool`](https://github.com/cdominik/optool) by Carsten Dominik, Michiel Min, and Ryo Tazaki. For general scientific use, I recommend `optool`; please use this crate at your own judgement.
>
> If you choose to use `kappa` in scientific work — **even if you do not run `optool` directly** — please cite **Dominik, Min & Tazaki (2021)** along with the relevant scattering theory and dust-model references below.

- **`optool` (required primary reference):** [Dominik, C., Min, M., & Tazaki, R. (2021)](https://ui.adsabs.harvard.edu/abs/2021ascl.soft04010D/abstract), *OpTool: Command-line driven tool for creating complex dust opacities*, Astrophysics Source Code Library, `ascl:2104.010`.
- **DHS:** [Min, Hovenier & de Koter (2005)](https://ui.adsabs.harvard.edu/abs/2005A%26A...432..909M/abstract) (stratified-sphere solver: [Toon & Ackerman 1981](https://ui.adsabs.harvard.edu/abs/1981ApOpt..20.3657T/abstract)).
- **Lorenz-Mie:** [De Rooij & Van der Stap (1984)](https://ui.adsabs.harvard.edu/abs/1984A%26A...131..237D/abstract).
- **MMF:** [Tazaki & Tanaka (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...860...79T/abstract) and [Tazaki (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2811T/abstract).
- **DIANA / DSHARP presets:** [Woitke et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...586A.103W/abstract) and [Birnstiel et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..45B/abstract).

```bibtex
@ARTICLE{2021ascl.soft04010D,
       author = {{Dominik}, Carsten and {Min}, Michiel and {Tazaki}, Ryo},
        title = "{OpTool: Command-line driven tool for creating complex dust opacities}",
     keywords = {Software},
         year = 2021,
        month = apr,
      journal = {Astrophysics Source Code Library},
          eid = {ascl:2104.010},
        pages = {ascl:2104.010},
archivePrefix = {ascl},
       eprint = {2104.010},
          url = {https://ascl.net/2104.010},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ascl.soft04010D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

---

## Usage

### Command-Line Interface

Build with default features (`cli` enabled):

```sh
cargo build --release
kappa --help
kappa materials
kappa --preset dsharp --method mie --lam 1300 --scatter --radmc -o out
```

### Rust Library

Library consumers that do not need the command-line executable can disable the `cli` feature:

```toml
kappa = { git = "https://github.com/PlanetesLAB/kappa", tag = "v1.0.0", default-features = false }
```

---

## Documentation

Detailed API documentation generated from the Rust source:

[![API Docs](https://img.shields.io/badge/docs-kappa-informational?style=for-the-badge&labelColor=20232a&color=4e7ca1)](https://planeteslab.github.io/documentation/kappa/)

---

## Primer

A primer on dust opacities in the interstellar medium written with mdBook, covering the theoretical background and implementation details of `kappa`:

[![Primer](https://img.shields.io/badge/book-mdBook-informational?style=for-the-badge&labelColor=20232a&color=4e7ca1)](https://planeteslab.github.io/tools/software/kappa/)
