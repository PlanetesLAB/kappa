//! File readers for optical constants (`.lnk`), size distributions, and wavelength grids.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use anyhow::Result;
use anyhow::anyhow;
use planetes_ext::types::RVector;

use crate::opac::Component;

fn is_comment_or_empty(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with('!')
        || trimmed.starts_with('*')
}

/// Reads a `.lnk` optical constants file and constructs a [`Component`].
///
/// # Errors
/// Returns an error if the file cannot be opened or read, if the header or density is missing or
/// invalid, or if any data row is missing or malformed.
pub fn read_lnk_file(file: &str, rho_in: Option<f64>) -> Result<Component> {
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;
    let mut lines_iter = all_lines.iter();

    let mut name = String::new();
    let mut class = String::new();
    let mut state = String::new();

    let header: String = lines_iter
        .find_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                if let Some((key, value)) = trimmed.trim_start_matches('#').trim().split_once(':') {
                    match key.trim() {
                        "Name" => name = value.trim().into(),
                        "Class" => class = value.trim().into(),
                        "State" => state = value.trim().into(),
                        _ => {}
                    }
                }
                None
            } else if is_comment_or_empty(trimmed) {
                None
            } else {
                Some(trimmed.into())
            }
        })
        .ok_or_else(|| anyhow!("Missing header in .lnk file"))?;

    // Header: nlam [rho]
    let mut header_parts = header.split_whitespace();
    let size: usize = header_parts
        .next()
        .ok_or_else(|| anyhow!("Missing wavelength count in .lnk header"))?
        .parse()
        .map_err(|_| anyhow!("Invalid wavelength count in .lnk header"))?;
    if size == 0 {
        return Err(anyhow!("Wavelength count in .lnk header must be positive"));
    }

    let rho: f64 = header_parts
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .or(rho_in)
        .ok_or_else(|| anyhow!("Missing material density `rho` in .lnk header"))?;
    if rho <= 0.0 || !rho.is_finite() {
        return Err(anyhow!("Material density `rho` must be finite and positive"));
    }

    let mut l0 = Vec::with_capacity(size);
    let mut n0 = Vec::with_capacity(size);
    let mut k0 = Vec::with_capacity(size);

    for line in lines_iter {
        if is_comment_or_empty(line) {
            continue;
        }
        let mut parts = line.split_whitespace();
        let l_val: f64 = parts
            .next()
            .ok_or_else(|| anyhow!("Missing wavelength column in .lnk file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid wavelength value in .lnk file"))?;
        let n_val: f64 = parts
            .next()
            .ok_or_else(|| anyhow!("Missing refractive index `n` column in .lnk file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid refractive index `n` in .lnk file"))?;
        let k_val: f64 = parts
            .next()
            .ok_or_else(|| anyhow!("Missing extinction coefficient `k` column in .lnk file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid extinction coefficient `k` in .lnk file"))?;
        if l_val <= 0.0 || !l_val.is_finite() || !n_val.is_finite() || !k_val.is_finite() {
            return Err(anyhow!("Non-finite or non-positive wavelength in .lnk file"));
        }
        l0.push(l_val);
        n0.push(n_val);
        k0.push(k_val);
        if l0.len() == size {
            break;
        }
    }

    if l0.len() < size {
        return Err(anyhow!(
            "Too few data rows in .lnk file: expected {size}, found {}",
            l0.len()
        ));
    }

    // Reverse if wavelengths are listed in descending order
    if l0[l0.len() - 1] < l0[0] {
        l0.reverse();
        n0.reverse();
        k0.reverse();
    }

    // Clamp non-positive n, k to 1e-10 for log-log interpolation
    for (n, k) in n0.iter_mut().zip(k0.iter_mut()) {
        if *n <= 0.0 {
            *n = 1e-10;
        }
        if *k <= 0.0 {
            *k = 1e-10;
        }
    }

    Ok(Component {
        name,
        class,
        state,
        rho,
        size,
        l0: RVector::from_vec(l0),
        n0: RVector::from_vec(n0),
        k0: RVector::from_vec(k0),
    })
}

/// Reads a tabulated grain size distribution file and returns `(radii, weights, moments)`.
///
/// # Errors
/// Returns an error if the file cannot be opened or read, if the header count or any data row is
/// missing or invalid, or if the total number weight is non-positive.
pub fn read_sizedis_file(file: &str) -> Result<(RVector, RVector, [f64; 3])> {
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;
    let mut lines_iter = all_lines.iter();

    let header = lines_iter
        .find(|line| !is_comment_or_empty(line))
        .ok_or_else(|| anyhow!("Missing header in size distribution file"))?;
    let na: usize = header
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("Missing bin count in size distribution file"))?
        .parse()
        .map_err(|_| anyhow!("Invalid bin count in size distribution file"))?;
    if na == 0 {
        return Err(anyhow!("Size distribution bin count must be positive"));
    }

    let mut r_vec = Vec::with_capacity(na);
    let mut nr_vec = Vec::with_capacity(na);
    let mut tot = [0.0; 3];
    let mut totn = 0.0;

    for line in lines_iter {
        if is_comment_or_empty(line) {
            continue;
        }
        let mut parts = line.split_whitespace();
        let r: f64 = parts
            .next()
            .ok_or_else(|| anyhow!("Missing radius column in size distribution file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid radius value in size distribution file"))?;
        let nr: f64 = parts
            .next()
            .ok_or_else(|| anyhow!("Missing number weight column in size distribution file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid number weight in size distribution file"))?;
        if r <= 0.0 || !r.is_finite() || nr < 0.0 || !nr.is_finite() {
            return Err(anyhow!(
                "Invalid size distribution entry: radius must be positive and weight non-negative"
            ));
        }
        r_vec.push(r);
        nr_vec.push(nr);
        totn += nr;
        tot[0] += nr * r;
        tot[1] += nr * r.powi(2);
        tot[2] += nr * r.powi(3);
        if r_vec.len() == na {
            break;
        }
    }

    if r_vec.len() < na {
        return Err(anyhow!(
            "Too few data rows in size distribution file: expected {na}, found {}",
            r_vec.len()
        ));
    }
    if totn <= 0.0 {
        return Err(anyhow!("Size distribution has zero total number weight"));
    }

    let moments = [
        tot[0] / totn,
        (tot[1] / totn).sqrt(),
        (tot[2] / totn).powf(1.0 / 3.0),
    ];

    Ok((RVector::from_vec(r_vec), RVector::from_vec(nr_vec), moments))
}

/// Reads a wavelength grid from a single-column or multi-column (e.g. `.lnk`) file.
///
/// # Errors
/// Returns an error if the file cannot be opened or read, if the header count or any wavelength
/// entry is missing, non-positive, or malformed, or if there are fewer rows than declared.
pub fn read_wavelength_grid(file: &str) -> Result<(usize, RVector)> {
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;
    let mut lines_iter = all_lines.iter();

    let header = lines_iter
        .find(|line| !is_comment_or_empty(line))
        .ok_or_else(|| anyhow!("Missing header in wavelength grid file"))?;
    let nlam: usize = header
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("Missing wavelength count in wavelength grid file"))?
        .parse()
        .map_err(|_| anyhow!("Invalid wavelength count in wavelength grid file"))?;
    if nlam == 0 {
        return Err(anyhow!("Wavelength count must be positive"));
    }

    let mut lam = Vec::with_capacity(nlam);
    for line in lines_iter {
        if is_comment_or_empty(line) {
            continue;
        }
        let val: f64 = line
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("Missing wavelength value in wavelength grid file"))?
            .parse()
            .map_err(|_| anyhow!("Invalid wavelength value in wavelength grid file"))?;
        if val <= 0.0 || !val.is_finite() {
            return Err(anyhow!(
                "Wavelength values must be finite and positive, got {val}"
            ));
        }
        lam.push(val);
        if lam.len() == nlam {
            break;
        }
    }

    if lam.len() < nlam {
        return Err(anyhow!(
            "Too few wavelength rows: expected {nlam}, found {}",
            lam.len()
        ));
    }

    Ok((nlam, RVector::from_vec(lam)))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use approx::assert_relative_eq;

    use super::{read_lnk_file, read_sizedis_file, read_wavelength_grid};

    fn temp_path(label: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/kappa_io_test_{label}_{nanos}.dat")
    }

    #[test]
    fn read_lnk_handles_comments_reversal_and_errors() {
        let path = temp_path("lnk");
        fs::write(
            &path,
            "# Name: TestMat\n# Class: Silicate\n# State: amorphous\n\n! comment\n3 2.5\n10.0 1.8 0.0\n* mid comment\n1.0 1.5 0.01\n0.1 1.4 -0.5\n",
        )
        .unwrap();

        let comp = read_lnk_file(&path, None).unwrap();
        assert_eq!(comp.name, "TestMat");
        assert_eq!(comp.class, "Silicate");
        assert_eq!(comp.state, "amorphous");
        assert_eq!(comp.size, 3);
        assert_relative_eq!(comp.rho, 2.5);
        assert_relative_eq!(comp.l0[0], 0.1);
        assert_relative_eq!(comp.l0[2], 10.0);
        // Clamped non-positive k values to 1e-10
        assert_relative_eq!(comp.k0[0], 1e-10);
        assert_relative_eq!(comp.k0[2], 1e-10);

        // Missing rho without fallback returns Err instead of panicking
        fs::write(&path, "2\n0.1 1.5 0.01\n1.0 1.6 0.02\n").unwrap();
        assert!(read_lnk_file(&path, None).is_err());
        let comp_fb = read_lnk_file(&path, Some(3.0)).unwrap();
        assert_relative_eq!(comp_fb.rho, 3.0);

        // Truncated file returns Err
        fs::write(&path, "5 2.0\n0.1 1.5 0.01\n").unwrap();
        assert!(read_lnk_file(&path, None).is_err());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_sizedis_returns_radii_weights_and_moments() {
        let path = temp_path("sd");
        fs::write(&path, "# comment\n! comment\n\n2\n1.0 3.0\n* mid\n3.0 1.0\n").unwrap();

        let (r, nr, moments) = read_sizedis_file(&path).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(nr.len(), 2);
        assert_relative_eq!(r[0], 1.0);
        assert_relative_eq!(r[1], 3.0);
        assert_relative_eq!(moments[0], 1.5, max_relative = 1e-12);
        assert_relative_eq!(moments[1], 3.0_f64.sqrt(), max_relative = 1e-12);
        assert_relative_eq!(moments[2], 7.5_f64.powf(1.0 / 3.0), max_relative = 1e-12);

        // Invalid header returns Err instead of panicking
        fs::write(&path, "not_a_number\n").unwrap();
        assert!(read_sizedis_file(&path).is_err());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_wavelength_grid_handles_comments_and_multicolumn() {
        let path = temp_path("lam");
        // Multi-column .lnk-like file with blank lines and !/*/# comments
        fs::write(
            &path,
            "\n! leading comment\n* another\n3 2.71\n# inline\n0.1 1.5 0.01\n1.0 1.6 0.02\n10.0 1.7 0.03\n",
        )
        .unwrap();

        let (nlam, lam) = read_wavelength_grid(&path).unwrap();
        assert_eq!(nlam, 3);
        assert_eq!(lam.len(), 3);
        assert_relative_eq!(lam[0], 0.1);
        assert_relative_eq!(lam[1], 1.0);
        assert_relative_eq!(lam[2], 10.0);

        // Non-positive wavelength returns Err
        fs::write(&path, "1\n-1.0\n").unwrap();
        assert!(read_wavelength_grid(&path).is_err());

        let _ = fs::remove_file(&path);
    }
}
