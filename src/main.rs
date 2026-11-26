//! Standalone command-line executable for `kappa` (built when the default
//! `cli` feature is enabled).

use std::process::ExitCode;

use kappa::cli::CliAction;
use kappa::cli::launch;

/// Entry point for the `kappa` binary; prints any [`kappa::opac::KappaError`]
/// cleanly to `stderr` and returns a non-zero exit code on failure.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Parses command-line arguments via [`launch`] and runs the requested
/// opacity calculation.
fn run() -> Result<(), kappa::opac::KappaError> {
    if let CliAction::Run(config) = launch()? {
        config.run()?;
    }
    Ok(())
}
