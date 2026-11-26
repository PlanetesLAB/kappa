use std::process::ExitCode;

use kappa::cli::CliAction;
use kappa::cli::launch;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), kappa::opac::KappaError> {
    if let CliAction::Run(config) = launch()? {
        config.run()?;
    }
    Ok(())
}
