use anyhow::Result;

use kappa::cli::CliAction;
use kappa::cli::launch;
use kappa::opac::KappaError;

fn main() -> Result<(), KappaError> {
    if let CliAction::Run(mut kpc) = launch()? {
        kpc.prepare_inputs()?;
        kpc.initialize()?;
        kpc.run()?;
    }

    Ok(())
}
