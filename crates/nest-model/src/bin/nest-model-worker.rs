//! A single offline operation under the manager's externally enforced limits.
use std::io::Write;

fn main() {
    if std::env::args_os().len() != 1 {
        eprintln!("usage: nest-model-worker < request.json > response.json");
        std::process::exit(2);
    }
    let result = nest_model::worker::Request::read(std::io::stdin().lock())
        .and_then(nest_model::worker::Request::run)
        .and_then(|response| {
            serde_json::to_writer(std::io::stdout().lock(), &response)?;
            std::io::stdout().flush()?;
            Ok(())
        });
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
