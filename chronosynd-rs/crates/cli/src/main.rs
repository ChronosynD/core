//! `chronosynd`, the operator CLI binary, thin shell around the library
//! entry point so unit tests can drive the CLI without spawning a subprocess

fn main() {
    let exit = chronosynd_cli::run(std::env::args_os(), std::io::stdout().lock());
    std::process::exit(exit);
}
