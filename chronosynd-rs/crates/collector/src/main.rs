//! Collector daemon binary, thin shell around `chronosynd_collector::run_daemon`

fn main() {
    let exit = chronosynd_collector::run_daemon(
        std::env::args_os(),
        std::io::stdout().lock(),
    );
    std::process::exit(exit);
}
