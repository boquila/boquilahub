use boquilahub::cli::Cli;
use clap::Parser;

#[tokio::main(worker_threads = 4)]
async fn main() {
    let cli = Cli::try_parse();

    match cli {
        Ok(cli) => {
            if cli.command.is_none() {
                #[cfg(all(windows, not(debug_assertions)))]
                winapi::um::wincon::FreeConsole();
                boquilahub::gui::run_gui();

                return;
            }

            cli.run().await;
        }
        Err(error) => error.exit(),
    }
}