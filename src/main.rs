use boquilahub::cli::{run_cli, Cli};
use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = Cli::try_parse();

    match cli {
        Ok(cli) => {
            if cli.command.is_none() {
                // #[cfg(all(windows, not(debug_assertions)))]
                // hide_window();
                boquilahub::gui::run_gui();

                return;
            }

            run_cli(cli.command.expect("Could not run CLI")).await;
        }
        Err(error) => error.exit(),
    }
}

#[cfg(all(windows, not(debug_assertions)))]
fn hide_window() {
    use winapi::um::wincon::FreeConsole;
    unsafe {
        FreeConsole(); // Detaches and closes the console window
    }
}
