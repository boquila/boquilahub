#![windows_subsystem = "windows"]

#[tokio::main]
async fn main() {
    boquilahub::cli::run_cli().await;
    boquilahub::gui::run_gui();
}
