#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

#[tokio::main]
async fn main() -> eframe::Result {
    boquilahub::cli::run_cli().await;
    boquilahub::gui::run_gui()
}
