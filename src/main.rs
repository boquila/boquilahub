#[tokio::main]
async fn main() {
    boquilahub::cli::run_cli().await;
    #[cfg(all(windows, not(debug_assertions)))]
    hide_window();
    boquilahub::gui::run_gui();
}

#[cfg(all(windows, not(debug_assertions)))]
fn hide_window() {
    use winapi::um::wincon::FreeConsole;
    unsafe {
        FreeConsole(); // Detaches and closes the console window
    }
}