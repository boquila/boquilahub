#[tokio::main]
async fn main() {
    boquilahub::cli::run_cli().await;
    // hide_window();
    boquilahub::gui::run_gui();
}

#[cfg(target_os = "windows")]
fn hide_window() {
    use winapi::um::wincon::FreeConsole;
    unsafe {
        FreeConsole(); // Detaches and closes the console window
    }
}