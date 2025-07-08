#[tokio::main]
async fn main() {
    boquilahub::cli::run_cli().await;
    // #[cfg(all(debug_assertions, target_os = "windows"))]
    // hide_window();
    boquilahub::gui::run_gui();
}

// fn hide_window() {
//     use winapi::um::wincon::FreeConsole;
//     unsafe {
//         FreeConsole(); // Detaches and closes the console window
//     }
// }