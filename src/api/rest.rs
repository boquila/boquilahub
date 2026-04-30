use super::abstractions::AIOutputs;
use super::bq::*;
use axum::{extract::Multipart, http::StatusCode, routing::{get, post}, Router};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Rgb, Rgba};
use reqwest::Client;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::process::Command;

async fn upload(mut multipart: Multipart) -> Result<String, StatusCode> {
    let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
    let imgbuf = image::load_from_memory(&data)
        .map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?
        .into_rgb8();

    let result = process_imgbuf(&imgbuf);
    serde_json::to_string(&result).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn root() -> &'static str {
    "BoquilaHUB Web API!"
}

pub async fn run_api(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app: Router = Router::new()
        .route("/", get(root))
        .route("/upload", post(upload))
        .layer(axum::extract::DefaultBodyLimit::max(10 * 1024 * 1024)); // 10MB limit;

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn detect_remotely(
    url: &str,
    buffer: Vec<u8>,
) -> Result<AIOutputs, Box<dyn std::error::Error>> {
    let client = Client::new();

    let response = client
        .post(url)
        .multipart(reqwest::multipart::Form::new().part(
            "file",
            reqwest::multipart::Part::bytes(buffer).mime_str("image/jpeg")?,
        ))
        .send()
        .await?;

    let response_text = response.text().await?;
    let deserialized: AIOutputs = serde_json::from_str(&response_text)?;

    Ok(deserialized)
}

pub fn rgba_image_to_jpeg_buffer(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, quality: u8) -> Vec<u8> {
    let mut buffer = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
    encoder
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            ColorType::Rgba8.into(),
        )
        .expect("Failed to encode image");
    buffer
}

pub fn rgb_image_to_jpeg_buffer(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, quality: u8) -> Vec<u8> {
    let mut buffer = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
    encoder
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            ColorType::Rgb8.into(),
        )
        .expect("Failed to encode image");
    buffer
}

pub const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn get_ipv4_address() -> Option<String> {
    #[cfg(windows)]
    {
        let output = Command::new("ipconfig")
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .expect("Failed to execute ipconfig");

        // Try UTF-8 first, fall back to lossy conversion if it fails
        let output_str = match std::str::from_utf8(&output.stdout) {
            Ok(s) => s.to_string(),
            Err(_) => String::from_utf8_lossy(&output.stdout).to_string(),
        };

        for line in output_str.lines() {
            if line.contains("IPv4") {
                // Extract the IP address (everything after the last ': ')
                return line.split(": ").last().map(|ip| ip.trim().to_string());
            }
        }
    }

    #[cfg(not(windows))]
    {
        let output = Command::new("ip")
            .args(["route", "get", "1.1.1.1"])
            .output()
            .ok()?
            .stdout;

        let output_str = String::from_utf8_lossy(&output);
        let parts: Vec<&str> = output_str.split_whitespace().collect();
        if let Some(pos) = parts.iter().position(|&p| p == "src") {
            return parts.get(pos + 1).map(|s| s.to_string());
        }
    }

    return None;
}

pub async fn check_boquila_hub_api(url: &str) -> bool {
    let Ok(response) = reqwest::get(url).await else {
        return false;
    };

    let Ok(body) = response.text().await else {
        return false;
    };

    body.trim() == "BoquilaHUB Web API!"
}
