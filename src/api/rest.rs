use super::inference::*;
use crate::api::abstractions::AIOutputs;
use axum::{extract::Multipart, routing::get, routing::post, Router};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Rgb, Rgba};
use reqwest::Client;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::process::Command;

use std::str;

async fn upload(mut multipart: Multipart) -> String {
    let mut serialized: String = String::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        // let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        let imgbuf = image::load_from_memory(&data.to_vec()).unwrap().into_rgb8();

        let test = process_imgbuf(&imgbuf);
        serialized = serde_json::to_string(&test).unwrap_or("Error".to_string());
    }
    return serialized;
}

async fn root() -> &'static str {
    "BoquilaHUB Web API!"
}

pub async fn run_api(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app: Router = Router::new()
        .route("/", get(root))
        .route("/upload", post(upload));

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn detect_remotely(url: &str, buffer: Vec<u8>) -> Result<AIOutputs, Box<dyn std::error::Error>> {
    let client = Client::new();

    let response = client
        .post(url)
        .multipart(
            reqwest::multipart::Form::new().part(
                "file",
                reqwest::multipart::Part::bytes(buffer)
                    .mime_str("image/jpeg")?,
            ),
        )
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
            ColorType::Rgba8.into()
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
            ColorType::Rgb8.into()
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
        let output_str = match str::from_utf8(&output.stdout) {
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
            .args(["addr", "show"])
            .output()
            .ok()?
            .stdout;

        let output_str = String::from_utf8_lossy(&output);

        for line in output_str.lines() {
            if line.trim().starts_with("inet ") {
                let ip = line.split('/').next()?.split_ascii_whitespace().last()?;

                if !ip.starts_with("127.") {
                    return Some(ip.to_string());
                }
            }
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
