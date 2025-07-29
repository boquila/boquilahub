use crate::api::models::processing::inference::AIOutputs;
use super::inference::*;
use axum::{extract::Multipart, routing::get, routing::post, Router};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageBuffer, Rgba};
use reqwest::blocking::Client;
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

pub fn detect_bbox_from_buf_remotely(url: &str, buffer: Vec<u8>) -> AIOutputs {
    let client = Client::new();
    let response = client
        .post(url)
        .multipart(
            reqwest::blocking::multipart::Form::new().part(
                "file",
                reqwest::blocking::multipart::Part::bytes(buffer)
                    .mime_str("image/jpeg")
                    .unwrap(),
            ),
        )
        .send()
        .expect("Failed to send request");

    let deserialized: AIOutputs = serde_json::from_str(&response.text().unwrap()).unwrap();
    return deserialized;
}

pub fn detect_bbox_from_imgbuf_remotely(url: &str, img: &ImageBuffer<Rgba<u8>, Vec<u8>>,) -> AIOutputs {
    let jpeg_buffer = rgba_image_to_jpeg_buffer(img, 95);
    detect_bbox_from_buf_remotely(url, jpeg_buffer)
}

pub fn rgba_image_to_jpeg_buffer(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    quality: u8,
) -> Vec<u8> {
    let dynamic_image = DynamicImage::ImageRgba8(img.clone());
    let mut buffer = Vec::new();
    {
        let mut encoder = JpegEncoder::new_with_quality(&mut buffer, quality);
        encoder.encode_image(&dynamic_image).expect("Failed to encode image");
    }
    buffer
}

pub fn detect_bbox_remotely(url: &str, file_path: &str) -> AIOutputs {
    let buf = std::fs::read(file_path).unwrap_or(vec![]);
    return detect_bbox_from_buf_remotely(url, buf);
}

pub const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn get_ipv4_address() -> Option<String> {
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

    None
}

pub async fn check_boquila_hub_api(url: &str) -> bool {
    let response = reqwest::get(url).await.unwrap();
    let body = response.text().await.unwrap();
    body.trim() == "BoquilaHUB Web API!"
}
