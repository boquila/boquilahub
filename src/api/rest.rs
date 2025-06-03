use super::abstractions::XYXYc;
use super::inference::*;
use axum::{extract::Multipart, routing::get, routing::post, Router};
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
        let test = detect_bbox_from_imgbuf(&imgbuf);
        serialized = serde_json::to_string(&test).unwrap_or("Error".to_string());
    }
    return serialized;
}

async fn root() -> &'static str {
    "BoquilaHUB Web API!"
}


pub async fn run_api(port: u16) {
    let app: Router = Router::new()
        .route("/", get(root))
        .route("/upload", post(upload));
    
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub fn detect_bbox_from_buf_remotely(url: String, buffer: Vec<u8>) -> Vec<XYXYc> {
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

    let deserialized: Vec<XYXYc> = serde_json::from_str(&response.text().unwrap()).unwrap();
    return deserialized;
}


pub fn detect_bbox_remotely(url: String, file_path: &str) -> Vec<XYXYc> {
    let buf = std::fs::read(file_path).unwrap_or(vec![]);
    return detect_bbox_from_buf_remotely(url, buf);
}

pub const CREATE_NO_WINDOW: u32 = 0x08000000;

fn get_ipv4_address() -> Option<String> {
    let output = Command::new("ipconfig")
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .expect("Failed to execute ipconfig");

    let output_str = str::from_utf8(&output.stdout).expect("Failed to convert output to string");

    for line in output_str.lines() {
        if line.contains("IPv4 Address") {
            // Extract the IP address (everything after the last ': ')
            return line.split(": ").last().map(|ip| ip.trim().to_string());
        }
    }

    None
}

pub fn get_ip() -> String {
    get_ipv4_address().unwrap()
}

pub async fn check_boquila_hub_api(url: &str) -> bool {
    let response = reqwest::get(url).await.unwrap();
    let body = response.text().await.unwrap();
    body.trim() == "BoquilaHUB Web API!"
}
