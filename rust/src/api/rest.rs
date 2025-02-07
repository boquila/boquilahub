use crate::api::abstractions::BBox;
use super::inference::*;
use axum::{extract::Multipart, routing::get, routing::post, Router};
use reqwest::blocking::Client;
use std::process::Command;
use std::str;

async fn upload(mut multipart: Multipart) -> String {
    let mut serialized: String = String::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        // let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        let test = detect_bbox_from_buf(data.to_vec());
        serialized = serde_json::to_string(&test).unwrap_or("Error".to_string());
    }
    return serialized;
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "BoquilaHUB Web API!"
}

#[flutter_rust_bridge::frb(dart_async)]
#[tokio::main]
pub async fn run_api() {
    let app: Router = Router::new()
        .route("/", get(root))
        .route("/upload", post(upload));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8791").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn detect_bbox_from_buf_remotely(url: String, buffer: Vec<u8>)  -> Vec<BBox>{
    let client = Client::new();
    let response = client
        .post(url)
        .multipart(reqwest::blocking::multipart::Form::new()
            .part("file", reqwest::blocking::multipart::Part::bytes(buffer)
                .mime_str("image/jpeg").unwrap())
        )
        .send()
        .expect("Failed to send request");

    let deserialized: Vec<BBox>  = serde_json::from_str(&response.text().unwrap()).unwrap();    
    return deserialized;
}

#[flutter_rust_bridge::frb(dart_async)]
pub fn detect_bbox_remotely(url: String, file_path: &str)  -> Vec<BBox>{
    let buf = std::fs::read(file_path).unwrap_or(vec![]);
    return detect_bbox_from_buf_remotely(url, buf);
}

fn get_ipv4_address() -> Option<String> {
    // Run the ipconfig command
    let output = Command::new("ipconfig")
        .output()
        .expect("Failed to execute ipconfig");
    
    // Convert output to a string
    let output_str = str::from_utf8(&output.stdout)
        .expect("Failed to convert output to string");
    
    // Split the output into lines and find the IPv4 address
    for line in output_str.lines() {
        if line.contains("IPv4 Address") {
            // Extract the IP address (everything after the last ': ')
            return line.split(": ").last()
                .map(|ip| ip.trim().to_string());
        }
    }

    None
}

#[flutter_rust_bridge::frb(sync)]
pub fn get_ip() -> String {
    get_ipv4_address().unwrap()
}

pub async fn check_boquila_hub_api(url: &str) -> bool {
    let response = reqwest::get(url).await.unwrap();
    let body = response.text().await.unwrap();
    body.trim() == "BoquilaHUB Web API!"
}