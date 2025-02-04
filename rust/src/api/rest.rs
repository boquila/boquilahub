use super::inference::*;
use axum::{extract::Multipart, routing::get, routing::post, Router};

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