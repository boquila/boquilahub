use anyhow::Result;
use boquilahub::api::eps::*;
use boquilahub::api::inference::*;

#[tokio::test]
async fn image_inference() -> Result<()> {
    let img = image::open("assets/img.jpg")?.to_rgb8();    
    let model_name = "yolov11n";
    let model_download_link = "https://huggingface.co/boquila/yolov11/resolve/main/yolov11n.bq";
    let filename = format!("{}.bq", model_name);
    
    println!("Testing inference with model: {}...", model_name);
    let path = std::path::Path::new(&filename);

    // Download model
    let bytes = reqwest::get(model_download_link).await?.bytes().await?;
    std::fs::write(&path, bytes)?;

    // Test inference
    set_model(&filename.to_owned(), &LIST_EPS[1], None)?;
    let aioutput = process_imgbuf(&img);
    println!("{:?}",aioutput);

    std::fs::remove_file(&path)?;
    
    Ok(())
}
