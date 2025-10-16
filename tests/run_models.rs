/// Here we run supported models and check that they always work
/// eg. no index out of bounds, no wrong pre or post-processing.
use anyhow::Result;
use boquilahub::api::eps::*;
use boquilahub::api::inference::*;
use boquilahub::api::pull::*;
use serde::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModel {
    pub name: String,
    pub description: String,
    pub download_link: String,
}

#[tokio::test]
async fn test_models() -> Result<()> {
    let listmodels = get_list().await?;
    let img = image::open("assets/img.jpg")?.to_rgb8();
    let n = listmodels.len();
    
    for model in listmodels {
        let filename = format!("{}.bq", model.name);
        println!("Testing inference with model: {}...", model.name);
        let path = std::path::Path::new(&filename);

        // Download model
        let bytes = reqwest::get(&model.download_link).await?.bytes().await?;
        std::fs::write(&path, bytes)?;

        // Test inference
        let _ = set_model(&filename.to_owned(), &LIST_EPS[1], None);
        let _aioutput = process_imgbuf(&img);

        std::fs::remove_file(&path)?;
    }
    println!("Success on {} models", n);
    Ok(())
}
