/// Here we run supported models and check that they always work
/// eg. no index out of bounds, no wrong pre or post-processing.
use anyhow::Result;
use boquilahub::api::ep::Ep;
use boquilahub::api::bq::*;

#[tokio::test]
#[ignore]
async fn test_models() -> Result<()> {
    let listmodels = BQModel::get_list_from_api().await?;
    let img = image::open("assets/test/img.jpg")?.to_rgb8();
    let n = listmodels.len();
    
    for model in listmodels {
        let filename = format!("{}.bq", model.name);
        println!("Testing inference with model: {}...", model.name);
        let path = std::path::Path::new("models").join(&filename);
        let model_path = path.to_string_lossy().into_owned();
        let should_download = !path.exists();

        if should_download {
            let bytes = reqwest::get(&model.download_link).await?.bytes().await?;
            std::fs::write(&path, bytes)?;
        }

        // Test inference
        GlobalBQ::First.set_model(&model_path, Ep::Cpu, None)?;
        process_imgbuf(&img);

        if should_download {
            std::fs::remove_file(&path)?;
        }
    }
    println!("Success on {} models", n);
    Ok(())
}
