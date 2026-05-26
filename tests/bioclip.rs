use anyhow::{anyhow, Result};
use boquilahub::api::abstractions::AIOutputs;
use boquilahub::api::bq::*;

#[tokio::test]
#[ignore]
async fn bioclip_produces_image_embedding() -> Result<()> {
    const MODEL_NAME: &str = "bioclip2";

    let path = std::path::Path::new("models").join(format!("{MODEL_NAME}.bq"));
    let model_path = path.to_string_lossy().into_owned();
    let should_download = !path.exists();

    if should_download {
        let listmodels = BQModel::get_list_from_api().await?;
        let model = listmodels
            .into_iter()
            .find(|m| m.name == MODEL_NAME)
            .ok_or_else(|| anyhow!("{MODEL_NAME} not listed in the API"))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = reqwest::get(&model.download_link).await?.bytes().await?;
        std::fs::write(&path, bytes)?;
    }

    // CUDA EP: the full bioclip vision tower won't fit in RAM on CI; CPU
    // execution OOMs. Requires CUDA + cuDNN locally.
    GlobalBQ::First.set_model(&model_path, Ep::Cuda, None)?;

    let result = (|| -> Result<()> {
        let img = image::open("tests/assets/img.jpg")?.to_rgb8();
        let aioutput = process_imgbuf(&img);
        let AIOutputs::Embed(emb) = &aioutput else {
            panic!("expected AIOutputs::Embed, got {:?}", aioutput);
        };

        println!(
            "model={}  dims={}  first5={:?}",
            emb.model,
            emb.values.len(),
            &emb.values[..emb.values.len().min(5)]
        );

        assert_eq!(emb.model, MODEL_NAME, "embedding model tag mismatch");
        assert_eq!(emb.values.len(), 768, "expected 768-dim CLIP embedding");
        assert!(
            emb.values.iter().any(|v| v.abs() > 1e-6),
            "embedding is all-zeros"
        );
        assert!(
            emb.values.iter().all(|v| v.is_finite()),
            "embedding contains NaN/inf"
        );

        // Self-cosine must be ~1.0 — a basic sanity check that values are
        // well-conditioned for downstream similarity ranking.
        let sim = emb.cosine(emb);
        assert!(
            (sim - 1.0).abs() < 1e-3,
            "self-cosine should be ~1.0, got {sim}"
        );

        Ok(())
    })();

    if should_download {
        let _ = std::fs::remove_file(&path);
    }

    result
}
