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
            "model={}  h={}  w={}  d={}  total={}  first5={:?}",
            emb.model,
            emb.h,
            emb.w,
            emb.d,
            emb.values.len(),
            &emb.values[..emb.values.len().min(5)]
        );

        assert_eq!(emb.model, MODEL_NAME, "embedding model tag mismatch");
        assert!(emb.d > 0, "embedding dim must be > 0");
        let expected_len = (emb.h as usize) * (emb.w as usize) * (emb.d as usize);
        assert_eq!(
            emb.values.len(),
            expected_len,
            "values len ({}) doesn't match h*w*d ({})",
            emb.values.len(),
            expected_len
        );
        assert!(
            emb.values.iter().all(|v| v.is_finite()),
            "embedding contains NaN/inf"
        );

        // Dense path: every token must be L2-normalised, since the GUI heatmap
        // treats per-token dot products as cosine similarities. Pooled path
        // (h == w == 1) skips this — we only normalise tokens for the dense case.
        if emb.h * emb.w > 1 {
            let d = emb.d as usize;
            for t in 0..(emb.h as usize * emb.w as usize) {
                let token = &emb.values[t * d..(t + 1) * d];
                let norm = token.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-3,
                    "token {t} not L2-normalised: norm={norm}"
                );
            }
        }

        // Self-cosine of the flattened vector is always 1.0 for a non-zero
        // vector — cheap sanity check that the cosine helper works.
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
