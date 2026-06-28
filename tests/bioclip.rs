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

    GlobalBQ::First.set_model(&model_path, Ep::gpu(), None)?;

    let result = (|| -> Result<()> {
        let img = image::open("tests/assets/img.jpg")?.to_rgb8();
        let aioutput = process_imgbuf(&img);
        let AIOutputs::Embed(emb) = &aioutput else {
            panic!("expected AIOutputs::Embed, got {:?}", aioutput);
        };

        let d = emb.d();
        println!(
            "model={}  h={}  w={}  d={}  total={}  first5={:?}",
            emb.model,
            emb.h,
            emb.w,
            d,
            emb.values.len(),
            &emb.values[..emb.values.len().min(5)]
        );

        assert_eq!(emb.model, MODEL_NAME);
        assert!(d > 0);
        assert!(emb.values.iter().all(|v| v.is_finite()));

        // Dense path: every token must be L2-normalised — the GUI dot product
        // assumes it. Pooled (h == w == 1) skips the check.
        if emb.h * emb.w > 1 {
            for t in 0..(emb.h as usize * emb.w as usize) {
                let token = &emb.values[t * d..(t + 1) * d];
                let norm = token.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-3, "token {t}: norm={norm}");
            }
        }

        let sim = emb.cosine(emb);
        assert!((sim - 1.0).abs() < 1e-3, "self-cosine {sim}");

        Ok(())
    })();

    if should_download {
        let _ = std::fs::remove_file(&path);
    }

    result
}
