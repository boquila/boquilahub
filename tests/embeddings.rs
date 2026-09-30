use anyhow::{anyhow, Result};
use boquilahub::api::abstractions::AIOutputs;
use boquilahub::api::bq::*;

async fn assert_image_embedding(model_name: &str) -> Result<()> {
    let path = std::path::Path::new("models").join(format!("{model_name}.bq"));
    let model_path = path.to_string_lossy().into_owned();
    let should_download = !path.exists();

    if should_download {
        let listmodels = BQModel::get_list_from_api().await?;
        let model = listmodels
            .into_iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| anyhow!("{model_name} not listed in the API"))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = reqwest::get(&model.download_link).await?.bytes().await?;
        std::fs::write(&path, bytes)?;
    }

    GlobalBQ::First.set_model(&model_path, Ep::gpu(), None)?;

    let result = (|| -> Result<()> {
        let img = image::open("tests/assets/img.jpg")?.to_rgb8();
        let aioutput = process_imgbuf(&img)?;
        let AIOutputs::Embed(emb) = &aioutput else {
            panic!("expected AIOutputs::Embed, got {:?}", aioutput);
        };

        println!(
            "model={}  total={}  first5={:?}",
            emb.model,
            emb.values.len(),
            &emb.values[..emb.values.len().min(5)]
        );

        assert_eq!(emb.model, model_name);
        assert!(!emb.values.is_empty());
        assert!(emb.values.iter().all(|v| v.is_finite()));

        let norm = emb.values.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "embedding not L2-normalised: norm={norm}");

        let sim = emb.cosine(emb);
        assert!((sim - 1.0).abs() < 1e-3, "self-cosine {sim}");

        Ok(())
    })();

    if should_download {
        let _ = std::fs::remove_file(&path);
    }

    result
}

#[tokio::test]
#[ignore]
async fn image_encoders_produce_embeddings() -> Result<()> {
    assert_image_embedding("bioclip2").await?;
    assert_image_embedding("dinov3-vitl16").await?;
    Ok(())
}

#[test]
#[ignore = "requires local models/dinov3-vits16.bq"]
fn segment_crops_accept_an_embedding_model_in_either_auxiliary_slot() -> Result<()> {
    let image = image::open("tests/assets/img.jpg")?.to_rgb8();
    GlobalBQ::First.set_model("tests/assets/yolo11n-seg.bq", Ep::Cpu, None)?;
    for slot in [GlobalBQ::Second, GlobalBQ::Third] {
        slot.set_model("models/dinov3-vits16.bq", Ep::Cpu, None)?;
        let output = process_imgbuf(&image)?;
        let AIOutputs::Segmentation(segments) = output else {
            panic!("expected segmentation output");
        };
        assert!(!segments.is_empty());
        assert!(segments.iter().all(|segment| {
            segment.bbox.embedding.as_ref().is_some_and(|embedding| {
                embedding.model == "dinov3-vits16"
                    && !embedding.values.is_empty()
                    && embedding.values.iter().all(|value| value.is_finite())
            })
        }));
        slot.clear();
    }
    GlobalBQ::Second.set_model("models/SpeciesNetv4.0.0a.bq", Ep::Cpu, None)?;
    GlobalBQ::Third.set_model("models/dinov3-vits16.bq", Ep::Cpu, None)?;
    let output = process_imgbuf(&image)?;
    let AIOutputs::Segmentation(segments) = output else {
        panic!("expected segmentation output");
    };
    assert!(segments.iter().all(|segment| {
        segment.bbox.extra_cls.as_ref().is_some_and(|classes| !classes.is_empty())
            && segment.bbox.embedding.is_some()
    }));
    GlobalBQ::Second.clear();
    GlobalBQ::Third.clear();
    GlobalBQ::First.clear();
    Ok(())
}
