use anyhow::{anyhow, Result};
use boquilahub::api::abstractions::{AIOutputs, AudioProbSugar};
use boquilahub::api::audio::AudioData;
use boquilahub::api::bq::*;

#[tokio::test]
#[ignore]
async fn perch_identifies_species() -> Result<()> {
    const MODEL_NAME: &str = "perch-v2";

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

    GlobalBQ::First.set_model(&model_path, Ep::Cpu, None)?;

    // Each fixture is named after the species we expect to find. Field
    // recordings have background species, so we only assert the target
    // appears as top-1 in *some* window — not every window.
    let cases = [
        ("tests/assets/perch/Cuculus canorus.mp3", "Cuculus canorus"),
        ("tests/assets/perch/Grus grus.mp3", "Grus grus"),
        ("tests/assets/perch/Parus major.wav", "Parus major"),
    ];

    let result = (|| -> Result<()> {
        for (path, expected) in cases {
            let audio = AudioData::from_file(path)?;
            let aioutput = process_audio(&audio);
            let AIOutputs::AudioClassification(probs) = &aioutput else {
                panic!("{path}: expected AudioClassification, got {:?}", aioutput);
            };
            assert!(!probs.is_empty(), "{path}: zero windows");

            println!(
                "\n{path}  windows={}  top={}",
                probs.len(),
                probs.highest_confidence()
            );
            for (i, p) in probs.iter().enumerate() {
                println!(
                    "  [{i}] {:>5.2}-{:>5.2}s  {:<40} p={:.3}",
                    p.start, p.end, p.prediction.label, p.prediction.prob
                );
            }

            assert!(
                probs.iter().any(|p| p.prediction.label == expected),
                "{path}: expected at least one window to predict '{expected}'"
            );
        }
        Ok(())
    })();

    if should_download {
        let _ = std::fs::remove_file(&path);
    }

    result
}
