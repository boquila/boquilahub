use anyhow::Result;
use boquilahub::api::abstractions::{sidecar_predictions_path, AIOutputs, XYXYc};
use boquilahub::api::audio::AudioData;
use boquilahub::api::bq::*;

fn boxes(output: AIOutputs) -> Vec<XYXYc> {
    let AIOutputs::ObjectDetection(boxes) = output else {
        unreachable!("batdetect2 is configured to emit ObjectDetection")
    };
    boxes
}

#[test]
fn batdetect2_golden() -> Result<()> {
    let close = |a: f32, b: f32, tol: f32| (a - b).abs() < tol;

    for ep in [Ep::Cpu, Ep::gpu()] {
        GlobalBQ::First.set_model("tests/assets/batdetect2-uk.bq", ep, None)?;

        for entry in std::fs::read_dir("tests/assets/bats")? {
            let wav = entry?.path();
            if wav.extension().is_none_or(|ext| ext != "wav") {
                continue;
            }

            let audio = AudioData::from_file(&wav)?;
            let actual = boxes(process_audio(&audio)?);
            let sidecar = std::fs::read(sidecar_predictions_path(&wav)?)?;
            let expected: Vec<_> = boxes(serde_json::from_slice(&sidecar)?);

            assert_eq!(actual.len(), expected.len(), "{ep:?} {wav:?}: count");
            for (a, g) in actual.iter().zip(&expected) {
                let ctx = format!("{ep:?} {wav:?} {}", g.label);
                assert_eq!(a.label, g.label, "{ctx}");
                assert!(close(a.xyxy.prob, g.xyxy.prob, 1e-3), "{ctx}: score");
                assert!(close(a.xyxy.x1, g.xyxy.x1, 1e-3), "{ctx}: start");
                assert!(close(a.xyxy.x2, g.xyxy.x2, 1e-3), "{ctx}: end");
                // Frequencies are in Hz, so 1.0 is tighter than 1e-3 seconds.
                assert!(close(a.xyxy.y1, g.xyxy.y1, 1.0), "{ctx}: low_freq");
                assert!(close(a.xyxy.y2, g.xyxy.y2, 1.0), "{ctx}: high_freq");
            }
        }
    }
    Ok(())
}
