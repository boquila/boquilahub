use anyhow::{ensure, Result};
use boquilahub::api::abstractions::AIOutputs;
use boquilahub::api::bq::{process_imgbuf, Ep, GlobalBQ};

#[test]
#[ignore = "requires the configured GPU and its runtime libraries"]
fn configured_gpu_matches_cpu() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let executable = std::env::current_exe()?;
        if let Some(build) = executable.ancestors().find(|path| path.ends_with("build")) {
            let profile = build.parent().unwrap();
            for library in ["onnxruntime_providers_shared", "onnxruntime_providers_cuda"] {
                let name = format!("{}{library}{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX);
                std::fs::copy(profile.join(&name), executable.parent().unwrap().join(&name))?;
            }
        }
    }
    println!("{}", ort::info());
    let image = image::open("tests/assets/img.jpg")?.to_rgb8();
    let mut results = Vec::new();
    for ep in [Ep::Cpu, Ep::gpu(), Ep::Cpu] {
        GlobalBQ::First.set_model("tests/assets/yolo11n-seg.bq", ep, None)?;
        let AIOutputs::Segmentation(segments) = process_imgbuf(&image)? else {
            anyhow::bail!("expected segmentation output");
        };
        ensure!(!segments.is_empty(), "no detections on {ep:?}");
        results.push(segments);
    }
    for actual in &results[1..] {
        ensure!(actual.len() == results[0].len(), "detection count changed");
        for (expected, actual) in results[0].iter().zip(actual) {
            ensure!(expected.bbox.label == actual.bbox.label, "label changed");
            ensure!(
                (expected.bbox.xyxy.prob - actual.bbox.xyxy.prob).abs() < 0.01,
                "confidence changed"
            );
        }
    }
    GlobalBQ::First.clear();
    Ok(())
}
