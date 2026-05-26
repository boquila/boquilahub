use anyhow::Result;
use boquilahub::api::abstractions::{AIOutputs, AudioProbSugar};
use boquilahub::api::audio::AudioData;
use boquilahub::api::bq::*;

#[test]
fn perch_identifies_species() -> Result<()> {
    GlobalBQ::First.set_model("models/perch-v2.bq", Ep::Cpu, None)?;

    // Each fixture is named after the species we expect to find. Field
    // recordings have background species, so we only assert the target
    // appears as top-1 in *some* window — not every window.
    let cases = [
        ("tests/assets/perch/Cuculus canorus.mp3", "Cuculus canorus"),
        ("tests/assets/perch/Grus grus.mp3", "Grus grus"),
        ("tests/assets/perch/Parus major.wav", "Parus major"),
    ];

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
}
