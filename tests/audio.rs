use anyhow::Result;
use boquilahub::api::audio::AudioData;
use boquilahub::api::bq::*;

#[test]
fn smoke() -> Result<()> {
    let a = AudioData::from_file("tests/assets/audio.mp3")?;
    let (min, max, rms) = a.amplitude_stats();

    println!(
        "samples={} rate={} channels={} duration={:.3}s",
        a.samples.len(),
        a.sample_rate,
        a.channels,
        a.duration()
    );
    println!("amp min={:.4} max={:.4} rms={:.4}", min, max, rms);
    println!("dc_offset={:.6}", a.dc_offset());
    println!("preview={:?}", a.preview(16));
    Ok(())
}

#[test]
#[ignore]
fn audio_inference() -> Result<()> {
    let audio = AudioData::from_file("tests/assets/bird.mp3")?;
    GlobalBQ::First.set_model("models/MD_AudioBirds_V1.bq", Ep::Cpu, None)?;

    let aioutput = boquilahub::api::bq::process_audio(&audio);
    println!("Inference success",);
    println!("AI Outputs: {:?}", aioutput);
    Ok(())
}
