use boquilahub::api::audio::AudioData;

#[test]
fn smoke() {
    let a = AudioData::from_file("assets/test/audio.mp3").unwrap();
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
}

#[test]
#[ignore]
fn audio_inference() {
    let audio = AudioData::from_file("assets/test/bird.mp3").unwrap();
    let _ = boquilahub::api::bq::GlobalBQ::First.set_model(
        "models/MD_AudioBirds_V1.bq",
        boquilahub::api::ep::Ep::Cpu,
        None,
    );
    
    let aioutput = boquilahub::api::bq::process_audio(&audio);
    println!("Inference success",);
    println!("AI Outputs: {:?}",aioutput);
}
