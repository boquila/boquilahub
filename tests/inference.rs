use anyhow::Result;
use boquilahub::api::bq::*;

#[tokio::test]
async fn image_inference() -> Result<()> {
    let img = image::open("tests/assets/img.jpg")?.to_rgb8();
    let model_path = "tests/assets/yolo11n-seg.bq";

    println!("Testing single image inference");

    // Test inference
    GlobalBQ::First.set_model(&model_path, Ep::Cpu, None)?;
    let aioutput = process_imgbuf(&img)?;
    println!("{:?}",aioutput);

    Ok(())
}
