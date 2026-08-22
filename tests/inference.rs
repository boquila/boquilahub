use anyhow::Result;
use boquilahub::api::abstractions::AIOutputs;
use boquilahub::api::bq::*;

#[derive(Debug)]
struct SegSummary {
    class_id: u32,
    label: String,
    prob: f32,
    bbox: [f32; 4],
    mask_w: usize,
    mask_h: usize,
    ones: usize,
}

fn run_seg(img: &image::RgbImage) -> Vec<SegSummary> {
    let aioutput = process_imgbuf(img).expect("inference failed");
    let AIOutputs::Segmentation(segs) = &aioutput else {
        panic!("expected AIOutputs::Segmentation, got {aioutput:?}");
    };
    segs.iter()
        .map(|s| SegSummary {
            class_id: s.bbox.xyxy.class_id,
            label: s.bbox.label.clone(),
            prob: s.bbox.xyxy.prob,
            bbox: [
                s.bbox.xyxy.x1,
                s.bbox.xyxy.y1,
                s.bbox.xyxy.x2,
                s.bbox.xyxy.y2,
            ],
            mask_w: s.mask.width,
            mask_h: s.mask.height,
            ones: s.mask.data.iter().filter(|b| **b).count(),
        })
        .collect()
}

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

/// Regression test for the segmentation post-processing (mask coefficients
/// matmul + reshape) in `Yolo::process_seg_output`. Golden values were
/// captured with the reference implementation; tolerances are small enough
/// to catch a wrong reshape/transpose while absorbing minor float
/// differences across CPU features.
#[tokio::test]
async fn yolo11n_seg_golden() -> Result<()> {
    let img = image::open("tests/assets/img.jpg")?.to_rgb8();
    GlobalBQ::First.set_model("tests/assets/yolo11n-seg.bq", Ep::Cpu, None)?;
    let segs = run_seg(&img);
    for s in &segs {
        println!("{s:?}");
    }

    let expected = [SegSummary {
        class_id: 21,
        label: "bear".to_string(),
        prob: 0.7373,
        bbox: [389.36, 91.56, 742.53, 861.18],
        mask_w: 63,
        mask_h: 137,
        ones: 5983,
    }];

    assert_eq!(segs.len(), expected.len(), "unexpected number of detections");
    for (a, e) in segs.iter().zip(&expected) {
        assert_eq!(a.class_id, e.class_id);
        assert_eq!(a.label, e.label);
        assert!(
            (a.prob - e.prob).abs() < 0.01,
            "prob: {a:?} vs {e:?}"
        );
        for i in 0..4 {
            assert!((a.bbox[i] - e.bbox[i]).abs() < 1.0, "bbox: {a:?} vs {e:?}");
        }
        assert_eq!((a.mask_w, a.mask_h), (e.mask_w, e.mask_h));
        let delta = a.ones.abs_diff(e.ones);
        assert!(delta <= e.ones / 20, "mask ones: {a:?} vs {e:?}");
    }
    Ok(())
}
