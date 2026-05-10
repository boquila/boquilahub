use boquilahub::api::abstractions::XYXY;
use boquilahub::api::processing::post::nms_indices;

#[test]
fn nms_per_class_suppresses_same_class_overlap() {
    let boxes = vec![
        XYXY::new(0.0, 0.0, 100.0, 100.0, 0.9, 0),
        XYXY::new(5.0, 5.0, 105.0, 105.0, 0.7, 0),
        XYXY::new(200.0, 200.0, 300.0, 300.0, 0.8, 1),
    ];
    let kept = nms_indices(&boxes, 0.5, true);
    assert!(kept.contains(&0), "high-prob same-class box should be kept");
    assert!(!kept.contains(&1), "low-prob overlapping same-class box should be suppressed");
    assert!(kept.contains(&2), "different class box should always be kept");
}

#[test]
fn nms_all_class_suppresses_across_classes() {
    let boxes = vec![
        XYXY::new(0.0, 0.0, 100.0, 100.0, 0.9, 0),
        XYXY::new(5.0, 5.0, 105.0, 105.0, 0.8, 1),
        XYXY::new(200.0, 200.0, 300.0, 300.0, 0.7, 2),
    ];
    let kept = nms_indices(&boxes, 0.5, false);
    assert!(kept.contains(&0), "highest prob box should be kept");
    assert!(!kept.contains(&1), "overlapping box of different class should be suppressed");
    assert!(kept.contains(&2), "non-overlapping box should be kept");
}