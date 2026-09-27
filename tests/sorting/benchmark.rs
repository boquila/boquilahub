// Exercises the production sorter with loaded Pred metadata; no GUI rendering.
use super::*;
use crate::api::abstractions::{AIOutputs, PredAudio, PredImg, PredVideo};
use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("boquila-sort-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn file(&self, name: &str, nanos: u64) -> PredImg {
        let path = self.0.join(name);
        let date = SystemTime::UNIX_EPOCH
            + Duration::from_secs(1_700_000_000)
            + Duration::from_nanos(nanos);
        std::fs::File::create(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(date))
            .unwrap();
        PredImg::new_simple(path)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn sorting_preserves_precision_ties_and_loaded_metadata() {
    let fixture = Files::new();
    let mut files = vec![
        fixture.file("later.jpg", 500),
        fixture.file("earlier.jpg", 100),
    ];
    files.push(files[1].clone());
    files.push(PredImg::new_simple(fixture.0.join("missing.jpg")));
    let mut browser = Browser::default();
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [0, 1, 2, 3]);
    browser.field = Some(DateField::Modified);
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [1, 2, 0, 3]);
    browser.descending = true;
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [0, 1, 2, 3]);
    fixture.file("earlier.jpg", 900);
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [0, 1, 2, 3]); // Snapshot changes only when reloaded.
    files[1] = PredImg::new_simple(files[1].file_path.clone());
    files[2] = files[1].clone();
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [1, 2, 0, 3]);
    std::fs::remove_dir_all(&fixture.0).unwrap();
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [1, 2, 0, 3]); // Sorting needs no filesystem access.
}

#[test]
fn metadata_is_available_for_every_pred_and_never_saved_in_sidecars() {
    let fixture = Files::new();
    let image = fixture.file("source.jpg", 100);
    let audio = PredAudio::new_simple(image.file_path.clone());
    let mut video = PredVideo::new_simple(image.file_path.clone());
    for file in [&image as &dyn Pred, &audio, &video] {
        let metadata = file.metadata().unwrap();
        assert_eq!(
            metadata.modified().unwrap(),
            image.metadata().unwrap().modified().unwrap()
        );
        assert_eq!(metadata.len(), 0);
    }
    video.hydrate(640, 480, 30.0, 10);
    video.record(3, AIOutputs::Classification(Vec::new()));
    video.write_predictions().unwrap();
    let json: serde_json::Value = serde_json::from_str(&video.predictions_json().unwrap()).unwrap();
    assert!(json.get("metadata").is_none());
    let updated = fixture.file("source.jpg", 900);
    let reloaded = PredVideo::new_simple(image.file_path.clone());
    assert_eq!(
        reloaded.metadata().unwrap().modified().unwrap(),
        updated.metadata().unwrap().modified().unwrap()
    );
    assert!(reloaded.prediction_at(3).is_some());
    assert!(
        PredVideo::new_simple(fixture.0.join("missing.mp4"))
            .metadata()
            .is_none()
    );
}

#[test]
#[ignore = "creates 100,000 temporary files; run with --release --ignored --nocapture"]
fn benchmark_date_sorting() {
    let fixture = Files::new();
    let start = Instant::now();
    let files: Vec<_> = (0..100_000)
        .map(|i| fixture.file(&format!("{i:06}.jpg"), (i * 7919 % 100_000) * 100))
        .collect();
    println!(
        "Create files + load predictions/metadata: {:?} (outside sort timing)",
        start.elapsed()
    );
    std::fs::remove_dir_all(&fixture.0).unwrap(); // Prove every timed sort uses RAM only.
    println!(
        "Metadata snapshot memory per 100k files: {} bytes",
        100_000 * std::mem::size_of::<Option<std::fs::Metadata>>()
    );
    for count in [1000, 10_000, 100_000] {
        let selected: Vec<_> = (0..count)
            .map(|i| files[i * 7919 % files.len()].clone())
            .collect();
        for field in DateField::ALL {
            for descending in [false, true] {
                let mut browser = Browser {
                    field: Some(field),
                    descending,
                    ..Default::default()
                };
                let mut samples = [Duration::ZERO; 5];
                for sample in &mut samples {
                    let start = Instant::now();
                    browser.sort_by_date(black_box(&selected));
                    *sample = start.elapsed();
                    black_box(&browser.order);
                }
                samples.sort();
                let name = field.label(&Lang::EN);
                println!(
                    "{count} files, {name}, descending={descending}: {samples:?} (sorted samples, median at index 2)"
                );
                let mut indexes = browser.order.clone();
                indexes.sort_unstable();
                assert_eq!(indexes, (0..count).collect::<Vec<_>>());
                let dates: Vec<_> = selected.iter().map(|file| field.read(file)).collect();
                for pair in browser.order.windows(2) {
                    let (a, b) = (dates[pair[0]], dates[pair[1]]);
                    match (a, b) {
                        (Some(a), Some(b)) if a != b => {
                            assert!(if descending { a > b } else { a < b })
                        }
                        (None, Some(_)) => panic!("missing date must sort last"),
                        _ if a == b => {
                            assert!(pair[0] < pair[1], "equal dates must retain input order")
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
