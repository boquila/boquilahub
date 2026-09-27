use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use boquilahub::api::{
    abstractions::{AIOutputs, Pred, PredAudio, PredImg, PredVideo},
    exif::{ExifMetadata, Field, In, Tag, Value},
};
use chrono::{NaiveDate, Timelike};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "boquilahub-exif-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> PathBuf {
        self.0.join("source.jpg")
    }

    fn write(&self, tags: &[(Tag, &str)]) {
        self.write_fields(
            tags.iter()
                .map(|(tag, value)| Field {
                    tag: *tag,
                    ifd_num: In::PRIMARY,
                    value: Value::Ascii(vec![value.as_bytes().to_vec()]),
                })
                .collect(),
        );
    }

    fn write_fields(&self, fields: Vec<Field>) {
        let mut writer = exif::experimental::Writer::new();
        for field in &fields {
            writer.push_field(field);
        }
        let mut tiff = Cursor::new(Vec::new());
        writer.write(&mut tiff, false).unwrap();
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe1];
        jpeg.extend_from_slice(&((tiff.get_ref().len() + 8) as u16).to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(tiff.get_ref());
        jpeg.extend_from_slice(&[0xff, 0xd9]);
        fs::write(self.path(), jpeg).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn dates_preserve_precision_offsets_and_all_tags() {
    let fixture = Fixture::new();
    fixture.write(&[
        (Tag::DateTimeOriginal, "2026:09:27 12:30:45"),
        (Tag::SubSecTimeOriginal, "1234567897"),
        (Tag::OffsetTimeOriginal, "+02:00"),
        (Tag::DateTimeDigitized, "2026:09:28 13:31:46"),
        (Tag::SubSecTimeDigitized, "5"),
        (Tag::OffsetTimeDigitized, "-03:30"),
        (Tag::DateTime, "2026:09:29 14:32:47"),
        (Tag::SubSecTime, "25"),
        (Tag::OffsetTime, "+00:00"),
        (Tag::Make, "Boquila"),
    ]);
    let metadata = ExifMetadata::from_file(fixture.path()).unwrap();
    let capture = metadata.captured_at.unwrap();
    assert_eq!(capture.local.to_string(), "2026-09-27 12:30:45.123456789");
    assert_eq!(capture.to_utc().unwrap().hour(), 10);
    assert_eq!(metadata.digitized_at.unwrap().to_utc().unwrap().hour(), 17);
    assert_eq!(
        metadata.modified_at.unwrap().local.nanosecond(),
        250_000_000
    );
    assert_eq!(metadata.created_at(), Some(capture));
    let make = metadata.get_field(Tag::Make, In::PRIMARY).unwrap();
    assert!(matches!(&make.value, Value::Ascii(values) if values[0] == b"Boquila"));
    assert_eq!(metadata.text(Tag::Make), Some("Boquila"));
    assert!(metadata.text(Tag::Model).is_none());
    let cloned = metadata.clone();
    assert!(std::sync::Arc::ptr_eq(&metadata.fields, &cloned.fields));
}

#[test]
fn creation_date_falls_back_past_missing_invalid_and_blank_dates() {
    let fixture = Fixture::new();
    for original in [
        "",
        "    :  :     :  :  ",
        "2026:02:30 12:00:00",
        "2026:13:01 12:00:00",
    ] {
        fixture.write(&[
            (Tag::DateTimeOriginal, original),
            (Tag::DateTimeDigitized, "2026:09:28 13:31:46"),
            (Tag::DateTime, "2026:09:29 14:32:47"),
        ]);
        let metadata = ExifMetadata::from_file(fixture.path()).unwrap();
        assert!(metadata.captured_at.is_none());
        assert_eq!(metadata.created_at(), metadata.digitized_at);
    }
    fixture.write(&[(Tag::DateTime, "2026:09:29 14:32:47")]);
    let metadata = ExifMetadata::from_file(fixture.path()).unwrap();
    assert_eq!(metadata.created_at(), metadata.modified_at);
    fixture.write(&[(Tag::Make, "Boquila")]);
    assert!(
        ExifMetadata::from_file(fixture.path())
            .unwrap()
            .created_at()
            .is_none()
    );
}

#[test]
fn bad_optional_components_do_not_discard_the_date_or_invent_utc() {
    let fixture = Fixture::new();
    for offset in [
        "",
        "      ",
        "+24:00",
        "+00:60",
        "+01:99",
        "invalid",
        "+01:00junk",
    ] {
        fixture.write(&[
            (Tag::DateTimeOriginal, "2026:09:27 12:30:45"),
            (Tag::SubSecTimeOriginal, "bad"),
            (Tag::OffsetTimeOriginal, offset),
        ]);
        let date = ExifMetadata::from_file(fixture.path())
            .unwrap()
            .created_at()
            .unwrap();
        assert_eq!(date.local.nanosecond(), 0);
        assert!(date.offset.is_none(), "offset: {offset}");
        assert!(date.to_utc().is_none());
    }
    fixture.write(&[(Tag::DateTimeOriginal, "2026:09:27 12:30:45")]);
    assert!(
        ExifMetadata::from_file(fixture.path())
            .unwrap()
            .created_at()
            .unwrap()
            .to_utc()
            .is_none()
    );
}

#[test]
fn thumbnail_dates_do_not_override_primary_image_dates() {
    let fixture = Fixture::new();
    fixture.write_fields(vec![
        Field {
            tag: Tag::Make,
            ifd_num: In::PRIMARY,
            value: Value::Ascii(vec![b"Boquila".to_vec()]),
        },
        Field {
            tag: Tag::DateTimeOriginal,
            ifd_num: In::THUMBNAIL,
            value: Value::Ascii(vec![b"2020:01:01 00:00:00".to_vec()]),
        },
        Field {
            tag: Tag::Orientation,
            ifd_num: In::PRIMARY,
            value: Value::Short(vec![6]),
        },
        Field {
            tag: Tag::GPSLatitude,
            ifd_num: In::PRIMARY,
            value: Value::Rational(vec![exif::Rational { num: 55, denom: 1 }]),
        },
    ]);
    let metadata = ExifMetadata::from_file(fixture.path()).unwrap();
    assert!(metadata.created_at().is_none());
    assert!(
        metadata
            .get_field(Tag::DateTimeOriginal, In::THUMBNAIL)
            .is_some()
    );
    assert_eq!(
        metadata
            .get_field(Tag::Orientation, In::PRIMARY)
            .unwrap()
            .value
            .get_uint(0),
        Some(6)
    );
    assert!(metadata.get_field(Tag::GPSLatitude, In::PRIMARY).is_some());
}

#[test]
fn all_prediction_constructors_read_source_exif_even_with_cached_predictions() {
    let fixture = Fixture::new();
    fixture.write(&[(Tag::DateTimeOriginal, "2020:01:01 00:00:00")]);
    let mut image = PredImg::new_simple(fixture.path());
    image.aioutput = Some(AIOutputs::Classification(Vec::new()));
    image.write_predictions().unwrap();
    fixture.write(&[(Tag::DateTimeOriginal, "2026:09:27 12:30:45")]);
    let image = PredImg::new_simple(fixture.path());
    let audio = PredAudio::new_simple(fixture.path());
    assert!(image.wasprocessed);
    assert!(audio.wasprocessed);
    for pred in [&image as &dyn Pred, &audio as &dyn Pred] {
        assert_eq!(
            pred.created_at().unwrap().local.date(),
            NaiveDate::from_ymd_opt(2026, 9, 27).unwrap()
        );
        assert!(!pred.predictions_json().unwrap().contains("exif"));
    }

    let mut video = PredVideo::new_simple(fixture.path());
    video.hydrate(1920, 1080, 30.0, 100);
    video.record(7, AIOutputs::Classification(Vec::new()));
    video.wasprocessed = true;
    video.write_predictions().unwrap();
    let mut cached: serde_json::Value =
        serde_json::from_str(&video.predictions_json().unwrap()).unwrap();
    assert!(cached.get("exif").is_none());
    // Even a stale EXIF field from another writer is ignored on deserialization.
    cached["exif"] = serde_json::json!({"captured_at": "stale"});
    fs::write(
        video.predictions_file_path().unwrap(),
        serde_json::to_string(&cached).unwrap(),
    )
    .unwrap();
    fixture.write(&[(Tag::DateTimeOriginal, "2027:01:02 03:04:05")]);
    let loaded = PredVideo::new_simple(fixture.path());
    assert_eq!(
        loaded.created_at().unwrap().local.date(),
        NaiveDate::from_ymd_opt(2027, 1, 2).unwrap()
    );
    assert!(loaded.wasprocessed);
    assert_eq!(loaded.processed_count(), 1);
    assert!(loaded.prediction_at(7).is_some());
    assert_eq!(loaded.width, 1920);
    let direct: PredVideo = serde_json::from_value(cached).unwrap();
    assert!(direct.exif.is_none());
    fs::write(fixture.path(), b"no EXIF anymore").unwrap();
    assert!(PredVideo::new_simple(fixture.path()).exif.is_none());
}

#[test]
fn missing_unsupported_and_corrupt_exif_do_not_prevent_loading_predictions() {
    let fixture = Fixture::new();
    for contents in [
        None,
        Some(b"RIFF----WAVE".as_slice()),
        Some(b"\xff\xd8\xff\xe1\0\x20Exif\0\0broken".as_slice()),
    ] {
        if let Some(contents) = contents {
            fs::write(fixture.path(), contents).unwrap();
        }
        let mut image = PredImg::new_simple(fixture.path());
        image.aioutput = Some(AIOutputs::Classification(Vec::new()));
        image.write_predictions().unwrap();
        let image = PredImg::new_simple(fixture.path());
        let audio = PredAudio::new_simple(fixture.path());
        assert!(image.wasprocessed && audio.wasprocessed);
        assert!(image.exif.is_none() && audio.exif.is_none());
        let video = PredVideo::new_simple(fixture.path());
        assert!(video.exif.is_none());
        assert!(video.frames.is_empty());
    }
}

#[test]
fn partially_broken_exif_retains_valid_fields() {
    let fixture = Fixture::new();
    // TIFF with two declared entries, but a truncated second entry.
    fs::write(fixture.path(), b"MM\0\x2a\0\0\0\x08\0\x02\x01\x00\0\x03\0\0\0\x01\0\x14\0\0\x01\x01\0\x03\0\0\0\x01\0\x15\0").unwrap();
    let metadata = ExifMetadata::from_file(fixture.path()).unwrap();
    assert_eq!(
        metadata
            .get_field(Tag::ImageWidth, In::PRIMARY)
            .unwrap()
            .value
            .get_uint(0),
        Some(20)
    );
}

#[test]
fn prediction_metadata_can_cross_gui_worker_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PredImg>();
    assert_send_sync::<PredAudio>();
    assert_send_sync::<PredVideo>();
}

#[test]
fn numeric_filter_keys_cover_values_without_accepting_invalid_numbers() {
    let fixture = Fixture::new();
    for (value, expected) in [
        (Value::Short(vec![6]), Some(6.0)),
        (Value::Long(vec![400]), Some(400.0)),
        (
            Value::Rational(vec![exif::Rational { num: 1, denom: 200 }]),
            Some(0.005),
        ),
        (
            Value::SRational(vec![exif::SRational { num: -1, denom: 2 }]),
            Some(-0.5),
        ),
        (Value::SByte(vec![-2]), Some(-2.0)),
        (Value::SShort(vec![-3]), Some(-3.0)),
        (Value::SLong(vec![-4]), Some(-4.0)),
        (Value::Float(vec![0.5]), Some(0.5)),
        (Value::Double(vec![1.25]), Some(1.25)),
        (Value::Double(vec![f64::NAN]), None),
        (
            Value::Rational(vec![exif::Rational { num: 1, denom: 0 }]),
            None,
        ),
        (Value::Ascii(vec![b"not numeric".to_vec()]), None),
    ] {
        fixture.write_fields(vec![Field {
            tag: Tag::ExposureTime,
            ifd_num: In::PRIMARY,
            value,
        }]);
        assert_eq!(
            ExifMetadata::from_file(fixture.path())
                .unwrap()
                .number(Tag::ExposureTime),
            expected
        );
    }
}
