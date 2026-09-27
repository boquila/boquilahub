//! EXIF read from source files at runtime, never from prediction sidecars.
//!
//! GUI sorting and filtering use ordinary Rust operations, with no IO:
//! ```no_run
//! use boquilahub::api::{abstractions::{Pred, PredImg}, exif::Tag};
//! let mut images = vec![PredImg::new_simple("photo.jpg".into())];
//! // Camera-local creation order, with undated files last.
//! images.sort_by_key(|image| {
//!     let date = image.created_at().map(|date| date.local);
//!     (date.is_none(), date)
//! });
//! images.retain(|image| {
//!     image.exif().and_then(|exif| exif.text(Tag::Make)) == Some("Canon")
//! });
//! // Numeric keys also cover exposure, focal length, ISO, orientation, etc.
//! let exposure = images.first().and_then(|image| image.exif())
//!     .and_then(|exif| exif.number(Tag::ExposureTime));
//! ```

use std::{fs::File, io::BufReader, path::Path, sync::Arc};

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, TimeZone, Utc};
pub use exif::{Field, In, Tag, Value};

/// Camera-local time, with a UTC offset only when the file actually provides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExifDateTime {
    pub local: NaiveDateTime,
    pub offset: Option<FixedOffset>,
}

impl ExifDateTime {
    /// An absolute timestamp is unavailable for files without a valid offset.
    pub fn to_utc(self) -> Option<DateTime<Utc>> {
        self.offset?
            .from_local_datetime(&self.local)
            .single()
            .map(|date| date.with_timezone(&Utc))
    }
}

/// An in-memory snapshot loaded by `Pred*::new_simple`.
///
/// All tags are retained for future sorting/filtering (camera, GPS, etc.).
/// Owned fields keep predictions Send + Sync; the parser itself is not Sync.
/// Cloning a prediction shares these immutable fields without rereading the file.
#[derive(Clone, Debug)]
pub struct ExifMetadata {
    pub captured_at: Option<ExifDateTime>,
    pub digitized_at: Option<ExifDateTime>,
    pub modified_at: Option<ExifDateTime>,
    pub fields: Arc<[Field]>,
}

impl ExifMetadata {
    /// Reads EXIF without decoding pixels. Unsupported formats, absent EXIF,
    /// and unreadable files return an error; prediction constructors use None.
    /// Valid fields survive non-fatal errors in other parts of the EXIF block.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, exif::Error> {
        let file = File::open(path)?;
        let data = exif::Reader::new()
            .continue_on_error(true)
            .read_from_container(&mut BufReader::new(file))
            .or_else(|error| error.distill_partial_result(|_| {}))?;
        Ok(Self {
            captured_at: parse_datetime(
                &data,
                Tag::DateTimeOriginal,
                Tag::SubSecTimeOriginal,
                Tag::OffsetTimeOriginal,
            ),
            digitized_at: parse_datetime(
                &data,
                Tag::DateTimeDigitized,
                Tag::SubSecTimeDigitized,
                Tag::OffsetTimeDigitized,
            ),
            modified_at: parse_datetime(&data, Tag::DateTime, Tag::SubSecTime, Tag::OffsetTime),
            fields: data.fields().cloned().collect(),
        })
    }

    /// Best available EXIF creation date: capture, digitization, then file change.
    /// This never falls back to filesystem timestamps. Use `captured_at` when
    /// sorting must strictly reflect when the camera took the picture.
    pub fn created_at(&self) -> Option<ExifDateTime> {
        self.captured_at.or(self.digitized_at).or(self.modified_at)
    }

    /// Look up any tag, keeping primary-image and thumbnail metadata distinct.
    pub fn get_field(&self, tag: Tag, ifd: In) -> Option<&Field> {
        self.fields
            .iter()
            .find(|field| field.tag == tag && field.ifd_num == ifd)
    }

    /// First ASCII value of a primary-image tag, e.g. camera make or model.
    pub fn text(&self, tag: Tag) -> Option<&str> {
        match &self.get_field(tag, In::PRIMARY)?.value {
            Value::Ascii(values) => std::str::from_utf8(values.first()?).ok(),
            _ => None,
        }
    }

    /// First numeric value of a primary-image tag. Invalid rationals and
    /// non-finite values return None; use `get_field` for arrays such as GPS.
    pub fn number(&self, tag: Tag) -> Option<f64> {
        let value = match &self.get_field(tag, In::PRIMARY)?.value {
            Value::Rational(values) => values.first().map(|value| value.to_f64()),
            Value::SRational(values) => values.first().map(|value| value.to_f64()),
            Value::SByte(values) => values.first().copied().map(f64::from),
            Value::SShort(values) => values.first().copied().map(f64::from),
            Value::SLong(values) => values.first().copied().map(f64::from),
            Value::Float(values) => values.first().copied().map(f64::from),
            Value::Double(values) => values.first().copied(),
            value => value.get_uint(0).map(f64::from),
        };
        value.filter(|value| value.is_finite())
    }
}

fn ascii(data: &exif::Exif, tag: Tag) -> Option<&[u8]> {
    match &data.get_field(tag, In::PRIMARY)?.value {
        Value::Ascii(values) => values.first().map(Vec::as_slice),
        _ => None,
    }
}

fn parse_datetime(
    data: &exif::Exif,
    date_tag: Tag,
    subsec_tag: Tag,
    offset_tag: Tag,
) -> Option<ExifDateTime> {
    let mut date = exif::DateTime::from_ascii(ascii(data, date_tag)?).ok()?;
    if let Some(subsec) = ascii(data, subsec_tag) {
        // A malformed optional component does not discard a valid base date.
        let _ = date.parse_subsec(subsec);
    }
    let local = NaiveDate::from_ymd_opt(date.year.into(), date.month.into(), date.day.into())?
        .and_hms_nano_opt(
            date.hour.into(),
            date.minute.into(),
            date.second.into(),
            date.nanosecond.unwrap_or(0),
        )?;
    // Chrono validates ranges; EXIF offsets must have the form +/-HH:MM.
    let offset = ascii(data, offset_tag)
        .and_then(|value| std::str::from_utf8(value).ok())
        .filter(|value| value.len() == 6 && value.as_bytes()[3] == b':')
        .and_then(|value| value.parse().ok());
    Some(ExifDateTime { local, offset })
}
