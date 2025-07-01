// The idea is to have the core funcionality that will alow us to do everything we need in the app
// but also, enough abstractions so we can experiment and build more complex tools in the future
#![allow(dead_code)]
use derive_new::new;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::api::models::processing::inference::AIOutputs;

/// Probabilities in the YOLO format
/// `classes` is a Vec with the names for each classification
/// `probs` is a Vec with the probabilities/confidence for each classification
#[derive(Serialize, Deserialize, Clone, new)]
pub struct ProbSpace {
    pub probs: Vec<f32>,
    pub classes_ids: Vec<usize>,
    pub classes: Vec<String>,
}

impl ProbSpace {
    pub fn highest_confidence(&self) -> String {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| self.classes[index].clone())
            .unwrap_or_else(|| String::from("no prediction"))
    }

    pub fn highest_confidence_detailed(&self) -> Option<(usize, String, f32)> {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, &prob)| (self.classes_ids[index], self.classes[index].clone(), prob))
    }
}

/// Segmentation in the YOLO format, normalized
/// # Fields
/// - `vertices` represents a polygon
#[derive(Serialize, Deserialize, Clone, new)]
pub struct SEGn {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub prob: f32,
    pub class_id: u16,
}

/// Segmentation in the YOLO format, not normalized
/// # Fields
/// - `vertices` represents a polygon
#[derive(Serialize, Deserialize, Clone, new)]
pub struct SEG {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub prob: f32,
    pub class_id: u16,
}

#[derive(Serialize, Deserialize, Clone, new)]
pub struct SEGc {
    pub seg: SEG,
    pub bbox: XYXY,
    pub label: String,
}

// Trait for all bounding boxes (that don't have a string)
pub trait BoundingBoxTrait: Copy {
    fn area(&self) -> f32;
    fn intersect(&self, other: &Self) -> f32;
    fn iou(&self, other: &Self) -> f32;
    fn get_coords(&self) -> (f32, f32, f32, f32);
    fn get_prob(&self) -> f32;
    fn get_class_id(&self) -> u16;
    fn check(&self) -> bool;
    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY;
    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn;
    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH;
    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn;
    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc;
    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc;
    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc;
    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc;
}

/// Bounding box in normalized XYXY format
/// # Fields
/// - `x1` and `y1` represent the top-left corner
/// - `x2` and `y2` represent the bottom-right  corner
#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYXYn {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u16,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, new)]
pub struct XYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u16,
}

/// Bounding box in normalized XYWH format
/// # Fields
/// - `x` and `y` represent the center
/// - `w` and `h` represent width and height
#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYWHn {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub prob: f32,
    pub class_id: u16,
}

#[derive(Serialize, Deserialize, Copy, Clone, new)]
pub struct XYWH {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub prob: f32,
    pub class_id: u16,
}

fn intersect_xyxys(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32, x4: f32, y4: f32) -> f32 {
    let x_left = x1.max(x3);
    let y_top = y1.max(y3);
    let x_right = x2.min(x4);
    let y_bottom = y2.min(y4);

    (x_right - x_left) * (y_bottom - y_top)
}

fn intersect_xywhs(x1: f32, y1: f32, w1: f32, h1: f32, x2: f32, y2: f32, w2: f32, h2: f32) -> f32 {
    let x_left = x1.max(x2);
    let y_top = y1.max(y2);
    let x_right = (x1 + w1).min(x2 + w2);
    let y_bottom = (y1 + h1).min(y2 + h2);

    (x_right - x_left) * (y_bottom - y_top)
}

fn iou<T: BoundingBoxTrait>(a: &T, b: &T) -> f32 {
    let intersection = a.intersect(b);
    let union = a.area() + b.area() - intersection;
    intersection / union
}

impl BoundingBoxTrait for XYXYn {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXYn) -> f32 {
        intersect_xyxys(
            self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2,
        )
    }

    fn iou(&self, other: &XYXYn) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u16 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x1 >= 0.0
            && self.x1 <= 1.0
            && self.y1 >= 0.0
            && self.y1 <= 1.0
            && self.x2 >= 0.0
            && self.x2 <= 1.0
            && self.y2 >= 0.0
            && self.y2 <= 1.0
            && self.x2 >= self.x1
            && self.y2 >= self.y1
            && self.prob >= 0.0
            && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    fn to_xyxyn(&self, _w: Option<f32>, _h: Option<f32>) -> XYXYn {
        return *self;
    }

    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY {
        let w = w.unwrap();
        let h = h.unwrap();
        XYXY::new(
            self.x1 * w,
            self.y1 * h,
            self.x2 * w,
            self.x2 * h,
            self.prob,
            self.class_id,
        )
    }

    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH {
        let temp = self.to_xyxy(w, h);
        return temp.to_xywh(w, h);
    }

    fn to_xywhn(&self, _w: Option<f32>, _h: Option<f32>) -> XYWHn {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        XYWHn::new(x, y, w, h, self.prob, self.class_id)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYXY {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXY) -> f32 {
        intersect_xyxys(
            self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2,
        )
    }

    fn iou(&self, other: &XYXY) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u16 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x2 >= self.x1 && self.y2 >= self.y1 && self.prob >= 0.0 && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn {
        let w = w.unwrap();
        let h = h.unwrap();
        return XYXYn::new(
            self.x1 / w,
            self.y1 / h,
            self.x2 / w,
            self.x2 / h,
            self.prob,
            self.class_id,
        );
    }

    fn to_xyxy(&self, _w: Option<f32>, _h: Option<f32>) -> XYXY {
        return *self;
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        return XYWH::new(x, y, w, h, self.prob, self.class_id);
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let temp = self.to_xyxyn(w, h);
        return temp.to_xywhn(w, h);
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYWHn {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersect(&self, other: &XYWHn) -> f32 {
        intersect_xywhs(
            self.x, self.y, self.w, self.h, other.x, other.y, other.w, other.h,
        )
    }

    fn iou(&self, other: &XYWHn) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u16 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.x >= 0.0
            && self.x <= 1.0
            && self.y >= 0.0
            && self.y <= 1.0
            && self.w >= 0.0
            && self.w <= 1.0
            && self.h >= 0.0
            && self.h <= 1.0
            && (self.x + self.w) <= 1.0
            && (self.y + self.h) <= 1.0
            && self.prob >= 0.0
            && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn to_xyxyn(&self, _w: Option<f32>, _h: Option<f32>) -> XYXYn {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        return XYXYn::new(x1, y1, x2, y2, self.prob, self.class_id);
    }

    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY {
        let temp = self.to_xyxyn(w, h);
        return temp.to_xyxy(w, h);
    }

    fn to_xywhn(&self, _w: Option<f32>, _h: Option<f32>) -> XYWHn {
        return *self;
    }

    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH {
        let w = w.unwrap();
        let h = h.unwrap();

        return XYWH {
            x: self.x * w,
            y: self.y * h,
            w: self.w * w,
            h: self.h * h,
            class_id: self.class_id,
            prob: self.prob,
        };
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

impl BoundingBoxTrait for XYWH {
    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersect(&self, other: &XYWH) -> f32 {
        intersect_xywhs(
            self.x, self.y, self.w, self.h, other.x, other.y, other.w, other.h,
        )
    }

    fn iou(&self, other: &XYWH) -> f32 {
        iou(self, other)
    }

    fn get_prob(&self) -> f32 {
        self.prob
    }

    fn get_class_id(&self) -> u16 {
        self.class_id
    }

    fn check(&self) -> bool {
        self.w >= 0.0 && self.h >= 0.0 && self.prob >= 0.0 && self.prob <= 1.0
    }

    fn get_coords(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn {
        let temp = self.to_xyxy(w, h);
        return temp.to_xyxyn(w, h);
    }

    fn to_xyxy(&self, _w: Option<f32>, _h: Option<f32>) -> XYXY {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        return XYXY::new(x1, y1, x2, y2, self.prob, self.class_id);
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let w = w.unwrap();
        let h = h.unwrap();

        return XYWHn {
            x: self.x / w,
            y: self.y / h,
            w: self.w / w,
            h: self.h / h,
            class_id: self.class_id,
            prob: self.prob,
        };
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        return *self;
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        let temp = self.to_xyxyn(w, h);
        return XYXYnc::new(temp, label);
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        let temp = self.to_xyxy(w, h);
        return XYXYc::new(temp, label);
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        let temp = self.to_xywh(w, h);
        return XYWHc::new(temp, label);
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        let temp = self.to_xywhn(w, h);
        return XYWHnc::new(temp, label);
    }
}

// AI model for Image Processing
#[derive(Deserialize, Clone, new)]
pub struct AI {
    pub name: String,
    pub version: f32, // to delete
    pub input_width: u32, // to delete
    pub input_height: u32, // to delete
    pub description: String,  // to think about it 
    pub color_code: String, // to delete
    pub task: String,       // "detect", "classify", "segment"
    pub post_processing: Vec<String>, // "NMS"
    pub classes: Vec<String>,
}

impl AI {
    pub fn get_path(&self) -> String {
        format!("models/{}.bq", self.name)
    }
}

#[derive(new, Clone)]
pub struct PredImg {
    pub file_path: PathBuf,
    pub aioutput: Option<AIOutputs>,
    pub wasprocessed: bool,
}

impl PredImg {
    // Simple constructor: only file_path is provided
    pub fn new_simple(file_path: PathBuf) -> Self {
        PredImg {
            file_path,
            aioutput: None,
            wasprocessed: false,
        }
    }

    #[inline(always)]
    pub fn draw(&self) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        let mut img = image::open(&self.file_path).unwrap().into_rgb8();
        if self.wasprocessed && !self.aioutput.as_ref().unwrap().is_empty() {
            super::render::draw_aioutput(&mut img, &self.aioutput.as_ref().unwrap());
        }
        return DynamicImage::ImageRgb8(img).to_rgba8();
    }

    pub fn draw2(&self) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let mut img = image::open(&self.file_path).unwrap().into_rgb8();
        if self.wasprocessed && !self.aioutput.as_ref().unwrap().is_empty() {
            super::render::draw_aioutput(&mut img, &self.aioutput.as_ref().unwrap());
        }
        return img;
    }

    pub fn save(&self) {
        let img_data = self.draw2();

        std::fs::create_dir_all("export").expect("Failed to create export directory");

        let filename = format!(
            "export/exported_{}.jpg",
            std::path::Path::new(&self.file_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image")
        );

        img_data.save(&filename).unwrap();
    }
}

pub trait PredImgSugar {
    fn count_processed_images(&self) -> usize;
    fn get_progress(&self) -> f32;
}

impl PredImgSugar for Vec<PredImg> {
    fn count_processed_images(&self) -> usize {
        self.iter().filter(|img| img.wasprocessed).count()
    }

    fn get_progress(&self) -> f32 {
        self.count_processed_images() as f32 / self.len() as f32
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, new)]
pub struct XYXYc {
    pub bbox: XYXY,
    pub label: String,
}

#[derive(Serialize, Deserialize, new)]
pub struct XYXYnc {
    pub bbox: XYXYn,
    pub label: String,
}

#[derive(Serialize, Deserialize, new)]
pub struct XYWHc {
    pub bbox: XYWH,
    pub label: String,
}

#[derive(Serialize, Deserialize, new)]
pub struct XYWHnc {
    pub bbox: XYWHn,
    pub label: String,
}

pub struct Dependency {
    version: f32,
    name: String,
}
