// The idea is to have the core funcionality that will alow us to do everything we need in the app
// but also, enough abstractions so we can experiment and build more complex tools in the future
#![allow(dead_code)]
use std::path::PathBuf;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

/// Probabilities in the YOLO format
/// `classes` is a Vec with the names for each classification
/// `probs` is a Vec with the probabilities/confidence for each classification
pub struct ProbSpace {
    pub classes: Vec<String>,
    pub probs: Vec<f32>,
}

/// Segmentation in the YOLO format, normalized
/// # Fields
/// - `vertices` represents a polygon
#[derive(Serialize, Deserialize, Clone)]
pub struct SEGn {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub prob: f32,
    pub class_id: u16,
}

/// Segmentation in the YOLO format, not normalized
/// # Fields
/// - `vertices` represents a polygon
#[derive(Serialize, Deserialize, Clone)]
struct SEG {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub prob: f32,
    pub class_id: u16,
}

pub struct T {}

// Trait for all bounding boxes (that don't have a string)
pub trait BoundingBoxTrait: Copy {
    fn new(a: f32, b: f32, c: f32, d: f32, prob: f32, class_id: u16) -> Self;
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
    fn jsonify(&self) -> String;
}

/// Bounding box in normalized XYXY format
/// # Fields
/// - `x1` and `y1` represent the top-left corner
/// - `x2` and `y2` represent the bottom-right  corner
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct XYXYn {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
    pub class_id: u16,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
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
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct XYWHn {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub prob: f32,
    pub class_id: u16,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
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
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, prob: f32, class_id: u16) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            prob,
            class_id,
        }
    }

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

    fn jsonify(&self) -> String {
        format!(
            "{{\"x1\":{},\"y1\":{},\"x2\":{},\"y2\":{},\"class_id\":{},\"prob\":{}}}",
            self.x1, self.y1, self.x2, self.y2, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYXY {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, prob: f32, class_id: u16) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            prob,
            class_id,
        }
    }

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

    fn jsonify(&self) -> String {
        format!(
            "{{\"x1\":{},\"y1\":{},\"x2\":{},\"y2\":{},\"class_id\":{},\"prob\":{}}}",
            self.x1, self.y1, self.x2, self.y2, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYWHn {
    fn new(x: f32, y: f32, w: f32, h: f32, prob: f32, class_id: u16) -> Self {
        Self {
            x,
            y,
            w,
            h,
            prob,
            class_id,
        }
    }

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

    fn jsonify(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"class_id\":{},\"prob\":{}}}",
            self.x, self.y, self.w, self.h, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYWH {
    fn new(x: f32, y: f32, w: f32, h: f32, prob: f32, class_id: u16) -> Self {
        Self {
            x,
            y,
            w,
            h,
            prob,
            class_id,
        }
    }

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

    fn jsonify(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"class_id\":{},\"prob\":{}}}",
            self.x, self.y, self.w, self.h, self.class_id, self.prob
        )
    }
}

// AI model for Image Processing
#[derive(Deserialize, Clone)]
pub struct AI {
    pub name: String,
    pub version: f32, // complement tothe name
    pub input_width: u32,
    pub input_height: u32,
    pub description: String,          // complement to the name
    pub color_code: String, // "terra", "fire", "green", depending on this, the app will show different colors hehe
    pub task: String,       // "detect", "classify", "segment"
    pub post_processing: Vec<String>, // "detect", "classify", "segment"
    pub classes: Vec<String>,
}

impl AI {
    pub fn new(
        name: String,
        version: f32,
        input_width: u32,
        input_height: u32,
        description: String,
        color_code: String,
        task: String,
        post_processing: Vec<String>,
        classes: Vec<String>,
    ) -> Self {
        Self {
            name,
            version,
            input_width,
            input_height,
            description,
            color_code,
            task,
            post_processing,
            classes,
        }
    }

    // The `default` function returns a dummy instance
    pub fn default() -> Self {
        AI::new(
            "boquilanet-gen".to_string(),
            0.1,
            1024,
            1024,
            "Generic animal detection".to_string(),
            "green".to_string(),
            "detect".to_string(),
            vec!["NMS".to_string()],
            vec!["animal".to_string()],
        )
    }

    // Method to get the path of the AI model
    pub fn get_path(&self) -> String {
        format!("models/{}.bq", self.name)
    }
}

pub struct PredImg {
    pub file_path: PathBuf,
    pub list_bbox: Vec<XYXYc>,
    pub wasprocessed: bool,
}

impl PredImg {
    pub fn new(file_path: PathBuf, list_bbox: Vec<XYXYc>, wasprocessed: bool) -> Self {
        PredImg {
            file_path,
            list_bbox,
            wasprocessed,
        }
    }

    // Simple constructor: only file_path is provided
    pub fn new_simple(file_path: PathBuf) -> Self {
        PredImg {
            file_path,
            list_bbox: Vec::new(),
            wasprocessed: false,
        }
    }

    pub fn draw(&self) -> Vec<u8> {
        let mut img = image::open(&self.file_path).unwrap().into_rgb8();
        super::render::draw_bbox_from_imgbuf(&mut img, &self.list_bbox);
        return super::utils::image_buffer_to_jpg_buffer(img);
    }

    pub fn draw2(&self) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        let mut img = image::open(&self.file_path).unwrap().into_rgb8();
        super::render::draw_bbox_from_imgbuf(&mut img, &self.list_bbox);
        return DynamicImage::ImageRgb8(img).to_rgba8();
        // return img
    }

    pub fn save(&self) {
        if self.wasprocessed {
            let jpg_data = &self.draw();
            let filename = &self.file_path;
            let path = std::path::Path::new(filename);

            // Create export directory if it doesn't exist
            std::fs::create_dir_all("export").unwrap_or_else(|e| {
                eprintln!("Failed to create export directory: {}", e);
            });

            // Extract file name component
            let file_stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed");

            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

            // Construct new path with "exported_" prefix
            let mut new_path = std::path::PathBuf::from("export");
            new_path.push(format!("exported_{}", file_stem));

            // Add extension if it exists
            if !extension.is_empty() {
                new_path.set_extension(extension);
            }

            let filepath = new_path
                .to_str()
                .unwrap_or("export/exported_file")
                .to_string();

            let mut file = std::fs::File::create(filepath).unwrap();
            std::io::Write::write_all(&mut file, &jpg_data).unwrap();
        }
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
        let scalar = self.count_processed_images();
        return scalar as f32 / self.len() as f32;
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct XYXYc {
    pub xyxy: XYXY,
    pub label: String,
}

#[derive(Serialize, Deserialize)]
pub struct XYXYnc {
    pub xyxyn: XYXYn,
    pub label: String,
}

#[derive(Serialize, Deserialize)]
pub struct XYWHc {
    pub xywh: XYWH,
    pub label: String,
}

#[derive(Serialize, Deserialize)]
pub struct XYWHnc {
    pub xywhn: XYWHn,
    pub label: String,
}

// Trait for all bounding boxes with a label string
pub trait BoundingBoxTraitC<T: BoundingBoxTrait> {
    fn new(boundingbox: T, label: String) -> Self;
    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc;
    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc;
    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc;
    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc;
    // The string that is used to render a bounding box in an image
    // "0.92% animal"
    fn strlabel(&self) -> String;
}

impl BoundingBoxTraitC<XYXY> for XYXYc {
    fn new(xyxy: XYXY, label: String) -> Self {
        Self { xyxy, label }
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        self.xyxy.to_xyxyc(w, h, label)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        self.xyxy.to_xyxync(w, h, label)
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        self.xyxy.to_xywhc(w, h, label)
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        self.xyxy.to_xywhnc(w, h, label)
    }

    fn strlabel(&self) -> String {
        detection_label(&self.label, &self.xyxy.prob)
    }
}

impl BoundingBoxTraitC<XYXYn> for XYXYnc {
    fn new(xyxyn: XYXYn, label: String) -> Self {
        Self { xyxyn, label }
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        self.xyxyn.to_xyxyc(w, h, label)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        self.xyxyn.to_xyxync(w, h, label)
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        self.xyxyn.to_xywhc(w, h, label)
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        self.xyxyn.to_xywhnc(w, h, label)
    }

    fn strlabel(&self) -> String {
        detection_label(&self.label, &self.xyxyn.prob)
    }
}

impl BoundingBoxTraitC<XYWH> for XYWHc {
    fn new(xywh: XYWH, label: String) -> Self {
        Self { xywh, label }
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        self.xywh.to_xyxyc(w, h, label)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        self.xywh.to_xyxync(w, h, label)
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        self.xywh.to_xywhc(w, h, label)
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        self.xywh.to_xywhnc(w, h, label)
    }

    fn strlabel(&self) -> String {
        detection_label(&self.label, &self.xywh.prob)
    }
}

impl BoundingBoxTraitC<XYWHn> for XYWHnc {
    fn new(xywhn: XYWHn, label: String) -> Self {
        Self { xywhn, label }
    }

    fn to_xyxyc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYc {
        self.xywhn.to_xyxyc(w, h, label)
    }

    fn to_xyxync(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYXYnc {
        self.xywhn.to_xyxync(w, h, label)
    }

    fn to_xywhc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHc {
        self.xywhn.to_xywhc(w, h, label)
    }

    fn to_xywhnc(&self, w: Option<f32>, h: Option<f32>, label: String) -> XYWHnc {
        self.xywhn.to_xywhnc(w, h, label)
    }

    fn strlabel(&self) -> String {
        detection_label(&self.label, &self.xywhn.prob)
    }
}

pub fn get_ai_by_description(list_ais: &[AI], description: &str) -> AI {
    list_ais
        .iter()
        .find(|ai| ai.name == description)
        .unwrap()
        .clone()
}

pub struct Dependency {
    version: f32,
    name: String,
}

fn detection_label(label: &String, conf: &f32) -> String {
    let conf = format!("{:.2}", conf);
    format!("{} {}", label, conf)
}
