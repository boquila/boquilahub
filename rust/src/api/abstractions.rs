#![allow(dead_code)]
use serde::Deserialize;

// Big TODO: class_ids should be string
// and this string should be defined right after the inference

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
pub struct SEGn {
    pub vertices: Vec<f32>,
    pub class_id: usize,
    pub prob: f32,
}

/// Segmentation in the YOLO format, not normalized
/// # Fields
/// - `vertices` represents a polygon
struct SEG {
    pub vertices: Vec<f32>,
    pub class_id: usize,
    pub prob: f32,
}

pub trait BoundingBoxTrait: Copy {
    fn new(a: f32, b: f32, c: f32, d: f32, class_id: usize, prob: f32) -> Self;
    fn area(&self) -> f32;
    fn intersect(&self, other: &Self) -> f32;
    fn iou(&self, other: &Self) -> f32;
    fn get_coords(&self) -> (f32, f32, f32, f32);
    fn get_prob(&self) -> f32;
    fn get_class_id(&self) -> usize;
    fn check(&self) -> bool;
    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY;
    fn to_xyxyn(&self, w: Option<f32>, h: Option<f32>) -> XYXYn;
    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH;
    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn;
    fn jsonify(&self) -> String;
}

/// Bounding box in normalized XYXY format
/// # Fields
/// - `x1` and `y1` represent the top-left corner
/// - `x2` and `y2` represent the bottom-right  corner
#[derive(Debug, Copy, Clone)]
pub struct XYXYn {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct XYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}

/// Bounding box in normalized XYWH format
/// # Fields
/// - `x` and `y` represent the center
/// - `w` and `h` represent width and height
#[derive(Debug, Copy, Clone)]
pub struct XYWHn {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub prob: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct XYWH {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub prob: f32,
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

// pub fn nms<T: BoundingBox>(mut boxes: Vec<T>) -> Vec<T> {
//     boxes.sort_by(|box1, box2| box2.get_prob().total_cmp(&box1.get_prob()));
//     let mut result = Vec::new();
//     while boxes.len() > 0 {
//         result.push(boxes[0]);
//         boxes = boxes
//             .iter()
//             .filter(|box1| boxes[0].iou(box1) < 0.7)
//             .map(|x| *x)
//             .collect()
//     }
//     return result
// }

impl BoundingBoxTrait for XYXYn {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, class_id: usize, prob: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            class_id,
            prob,
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

    fn get_class_id(&self) -> usize {
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
            self.class_id,
            self.prob,
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
        XYWHn::new(x, y, w, h, self.class_id, self.prob)
    }

    fn jsonify(&self) -> String {
        format!(
            "{{\"x1\":{},\"y1\":{},\"x2\":{},\"y2\":{},\"class_id\":{},\"prob\":{}}}",
            self.x1, self.y1, self.x2, self.y2, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYXY {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32, class_id: usize, prob: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            class_id,
            prob,
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

    fn get_class_id(&self) -> usize {
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
        XYXYn::new(
            self.x1 / w,
            self.y1 / h,
            self.x2 / w,
            self.x2 / h,
            self.class_id,
            self.prob,
        )
    }

    fn to_xyxy(&self, _w: Option<f32>, _h: Option<f32>) -> XYXY {
        return *self;
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        XYWH::new(x, y, w, h, self.class_id, self.prob)
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let temp = self.to_xyxyn(w, h);
        return temp.to_xywhn(w, h);
    }

    fn jsonify(&self) -> String {
        format!(
            "{{\"x1\":{},\"y1\":{},\"x2\":{},\"y2\":{},\"class_id\":{},\"prob\":{}}}",
            self.x1, self.y1, self.x2, self.y2, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYWHn {
    fn new(x: f32, y: f32, w: f32, h: f32, class_id: usize, prob: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            class_id,
            prob,
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

    fn get_class_id(&self) -> usize {
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
        XYXYn::new(x1, y1, x2, y2, self.class_id, self.prob)
    }

    fn to_xyxy(&self, w: Option<f32>, h: Option<f32>) -> XYXY {
        let temp = self.to_xyxyn(w, h);
        temp.to_xyxy(w, h)
    }

    fn to_xywhn(&self, _w: Option<f32>, _h: Option<f32>) -> XYWHn {
        return *self;
    }

    fn to_xywh(&self, w: Option<f32>, h: Option<f32>) -> XYWH {
        let w = w.unwrap();
        let h = h.unwrap();

        XYWH {
            x: self.x * w,
            y: self.y * h,
            w: self.w * w,
            h: self.h * h,
            class_id: self.class_id,
            prob: self.prob,
        }
    }

    fn jsonify(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"class_id\":{},\"prob\":{}}}",
            self.x, self.y, self.w, self.h, self.class_id, self.prob
        )
    }
}

impl BoundingBoxTrait for XYWH {
    fn new(x: f32, y: f32, w: f32, h: f32, class_id: usize, prob: f32) -> Self {
        Self {
            x,
            y,
            w,
            h,
            class_id,
            prob,
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

    fn get_class_id(&self) -> usize {
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
        XYXY::new(x1, y1, x2, y2, self.class_id, self.prob)
    }

    fn to_xywhn(&self, w: Option<f32>, h: Option<f32>) -> XYWHn {
        let w = w.unwrap();
        let h = h.unwrap();

        XYWHn {
            x: self.x / w,
            y: self.y / h,
            w: self.w / w,
            h: self.h / h,
            class_id: self.class_id,
            prob: self.prob,
        }
    }

    fn to_xywh(&self, _w: Option<f32>, _h: Option<f32>) -> XYWH {
        return *self;
    }

    fn jsonify(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"class_id\":{},\"prob\":{}}}",
            self.x, self.y, self.w, self.h, self.class_id, self.prob
        )
    }
}

pub fn nms<T: BoundingBoxTrait>(mut boxes: Vec<T>, iou_threshold: f32) -> Vec<T> {
    boxes.sort_by(|a, b| {
        b.get_prob()
            .partial_cmp(&a.get_prob())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();

    while let Some(current) = boxes.first() {
        let current = current.clone();
        keep.push(current);

        boxes = boxes
            .into_iter()
            .skip(1)
            .filter(|b| {
                b.get_class_id() != current.get_class_id() || b.iou(&current) <= iou_threshold
            })
            .collect();
    }

    keep
}

// AI model for Image Processing
#[derive(Deserialize, Debug, Clone)]
pub struct AImodel {
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

impl AImodel {
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
        AImodel::new(
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

pub struct ImgPred {
    pub file_path: String,
    pub list_bbox: Vec<XYXY>,
}
