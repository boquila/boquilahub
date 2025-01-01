#![allow(dead_code)]

// Big TODO: class_ids should be string
// and this string should be defined right after the inference

/// Probabilities in the YOLO format
/// `classes` is a Vec with the names for each classification
/// `probs` is a Vec with the probabilities/confidence for each classification
struct ProbSpace {
    pub classes: Vec<String>,
    pub probs: Vec<f32>,
}

/// Segmentation in the YOLO format, normalized
/// # Fields
/// - `vertices` represents a polygon
struct SEGn {
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
    fn get_coords(&self) -> (f32,f32,f32,f32);
    fn get_prob(&self) -> f32;
    fn get_class_id(&self) -> usize;
    fn check(&self) -> bool;
    // fn transform(&self) -> BoundingBox;
    // fn n(&self) -> BoundingBox;
}

// This should be the final implementation
// struct PredImg<T: BoundingBox, const N: usize> {
//     path: String,
//     items: Vec<T>,
// }
// NMS

enum BoundingBox {
    XYXY(XYXY),
    XYXYn(XYXYn),
    XYWH(XYWH),
    XYWHn(XYWHn),
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

impl BoundingBox {
    fn to_xyxy(&self) -> XYXY {
        match self {
            BoundingBox::XYXY(b) => *b,
            BoundingBox::XYXYn(b) => XYXY {
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWH(b) => XYXY {
                x1: b.x - b.w / 2.0,
                y1: b.y - b.h / 2.0,
                x2: b.x + b.w / 2.0,
                y2: b.y + b.h / 2.0,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWHn(b) => XYXY {
                x1: b.x - b.w / 2.0,
                y1: b.y - b.h / 2.0,
                x2: b.x + b.w / 2.0,
                y2: b.y + b.h / 2.0,
                class_id: b.class_id,
                prob: b.prob,
            },
        }
    }

    fn to_xyxyn(&self) -> XYXYn {
        match self {
            BoundingBox::XYXY(b) => XYXYn {
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYXYn(b) => *b,
            BoundingBox::XYWH(b) => XYXYn {
                x1: b.x - b.w / 2.0,
                y1: b.y - b.h / 2.0,
                x2: b.x + b.w / 2.0,
                y2: b.y + b.h / 2.0,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWHn(b) => XYXYn {
                x1: b.x - b.w / 2.0,
                y1: b.y - b.h / 2.0,
                x2: b.x + b.w / 2.0,
                y2: b.y + b.h / 2.0,
                class_id: b.class_id,
                prob: b.prob,
            },
        }
    }

    fn to_xywh(&self) -> XYWH {
        match self {
            BoundingBox::XYXY(b) => XYWH {
                x: (b.x1 + b.x2) / 2.0,
                y: (b.y1 + b.y2) / 2.0,
                w: b.x2 - b.x1,
                h: b.y2 - b.y1,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYXYn(b) => XYWH {
                x: (b.x1 + b.x2) / 2.0,
                y: (b.y1 + b.y2) / 2.0,
                w: b.x2 - b.x1,
                h: b.y2 - b.y1,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWH(b) => *b,
            BoundingBox::XYWHn(b) => XYWH {
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
                class_id: b.class_id,
                prob: b.prob,
            },
        }
    }

    fn to_xywhn(&self) -> XYWHn {
        match self {
            BoundingBox::XYXY(b) => XYWHn {
                x: (b.x1 + b.x2) / 2.0,
                y: (b.y1 + b.y2) / 2.0,
                w: b.x2 - b.x1,
                h: b.y2 - b.y1,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYXYn(b) => XYWHn {
                x: (b.x1 + b.x2) / 2.0,
                y: (b.y1 + b.y2) / 2.0,
                w: b.x2 - b.x1,
                h: b.y2 - b.y1,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWH(b) => XYWHn {
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
                class_id: b.class_id,
                prob: b.prob,
            },
            BoundingBox::XYWHn(b) => *b,
        }
    }
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

    fn get_coords(&self) -> (f32,f32,f32,f32) {
        (self.x1,self.y1,self.x2,self.y2)
    }

    // fn transform(&self) -> BoundingBox {
    //     let x = (self.x1 + self.x2) / 2.0;
    //     let y = (self.y1 + self.y2) / 2.0;
    //     let w = self.x2 - self.x1;
    //     let h = self.y2 - self.y1;
    //     BoundingBox::XYWHn(XYWHn::new(x, y, w, h, self.class_id, self.prob))
    // }

    // fn n(&self) -> BoundingBox {
    //     todo!()
    // }
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

    fn get_coords(&self) -> (f32,f32,f32,f32) {
        (self.x1,self.y1,self.x2,self.y2)
    }

    // fn transform(&self) -> BoundingBox {
    //     let x = (self.x1 + self.x2) / 2.0;
    //     let y = (self.y1 + self.y2) / 2.0;
    //     let w = self.x2 - self.x1;
    //     let h = self.y2 - self.y1;
    //     BoundingBox::XYWH(XYWH::new(x, y, w, h, self.class_id, self.prob))
    // }

    // fn n(&self) -> BoundingBox {
    //     todo!()
    // }
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

    fn get_coords(&self) -> (f32,f32,f32,f32) {
        (self.x,self.y,self.w,self.h)
    }

    // fn transform(&self) -> BoundingBox {
    //     let x1 = self.x - self.w / 2.0;
    //     let y1 = self.y - self.h / 2.0;
    //     let x2 = self.x + self.w / 2.0;
    //     let y2 = self.y + self.h / 2.0;
    //     BoundingBox::XYXYn(XYXYn::new(x1, y1, x2, y2, self.class_id, self.prob))
    // }

    // fn n(&self) -> BoundingBox {
    //     todo!()
    // }
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

    fn get_coords(&self) -> (f32,f32,f32,f32) {
        (self.x,self.y,self.w,self.h)
    }

    // fn transform(&self) -> BoundingBox {
    //     let x1 = self.x - self.w / 2.0;
    //     let y1 = self.y - self.h / 2.0;
    //     let x2 = self.x + self.w / 2.0;
    //     let y2 = self.y + self.h / 2.0;
    //     BoundingBox::XYXY(XYXY::new(x1, y1, x2, y2, self.class_id, self.prob))
    // }

    // fn n(&self) -> BoundingBox {
    //     todo!()
    // }
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

// AI model for Object Detection
struct AImodel {
    pub name: String,
    pub version: f32, // complement tothe name
    pub input_width: u32,
    pub input_height: u32,
    pub description: String, // complement to the name
    pub color_code: String, // "terra", "fire", "green", depending on this, the app will show different colors hehe
    pub task: String, // "detect", "classify", "segment"
    pub classes: Vec<String>,
}

impl AImodel {
    // Constructor for the AI struct
    pub fn new(
        name: String,
        version: f32,
        input_width: u32,
        input_height: u32,
        description: String,
        color_code: String,
        task: String,
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
            classes,
        }
    }

    // Method to get the path of the AI model
    pub fn get_path(&self) -> String {
        format!("models/{}.bq", self.name)
    }
}

pub struct ImgPred{
    pub file_path: String,
    pub list_bbox: Vec<XYXY>,
}