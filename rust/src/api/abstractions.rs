#![allow(dead_code)]

#[derive(Clone)]
pub struct ProbSpace {
    pub classes: Vec<String>,
    pub confidences: Vec<f32>,
}

pub trait BoundingBox {
    fn new(a: f32, b: f32, c: f32, d: f32, class_id: usize, prob: f32) -> Self;
    fn area(&self) -> f32;
    fn intersect(&self, other: &Self) -> f32;
    fn iou(&self, other: &Self) -> f32;
}

#[derive(Clone)]
/// Bounding box in normalized XYXY format
/// # Fields
/// - `x1` and `y1` represent the top-left corner
/// - `x2` and `y2` represent the bottom-right  corner
pub struct XYXYn {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}

impl XYXYn {
    pub fn toxywhn(&self) -> XYWHn {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        XYWHn::new(x, y, w, h, self.class_id, self.prob)
    }
}

#[derive(Clone)]
pub struct XYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}

struct PredImg<const N: usize> {
    path: String,
    items: [XYXYn; N],
}
// TODO: implement NMS here

#[derive(Clone)]
/// Bounding box in normalized XYWH format
/// # Fields
/// - `x` and `y` represent the center
/// - `w` and `h` represent width and height
pub struct XYWHn {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub prob: f32,
}

impl XYWHn {
    pub fn toxyxyn(&self) -> XYXYn {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        XYXYn::new(x1, y1, x2, y2, self.class_id, self.prob)
    }
}

#[derive(Clone)]
pub struct XYWH {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub prob: f32,
}

#[derive(Clone)]
/// Segmentation in the YOLO format, normalized
/// # Fields
/// - `vertices` represents a polygon
pub struct SEGn {
    pub vertices: Vec<f32>,
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

impl BoundingBox for XYXYn {
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
        intersect_xyxys(self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2)
    }

    fn iou(&self, other: &XYXYn) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

impl BoundingBox for XYXY {
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
        intersect_xyxys(self.x1, self.y1, self.x2, self.y2, other.x1, other.y1, other.x2, other.y2)
    }

    fn iou(&self, other: &XYXY) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

impl BoundingBox for XYWHn {
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
        let x_left = self.x.max(other.x);
        let y_top = self.y.max(other.y);
        let x_right = (self.x + self.w).min(other.x + other.w);
        let y_bottom = (self.y + self.h).min(other.y + other.h);

        (x_right - x_left) * (y_bottom - y_top)
    }

    fn iou(&self, other: &XYWHn) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

impl BoundingBox for XYWH {
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
        let x_left = self.x.max(other.x);
        let y_top = self.y.max(other.y);
        let x_right = (self.x + self.w).min(other.x + other.w);
        let y_bottom = (self.y + self.h).min(other.y + other.h);

        (x_right - x_left) * (y_bottom - y_top)
    }

    fn iou(&self, other: &XYWH) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

// TODO: intersect for xywhs
// TODO: general iou function?
