#![allow(dead_code)]

#[derive(Clone)]
pub struct ProbSpace {
    pub classes: Vec<String>,
    pub confidences: Vec<f32>,
}

#[derive(Clone)]
pub struct XYXYBBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub probability: f32,
}

#[derive(Clone)]
pub struct Segmentation {
    pub vertices: Vec<f32>,
    pub class_id: usize,
    pub probability: f32,
}

impl XYXYBBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, class_id: usize, probability: f32) -> Self {
        XYXYBBox {
            x1,
            y1,
            x2,
            y2,
            class_id,
            probability
        }
    }

    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXYBBox) -> f32 {
        let x_left = self.x1.max(other.x1);
        let y_top = self.y1.max(other.y1);
        let x_right = self.x2.min(other.x2);
        let y_bottom = self.y2.min(other.y2);

        if x_right < x_left || y_bottom < y_top {
            0.0
        } else {
            (x_right - x_left) * (y_bottom - y_top)
        }
    }

    fn iou(&self, other: &XYXYBBox) -> f32 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

