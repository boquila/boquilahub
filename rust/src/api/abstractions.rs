#![allow(dead_code)]

#[derive(Clone)]
pub struct ProbSpace {
    pub labels: Vec<String>,
    pub confidences: Vec<f64>,
}

#[derive(Clone)]
pub struct XYXYBBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub probability: f64,
    pub label: String,
}

impl XYXYBBox {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, probability: f64, label: &str) -> Self {
        XYXYBBox {
            x1,
            y1,
            x2,
            y2,
            probability,
            label: label.to_string(),
        }
    }

    fn area(&self) -> f64 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersect(&self, other: &XYXYBBox) -> f64 {
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

    fn iou(&self, other: &XYXYBBox) -> f64 {
        let intersection = self.intersect(other);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

