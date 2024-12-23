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
pub struct XYWHBBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub class_id: usize,
    pub probability: f32,
}

#[derive(Clone)]
pub struct Segmentation {
    pub vertices: Vec<f32>,
    pub class_id: usize,
    pub probability: f32,
}

impl XYWHBBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32, class_id: usize, probability: f32) -> Self {
        Self { x, y, w, h, class_id, probability }
    }

    pub fn toxyxy(&self) -> XYXYBBox {
        let x1 = self.x - self.w / 2.0;
        let y1 = self.y - self.h / 2.0;
        let x2 = self.x + self.w / 2.0;
        let y2 = self.y + self.h / 2.0;
        XYXYBBox::new(x1,y1,x2,y2,self.class_id,self.probability)
    }
}

impl XYXYBBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, class_id: usize, probability: f32) -> Self {
        Self { x1, y1, x2, y2, class_id, probability }
    }

    pub fn toxywh(&self) -> XYWHBBox {
        let x = (self.x1 + self.x2) / 2.0;
        let y = (self.y1 + self.y2) / 2.0;
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        XYWHBBox::new(x,y,w,h,self.class_id,self.probability)
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

