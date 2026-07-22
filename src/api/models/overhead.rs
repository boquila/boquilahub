use super::*;
use crate::api::{
    abstractions::{AIOutputs, XY, XYc},
    processing::{inference::inference, pre::imgbuf_to_dinov3_input},
};
use anyhow::{bail, Error, Result};
use image::{ImageBuffer, Rgb};
use ndarray::ArrayViewD;
use ort::{session::Session, value::ValueType};

const DEFAULT_INPUT_SIZE: u32 = 512;

/// Heatmap-based overhead point detector. Handles single-head models (one
/// heatmap, class always 0) and dual-head models (`loc` + `cls` maps, argmax
/// over class channels), auto-detected from the number of ONNX outputs. Emits
/// centroid points via `AIOutputs::PointDetection`.
pub struct Overhead {
    pub classes: Vec<String>,
    pub input_width: u32,
    pub input_height: u32,
    pub input_name: String,
    pub loc_name: String,
    pub cls_name: Option<String>,
    pub session: Session,
    pub config: ModelConfig,
}

impl Overhead {
    pub fn new(
        metadata: AIMetadata,
        session: Session,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        let (input_height, input_width) = match &session.inputs()[0].dtype() {
            ValueType::Tensor { shape: dimensions, .. } => {
                let resolve = |d: i64| if d > 0 { d as u32 } else { DEFAULT_INPUT_SIZE };
                (resolve(dimensions[2]), resolve(dimensions[3]))
            }
            _ => bail!("expected tensor input for Overhead"),
        };

        let outputs = session.outputs();
        if outputs.is_empty() {
            bail!("Overhead model has no outputs");
        }
        let loc_name = outputs[0].name().to_string();
        let cls_name = outputs.get(1).map(|o| o.name().to_string());

        Ok(Overhead {
            classes: metadata.classes,
            input_width,
            input_height,
            input_name: session.inputs()[0].name().to_string(),
            loc_name,
            cls_name,
            session,
            config,
        })
    }

    pub fn run_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> AIOutputs {
        let (img_w, img_h) = img.dimensions();

        // Resize + ImageNet-normalized NCHW.
        let input = imgbuf_to_dinov3_input(self.input_height, self.input_width, img);
        let outputs = inference(&self.session, &input, &self.input_name).unwrap();

        // loc heatmap [1, 1, H, W]; keep natural [N,C,H,W] order (no transpose).
        let loc_owned = outputs[self.loc_name.as_str()]
            .try_extract_array::<f32>()
            .unwrap()
            .into_owned();
        let loc = loc_owned.view();
        let ls = loc.shape();
        let (loc_h, loc_w) = (ls[2], ls[3]);

        let cls = self.cls_name.as_ref().map(|name| {
            outputs[name.as_str()].try_extract_array::<f32>().unwrap().into_owned()
        });

        // Detection score is the heatmap value; the frontend confidence slider
        // filters on it via `config`.
        let conf_thr = self.config.confidence_threshold;

        let scale_x = img_w as f32 / loc_w as f32;
        let scale_y = img_h as f32 / loc_h as f32;

        let mut points: Vec<XYc> = Vec::new();
        for py in 0..loc_h {
            for px in 0..loc_w {
                let score = loc[[0, 0, py, px]];
                if score < conf_thr {
                    continue;
                }
                if !is_local_maximum(&loc, py, px, loc_h, loc_w) {
                    continue;
                }

                // Channel 0 is background — argmax over species channels only.
                let class_id = match &cls {
                    Some(cls) => {
                        let cv = cls.view();
                        let cs = cv.shape();
                        let (num_classes, cls_h, cls_w) = (cs[1], cs[2], cs[3]);
                        let cy = (py * cls_h / loc_h).min(cls_h - 1);
                        let cx = (px * cls_w / loc_w).min(cls_w - 1);
                        (1..num_classes)
                            .max_by(|&a, &b| {
                                cv[[0, a, cy, cx]]
                                    .partial_cmp(&cv[[0, b, cy, cx]])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or(0) as u32
                    }
                    None => 0,
                };

                let x = px as f32 * scale_x;
                let y = py as f32 * scale_y;
                let label = self
                    .classes
                    .get(class_id as usize)
                    .cloned()
                    .unwrap_or_else(|| class_id.to_string());
                points.push(XYc::new(XY::new(x, y, score, class_id), label));
            }
        }

        AIOutputs::PointDetection(points)
    }
}

/// 8-connected 3×3 local-maximum test. Plateau tie-break (`>=` toward
/// south/east, `>` toward north/west) keeps only the bottom-right pixel of a
/// flat maximum, so a plateau fires exactly once.
fn is_local_maximum(loc: &ArrayViewD<'_, f32>, py: usize, px: usize, h: usize, w: usize) -> bool {
    let val = loc[[0, 0, py, px]];
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let ny = py as i32 + dy;
            let nx = px as i32 + dx;
            if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                continue;
            }
            let neighbor = loc[[0, 0, ny as usize, nx as usize]];
            let is_south_east = dy > 0 || (dy == 0 && dx > 0);
            if is_south_east {
                if neighbor >= val {
                    return false;
                }
            } else if neighbor > val {
                return false;
            }
        }
    }
    true
}
