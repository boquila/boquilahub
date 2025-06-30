use crate::api::abstractions::SEGc;
use crate::api::models::processing::inference::AIOutputs;
use std::sync::LazyLock;

use super::abstractions::XYXYc;
use ab_glyph::FontRef;
use image::{ImageBuffer, Rgb};
use imageproc::drawing::{
    draw_filled_rect_mut, draw_hollow_polygon, draw_hollow_rect_mut, draw_text_mut,
};
use imageproc::point::Point;
use imageproc::rect::Rect;

const BBOX_COLORS: [Rgb<u8>; 90] = [
    Rgb([255, 0, 0]),     // Red
    Rgb([103, 58, 183]),  // Deep Purple
    Rgb([3, 169, 244]),   // Light Blue Accent
    Rgb([139, 195, 74]),  // Light Green
    Rgb([205, 220, 57]),  // Lime
    Rgb([255, 152, 0]),   // Orange
    Rgb([255, 193, 7]),   // Amber
    Rgb([174, 0, 255]),   // Purple Accent
    Rgb([33, 150, 243]),  // Blue
    Rgb([255, 87, 34]),   // Deep Orange
    Rgb([156, 39, 176]),  // Purple
    Rgb([255, 235, 59]),  // Yellow
    Rgb([0, 188, 212]),   // Cyan
    Rgb([121, 85, 72]),   // Brown
    Rgb([255, 64, 129]),  // Pink Accent
    Rgb([83, 109, 254]),  // Indigo Accent
    Rgb([0, 150, 136]),   // Teal
    Rgb([233, 30, 99]),   // Pink
    Rgb([63, 81, 181]),   // Indigo
    Rgb([128, 169, 179]), // Blue Gray
    Rgb([153, 102, 153]), // Dark Lilac
    Rgb([85, 107, 47]),   // Dark Olive Green
    Rgb([240, 230, 140]), // Khaki
    Rgb([210, 180, 140]), // Tan
    Rgb([219, 112, 147]), // Dusty Rose
    Rgb([255, 218, 185]), // Peach
    Rgb([139, 117, 85]),  // Rosy Brown
    Rgb([255, 160, 122]), // Light Salmon
    Rgb([60, 179, 113]),  // Medium Sea Green
    Rgb([128, 0, 128]),   // Purple
    Rgb([107, 142, 35]),  // Olive Drab
    Rgb([70, 130, 180]),  // Steel Blue
    Rgb([255, 182, 193]), // Light Pink
    Rgb([205, 133, 63]),  // Peru
    Rgb([107, 142, 35]),  // Olive Drab
    Rgb([143, 188, 143]), // Dark Sea Green
    Rgb([255, 20, 147]),  // Deep Pink
    Rgb([255, 105, 180]), // Hot Pink
    Rgb([218, 112, 214]), // Orchid
    Rgb([128, 0, 0]),     // Maroon
    Rgb([178, 34, 34]),   // Fire Brick
    Rgb([160, 32, 240]),  // Purple
    Rgb([199, 21, 133]),  // Medium Violet Red
    Rgb([255, 228, 196]), // Bisque
    Rgb([176, 224, 230]), // Powder Blue
    Rgb([221, 160, 221]), // Plum
    Rgb([255, 51, 102]),  // Cerise
    Rgb([204, 102, 153]), // Pale Magenta
    Rgb([153, 204, 255]), // Sky Blue
    Rgb([102, 204, 204]), // Aquamarine
    Rgb([255, 153, 153]), // Salmon
    Rgb([204, 153, 255]), // Lavender
    Rgb([255, 204, 51]),  // Mustard
    Rgb([153, 102, 102]), // Brick
    Rgb([102, 153, 153]), // Teal
    Rgb([255, 102, 102]), // Coral
    Rgb([102, 204, 102]), // Lime
    Rgb([204, 102, 102]), // Terracotta
    Rgb([51, 153, 102]),  // Viridian
    Rgb([204, 102, 255]), // Orchid
    Rgb([153, 255, 153]), // Pale Green
    Rgb([255, 153, 204]), // Blush
    Rgb([255, 102, 204]), // Fuchsia
    Rgb([153, 102, 204]), // Indigo
    Rgb([102, 255, 255]), // Turquoise
    Rgb([204, 102, 153]), // Mauve
    Rgb([102, 255, 102]), // Spring Green
    Rgb([255, 153, 102]), // Tangerine
    Rgb([102, 153, 102]), // Olive
    Rgb([255, 204, 153]), // Apricot
    Rgb([102, 153, 204]), // Cornflower
    Rgb([204, 153, 102]), // Copper
    Rgb([153, 204, 102]), // Chartreuse
    Rgb([204, 102, 204]), // Plum
    Rgb([75, 0, 130]),    // Indigo
    Rgb([64, 224, 208]),  // Turquoise
    Rgb([255, 140, 0]),   // Dark Orange
    Rgb([147, 112, 219]), // Medium Purple
    Rgb([0, 250, 154]),   // Medium Spring Green
    Rgb([255, 99, 71]),   // Tomato
    Rgb([186, 85, 211]),  // Medium Orchid
    Rgb([152, 251, 152]), // Pale Green
    Rgb([219, 112, 147]), // Pale Violet Red
    Rgb([244, 164, 96]),  // Sandy Brown
    Rgb([176, 196, 222]), // Light Steel Blue
    Rgb([255, 127, 80]),  // Coral
    Rgb([135, 206, 250]), // Light Sky Blue
    Rgb([218, 165, 32]),  // Golden Rod
    Rgb([72, 61, 139]),   // Dark Slate Blue
    Rgb([250, 128, 114]), // Salmon
];

const FONT_SCALE: f32 = 36.4;
const LABEL_PADDING: f32 = FONT_SCALE / 6.13;
const CHAR_WIDTH: f32 = FONT_SCALE / 1.84;
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const FONT_BYTES: &[u8] = include_bytes!("../../assets//DejaVuSans.ttf");
static FONT: LazyLock<FontRef<'static>> =
    LazyLock::new(|| FontRef::try_from_slice(FONT_BYTES).expect("Failed to load font"));

// only this should be public
pub fn draw_aioutput(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, predictions: &AIOutputs) {
    match predictions {
        AIOutputs::ObjectDetection(detections) => {
            draw_bbox_from_imgbuf(img, detections);
        }
        AIOutputs::Segmentation(segs) => {
            draw_seg_from_imgbuf(img, segs);
        }
        AIOutputs::Classification(prob_space) => {
            todo!()
        }
    }
}

fn draw_bbox_from_imgbuf(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, detections: &Vec<XYXYc>) {
    let font = &*FONT; // Dereference the LazyLock

    for bbox in detections {
        let w = bbox.bbox.x2 - bbox.bbox.x1;
        let h = bbox.bbox.y2 - bbox.bbox.y1;
        let color = BBOX_COLORS[bbox.bbox.class_id as usize];
        let text = str_label(&bbox.label, bbox.bbox.prob);

        draw_hollow_rect_mut(
            img,
            Rect::at(bbox.bbox.x1 as i32, bbox.bbox.y1 as i32).of_size(w as u32, h as u32),
            color,
        );
        draw_filled_rect_mut(
            img,
            Rect::at(
                bbox.bbox.x1 as i32,
                (bbox.bbox.y1 - FONT_SCALE + LABEL_PADDING) as i32,
            )
            .of_size(
                (text.len() as f32 * CHAR_WIDTH) as u32,
                FONT_SCALE as u32 + 4,
            ),
            color,
        );
        draw_text_mut(
            img,
            WHITE,
            bbox.bbox.x1 as i32,
            (bbox.bbox.y1 - FONT_SCALE + LABEL_PADDING) as i32,
            FONT_SCALE,
            &font,
            &text,
        );
    }
}

fn draw_seg_from_imgbuf(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, segmentations: &Vec<SEGc>) {
    let font = &*FONT; // Dereference the LazyLock

    for seg in segmentations {
        let color = BBOX_COLORS[seg.seg.class_id as usize];
        let text = str_label(&seg.label, seg.seg.prob);

        let poly: Vec<Point<f32>> = seg
            .seg
            .x
            .iter()
            .zip(seg.seg.y.iter())
            .map(|(&x, &y)| Point {
                x: x as f32,
                y: y as f32,
            })
            .collect();

        draw_hollow_polygon(img, &poly, color);

        // Draw label background
        let label_width = (text.len() as f32 * CHAR_WIDTH) as u32;
        let label_height = FONT_SCALE as u32 + 4;

        // Ensure label stays within image bounds
        let label_x = std::cmp::max(0, seg.bbox.x1 as i32) as i32;
        let label_y = std::cmp::max(0, seg.bbox.y1 as i32 - FONT_SCALE as i32 + LABEL_PADDING as i32) as i32;

        draw_filled_rect_mut(
            img,
            Rect::at(label_x, label_y).of_size(label_width, label_height),
            color,
        );

        // Draw label text
        draw_text_mut(img, WHITE, label_x, label_y, FONT_SCALE, &font, &text);
    }
}

fn str_label(label: &str, prob: f32) -> String {
    format!("{} {:.2}", label, prob)
}
