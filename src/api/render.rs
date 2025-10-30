use crate::api::abstractions::{AIOutputs, BitMatrix, PredImg, ProbSpace, SEGc, XYXYc};
use crate::localization::translate;
use ab_glyph::FontRef;
use image::{DynamicImage, ImageBuffer, Rgb};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::sync::LazyLock;

fn blend_pixel(base: Rgb<u8>, overlay: Rgb<u8>, alpha: f32) -> Rgb<u8> {
    let r = (1.0 - alpha) * base[0] as f32 + alpha * overlay[0] as f32;
    let g = (1.0 - alpha) * base[1] as f32 + alpha * overlay[1] as f32;
    let b = (1.0 - alpha) * base[2] as f32 + alpha * overlay[2] as f32;

    Rgb([r as u8, g as u8, b as u8])
}

const BBOX_COLORS: [Rgb<u8>; 90] = [
    Rgb([220, 20, 60]),   // Rich crimson
    Rgb([103, 58, 183]),  // Deep Purple
    Rgb([3, 169, 244]),   // Light Blue Accent
    Rgb([139, 195, 74]),  // Light Green
    Rgb([150, 160, 40]),  // Lime
    Rgb([255, 152, 0]),   // Orange
    Rgb([200, 150, 0]),   // Amber
    Rgb([174, 0, 255]),   // Purple Accent
    Rgb([33, 150, 243]),  // Blue
    Rgb([255, 87, 34]),   // Deep Orange
    Rgb([156, 39, 176]),  // Purple
    Rgb([180, 160, 20]),  // Yellow
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
    Rgb([180, 160, 90]),  // Khaki
    Rgb([210, 180, 140]), // Tan
    Rgb([219, 112, 147]), // Dusty Rose
    Rgb([200, 150, 120]), // Peach
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
    Rgb([180, 140, 120]), // Bisque
    Rgb([120, 160, 180]), // Powder Blue
    Rgb([221, 160, 221]), // Plum
    Rgb([255, 51, 102]),  // Cerise
    Rgb([204, 102, 153]), // Pale Magenta
    Rgb([153, 204, 255]), // Sky Blue
    Rgb([102, 204, 204]), // Aquamarine
    Rgb([255, 153, 153]), // Salmon
    Rgb([204, 153, 255]), // Lavender
    Rgb([180, 140, 30]),  // Mustard
    Rgb([153, 102, 102]), // Brick
    Rgb([102, 153, 153]), // Teal
    Rgb([255, 102, 102]), // Coral
    Rgb([102, 204, 102]), // Lime
    Rgb([204, 102, 102]), // Terracotta
    Rgb([51, 153, 102]),  // Viridian
    Rgb([204, 102, 255]), // Orchid
    Rgb([100, 180, 100]), // Pale Green
    Rgb([255, 153, 204]), // Blush
    Rgb([255, 102, 204]), // Fuchsia
    Rgb([153, 102, 204]), // Indigo
    Rgb([60, 180, 180]),  // Turquoise
    Rgb([204, 102, 153]), // Mauve
    Rgb([60, 180, 60]),   // Spring Green
    Rgb([255, 153, 102]), // Tangerine
    Rgb([102, 153, 102]), // Olive
    Rgb([200, 150, 100]), // Apricot
    Rgb([102, 153, 204]), // Cornflower
    Rgb([204, 153, 102]), // Copper
    Rgb([153, 204, 102]), // Chartreuse
    Rgb([204, 102, 204]), // Plum
    Rgb([75, 0, 130]),    // Indigo
    Rgb([64, 224, 208]),  // Turquoise
    Rgb([255, 140, 0]),   // Dark Orange
    Rgb([147, 112, 219]), // Medium Purple
    Rgb([0, 180, 120]),   // Medium Spring Green
    Rgb([255, 99, 71]),   // Tomato
    Rgb([186, 85, 211]),  // Medium Orchid
    Rgb([100, 180, 100]), // Pale Green
    Rgb([219, 112, 147]), // Pale Violet Red
    Rgb([244, 164, 96]),  // Sandy Brown
    Rgb([176, 196, 222]), // Light Steel Blue
    Rgb([255, 127, 80]),  // Coral
    Rgb([135, 206, 250]), // Light Sky Blue
    Rgb([218, 165, 32]),  // Golden Rod
    Rgb([72, 61, 139]),   // Dark Slate Blue
    Rgb([250, 128, 114]), // Salmon
];

const FONT_SCALE: f32 = 32.0;
const CHAR_WIDTH: f32 = FONT_SCALE / 2.55;
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
pub const FONT_BYTES: &[u8] = include_bytes!("../../assets/NotoSansSC-Regular.ttf");
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
            draw_cls_from_imgbuf(img, prob_space);
        }
    }
}

fn draw_bbox_from_imgbuf(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, detections: &[XYXYc]) {
    let font = &*FONT; // Dereference the LazyLock

    for bbox in detections {
        let w = bbox.xyxy.x2 - bbox.xyxy.x1;
        let h = bbox.xyxy.y2 - bbox.xyxy.y1;
        let color = get_color(bbox);
        let text = str_label(&bbox);

        draw_hollow_rect_mut(
            img,
            Rect::at(bbox.xyxy.x1 as i32, bbox.xyxy.y1 as i32).of_size(w as u32, h as u32),
            color,
        );

        let len = text.lines().map(|line| line.len()).max().unwrap_or(0);
        let line_count = text.lines().count().max(1) as u32;
        let rect_height = FONT_SCALE as u32 * line_count + 4;

        // Clamp the label position to stay within image bounds
        let preferred_y = bbox.xyxy.y1 as i32 - rect_height as i32;
        let label_y = preferred_y
            .max(0)
            .min(img.height() as i32 - rect_height as i32);

        draw_filled_rect_mut(
            img,
            Rect::at(bbox.xyxy.x1 as i32, label_y)
                .of_size((len as f32 * CHAR_WIDTH) as u32, rect_height),
            color,
        );
        draw_multiline_text(
            img,
            WHITE,
            bbox.xyxy.x1 as i32,
            label_y,
            FONT_SCALE,
            &font,
            &text,
        );
    }
}

fn draw_seg_from_imgbuf(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, segmentations: &[SEGc]) {
    for seg in segmentations {
        let w = (seg.bbox.xyxy.x2 - seg.bbox.xyxy.x1) as usize;
        let h = (seg.bbox.xyxy.y2 - seg.bbox.xyxy.y1) as usize;

        let color = get_color(&seg.bbox);
        let mask: &BitMatrix = &seg.mask;

        // Convert bbox float coordinates to integers safely
        let x_offset = seg.bbox.xyxy.x1.floor() as i32;
        let y_offset = seg.bbox.xyxy.y1.floor() as i32;

        // Resize mask to match actual bbox dimensions
        for y in 0..h {
            for x in 0..w {
                // Map current pixel to mask coordinates using nearest neighbor
                let mask_x = (x as f32 / w as f32 * mask.width as f32).floor() as usize;
                let mask_y = (y as f32 / h as f32 * mask.height as f32).floor() as usize;

                // Ensure mask coordinates are within bounds
                if mask_y < mask.height && mask_x < mask.width {
                    // Access bit at (mask_y, mask_x) using row-major indexing
                    let mask_index = mask_y * mask.width + mask_x;
                    if mask.data[mask_index] {
                        let img_x = x_offset + x as i32;
                        let img_y = y_offset + y as i32;

                        // Ensure coordinates are within image bounds
                        if img_x >= 0
                            && img_y >= 0
                            && (img_x as u32) < img.width()
                            && (img_y as u32) < img.height()
                        {
                            let alpha = 0.4; // 40% intensity
                            let img_pixel = img.get_pixel(img_x as u32, img_y as u32);
                            let blended = blend_pixel(*img_pixel, color, alpha);
                            img.put_pixel(img_x as u32, img_y as u32, blended);
                        }
                    }
                }
            }
        }
        draw_bbox_from_imgbuf(img, std::slice::from_ref(&seg.bbox));
    }
}

pub fn draw_no_predictions(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    lang: Option<&crate::localization::Lang>,
) {
    let font = &*FONT; // Dereference the LazyLock
    let mut text = "no predictions";
    if let Some(lang) = lang {
        text = translate(crate::localization::Key::no_predictions, lang);
    }

    let start_x = 10i32;
    let start_y = 10i32;

    // Calculate background dimensions for fallback text
    let bg_width = (text.len() as f32 * CHAR_WIDTH + 20.0) as u32;
    let bg_height = (FONT_SCALE + 18.0) as u32;

    // Draw semi-transparent background
    draw_filled_rect_mut(
        img,
        Rect::at(start_x - 5, start_y - 5).of_size(bg_width, bg_height),
        Rgb([0, 0, 0]), // Black background
    );

    // Draw border
    draw_hollow_rect_mut(
        img,
        Rect::at(start_x - 5, start_y - 5).of_size(bg_width, bg_height),
        WHITE,
    );

    // Draw fallback text
    draw_text_mut(img, WHITE, start_x, start_y, FONT_SCALE, &font, text);
    return;
}

fn draw_cls_from_imgbuf(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, prob_space: &ProbSpace) {
    let font = &*FONT; // Dereference the LazyLock

    // Create pairs of (class, prob) and sort by probability descending
    let mut class_probs: Vec<(&String, f32)> = prob_space
        .classes
        .iter()
        .zip(prob_space.probs.iter())
        .map(|(class, &prob)| (class, prob))
        .collect();

    class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top 3
    let top3 = class_probs.into_iter().take(3).collect::<Vec<_>>();

    // Starting position in top left corner
    let start_x = 10i32;
    let start_y = 10i32;
    let line_height = (FONT_SCALE + 8.0) as i32;

    // Calculate background dimensions
    let max_text_width = top3
        .iter()
        .map(|(class, prob)| format!("{}: {:.1}%", class, prob * 100.0).len())
        .max()
        .unwrap_or(0);

    let bg_width = (max_text_width as f32 * CHAR_WIDTH + 20.0) as u32;
    let bg_height = (top3.len() as i32 * line_height + 10) as u32;

    // Draw semi-transparent background
    draw_filled_rect_mut(
        img,
        Rect::at(start_x - 5, start_y - 5).of_size(bg_width, bg_height),
        Rgb([0, 0, 0]), // Black background
    );

    // Draw border
    draw_hollow_rect_mut(
        img,
        Rect::at(start_x - 5, start_y - 5).of_size(bg_width, bg_height),
        WHITE,
    );

    // Draw each classification
    for (i, (class, prob)) in top3.iter().enumerate() {
        let text = format!("{}: {:.1}%", class, prob * 100.0);
        let y_pos = start_y + (i as i32 * line_height);

        draw_text_mut(img, WHITE, start_x, y_pos, FONT_SCALE, &font, &text);
    }
}

// Modified drawing function to handle multi-line text
fn draw_multiline_text(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    color: Rgb<u8>,
    x: i32,
    y: i32,
    font_scale: f32,
    font: &FontRef<'static>,
    text: &str,
) {
    let lines: Vec<&str> = text.split('\n').collect();
    let line_height = font_scale as i32 + 2; // Add some spacing between lines

    for (i, line) in lines.iter().enumerate() {
        draw_text_mut(
            img,
            color,
            x,
            y + (i as i32 * line_height),
            font_scale,
            font,
            line,
        );
    }
}

fn str_label(xyxyc: &XYXYc) -> String {
    let base = format!("{} {:.2}", xyxyc.label, xyxyc.xyxy.prob);
    match &xyxyc.extra_cls {
        Some(extra) => {
            let (str, cls_prob, _id) = extra.highest_confidence_full();
            format!("{}\n{} {:.2}", base, str, cls_prob)
        }
        None => base,
    }
}

fn get_color(bbox: &XYXYc) -> Rgb<u8> {
    let class_id = match &bbox.extra_cls {
        Some(extra) => {
            let (_str, _cls_prob, id) = extra.highest_confidence_full();
            id
        }
        None => bbox.xyxy.class_id,
    };
    BBOX_COLORS[class_id as usize % BBOX_COLORS.len()]
}

impl PredImg {
    #[inline(always)]
    pub fn draw(&self) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        let mut img = image::open(&self.file_path).unwrap().into_rgb8();
        if self.wasprocessed && !self.aioutput.as_ref().unwrap().is_empty() {
            super::render::draw_aioutput(&mut img, &self.aioutput.as_ref().unwrap());
        }
        return DynamicImage::ImageRgb8(img).to_rgba8();
    }
}
