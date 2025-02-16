use ab_glyph::FontRef;
use image::Rgb;
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use super::abstractions::BBox;

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

pub fn draw_bbox_from_file_path(file_path: &str, predictions: &Vec<BBox>) -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
    let buf = std::fs::read(file_path).unwrap();
    let img = draw_bbox_from_buf(&buf,predictions);
    return img
}

// #[flutter_rust_bridge::frb(sync)]
// pub fn draw_bbox(file_path: &str, predictions: &Vec<BBox>) -> Vec<u8> {
//     let image_buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
//         draw_bbox_from_file_path(file_path, predictions);
//     let mut jpeg_data = Vec::new();
//     let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_data, 95);
//     encoder.encode_image(&image_buffer).unwrap();
//     return jpeg_data;
// }

const FONT_SCALE: f32 = 18.4;
const LABEL_PADDING: f32 = 3.0;
const CHAR_WIDTH: usize = 10;
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const FONT_BYTES: &[u8] = include_bytes!("../../../assets//DejaVuSans.ttf");

fn draw_bbox_from_buf(buf: &[u8], predictions: &Vec<BBox>) -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::load_from_memory(buf).unwrap().to_rgb8();  
    let font: FontRef<'_> = FontRef::try_from_slice(FONT_BYTES).unwrap();    
    
    for bbox in predictions {        
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;
        let color = BBOX_COLORS[bbox.class_id as usize];
        let text = bbox.strlabel();

        draw_hollow_rect_mut(
            &mut img,
            Rect::at(bbox.x1 as i32, bbox.y1 as i32).of_size(w as u32, h as u32),
            color,
        );
        draw_filled_rect_mut(
            &mut img,
            Rect::at(bbox.x1 as i32, (bbox.y1 - FONT_SCALE + LABEL_PADDING) as i32)
                .of_size((text.len() * CHAR_WIDTH) as u32, FONT_SCALE as u32 + 4),
            color,
        );
        draw_text_mut(
            &mut img,
            WHITE,
            bbox.x1 as i32,
            (bbox.y1 - FONT_SCALE + LABEL_PADDING) as i32,
            FONT_SCALE,
            &font,
            &text,
        );
    }
    return img
}