use crate::api::abstractions::XYXY;
use image::{
    imageops::{resize, FilterType},
    ImageBuffer, Rgb,
};
use ndarray::{Array, Ix4};

const SCALE: f32 = 1.0 / 255.0;

pub enum TensorFormat {
    NCHW, // Batch, Channel, Height, Width
    NHWC, // Batch, Height, Width, Channel
}

pub fn imgbuf_to_input_array(
    batch_size: usize,
    input_depth: usize,
    input_height: u32,
    input_width: u32,
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    format: TensorFormat,
) -> (Array<f32, Ix4>, u32, u32) {
    let (img_width, img_height) = img.dimensions();
    let resized = resize(img, input_width, input_height, FilterType::Nearest);

    let (h, w) = (input_height as usize, input_width as usize);
    let mut input = match format {
        TensorFormat::NCHW => Array::zeros((batch_size, input_depth, h, w)),
        TensorFormat::NHWC => Array::zeros((batch_size, h, w, input_depth)),
    };

    for (x, y, pixel) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        let [r, g, b, ..] = pixel.0;
        let rgb: [f32; 3] = [r as f32 * SCALE, g as f32 * SCALE, b as f32 * SCALE];

        match format {
            TensorFormat::NCHW => {
                input[[0, 0, y, x]] = rgb[0];
                input[[0, 1, y, x]] = rgb[1];
                input[[0, 2, y, x]] = rgb[2];
            }
            TensorFormat::NHWC => {
                input[[0, y, x, 0]] = rgb[0];
                input[[0, y, x, 1]] = rgb[1];
                input[[0, y, x, 2]] = rgb[2];
            }
        }
    }
    (input, img_width, img_height)
}

pub fn slice_image(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    bbox: &XYXY,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (img_width, img_height) = img.dimensions();

    let x1 = (bbox.x1.max(0.0) as u32).min(img_width);
    let y1 = (bbox.y1.max(0.0) as u32).min(img_height);
    let x2 = (bbox.x2.max(0.0) as u32).min(img_width);
    let y2 = (bbox.y2.max(0.0) as u32).min(img_height);

    let width = x2 - x1;
    let height = y2 - y1;

    let mut sliced = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x1 + x, y1 + y);
            sliced.put_pixel(x, y, *pixel);
        }
    }

    sliced
}
