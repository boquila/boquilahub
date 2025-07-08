use image::{
    imageops::{resize, FilterType},
    ImageBuffer, Rgb,
};
use ndarray::{Array, Ix4};

pub fn imgbuf_to_input_array(
    batch_size: usize,
    input_depth: usize,
    input_height: u32,
    input_width: u32,
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> (Array<f32, Ix4>, u32, u32) {
    let (img_width, img_height) = (img.width(), img.height());

    let resized = resize(img, input_width, input_height, FilterType::Nearest);

    let mut input = Array::zeros((
        batch_size,
        input_depth,
        input_height as usize,
        input_width as usize,
    ));

    for (x, y, pixel) in resized.enumerate_pixels() {
        let x_u = x as usize;
        let y_u = y as usize;
        input[[0, 2, y_u, x_u]] = (pixel[2] as f32) / 255.0;
        input[[0, 1, y_u, x_u]] = (pixel[1] as f32) / 255.0;
        input[[0, 0, y_u, x_u]] = (pixel[0] as f32) / 255.0;
    }

    (input, img_width, img_height)
}

pub fn imgbuf_to_input_array_nhwc(
    batch_size: usize,
    input_depth: usize,
    input_height: u32,
    input_width: u32,
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> Array<f32, Ix4>{
    let resized = resize(img, input_width, input_height, FilterType::Nearest);

    let mut input = Array::zeros((
        batch_size,
        input_height as usize,
        input_width as usize,
        input_depth,
    ));

    for (x, y, pixel) in resized.enumerate_pixels() {
        let x_u = x as usize;
        let y_u = y as usize;
        input[[0, y_u, x_u, 2]] = (pixel[2] as f32) / 255.0; // Blue
        input[[0, y_u, x_u, 1]] = (pixel[1] as f32) / 255.0; // Green
        input[[0, y_u, x_u, 0]] = (pixel[0] as f32) / 255.0; // Red
    }

    return input
}