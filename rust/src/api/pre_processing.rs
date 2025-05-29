use image::{
    imageops::{resize, FilterType}, open, ImageBuffer, Rgb
};
use ndarray::{Array, Ix4};

pub fn prepare_input_from_filepath(
    file_path: &str,
    input_width: u32,
    input_height: u32,
) -> (Array<f32, Ix4>, u32, u32) {
    let img = open(file_path).unwrap().into_rgb8();
    return prepare_input_from_imgbuf(&img, input_width, input_height);
}

pub fn prepare_input_from_buf(
    buf: &[u8],
    input_width: u32,
    input_height: u32,
) -> (Array<f32, Ix4>, u32, u32) {
    let img = image::load_from_memory(buf).unwrap().into_rgb8();
        
    return prepare_input_from_imgbuf(&img, input_width, input_height);
}

pub fn prepare_input_from_imgbuf(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    input_width: u32,
    input_height: u32,
) -> (Array<f32, Ix4>, u32, u32) {
    let (img_width, img_height) = (img.width(), img.height());

    let resized = resize(img, input_width, input_height, FilterType::Nearest);

    let mut input = Array::zeros((1, 3, input_height as usize, input_width as usize));

    for (x, y, pixel) in resized.enumerate_pixels() {
        let x_u = x as usize;
        let y_u = y as usize;
        input[[0, 2, y_u, x_u]] = (pixel[2] as f32) / 255.0;
        input[[0, 1, y_u, x_u]] = (pixel[1] as f32) / 255.0;
        input[[0, 0, y_u, x_u]] = (pixel[0] as f32) / 255.0;
    }

    (input, img_width, img_height)
}
