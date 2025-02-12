use image::{imageops::FilterType, GenericImageView};
use ndarray::{Array, Ix4};

// TODO: rewrite, could be more efficient
pub fn prepare_input(buf: &Vec<u8>,input_width: u32, input_height: u32) -> (Array<f32, Ix4>, u32, u32) {
    let img = image::load_from_memory(buf).unwrap();    
    let (img_width, img_height) = (img.width(), img.height());
    let img = img.resize_exact(input_width, input_height, FilterType::CatmullRom);

    let mut input = Array::zeros((1, 3, input_width as usize, input_height as usize));
    
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.0;
        input[[0, 1, y, x]] = (g as f32) / 255.0;
        input[[0, 2, y, x]] = (b as f32) / 255.0;
    }

    return (input, img_width, img_height);
}