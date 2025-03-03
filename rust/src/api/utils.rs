use image::{ImageBuffer, Rgb};
use ndarray::{Array3, ArrayBase, Dim, OwnedRepr};

pub fn image_buffer_to_ndarray(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>> {
    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;
    
    // Create a new 3D array with dimensions [height, width, 3]
    let mut array = Array3::<u8>::zeros((height, width, 3));
    
    // Fill the array with pixel data from the image
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x as u32, y as u32);
            array[[y, x, 2]] = pixel[2]; // B
            array[[y, x, 1]] = pixel[1]; // G
            array[[y, x, 0]] = pixel[0]; // R            
        }
    }
    
    array
}