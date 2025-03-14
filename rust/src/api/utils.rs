use image::{codecs::jpeg::JpegEncoder, ImageBuffer, Rgb};
use ndarray::{Array3, ArrayBase, Dim, OwnedRepr};

pub fn image_buffer_to_ndarray(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>> {
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

pub fn ndarray_to_image_buffer(
    ndarray: &ArrayBase<OwnedRepr<u8>, Dim<[usize; 3]>>,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (height,width,_) = ndarray.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let b = ndarray[[y, x, 2]];
            let g = ndarray[[y, x, 1]];
            let r = ndarray[[y, x, 0]];
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    return img;
}

pub fn image_buffer_to_jpg_buffer(image_buffer: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<u8> {
    let mut jpeg_data = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_data, 95);
    encoder.encode_image(&image_buffer).unwrap();
    return jpeg_data;
}