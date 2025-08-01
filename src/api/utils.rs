use image::{ImageBuffer, Rgb};
use ndarray::{Array3, ArrayBase, Dim, OwnedRepr};
use std::path::{Path, PathBuf};
use std::io::{self};

/// Creates the predictions file path based on the input file path
/// For file 'img.jpg', creates path 'img_predictions.json'
pub fn create_predictions_file_path(input_path: &Path) -> io::Result<PathBuf> {
    let file_stem = input_path
        .file_stem()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid input path"))?;
    let parent = input_path.parent().unwrap_or(Path::new(""));
    let output_path = parent.join(format!("{}_predictions.json", file_stem.to_string_lossy()));
    Ok(output_path)
}

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
    let (height, width, _) = ndarray.dim();
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