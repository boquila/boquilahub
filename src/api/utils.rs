use ffmpeg_next as ffmpeg;
use image::{ImageBuffer, Rgb};
use ndarray::{Array3, ArrayBase, Dim, OwnedRepr};
use std::io::{self};
use std::path::{Path, PathBuf};

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

pub const COUNTRY_CODES: [&str; 250] = [
    "", "ABW", "AFG", "AGO", "AIA", "ALA", "ALB", "AND", "ARE", "ARG", "ARM", "ASM", "ATA", "ATF",
    "ATG", "AUS", "AUT", "AZE", "BDI", "BEL", "BEN", "BES", "BFA", "BGD", "BGR", "BHR", "BHS",
    "BIH", "BLM", "BLR", "BLZ", "BMU", "BOL", "BRA", "BRB", "BRN", "BTN", "BVT", "BWA", "CAF",
    "CAN", "CCK", "CHE", "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL", "COM", "CPV",
    "CRI", "CUB", "CUW", "CXR", "CYM", "CYP", "CZE", "DEU", "DJI", "DMA", "DNK", "DOM", "DZA",
    "ECU", "EGY", "ERI", "ESH", "ESP", "EST", "ETH", "FIN", "FJI", "FLK", "FRA", "FRO", "FSM",
    "GAB", "GBR", "GEO", "GGY", "GHA", "GIB", "GIN", "GLP", "GMB", "GNB", "GNQ", "GRC", "GRD",
    "GRL", "GTM", "GUF", "GUM", "GUY", "HKG", "HMD", "HND", "HRV", "HTI", "HUN", "IDN", "IMN",
    "IND", "IOT", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA", "JAM", "JEY", "JOR", "JPN", "KAZ",
    "KEN", "KGZ", "KHM", "KIR", "KNA", "KOR", "KWT", "LAO", "LBN", "LBR", "LBY", "LCA", "LIE",
    "LKA", "LSO", "LTU", "LUX", "LVA", "MAC", "MAF", "MAR", "MCO", "MDA", "MDG", "MDV", "MEX",
    "MHL", "MKD", "MLI", "MLT", "MMR", "MNE", "MNG", "MNP", "MOZ", "MRT", "MSR", "MTQ", "MUS",
    "MWI", "MYS", "MYT", "NAM", "NCL", "NER", "NFK", "NGA", "NIC", "NIU", "NLD", "NOR", "NPL",
    "NRU", "NZL", "OMN", "PAK", "PAN", "PCN", "PER", "PHL", "PLW", "PNG", "POL", "PRI", "PRK",
    "PRT", "PRY", "PSE", "PYF", "QAT", "REU", "ROU", "RUS", "RWA", "SAU", "SDN", "SEN", "SGP",
    "SGS", "SHN", "SJM", "SLB", "SLE", "SLV", "SMR", "SOM", "SPM", "SRB", "SSD", "STP", "SUR",
    "SVK", "SVN", "SWE", "SWZ", "SXM", "SYC", "SYR", "TCA", "TCD", "TGO", "THA", "TJK", "TKL",
    "TKM", "TLS", "TON", "TTO", "TUN", "TUR", "TUV", "TWN", "TZA", "UGA", "UKR", "UMI", "URY",
    "USA", "UZB", "VAT", "VCT", "VEN", "VGB", "VIR", "VNM", "VUT", "WLF", "WSM", "YEM", "ZAF",
    "ZMB", "ZWE",
];

pub fn extract_img(frame: &ffmpeg::frame::Video) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;

    // Create a new RGB image buffer
    let mut img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);

    // Convert the FFmpeg frame data to RGB format
    let data = frame.data(0);
    let linesize = frame.stride(0);

    // Copy data from FFmpeg frame to image buffer
    match frame.format() {
        ffmpeg::format::Pixel::RGB24 => {
            for y in 0..height {
                for x in 0..width {
                    let offset = y * linesize + x * 3;
                    let b = data[offset + 2];
                    let g = data[offset + 1];
                    let r = data[offset];
                    img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
        }
        ffmpeg::format::Pixel::YUV420P | ffmpeg::format::Pixel::YUVJ420P => {
            // For YUV formats, we need to convert from YUV to RGB
            // This requires accessing Y, U, and V planes separately
            let y_plane = frame.data(0);
            let y_stride = frame.stride(0);
            let u_plane = frame.data(1);
            let u_stride = frame.stride(1);
            let v_plane = frame.data(2);
            let v_stride = frame.stride(2);

            for y in 0..height {
                for x in 0..width {
                    let y_value = y_plane[y * y_stride + x] as f32;

                    // Subsample U and V (they are at quarter resolution in YUV420)
                    let u_x = x / 2;
                    let u_y = y / 2;
                    let v_x = x / 2;
                    let v_y = y / 2;

                    let u_value = u_plane[u_y * u_stride + u_x] as f32 - 128.0;
                    let v_value = v_plane[v_y * v_stride + v_x] as f32 - 128.0;

                    // YUV to RGB conversion
                    let r = (y_value + 1.402 * v_value).clamp(0.0, 255.0) as u8;
                    let g =
                        (y_value - 0.344136 * u_value - 0.714136 * v_value).clamp(0.0, 255.0) as u8;
                    let b = (y_value + 1.772 * u_value).clamp(0.0, 255.0) as u8;

                    img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
        }
        _ => {
            // For other formats, we would need to convert using ffmpeg's software scaler (SwsContext)
            // This is more complicated and would require additional code
        }
    }

    return img_buffer;
}
