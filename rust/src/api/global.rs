#[flutter_rust_bridge::frb(ignore)]
pub const IMAGE_FORMATS: [&'static str; 15] = [
    "avif",
    "bmp",
    "dds",
    "ff", // Farbfeld commonly uses .ff extension
    "gif",
    "hdr",
    "ico",
    "jpg", // JPEG commonly uses .jpg extension
    "exr",
    "png",
    "pnm", // PNM can also be pbm, pgm, or ppm
    "qoi",
    "tga",
    "tiff", // Sometimes also .tif
    "webp"
];

#[flutter_rust_bridge::frb(ignore)]
pub const VIDEO_FORMATS: [&'static str; 25] = [
    "mp4",     // MPEG-4 Part 14
    "mkv",     // Matroska
    "avi",     // Audio Video Interleave
    "mov",     // QuickTime
    "wmv",     // Windows Media Video
    "flv",     // Flash Video
    "webm",    // WebM
    "mpg",     // MPEG-1
    "mpeg",    // MPEG-1/2
    "m4v",     // MPEG-4 Part 14 video
    "3gp",     // 3GPP multimedia
    "ts",      // MPEG Transport Stream
    "mxf",     // Material Exchange Format
    "vob",     // DVD Video Object
    "asf",     // Advanced Systems Format
    "rm",      // RealMedia
    "rmvb",    // RealMedia Variable Bitrate
    "ogv",     // Ogg Video
    "m2ts",    // Blu-ray BDAV
    "mts",     // AVCHD
    "divx",    // DivX
    "f4v",     // Flash MP4 Video
    "m2v",     // MPEG-2 Video
    "swf",     // Small Web Format (Flash)
    "wtv"      // Windows Recorded TV Show
];