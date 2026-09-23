#!/bin/sh
# Check the executable and, after bundling, every packaged dylib.
set -eu
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <Mach-O file> [...]" >&2
    exit 1
fi
for binary do
    # Capture separately so an otool failure cannot pass as an empty match.
    dependencies=$(otool -L "$binary")
    if printf '%s\n' "$dependencies" | grep -E '(^|[[:space:]/])lib(avcodec|avdevice|avfilter|avformat|avresample|avutil|swresample|swscale|postproc)(\.[0-9]+)*\.dylib([[:space:]]|$)'; then
        echo "ERROR: $binary dynamically links FFmpeg" >&2
        exit 1
    fi
done
