#!/usr/bin/env bash
# Verify that the packaging guard rejects dynamic FFmpeg and tool failures.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
test_dir=$(mktemp -d)
trap 'rm -f "$test_dir"/*; rmdir "$test_dir"' EXIT
export PATH="$test_dir:$PATH"
export OTOOL_FIXTURE="$test_dir/dependencies"
cat > "$test_dir/otool" <<'MOCK'
#!/bin/sh
test "$1" = -L || exit 2
cat "$OTOOL_FIXTURE"
exit "${OTOOL_STATUS:-0}"
MOCK
chmod +x "$test_dir/otool"
check="$repo/.github/assert-macos-ffmpeg-static.sh"

printf '%s\n' 'boquilahub:' \
    '    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0)' \
    '    @executable_path/libx264.164.dylib (compatibility version 0.0.0)' \
    '    @executable_path/libonnxruntime.1.26.0.dylib (compatibility version 1.0.0)' \
    > "$OTOOL_FIXTURE"
sh "$check" boquilahub libx264.164.dylib

for lib in avcodec avdevice avfilter avformat avresample avutil swresample swscale postproc; do
    for prefix in /opt/homebrew/lib/ @rpath/ @executable_path/ ''; do
        for version in '' .62 .62.1.0; do
            printf '%s\n' 'boquilahub:' \
                "    ${prefix}lib${lib}${version}.dylib (compatibility version 1.0.0)" \
                > "$OTOOL_FIXTURE"
            if sh "$check" boquilahub > /dev/null 2>&1; then
                echo "ERROR: accepted dynamic dependency: ${prefix}lib${lib}${version}.dylib" >&2
                exit 1
            fi
        done
    done
done

# A failed inspection must not be mistaken for absence of FFmpeg dependencies.
printf '%s\n' 'boquilahub:' > "$OTOOL_FIXTURE"
if OTOOL_STATUS=1 sh "$check" boquilahub > /dev/null 2>&1; then
    echo 'ERROR: accepted failed otool inspection' >&2
    exit 1
fi
if sh "$check" > /dev/null 2>&1; then
    echo 'ERROR: accepted an empty file list' >&2
    exit 1
fi
echo 'macOS FFmpeg dependency guard tests passed'
