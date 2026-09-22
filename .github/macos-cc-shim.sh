#!/bin/sh
# macOS linker shim.
#
# ffmpeg-sys-next adds the GNU-only flag `-Wl,--no-as-needed` whenever it links
# FFmpeg through the FFMPEG_DIR env var (which .cargo/config.toml sets on every
# platform). Apple's `ld` rejects that flag, so a plain macOS link fails.
#
# Set as the linker on macOS only (via CARGO_TARGET_*_APPLE_DARWIN_LINKER), this
# strips that flag — whether rustc passes it directly or inside an
# @response-file — and forwards everything else to the real compiler/linker.
# Dropping it is safe: --no-as-needed only changes whether *unused* shared libs
# are recorded, and the FFmpeg libs here are used.
#
# ffmpeg-sys-next 8.1 also unconditionally links QTKit and
# VideoDecodeAcceleration for static Apple builds. Modern SDKs no longer ship
# these frameworks, and FFmpeg 8 does not use them. Keep all other frameworks.
set -eu
response_dir=''
trap 'if [ -n "$response_dir" ]; then rm -f "$response_dir"/*.rsp; rmdir "$response_dir"; fi' EXIT
flag='-Wl,--no-as-needed'
n=$#
while [ "$n" -gt 0 ]; do
    a="$1"; shift; n=$((n - 1))
    case "$a" in
        "$flag") ;;                                  # drop when passed directly
        -framework)
            if [ "$n" -gt 0 ]; then
                framework="$1"; shift; n=$((n - 1))
                case "$framework" in
                    QTKit|VideoDecodeAcceleration) ;;
                    *) set -- "$@" "$a" "$framework" ;;
                esac
            else
                set -- "$@" "$a"
            fi ;;
        @*)                                          # rustc linker response file
            if [ -z "$response_dir" ]; then response_dir="$(mktemp -d)"; fi
            tmp="$response_dir/args-$n.rsp"
            # rustc writes one argument per line, possibly double-quoted.
            # Compare unquoted flag names, but preserve retained lines verbatim.
            awk '
                {
                    arg = $0
                    sub(/^"/, "", arg)
                    sub(/"$/, "", arg)
                    if (pending) {
                        pending = 0
                        if (arg == "QTKit" || arg == "VideoDecodeAcceleration") next
                        print framework
                        print $0
                        next
                    }
                    if (arg == "-framework") {
                        framework = $0
                        pending = 1
                        next
                    }
                    if (arg != "-Wl,--no-as-needed") print $0
                }
                END { if (pending) print framework }
            ' "${a#@}" > "$tmp"
            set -- "$@" "@$tmp" ;;
        *) set -- "$@" "$a" ;;
    esac
done
/usr/bin/cc "$@"
