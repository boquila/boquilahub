#!/bin/sh
# macOS linker shim.
#
# ffmpeg-sys-next adds the GNU-only flag `-Wl,--no-as-needed` whenever it links
# FFmpeg through the FFMPEG_DIR env var (which .cargo/config.toml sets on every
# platform). Apple's `ld` rejects that flag, so a plain macOS link fails.
#
# Set as the linker on macOS only (via CARGO_TARGET_*_APPLE_DARWIN_LINKER), this
# strips that one flag — whether rustc passes it directly or inside an
# @response-file — and forwards everything else to the real compiler/linker.
# Dropping it is safe: --no-as-needed only changes whether *unused* shared libs
# are recorded, and the FFmpeg libs here are used.
flag='-Wl,--no-as-needed'
n=$#
while [ "$n" -gt 0 ]; do
    a="$1"; shift; n=$((n - 1))
    case "$a" in
        "$flag") ;;                                  # drop when passed directly
        @*)                                          # rustc linker response file
            tmp="$(mktemp)"
            grep -vFx "$flag" "${a#@}" > "$tmp"
            set -- "$@" "@$tmp" ;;
        *) set -- "$@" "$a" ;;
    esac
done
exec /usr/bin/cc "$@"
