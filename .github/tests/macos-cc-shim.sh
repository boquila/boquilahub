#!/usr/bin/env bash
# Exercise argument filtering without requiring Apple's compiler or SDK.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
test_dir=$(mktemp -d)
trap 'rm -f "$test_dir"/*; rmdir "$test_dir"' EXIT
export TEST_CC="$test_dir/cc.sh"
export TMPDIR="$test_dir"

# Replace only the final compiler invocation; run the actual shim logic.
sed 's|^/usr/bin/cc "\$@"$|sh "$TEST_CC" "$@"|' \
    "$repo/.github/macos-cc-shim.sh" > "$test_dir/shim.sh"
cat > "$TEST_CC" <<'MOCK'
#!/bin/sh
for arg do
    case "$arg" in
        @*) cat "${arg#@}" ;;
        *) printf '%s\n' "$arg" ;;
    esac
done
exit "${TEST_CC_STATUS:-0}"
MOCK

# Ordinary shared builds: preserve argument boundaries and all useful flags.
sh "$test_dir/shim.sh" 'object with spaces.o' -Wl,--no-as-needed \
    -framework CoreFoundation -lavcodec -o 'output with spaces' > "$test_dir/actual"
printf '%s\n' 'object with spaces.o' -framework CoreFoundation -lavcodec \
    -o 'output with spaces' > "$test_dir/expected"
diff -u "$test_dir/expected" "$test_dir/actual"

# Static builds: strip only the obsolete framework pairs, wherever they occur.
sh "$test_dir/shim.sh" -framework QTKit -framework CoreMedia \
    -framework VideoDecodeAcceleration -framework VideoToolbox \
    -Wl,--no-as-needed -lx264 > "$test_dir/actual"
printf '%s\n' -framework CoreMedia -framework VideoToolbox -lx264 > "$test_dir/expected"
diff -u "$test_dir/expected" "$test_dir/actual"

# Response files: preserve quoted paths, valid frameworks, and multiple files.
printf '%s\n' '"object with spaces.o"' '"-framework"' '"QTKit"' \
    -framework CoreVideo '"-Wl,--no-as-needed"' > "$test_dir/first.rsp"
printf '%s\n' -framework VideoDecodeAcceleration -Wl,--no-as-needed \
    '"-framework"' '"VideoToolbox"' -lx264 > "$test_dir/second.rsp"
sh "$test_dir/shim.sh" "@$test_dir/first.rsp" -o 'output with spaces' \
    "@$test_dir/second.rsp" > "$test_dir/actual"
printf '%s\n' '"object with spaces.o"' -framework CoreVideo -o \
    'output with spaces' '"-framework"' '"VideoToolbox"' -lx264 > "$test_dir/expected"
diff -u "$test_dir/expected" "$test_dir/actual"

# An entirely filtered response file is valid, and compiler failures propagate.
printf '%s\n' -Wl,--no-as-needed -framework QTKit > "$test_dir/empty.rsp"
sh "$test_dir/shim.sh" "@$test_dir/empty.rsp" > "$test_dir/actual"
test ! -s "$test_dir/actual"
status=0
TEST_CC_STATUS=42 sh "$test_dir/shim.sh" "@$test_dir/first.rsp" > /dev/null || status=$?
test "$status" -eq 42

# Missing response files must fail before invoking the compiler.
if sh "$test_dir/shim.sh" "@$test_dir/missing.rsp" > /dev/null 2>&1; then
    echo 'ERROR: missing response file was accepted' >&2
    exit 1
fi
test -z "$(find "$test_dir" -mindepth 1 -type d -print)"
echo 'macOS linker shim tests passed'
