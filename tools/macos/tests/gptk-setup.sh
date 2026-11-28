#!/bin/sh
# Exercise setup with a complete synthetic Wine tree and disk-image fixtures.
set -eu
root=$(cd "$(dirname "$0")/../../.." && pwd)
tmp=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$tmp"' EXIT
export HOME="$tmp/home" FAKE_BREW="$tmp/brew" FAKE_OUTER="$tmp/outer" FAKE_INNER="$tmp/inner"
export PATH="$tmp/bin:$PATH"
unset DEPS GPTK_DMG
export CROSSOVER_VERSION=26.0.1

work="$tmp/checkout"
wine="$work/target/wine-cx"
installed="$wine/lib"
redist="$FAKE_INNER/redist/lib"
mkdir -p "$HOME" "$tmp/bin" "$work/tools/macos" "$wine/bin" "$FAKE_OUTER" "$FAKE_INNER" \
    "$work/cache" "$work/saves"
cp "$root/setup.sh" "$root/victory.example.toml" "$work/"
cp "$root/tools/macos"/gptk-*.sh "$work/tools/macos/" 2>/dev/null || true
cp "$root/tools/macos/victory.sh" "$tmp/launcher.sh"
builder="$work/tools/macos/build-crossover-real.sh"
cp "$root/tools/macos/build-crossover-wine.sh" "$builder"
current_stamp=$(/bin/zsh "$builder" --stamp)
stale_stamp="$CROSSOVER_VERSION:stale-stamp"
export FULL_REBUILD_MARKER="$tmp/full-rebuild"
touch "$FAKE_OUTER/Evaluation test.dmg" "$tmp/Selected Toolkit.dmg"
printf '%s\n' "$current_stamp" > "$wine/.victory-patches"
touch "$wine/.victory-patch-paths"
printf 'cache\n' > "$work/cache/sentinel"
printf 'save\n' > "$work/saves/sentinel"

# Only platform/build boundaries are stubbed. plutil, the file operations,
# and setup itself operate on real fixtures.
cat > "$tmp/bin/uname" <<'STUB'
#!/bin/sh
echo arm64
STUB
cat > "$tmp/bin/sw_vers" <<'STUB'
#!/bin/sh
echo 15.0
STUB
cat > "$tmp/bin/brew" <<'STUB'
#!/bin/sh
case "$1" in
    --prefix) echo "$FAKE_BREW" ;;
    list) ;;
    *) exit 99 ;;
esac
STUB
cat > "$tmp/bin/hdiutil" <<'STUB'
#!/bin/sh
set -eu
case "$1" in
    attach)
        [ "$5" = -mountpoint ]
        case "$7" in
            */Selected\ Toolkit.dmg)
                [ "${FAIL_EXTRACT:-0}" = 0 ]
                [ "${FAIL_AFTER_FULL_REBUILD:-0}" = 0 ] || [ ! -f "$FULL_REBUILD_MARKER" ]
                ln -s "$FAKE_OUTER/Evaluation test.dmg" "$6/Evaluation test.dmg"
                ;;
            */Evaluation\ test.dmg)
                case "${LINK_FRAMEWORK:-0}" in
                    0) ln -s "$FAKE_INNER/redist" "$6/redist" ;;
                    outside|absolute)
                        cp -Rp "$FAKE_INNER/redist" "$6/redist"
                        framework="$6/redist/lib/external/D3DMetal.framework"
                        destination="$6/Mounted.framework"
                        [ "$LINK_FRAMEWORK" != absolute ] || destination="$6/redist/lib/external/Mounted.framework"
                        /bin/mv "$framework" "$destination"
                        ln -s "$destination" "$framework"
                        ;;
                esac
                ;;
            *) exit 99 ;;
        esac
        ;;
    detach)
        rm -rf "$3/Evaluation test.dmg" "$3/redist" "$3/Mounted.framework"
        ;;
    verify) ;;
    *) exit 99 ;;
esac
STUB
cat > "$tmp/bin/ditto" <<'STUB'
#!/bin/sh
set -eu
case "$1" in
    */redist/lib/)
        if [ "${FAIL_COPY:-0}" = 1 ]; then
            mkdir -p "$2/wine/x86_64-windows"
            printf 'partial copy\n' > "$2/wine/x86_64-windows/d3d12.dll"
            exit 1
        fi
        ;;
esac
exec /usr/bin/ditto "$@"
STUB
cat > "$tmp/bin/mv" <<'STUB'
#!/bin/sh
set -eu
case "$*" in
    *new-lib*" $TEST_WINE_DIR/lib")
        if [ "${FAIL_ACTIVATE:-0}" = 1 ]; then exit 1; fi
        ;;
esac
exec /bin/mv "$@"
STUB
export TEST_WINE_DIR="$wine"
for command in arch xcode-select rustup; do
    ln -s /usr/bin/true "$tmp/bin/$command"
done
for script in "$wine/bin/wine" "$work/tools/macos/build-deps.sh"; do
    cp /usr/bin/true "$script"
done
cp /usr/bin/true "$work/tools/macos/victory.sh"
cat > "$work/tools/macos/build-crossover-wine.sh" <<'STUB'
#!/bin/sh
builder="$(dirname "$0")/build-crossover-real.sh"
case "${1:-}" in
    --stamp|--can-update-in-place) exec /bin/zsh "$builder" "$@" ;;
    --incremental-only) /bin/zsh "$builder" --stamp > "$TEST_WINE_DIR/.victory-patches" ;;
    *)
        if [ ! -f "$TEST_WINE_DIR/.victory-patch-paths" ]; then
            touch "$FULL_REBUILD_MARKER"
            rm -rf "$TEST_WINE_DIR/lib/external/D3DMetal.framework"
        else
            echo "unexpected Wine rebuild" >&2
            exit 99
        fi
        ;;
esac
STUB
chmod +x "$tmp/bin/"* "$work/tools/macos/build-crossover-wine.sh"

fail() { echo "FAIL: $*" >&2; cat "$tmp/output" >&2; exit 1; }
write_redist() {
    lib=$1 version=$2 label=$3
    framework="$lib/external/D3DMetal.framework"
    mkdir -p "$framework/Versions/A/Resources" "$lib/wine/x86_64-windows"
    ln -sfn A "$framework/Versions/Current"
    ln -sfn Versions/Current/Resources "$framework/Resources"
    ln -sfn Versions/Current/D3DMetal "$framework/D3DMetal"
    cat > "$framework/Resources/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>CFBundleShortVersionString</key><string>$version</string>
</dict></plist>
PLIST
    for path in external/libd3dshared.dylib external/D3DMetal.framework/Versions/A/D3DMetal \
        wine/x86_64-windows/d3d12.dll wine/x86_64-windows/dxgi.dll; do
        printf '%s\n' "$label" > "$lib/$path"
    done
}

setup() (cd "$tmp" && env "$@" /bin/zsh "$work/setup.sh" > "$tmp/output" 2>&1)
run_builder() { /bin/zsh "$builder" "$@" > "$tmp/output" 2>&1; }
assert_output() { grep -F -- "$1" "$tmp/output" >/dev/null || fail "missing diagnostic $1"; }
assert_preserved() {
    diff -r --no-dereference "$tmp/original-lib" "$installed" >/dev/null || fail "failed setup modified the installed Wine library tree"
    [ "$(cat "$work/cache/sentinel")" = cache ] && [ "$(cat "$work/saves/sentinel")" = save ] ||
        fail "failed setup modified cache or saves"
}

write_redist "$installed" 4.0b1 old
printf 'Wine library\n' > "$installed/wine/x86_64-windows/other.dll"
touch "$installed/external/D3DMetal.framework/obsolete"
cp -Rp "$installed" "$tmp/original-lib"

if setup; then fail "setup accepted an existing 4.0b1 framework"; fi
assert_output '4.0b1'
assert_output 'GPTK_DMG'
assert_preserved

cp "$tmp/launcher.sh" "$work/tools/macos/launcher.sh"
if WINE="$wine/bin/wine" sh "$work/tools/macos/launcher.sh" --version > "$tmp/output" 2>&1; then
    fail "launcher accepted an existing 4.0b1 framework"
fi
assert_output '4.0b1'
assert_output 'GPTK_DMG'

export GPTK_DMG="$tmp/Selected Toolkit.dmg"
write_redist "$redist" 4.0b2 candidate
printf '%s\n' "$stale_stamp" > "$wine/.victory-patches"
if setup; then fail "setup accepted an older toolkit"; fi
assert_output '4.0b2'
assert_output '4.0b3'
assert_preserved
printf '%s\n' "$current_stamp" > "$wine/.victory-patches"

write_redist "$redist" '4.0beta3' candidate
if setup; then fail "setup accepted unknown toolkit spelling"; fi
assert_output '4.0beta3'
assert_output 'GPTK_DMG'
assert_preserved

write_redist "$redist" "4.0b3
" candidate
if setup; then fail "setup accepted a newline in the toolkit version"; fi
assert_output "GPTK_DMG"
assert_preserved

write_redist "$redist" 4.0b3 candidate
if setup FAIL_EXTRACT=1; then fail "setup accepted a failed extraction"; fi
assert_output 'could not mount'
assert_preserved

rm "$redist/wine/x86_64-windows/dxgi.dll"
if setup; then fail "setup accepted an incomplete toolkit"; fi
assert_output 'dxgi.dll'
assert_preserved
write_redist "$redist" 4.0b3 candidate

if setup FAIL_COPY=1; then fail "setup accepted an incomplete staged copy"; fi
assert_output 'copy'
assert_preserved

if setup FAIL_ACTIVATE=1; then fail "setup accepted a failed activation"; fi
assert_output 'activate'
assert_preserved

case "${1:-all}" in
    all|relative)
        setup GPTK_DMG='Selected Toolkit.dmg' || fail "setup rejected an image relative to its caller"
        ;;
    *) setup || fail "setup rejected a complete 4.0b3 toolkit" ;;
esac
assert_output '4.0b3'
for path in external/libd3dshared.dylib external/D3DMetal.framework/D3DMetal \
    wine/x86_64-windows/d3d12.dll wine/x86_64-windows/dxgi.dll; do
    [ "$(cat "$installed/$path")" = candidate ] || fail "refresh did not update $path"
done
[ ! -e "$installed/external/D3DMetal.framework/obsolete" ] || fail "refresh retained an obsolete framework file"
[ "$(cat "$installed/wine/x86_64-windows/other.dll")" = 'Wine library' ] || fail "refresh replaced an unrelated Wine library"

unset GPTK_DMG
setup || fail "setup rejected a repeat with an accepted installation"
assert_output 'Wine already built'
assert_output '4.0b3'
[ "$(cat "$work/cache/sentinel")" = cache ] && [ "$(cat "$work/saves/sentinel")" = save ] ||
    fail "successful setup modified cache or saves"
case "${1:-all}" in
    all|rebuild-no-image|rebuild-image)
        rm "$wine/.victory-patch-paths"
        cp -Rp "$wine" "$tmp/original-wine"
        for case_name in rebuild-no-image rebuild-image; do
            [ "${1:-all}" = all ] || [ "$1" = "$case_name" ] || continue
            if [ "$case_name" = rebuild-image ]; then
                write_redist "$redist" 4.0b10 candidate
                export GPTK_DMG="$tmp/Selected Toolkit.dmg"
            fi
            if setup FAIL_AFTER_FULL_REBUILD=1; then fail "$case_name unexpectedly ran a full rebuild"; fi
            diff -r --no-dereference "$tmp/original-wine" "$wine" >/dev/null ||
                fail "$case_name lost the original Wine tree"
            [ ! -e "$FULL_REBUILD_MARKER" ] || fail "$case_name started the full rebuild"
            assert_output 'full Wine rebuild'
            assert_output 'backup'
            assert_output 'GPTK_DMG'
        done
        touch "$wine/.victory-patch-paths"
        unset GPTK_DMG
        ;;
esac

case "${1:-all}" in
    all|framework-link|framework-staged-link)
        rm -rf "$tmp/original-lib"
        cp -Rp "$installed" "$tmp/original-lib"
        write_redist "$redist" 4.0b10 candidate
        export GPTK_DMG="$tmp/Selected Toolkit.dmg"
        for link_mode in outside absolute; do
            [ "${1:-all}" = all ] ||
                [ "$1:$link_mode" = framework-link:outside ] ||
                [ "$1:$link_mode" = framework-staged-link:absolute ] || continue
            if setup LINK_FRAMEWORK="$link_mode"; then fail "setup accepted a framework linked to the mounted image ($link_mode)"; fi
            assert_output 'resolves outside'
            assert_preserved
        done
        unset GPTK_DMG
        ;;
esac

case "${1:-all}" in
    all|incremental)
        printf '%s\n' "$stale_stamp" > "$wine/.victory-patches"
        setup || fail "setup rejected an incremental Wine rebuild"
        assert_output "rebuilding Wine's server"
        [ "$(cat "$wine/.victory-patches")" = "$current_stamp" ] || fail "setup did not invoke the incremental builder"
        ;;
esac

case "${1:-all}" in
    all|classifier)
        printf 'server/fd.c\ndlls/ntdll/unix/virtual.c\n' > "$wine/.victory-patch-paths"
        run_builder --can-update-in-place || fail "classifier rejected Wine's server and ntdll patches"
        # Without DEPS, the real builder stops once it has accepted a partial build.
        if run_builder --incremental-only; then fail "incremental builder ran without DEPS"; fi
        assert_output 'set DEPS to an x86_64 prefix'
        printf 'dlls/win32u/main.c\n' > "$wine/.victory-patch-paths"
        if run_builder --can-update-in-place; then fail "classifier accepted a previously installed patch that needs a full rebuild"; fi
        mkdir -p "$work/tools/macos/patches"
        printf '+++ b/dlls/win32u/main.c\n' > "$work/tools/macos/patches/full.patch"
        : > "$wine/.victory-patch-paths"
        if run_builder --can-update-in-place; then fail "classifier accepted an incoming patch that needs a full rebuild"; fi
        rm "$work/tools/macos/patches/full.patch" "$wine/.victory-patch-paths"
        if run_builder --can-update-in-place; then fail "classifier accepted missing Wine metadata"; fi
        if run_builder --incremental-only; then
            fail "incremental-only Wine builder accepted a full rebuild"
        fi
        assert_output 'full rebuild'
        touch "$wine/.victory-patch-paths"
        ;;
esac

case "${1:-all}" in
    all|release)
        cp -Rp "$wine" "$tmp/release-wine"
        write_redist "$redist" 4.0b10 candidate
        export CROSSOVER_VERSION=26.0.2
        if run_builder --can-update-in-place; then fail "classifier accepted a different CrossOver release"; fi
        if run_builder --incremental-only; then fail "incremental builder accepted a different CrossOver release"; fi
        assert_output 'full rebuild'
        if setup GPTK_DMG="$tmp/Selected Toolkit.dmg"; then fail "setup accepted a different CrossOver release"; fi
        assert_output 'full Wine rebuild'
        diff -r --no-dereference "$tmp/release-wine" "$wine" >/dev/null ||
            fail "the release change modified the installed Wine tree"
        [ ! -e "$FULL_REBUILD_MARKER" ] || fail "the release change started a full rebuild"
        export CROSSOVER_VERSION=26.0.1

        printf '%s\n' "${current_stamp#*:}" > "$wine/.victory-patches"
        if run_builder --can-update-in-place; then fail "classifier accepted an unversioned CrossOver stamp"; fi
        if run_builder --incremental-only; then fail "incremental builder accepted an unversioned CrossOver stamp"; fi
        assert_output 'full rebuild'
        printf '%s\n' "$current_stamp" > "$wine/.victory-patches"
        ;;
esac

rm "$installed/wine/x86_64-windows/dxgi.dll"
if setup; then fail "setup accepted an installed library tree with a missing DLL"; fi
assert_output "dxgi.dll"
assert_output "GPTK_DMG"
echo "GPTK setup checks passed"
