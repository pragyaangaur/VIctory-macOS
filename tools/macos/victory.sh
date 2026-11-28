#!/bin/sh
# Builds VIctory for Windows and runs it on macOS under CrossOver Wine with
# Apple's D3DMetal. Arguments go to victory.exe, for example:
#   tools/macos/victory.sh config
#   tools/macos/victory.sh run --execute <path to the game's executable>
# WINE defaults to the Wine that tools/macos/build-crossover-wine.sh builds:
# this checkout's own (target/wine-cx), else the one behind the space-free
# link in ~/Library/Caches/victory-macos. Every checkout's setup points that
# link at its own target/ (a fleet task's too), so it names whichever ran
# setup last; only the build needs it.
# VICTORY_DIAGNOSTICS=1 builds with the diagnostics feature (debug logs,
# black-frame reports, frame capture and the other debugging switches).
set -e
caller="$PWD"
cd "$(dirname "$0")/../.."
ROOT="$PWD"

# VICTORY_SET=<name> reads the switches in tools/macos/sets/<name>.txt (one
# NAME=value a line, # comments), as a named set testers can compare runs by
# (#555). A variable already set in the environment wins over the set.
if [ -n "${VICTORY_SET:-}" ]; then
    case "$VICTORY_SET" in
        *[!A-Za-z0-9_-]*) echo "VICTORY_SET names a file in tools/macos/sets: $VICTORY_SET" >&2; exit 1 ;;
    esac
    applied=
    set_file="$ROOT/tools/macos/sets/$VICTORY_SET.txt"
    [ -f "$set_file" ] || { echo "no switch set $VICTORY_SET: the sets are $(ls "$ROOT/tools/macos/sets" | sed 's/\.txt$//' | tr '\n' ' ')" >&2; exit 1; }
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*) continue ;;
        esac
        name=${line%%=*}
        case "$name" in
            VICTORY_*|WINE*) ;;
            *) echo "$set_file: not a VICTORY_ or WINE variable: $line" >&2; exit 1 ;;
        esac
        if ! printenv "$name" > /dev/null; then
            export "$name=${line#*=}"
        fi
        applied="${applied:+$applied }$name=$(printenv "$name")"
    done < "$set_file"
    echo "switch set $VICTORY_SET: $applied" >&2
fi

# Resolve a relative Wine override from the environment or the switch set.
case "${WINE:-}" in
    ''|/*) ;;
    *) WINE="$caller/$WINE" ;;
esac
TARGET=x86_64-pc-windows-gnullvm
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if [ -z "${WINE:-}" ]; then
    WINE="$ROOT/target/wine-cx/bin/wine"
    [ -x "$WINE" ] || WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"
fi
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ -d "$(dirname "$WINE")/../lib/external/D3DMetal.framework" ] || {
    echo "D3DMetal is missing: copy the redist/lib folder of Apple's evaluation environment over $(dirname "$WINE")/../lib" >&2
    exit 1
}
# The 64-thread argument clamp is compiled at build time with the DXC that
# D3DMetal ships: CrossOver's own compiler refuses it, and without it the
# one-thread form runs instead. A VICTORY_HELPER_DXC already set wins.
DXC_LIB="$(dirname "$WINE")/../lib/external/D3DMetal.framework/Versions/A/Resources/libdxcompiler.dylib"
if [ -z "${VICTORY_HELPER_DXC:-}" ] && [ -f "$DXC_LIB" ]; then
    # A build product of this checkout, as its target/ holds them.
    helper="$ROOT/target/dxc-d3dmetal"
    mkdir -p "$ROOT/target"
    if [ ! -x "$helper" ] || [ tools/macos/dxc-d3dmetal.sh -nt "$helper" ]; then
        sh tools/macos/dxc-d3dmetal.sh "$helper" || helper=
    fi
    if [ -n "$helper" ]; then
        export DXC_LIB VICTORY_HELPER_DXC="$helper"
    fi
fi
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ]; then
    cargo build --quiet --release -p victory --features diagnostics --target "$TARGET"
else
    cargo build --quiet --release -p victory --target "$TARGET"
fi
# Ask Cargo where it built the executable, including any configured target directory.
metadata=$(cargo metadata --format-version 1 --no-deps)
target_dir=$(printf '%s\n' "$metadata" | /usr/bin/plutil -extract target_directory raw -o - -)
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-play}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
# Rosetta tells the game about AVX and AVX2, which it may check for.
export ROSETTA_ADVERTISE_AVX=1
# macOS 15's Rosetta runs AVX2 but its CPUID leaves it out, so victory's
# AVX2 check refuses a Mac that can run the game (M1 Max, 15.7.3).
if [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 15 ]; then
    export VICTORY_SKIP_CPU_CHECK="${VICTORY_SKIP_CPU_CHECK:-1}"
fi
# A game left in the prefix (hung, or its window closed while it still ran)
# keeps its GPU memory; a second one on top ran a 32 GB Mac out of it and
# froze the whole display. End it first.
if [ "${1:-}" = run ]; then
    "$(dirname "$WINE")/wineserver" -k 2>/dev/null || true
    "$(dirname "$WINE")/wineserver" -w 2>/dev/null || true
fi
# With fault tracking on, Rosetta stops on a 256-bit store that writes across
# into a guarded page unless the loader splits it, and the loader splits the
# stores listed in <cache_dir>/wide. Without those lists the run ends in
# Rosetta's "EmulateForward" with no crash report. Make any that are missing
# before the title starts: a first run binds without --execute, so the cache
# holds the images to list (about two minutes for the executable, once).
# Skipped when the lists are named by hand (VICTORY_PATCH_WIDE or
# VICTORY_PATCH_WIDE_DIR), when the cache is moved (VICTORY_CACHE_DIR or
# cache_dir in the config, a Windows path this script does not resolve), and
# with VICTORY_WIDE_LISTS=0.
wide_lists() {
    [ "${VICTORY_FAULT_TRACKING:-}" = 1 ] || return 0
    [ "${VICTORY_WIDE_LISTS:-1}" != 0 ] || return 0
    [ -z "${VICTORY_PATCH_WIDE:-}" ] && [ -z "${VICTORY_PATCH_WIDE_DIR:-}" ] || return 0
    case " $* " in *" --execute "*) ;; *) return 0 ;; esac
    config="${VICTORY_CONFIG:-$caller/victory.toml}"
    if [ -n "${VICTORY_CACHE_DIR:-}" ] || { [ -f "$config" ] && grep -Eq '^[[:space:]]*cache_dir[[:space:]]*=' "$config"; }; then
        echo "wide-store lists: the cache is moved; make them with cargo xtask wide-stores --cache <cache_dir as a Mac path>" >&2
        return 0
    fi
    caches() { ls -d "$WINEPREFIX"/drive_c/users/*/AppData/Local/victory/cache 2>/dev/null; }
    missing() {
        caches | while IFS= read -r cache; do
            for image in "$cache"/images/victory-image-*.dll; do
                [ -f "$image" ] || continue
                hash=${image##*/victory-image-}
                [ -f "$cache/wide/${hash%.dll}.txt" ] || { echo "$cache"; break; }
            done
        done
    }
    if [ -z "$(caches)" ] || ! ls "$(caches | head -n 1)"/images/victory-image-*.dll > /dev/null 2>&1; then
        echo "wide-store lists: binding the title once so the cache holds its images" >&2
        # The same arguments without --execute: map and bind, then stop.
        prime=
        for arg in "$@"; do
            [ "$arg" = --execute ] || prime="$prime '$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")'"
        done
        eval "set -- $prime"
        "$WINE" "$target_dir/$TARGET/release/victory.exe" "$@" > /dev/null 2>&1 || true
        "$(dirname "$WINE")/wineserver" -w 2>/dev/null || true
    fi
    missing | sort -u | while IFS= read -r cache; do
        echo "wide-store lists: making the missing ones in $cache (about two minutes the first time)" >&2
        cargo xtask wide-stores --cache "$cache" < /dev/null >&2 ||
            echo "wide-store lists: cargo xtask wide-stores failed; with fault tracking on, Rosetta may stop on a 256-bit store" >&2
    done
}
if [ "${1:-}" = run ]; then
    (cd "$ROOT" && wide_lists "$@")
fi
# Keep the Mac and its display awake while VIctory runs. Runs left alone
# stopped on the loading artwork at 0.8 fps, and a --cache pre-build paused
# for about 16 minutes, while the same builds went on with someone at the Mac
# (#554): display sleep hides the game's window, and macOS then throttles a
# hidden app (App Nap). caffeinate holds its assertions on behalf of Wine and
# lets them go when Wine exits, with Wine's exit status.
# VICTORY_CAFFEINATE=0 leaves sleep to macOS.
if [ "${VICTORY_CAFFEINATE:-1}" != 0 ] && [ -x /usr/bin/caffeinate ]; then
    exec /usr/bin/caffeinate -dim "$WINE" "$target_dir/$TARGET/release/victory.exe" "$@"
fi
exec "$WINE" "$target_dir/$TARGET/release/victory.exe" "$@"
