#!/bin/zsh
# One-step setup of VIctory for macOS on an Apple Silicon Mac.
#
#   ./setup.sh
#
# It installs the tools it needs with Homebrew, builds an x86_64 Wine from
# CodeWeavers' CrossOver source, adds Apple's D3DMetal from the Game Porting
# Toolkit in your Downloads folder, and builds VIctory. Run it again at any
# time: finished steps are skipped. The first run takes about an hour.
set -eu
ROOT="${0:A:h}"
# Resolve an explicit image path relative to the caller before changing folders.
dmg="${GPTK_DMG:-}"
case "$dmg" in
  ''|/*) ;;
  *) dmg="$PWD/$dmg" ;;
esac
cd "$ROOT"
. tools/macos/gptk-version.sh
say() { print -P "%F{cyan}==>%f $*"; }
die() { print -P "%F{red}error:%f $*" >&2; exit 1; }

# 1. The Mac itself.
[[ $(uname -m) == arm64 ]] || die "this needs an Apple Silicon Mac (M1 or newer)"
(( ${$(sw_vers -productVersion)%%.*} >= 15 )) || die "this needs macOS 15 Sequoia or newer"
arch -x86_64 /usr/bin/true 2> /dev/null || die "Rosetta 2 is missing. Install it with: softwareupdate --install-rosetta"
xcode-select -p > /dev/null 2>&1 || die "the Xcode command line tools are missing. Install them with: xcode-select --install"
command -v brew > /dev/null || die "Homebrew is missing. Install it from https://brew.sh and run ./setup.sh again"
# The first run builds Wine and needs about 15 GB. Later runs reuse that build,
# so only check the space before Wine exists.
if [[ ! -x $ROOT/target/wine-cx/bin/wine ]]; then
  free_gb=$(( $(df -Pk "$ROOT" | awk 'NR == 2 { print $4 }') / 1024 / 1024 ))
  (( free_gb >= 15 )) || die "the first setup needs about 15 GB free and this disk has ${free_gb} GB. Free some space and run ./setup.sh again"
fi

# 2. Tools from Homebrew.
BREW="$(brew --prefix)"
wanted=(rustup llvm lld mingw-w64 bison flex pkgconf)
missing=()
for f in $wanted; do brew list --formula "$f" > /dev/null 2>&1 || missing+=("$f"); done
if (( ${#missing} )); then
  say "installing ${missing[*]} with Homebrew"
  # brew exits with an error when a post-install step fails, even if the
  # formula itself installed (rustup's can fail on a completion link it may
  # not write). Ask Homebrew what is installed instead of trusting the exit.
  brew install "${missing[@]}" || true
  failed=()
  for f in $missing; do brew list --formula "$f" > /dev/null 2>&1 || failed+=("$f"); done
  (( ${#failed} == 0 )) || die "Homebrew could not install ${failed[*]}. See its output above, fix that and run ./setup.sh again"
fi
export PATH="$BREW/opt/rustup/bin:$PATH"

# 3. Rust with the Windows target VIctory is built for on a Mac.
say "setting up Rust"
rustup show active-toolchain > /dev/null 2>&1 || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)" --profile minimal
rustup target add x86_64-pc-windows-gnullvm > /dev/null

# Wine's build breaks on paths with spaces, so the large build folders live
# under a space-free link in the cache folder. There is one link for every
# checkout, so it names whichever checkout ran setup last (a fleet task's
# too): the build goes through it, and tools/macos/victory.sh and
# test-wine.sh take this checkout's own target/wine-cx first.
ALIAS="$HOME/Library/Caches/victory-macos"
WINE_DIR="$ROOT/target/wine-cx"

# Validate the candidate before rebuilding an existing Wine or touching its lib.
validate_gptk() {
  local lib=${1:A} description=$2 version component file
  for component in external/D3DMetal.framework/Resources/Info.plist \
      external/D3DMetal.framework/D3DMetal external/libd3dshared.dylib \
      wine/x86_64-windows/d3d12.dll wine/x86_64-windows/dxgi.dll; do
    file="$lib/$component"
    [[ -f $file && -s $file ]] \
      || die "$description is missing $component. Check that GPTK_DMG points to Apple's Game Porting Toolkit .dmg"
    [[ ${file:A} == "$lib"/* ]] \
      || die "$component in $description resolves outside its library tree. Check the selected GPTK_DMG"
  done
  version=$(gptk_framework_version "$lib/external/D3DMetal.framework") \
    || die "could not read D3DMetal's version in $description. Set GPTK_DMG to a Game Porting Toolkit .dmg with D3DMetal 4.0b3 or newer"
  gptk_version_supported "$version" \
    || die "$description has unsupported D3DMetal version '$version'; set GPTK_DMG to a .dmg with D3DMetal 4.0b3 or newer"
  print -r -- "$version"
}

setup_gptk() (
  lib="$WINE_DIR/lib"
  framework="$lib/external/D3DMetal.framework"
  stage="$WINE_DIR/.gptk-refresh"
  [[ ! -e $stage ]] || die "a D3DMetal refresh is in progress or was interrupted at $stage. See docs/macos.md for recovery"
  if [[ -e $framework || -L $framework ]]; then
    if installed=$(gptk_framework_version "$framework"); then
      say "installed D3DMetal version $installed"
      if gptk_version_supported "$installed" && [[ -z $dmg ]]; then
        validate_gptk "$lib" "the installed D3DMetal runtime" > /dev/null
        return
      fi
    else
      say "could not read the installed D3DMetal version from $framework/Resources/Info.plist"
    fi
    [[ -n $dmg ]] || die "set GPTK_DMG to the path of a Game Porting Toolkit .dmg with D3DMetal 4.0b3 or newer, then run ./setup.sh"
  elif [[ -z $dmg ]]; then
    # (N) keeps zsh quiet when nothing matches, (om) puts the newest first.
    downloads=("$HOME"/Downloads/Game_Porting_Toolkit*.dmg(Nom))
    dmg="${downloads[1]:-}"
  fi
  if [[ -z $dmg ]]; then
    print
    print -P "%F{yellow}One manual step is left.%f Apple requires a sign-in to download D3DMetal:"
    print "  1. Open https://developer.apple.com/download/all/?q=game%20porting%20toolkit"
    print "  2. Sign in and download a Game Porting Toolkit .dmg with D3DMetal 4.0b3 or newer"
    print "  3. Leave it in your Downloads folder and run ./setup.sh again"
    exit 1
  fi
  [[ -f $dmg ]] || die "GPTK_DMG does not name a file: $dmg"
  [[ -d $lib && ! -L $lib ]] || die "expected Wine's library directory at $lib; reinstall Wine before refreshing D3DMetal"
  say "checking D3DMetal from ${dmg:t}"
  mkdir "$stage" || die "could not start a D3DMetal refresh at $stage"
  outer= inner=
  cleanup_gptk() {
    local restored=1
    if [[ -d $stage/previous-lib && ! -e $lib ]] && ! mv -f "$stage/previous-lib" "$lib"; then
      print -r -- "Could not restore Wine's library directory. The previous set is at $stage/previous-lib" >&2
      restored=0
    fi
    if [[ -n $inner ]]; then
      hdiutil detach -quiet "$inner" 2> /dev/null || true
      rmdir "$inner" 2> /dev/null || true
    fi
    if [[ -n $outer ]]; then
      hdiutil detach -quiet "$outer" 2> /dev/null || true
      rmdir "$outer" 2> /dev/null || true
    fi
    (( restored )) && rm -rf "$stage"
  }
  trap cleanup_gptk EXIT
  trap 'exit 1' HUP INT TERM
  outer="$(mktemp -d)"
  inner="$(mktemp -d)"
  if ! hdiutil attach -quiet -nobrowse -readonly -mountpoint "$outer" "$dmg"; then
    if hdiutil verify -quiet "$dmg" 2> /dev/null; then
      die "macOS could not mount ${dmg:t}, though the file is intact. Run ./setup.sh again; if an AI agent or another sandboxed shell runs it and it fails again, run it from your own Terminal"
    fi
    die "could not open ${dmg:t}. It may be an incomplete download: delete it, download it again and run ./setup.sh again"
  fi
  eval_dmg=("$outer"/Evaluation*.dmg(N))
  (( ${#eval_dmg} )) || die "${dmg:t} has no Evaluation .dmg inside. Check that it is Apple's Game Porting Toolkit"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$inner" "$eval_dmg[1]" \
    || die "could not open ${eval_dmg[1]:t} inside ${dmg:t}"
  candidate=$(validate_gptk "$inner/redist/lib" "${dmg:t}")

  # Clone on APFS and overlay in staging so neither Wine nor the old toolkit
  # is modified if the copy runs out of disk space or a component is missing.
  cp -cRp "$lib" "$stage/new-lib" \
    || die "could not copy Wine's library directory for staging. Check the disk's free space"
  [[ ! -L $stage/new-lib/external ]] \
    || die "Wine's external library directory is a symlink. Reinstall Wine before refreshing D3DMetal"
  rm -rf "$stage/new-lib/external/D3DMetal.framework"
  ditto "$inner/redist/lib/" "$stage/new-lib/" \
    || die "could not copy the D3DMetal redistributables to staging. Check the disk's free space"
  staged=$(validate_gptk "$stage/new-lib" "the staged D3DMetal copy")
  [[ $staged == $candidate ]] || die "the staged D3DMetal version is '$staged'; expected '$candidate'"
  # Both renames stay on the Wine volume. The exit trap restores the previous
  # library directory if the second rename fails.
  mv -f "$lib" "$stage/previous-lib" && mv -f "$stage/new-lib" "$lib" \
    || die "could not activate the staged D3DMetal $candidate"
  say "installed D3DMetal $candidate"
)

# Protect any existing runtime before deciding whether setup can rebuild Wine.
wine_existing=0
wine_current=0
framework="$WINE_DIR/lib/external/D3DMetal.framework"
if [[ -x $WINE_DIR/bin/wine || -e $framework || -L $framework ]]; then
  wine_existing=1
  if [[ -x $WINE_DIR/bin/wine && -f $WINE_DIR/.victory-patches && -f $WINE_DIR/.victory-patch-paths && "$(cat "$WINE_DIR/.victory-patches" 2> /dev/null)" == "$(tools/macos/build-crossover-wine.sh --stamp)" ]]; then
    wine_current=1
  elif ! tools/macos/build-crossover-wine.sh --can-update-in-place; then
    die "this installation needs a full Wine rebuild. Move $WINE_DIR to a backup, then set GPTK_DMG to a qualifying toolkit .dmg and run ./setup.sh. See docs/macos.md"
  fi
  setup_gptk
fi

# 4. x86_64 libraries for Wine.
mkdir -p "$ALIAS" "$ROOT/target"
ln -sfn "$ROOT/target" "$ALIAS/target"
say "building the x86_64 libraries Wine needs (about 10 minutes the first time)"
tools/macos/build-deps.sh "$ALIAS/target/deps-x86"

# 5. CrossOver Wine with the patches in tools/macos/patches and SDL2.
if (( wine_current )); then
  say "Wine already built"
elif (( wine_existing )); then
  say "rebuilding Wine's server, ntdll and HID bus (a few minutes)"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh --incremental-only
else
  say "building CrossOver Wine (about 30 minutes the first time)"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh
  setup_gptk
fi

# 6. VIctory itself.
say "building VIctory"
[[ -f victory.toml ]] || cp victory.example.toml victory.toml
# victory.sh owns the single release build: it sets VICTORY_HELPER_DXC when
# D3DMetal's DXC is there, and a separate build here without it would make
# victory-gpu's build.rs see the variable appear and rebuild and relink
# victory a second time. --version starts VIctory under Wine without reading
# victory.toml, whose example game_dir does not exist yet, so a fresh setup
# does not fail here. Stderr stays visible, and captured stdout is printed on
# failure so diagnostics from either Cargo or a custom Wine wrapper are kept.
version_out=$(tools/macos/victory.sh --version) || {
  [[ -n "$version_out" ]] && print -r -- "$version_out" >&2
  die "VIctory failed to build or start under Wine. Run tools/macos/victory.sh --version to see why"
}

print
print -P "%F{green}Setup finished.%f"
print "Put your own copy of the game in a folder, set game_dir in victory.toml, check it with"
print "  tools/macos/victory.sh config"
print "and start it with:"
print "  tools/macos/victory.sh run --execute <the game's executable>"
