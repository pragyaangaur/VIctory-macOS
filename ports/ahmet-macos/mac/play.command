#!/bin/zsh
# Plays GTA VI on an Apple Silicon Mac with this branch's victory.exe under CrossOver 26.3 + D3DMetal.
# The settings below are the ones that give ~6-8 fps in the world on an M5 Max (up to ~10) (see mac/README.md).
#
# Set these three for your machine (or export them before running):
GAME_DIR=${GAME_DIR:-$HOME/Games/GTA6}                 # folder with game_scarlett_master.exe (your own copy)
PLAY_DIR=${PLAY_DIR:-${0:A:h}/play}                     # victory.exe, dxcompiler.dll, dxil.dll, victory.toml, cache, saves
BOTTLE=${BOTTLE:-victory}                               # a CrossOver bottle (any Windows 10 bottle works)
# Optional, see mac/README.md:
FAST_SERVER=${FAST_SERVER:-}                            # path to a wineserver built with mac/patches/wineserver-views-rbtree.patch
AUDIO_WINE=${AUDIO_WINE:-}                              # path to a wine whose winecoreaudio has mac/patches/winecoreaudio-follow-default.patch

CXROOT="$HOME/Applications/CrossOver.app/Contents"
[ -d "$CXROOT" ] || CXROOT="/Applications/CrossOver.app/Contents"
WINE_BIN=${AUDIO_WINE:-"$CXROOT/SharedSupport/CrossOver/bin/wine"}

if ps -axo command= | /usr/bin/grep -Eq '^Z:/.*victory[^/ ]*\.exe run( |$)'; then
  echo "GTA is already running. Quit it first."
  exit 1
fi
[ -x "$PLAY_DIR/victory.exe" ] || { echo "No victory.exe in $PLAY_DIR (build: cargo build --release --target x86_64-pc-windows-gnu -p victory)"; exit 1; }
[ -f "$GAME_DIR/game_scarlett_master.exe" ] || { echo "No game_scarlett_master.exe in $GAME_DIR"; exit 1; }
if [ ! -f "$PLAY_DIR/victory.toml" ]; then
  mkdir -p "$PLAY_DIR/cache" "$PLAY_DIR/data"
  cat > "$PLAY_DIR/victory.toml" <<EOT
[paths]
game_dir = 'Z:${GAME_DIR//\//\\}'
cache_dir = 'Z:${PLAY_DIR//\//\\}\\cache'
data_dir = 'Z:${PLAY_DIR//\//\\}\\data'
EOT
fi

# The patched server must be the bottle's server before any client starts (Wine's loader starts
# its own otherwise). It has to match CrossOver's version: it is built from CrossOver 26.3's source.
if [ -n "$FAST_SERVER" ] && [ -x "$FAST_SERVER" ]; then
  "$WINE_BIN" --bottle "$BOTTLE" --ux-app "$FAST_SERVER" -p > /dev/null 2>&1 < /dev/null
  sleep 1
fi

cd "$PLAY_DIR"
echo "Starting GTA VI. Log: $PLAY_DIR/play.log (the first start translates shaders: expect minutes of slow frames)."
VICTORY_CONFIG="Z:$PLAY_DIR/victory.toml" VICTORY_LOG=${VICTORY_LOG:-info} WINEDEBUG=-all \
VICTORY_CONSOLE=${VICTORY_CONSOLE-series-s} \
VICTORY_OWN_UPSCALER=${VICTORY_OWN_UPSCALER-1} \
VICTORY_ALIAS=${VICTORY_ALIAS:-1} VICTORY_ALIAS_POOL=${VICTORY_ALIAS_POOL:-1} VICTORY_ALIAS_TILES=${VICTORY_ALIAS_TILES:-1} \
VICTORY_STEP_OVER=${VICTORY_STEP_OVER-pool} \
VICTORY_WAVE_PAIRS=${VICTORY_WAVE_PAIRS:-1} \
VICTORY_MAC_PRIMITIVE_VARYINGS=${VICTORY_MAC_PRIMITIVE_VARYINGS:-1} \
VICTORY_PER_VERTEX_SOLVED=${VICTORY_PER_VERTEX_SOLVED:-0} \
VICTORY_LAND_ECHO=${VICTORY_LAND_ECHO:-0} \
VICTORY_AMBIENT=${VICTORY_AMBIENT:-1} \
VICTORY_AUDIO_DOWNMIX=${VICTORY_AUDIO_DOWNMIX:-1} \
  "$WINE_BIN" --bottle "$BOTTLE" "Z:$PLAY_DIR/victory.exe" run "Z:$GAME_DIR/game_scarlett_master.exe" --execute \
  > "$PLAY_DIR/play.log" 2>&1 < /dev/null
echo "Game ended. Last error line: $(grep -a -i -E 'crash|panic|error' "$PLAY_DIR/play.log" | tail -1 | cut -c1-160)"
