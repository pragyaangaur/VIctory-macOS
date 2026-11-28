# GTA VI on an Apple Silicon Mac: how this branch gets 6-8 fps in the world

This branch is avatarmanz/VIctory-7900XTX `main` (25 Nov, `0a435665`) plus a Mac layer. On an M5 Max
(64 GB, macOS 27.3, CrossOver 26.3, D3DMetal 4.0 beta 3) it plays the prologue and the open world at about
6-8 fps once shaders are translated (median 5.7-7.7 over five sessions, up to ~10 in light scenes), with clothes, hair, cars, tires and TV pictures drawn. Other Mac
builds I know of were at 1-5 fps in the world on 27 Nov. It is still slow, not "playable" in a normal sense,
but it is the setup below, so anyone can reproduce it. No game files are here; you need your own copy.

## What you need

- An Apple Silicon Mac. More cores help: the game records its frame on 5 threads under Rosetta.
- CrossOver 26.3 (D3DMetal 4.0 beta 3 comes with it). The patched wineserver below only fits 26.3.
- Rust with the `x86_64-pc-windows-gnu` target and mingw-w64 (`brew install mingw-w64`).
- `dxcompiler.dll` and `dxil.dll` from a [DirectXShaderCompiler release](https://github.com/microsoft/DirectXShaderCompiler/releases) (x64), next to `victory.exe`.
- Your own copy of the game (the folder with `game_scarlett_master.exe`).

## Build and run

```zsh
cargo build --release --target x86_64-pc-windows-gnu -p victory
mkdir -p mac/play && cp target/x86_64-pc-windows-gnu/release/victory.exe mac/play/
# put dxcompiler.dll and dxil.dll in mac/play/ too
GAME_DIR=/path/to/your/game BOTTLE=<a CrossOver bottle> zsh mac/play.command
```

The first start translates about 4,000 shaders: expect several minutes of slow frames and missing
objects. After that the cache is warm. Changing the shader translator (any code under
`crates/victory-shader`) invalidates the whole cache, so the next start is slow again.

## The switches and why

| Switch | What it does on the Mac |
|---|---|
| `VICTORY_CONSOLE=series-s` | The game picks the Series S settings: 1080p-class image, lighter scene. The largest single speedup. |
| `VICTORY_OWN_UPSCALER=1` | Series S upscales with FidelityFX compute that does not translate; this makes the game use its own temporal upscaler. |
| `VICTORY_ALIAS=1`, `_POOL=1`, `_TILES=1` | Zero copy: the game's memory is mapped onto D3DMetal's buffers with `mach_vm_remap`, so most per-frame copies and copy-backs go away. On our runs the world frame went 901 -> 555 ms with it alone, before the other work. |
| `VICTORY_STEP_OVER=pool` | Steps over a fault in the title's page-pool code instead of dying. |
| `VICTORY_WAVE_PAIRS=1` | Console wave64 compute programs run as paired 32-lane waves on the Mac GPU where nothing exact fits. |
| `VICTORY_MAC_PRIMITIVE_VARYINGS=1`, `VICTORY_PER_VERTEX_SOLVED=0` | How mesh shader outputs reach pixel shaders under Metal's 32 KiB limit (hair, cars, tires). |
| `VICTORY_LAND_ECHO=0` | With it on, faces lose their skin under Wine. |
| `VICTORY_AMBIENT=1` | A stand-in for the indirect light (the ray-tracing builder does not run), so shaded areas are not black. |
| `VICTORY_AUDIO_DOWNMIX=1` | The game's spatial audio downmixed for normal outputs. |
| `VICTORY_RENDER_SCALE=0.5` (optional) | The 3D world at half width and height: softer, less GPU work. |

## Two Wine patches (optional, both for CrossOver 26.3's source)

- `patches/wineserver-views-rbtree.patch`: the game maps and unmaps 64 KiB views thousands of times a frame,
  and Wine's server walked a list of every view on each call. This indexes them in a tree (UwUCammie's patch
  from fxceless/victory-fork#184). Here it took world frames from 3.6-8 s to 1.4-2.4 s at the time; it
  matters more the slower everything else is. Build steps: pragyaangaur/VIctory-macOS#155. Point
  `FAST_SERVER` at the built `wineserver`.
- `patches/winecoreaudio-follow-default.patch`: CrossOver pins game audio to the output that was default at
  start, so Bluetooth headphones connected later stay silent. This makes render streams follow the macOS
  default output. Point `AUDIO_WINE` at a wine using the patched `winecoreaudio.so`.

## Known problems

- TV and phone video has no sound (the video's audio never reaches XAudio2).
- Shaded ground near the water can be black.
- A few square blocks along big shadow edges now and then.
- The frame is CPU-bound: about 1,000 `ExecuteIndirect` calls a frame at ~110 us each in the GPU layer,
  plus waits for the memory guards' lock under Wine. GPU-side tricks from the Windows forks (DLSS, frame
  generation) do not apply here.
