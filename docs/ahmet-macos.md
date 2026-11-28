# Ahmet’s CrossOver build for Apple Silicon

This is Ahmet Dedeler’s complete `mac-newbase` revision `d28238f9da092b4d6a851de01ff5191cfaf295c1`, based on avatarmanz/VIctory-7900XTX. The independent Rust workspace includes its renderer, shader translator, HLE, launcher, configuration and both optional CrossOver 26.3 source patches. Its Git tree is `01d52914155333924b042d41d24889319c948e25`. [Original source and author’s notes](https://github.com/Ahmet-Dedeler/VIctory/tree/d28238f9da092b4d6a851de01ff5191cfaf295c1/mac)

## Run the packaged build

1. Download **VIctory-Ahmet-CrossOver** from the **Ahmet CrossOver build** workflow. Unzip the Actions download and extract `VIctory-Ahmet-CrossOver.tar.gz` outside the repository.
2. Put the x64 `dxcompiler.dll` and `dxil.dll` beside the included `victory.exe`. Use your own game files and a CrossOver 26.3 Windows 10 bottle configured for D3DMetal. The M5 Max benchmark used a local runtime with the Wine view-tree patch. [Microsoft DXC releases](https://github.com/microsoft/DirectXShaderCompiler/releases) · [Optional Wine patch instructions](https://github.com/Ahmet-Dedeler/VIctory/blob/d28238f9da092b4d6a851de01ff5191cfaf295c1/mac/README.md#two-wine-patches-optional-both-for-crossover-263s-source)
3. In Terminal, change into the extracted `VIctory-Ahmet` directory. Set the absolute path to your game and the name of your bottle in the launch command.

   ```zsh
   GAME_DIR="/absolute/path/to/your/game" BOTTLE="victory" PLAY_DIR="$PWD" zsh ./play.command
   ```

The launcher creates the configuration, cache and save data in this directory on first run. The shader cache takes several minutes to warm. Keep the extracted directory separate from other builds and saves. The archive contains the complete corresponding source, the licence and the Git tree ID. The source archive includes the optional Wine patches and the BCPack table notice in `source/docs/asset-policy.md`.

## Build the package locally

From the root of this checkout on an Apple Silicon Mac with Homebrew and rustup, run the following commands.

```zsh
brew install mingw-w64
rustup toolchain install 1.98.1 --profile minimal --target x86_64-pc-windows-gnu
bash tools/macos/package-ahmet.sh "$HOME/Downloads"
```

The packaging script builds the committed `ports/ahmet-macos` tree with the Cargo lockfile. To build a working copy directly, run `cargo +1.98.1 build --locked --release --target x86_64-pc-windows-gnu -p victory` from `ports/ahmet-macos`.

## Measured on a 128 GB M5 Max

The comparison used the same starting save in separate runs on macOS 27.2.1, CrossOver 26.3 and D3DMetal. Both builds used Series S settings, muted audio and their own warm shader caches. Timing probes counted rendered DXGI presentations. The packaged executable uses the original source without those probes.

| Prison cell at the window | Sample | Median rendered FPS | Average rendered FPS | 1% low FPS |
| --- | --- | ---: | ---: | ---: |
| Local Wave 20, first control | 93 intervals | 2.30 | 2.30 | 2.05 |
| Local Wave 20, second control | 93 intervals | 2.40 | 2.38 | 2.08 |
| Ahmet, warm default | 93 intervals | 10.66 | 10.54 | 7.29 |
| Ahmet, default, first timed run | 810 intervals, 120.004 s | 6.81 | 6.75 | 4.43 |
| Ahmet, default, independent repeat | 760 intervals, 120.136 s | 6.26 | 6.33 | 3.32 |

The warm short sample was **4.45–4.64×** the median FPS of the two Wave 20 controls. Wave 20 was a local integration at `b7dbb1b9` with idle eviction enabled and quick sync disabled. An equivalent comparison against this repository’s current main is still open. Ahmet’s build combines a different renderer, zero-copy memory aliasing, paired waves and the Series S upscaling path. The builds also draw hair and lighting differently. This comparison measures the complete configuration.

The two-minute timers started after visual confirmation of the cell. The runs had seven and nine intervals over 200 ms, respectively. One other launch crashed before reaching the timed scene. Sustained 10–12 FPS and driving performance remain unverified.

### Resolution

Ahmet’s default prison viewport was **1280×720** with a **1920×1080** presentation source. The Retina swap chain was larger again. Setting `VICTORY_RENDER_SCALE=0.5` reduced the viewport to 960×540. One matched 93-interval sample gave 10.71 median FPS versus 10.66 on default. The renderer also accepts `VICTORY_RESOLUTION=1280x720` to cap the title’s output ladder. The two settings control different stages. [Resolution implementation](https://github.com/Ahmet-Dedeler/VIctory/blob/d28238f9da092b4d6a851de01ff5191cfaf295c1/apps/victory/src/run.rs#L97-L135)
