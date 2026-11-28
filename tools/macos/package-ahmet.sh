#!/usr/bin/env bash
# Build the exact checked-in Ahmet source and bundle its corresponding source with the executable.
set -euo pipefail

output_dir=${1:-"$PWD"}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
package="$scratch/VIctory-Ahmet"
mkdir "$package"

tree=$(git -C "$root" rev-parse HEAD:ports/ahmet-macos)
git -C "$root" archive --format=tar.gz --prefix=source/ \
    --output="$package/source.tar.gz" "$tree"
tar -xzf "$package/source.tar.gz" -C "$scratch"
source_dir="$scratch/source"
target_dir=${CARGO_TARGET_DIR:-"$scratch/target"}
mkdir -p "$target_dir"
target_dir=$(cd "$target_dir" && pwd)
(
    cd "$source_dir"
    CARGO_TARGET_DIR="$target_dir" cargo +1.98.1 build --locked --release \
        --target x86_64-pc-windows-gnu --package victory
)

cp "$target_dir/x86_64-pc-windows-gnu/release/victory.exe" "$package/"
cp "$source_dir/mac/play.command" "$source_dir/LICENSE" "$package/"
cp "$root/docs/ahmet-macos.md" "$package/README.md"
printf '%s\n' "$tree" > "$package/source-tree.txt"
tar -czf "$output_dir/VIctory-Ahmet-CrossOver.tar.gz" -C "$scratch" VIctory-Ahmet
echo "$output_dir/VIctory-Ahmet-CrossOver.tar.gz"
