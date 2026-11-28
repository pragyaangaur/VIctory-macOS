#!/bin/sh
set -eu
tool_dir=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$tool_dir/gptk-version.sh" ] || { echo "FAIL: no shared GPTK version check" >&2; exit 1; }
. "$tool_dir/gptk-version.sh"

for version in 3.0 4.0b1 4.0b2 '' 4 4.0b 4.0b0 4.1b0 4.0beta3 '4.0 beta 3' '4\0560' '4.0b3+' '4.0
4.1' 3.99b99; do
    if gptk_version_supported "$version"; then
        printf 'FAIL: accepted <%s>\n' "$version" >&2
        exit 1
    fi
done
for version in 4.0b3 4.0b10 4.0 4.1 4.1b1 4.0.1 5.0b1; do
    gptk_version_supported "$version" || { echo "FAIL: rejected $version" >&2; exit 1; }
done
echo "GPTK version checks passed"
