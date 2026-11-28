# Shared by setup.sh and the macOS launcher.
gptk_framework_version() (
    # The suffix preserves any trailing newline in the plist value for rejection.
    version=$(/usr/bin/plutil -extract CFBundleShortVersionString raw -expect string -n -o - \
        "$1/Resources/Info.plist" 2>/dev/null && printf '.') || exit 1
    case "$version" in
        *'
'*) exit 1 ;;
    esac
    printf '%s\n' "${version%.}"
)

gptk_version_supported() {
    case "${1-}" in
        ''|*[!0-9.b]*) return 1 ;;
    esac
    printf '%s\n' "$1" | LC_ALL=C awk -F '[.b]' '
        /^[0-9]+\.[0-9]+(\.[0-9]+|b[1-9][0-9]*)?$/ {
            supported = $1 > 4 || ($1 == 4 && ($2 > 0 || $0 !~ /b/ || $3 >= 3))
        }
        END { exit !supported }'
}
