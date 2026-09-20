#!/usr/bin/env bash
set -euo pipefail

# macOS Package (.pkg) Build Script for JEPA
# Builds release binary, creates /Applications/JEPA.app native desktop bundle,
# packages CLI into /usr/local/bin/jepa, and installs LaunchAgent.

VERSION="0.1.0"
IDENTIFIER="com.jepa.pkg"
BUILD_DIR="target/packaging_macos"
PKG_ROOT="${BUILD_DIR}/root"
SCRIPTS_DIR="${BUILD_DIR}/scripts"
OUTPUT_PKG="target/jepa-${VERSION}-macos-arm64.pkg"

echo "Building release binary for Apple Silicon (arm64)..."
cargo build --release

rm -rf "${PKG_ROOT}" "${SCRIPTS_DIR}"
mkdir -p "${PKG_ROOT}/usr/local/bin"
mkdir -p "${PKG_ROOT}/Library/LaunchAgents"
mkdir -p "${PKG_ROOT}/Applications/JEPA.app/Contents/MacOS"
mkdir -p "${PKG_ROOT}/Applications/JEPA.app/Contents/Resources"
mkdir -p "${SCRIPTS_DIR}"

# Copy binary to CLI location
cp "target/release/jepa" "${PKG_ROOT}/usr/local/bin/jepa"
chmod 755 "${PKG_ROOT}/usr/local/bin/jepa"

# Create native macOS Application Bundle (/Applications/JEPA.app)
cp "target/release/jepa" "${PKG_ROOT}/Applications/JEPA.app/Contents/MacOS/jepa"
cp "packaging/macos/Info.plist" "${PKG_ROOT}/Applications/JEPA.app/Contents/Info.plist"

# App Launcher that defaults to desktop GUI mode
cat <<'EOF' > "${PKG_ROOT}/Applications/JEPA.app/Contents/MacOS/jepa_launcher"
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${DIR}/jepa" app
EOF
chmod 755 "${PKG_ROOT}/Applications/JEPA.app/Contents/MacOS/jepa_launcher"

# Update Info.plist Executable to launcher
sed -i '' 's|<string>jepa</string>|<string>jepa_launcher</string>|g' "${PKG_ROOT}/Applications/JEPA.app/Contents/Info.plist"

# Copy LaunchAgent template
cp "packaging/macos/com.jepa.daemon.plist" "${PKG_ROOT}/Library/LaunchAgents/com.jepa.daemon.plist"

# Postinstall script to set permissions
cat <<'EOF' > "${SCRIPTS_DIR}/postinstall"
#!/bin/bash
chmod 644 /Library/LaunchAgents/com.jepa.daemon.plist || true
chmod -R 755 /Applications/JEPA.app || true
mkdir -p "$HOME/.jepa"
chmod 700 "$HOME/.jepa"
echo "JEPA Desktop and CLI installation completed successfully."
exit 0
EOF
chmod +x "${SCRIPTS_DIR}/postinstall"

# Build package using pkgbuild
echo "Generating macOS installer package: ${OUTPUT_PKG}"
pkgbuild \
    --root "${PKG_ROOT}" \
    --scripts "${SCRIPTS_DIR}" \
    --identifier "${IDENTIFIER}" \
    --version "${VERSION}" \
    --install-location "/" \
    "${OUTPUT_PKG}"

echo "Successfully built ${OUTPUT_PKG}"
