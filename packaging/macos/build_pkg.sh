#!/usr/bin/env bash
set -euo pipefail

# macOS Package (.pkg) Build Script for jepctl
# Builds release binary, creates /Applications/jepctl.app native desktop bundle,
# packages CLI into /usr/local/bin/jepctl, and installs LaunchAgent.

VERSION="$(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)"
IDENTIFIER="com.jepctl.pkg"
BUILD_DIR="target/packaging_macos"
PKG_ROOT="${BUILD_DIR}/root"
SCRIPTS_DIR="${BUILD_DIR}/scripts"
OUTPUT_PKG="target/jepctl-${VERSION}-macos-arm64.pkg"

echo "Building release binary for Apple Silicon (arm64)..."
cargo build --release --features metal

rm -rf "${PKG_ROOT}" "${SCRIPTS_DIR}"
mkdir -p "${PKG_ROOT}/usr/local/bin"
mkdir -p "${PKG_ROOT}/Library/LaunchAgents"
mkdir -p "${PKG_ROOT}/Applications/jepctl.app/Contents/MacOS"
mkdir -p "${PKG_ROOT}/Applications/jepctl.app/Contents/Resources"
mkdir -p "${SCRIPTS_DIR}"

# Copy binary to CLI location
cp "target/release/jepctl" "${PKG_ROOT}/usr/local/bin/jepctl"
chmod 755 "${PKG_ROOT}/usr/local/bin/jepctl"

# Create native macOS Application Bundle (/Applications/jepctl.app)
cp "target/release/jepctl" "${PKG_ROOT}/Applications/jepctl.app/Contents/MacOS/jepctl"
cp "packaging/macos/Info.plist" "${PKG_ROOT}/Applications/jepctl.app/Contents/Info.plist"

# App Launcher that defaults to desktop GUI mode
cat <<'EOF' > "${PKG_ROOT}/Applications/jepctl.app/Contents/MacOS/jepa_launcher"
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${DIR}/jepctl" app
EOF
chmod 755 "${PKG_ROOT}/Applications/jepctl.app/Contents/MacOS/jepa_launcher"

# Update Info.plist Executable to launcher
sed -i '' 's|<string>jepctl</string>|<string>jepa_launcher</string>|g' "${PKG_ROOT}/Applications/jepctl.app/Contents/Info.plist"

# Copy LaunchAgent template
cp "packaging/macos/com.jepctl.daemon.plist" "${PKG_ROOT}/Library/LaunchAgents/com.jepctl.daemon.plist"

# Postinstall script to set permissions
cat <<'EOF' > "${SCRIPTS_DIR}/postinstall"
#!/bin/bash
chmod 644 /Library/LaunchAgents/com.jepctl.daemon.plist || true
chmod -R 755 /Applications/jepctl.app || true
mkdir -p "$HOME/.jepctl"
chmod 700 "$HOME/.jepctl"
echo "jepctl desktop and CLI installation completed successfully."
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
