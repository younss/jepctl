#!/usr/bin/env bash
set -euo pipefail

# Linux Debian (.deb) Package and Tarball Generator for jepctl
VERSION="$(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo "amd64")"
PKG_NAME="jepa_${VERSION}_${ARCH}"
BUILD_DIR="target/packaging_linux/${PKG_NAME}"

echo "Building release binary for Linux..."
cargo build --release ${JEPA_FEATURES:+--features "$JEPA_FEATURES"}

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/DEBIAN"
mkdir -p "${BUILD_DIR}/usr/bin"
mkdir -p "${BUILD_DIR}/usr/lib/systemd/user"
mkdir -p "${BUILD_DIR}/usr/share/applications"
mkdir -p "${BUILD_DIR}/etc/udev/rules.d"

# Copy binary
cp "target/release/jepctl" "${BUILD_DIR}/usr/bin/jepctl"
chmod 755 "${BUILD_DIR}/usr/bin/jepctl"

# Copy systemd unit
cp "packaging/linux/jepctl.service" "${BUILD_DIR}/usr/lib/systemd/user/jepctl.service"

# Create Desktop Application entry
cat <<'EOF' > "${BUILD_DIR}/usr/share/applications/jepctl.desktop"
[Desktop Entry]
Name=jepctl
Comment=Joint-Embedding Predictive Architecture Desktop App
Exec=/usr/bin/jepctl app
Terminal=false
Type=Application
Categories=Development;Science;ArtificialIntelligence;
EOF

# Udev rule for camera permissions
cat <<'EOF' > "${BUILD_DIR}/etc/udev/rules.d/99-jepctl-camera.rules"
KERNEL=="video[0-9]*", GROUP="video", MODE="0660"
EOF

# Debian Control file
cat <<EOF > "${BUILD_DIR}/DEBIAN/control"
Package: jepctl
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: jepctl contributors <support@jepctl.local>
Description: High-performance local runtime for Joint-Embedding Predictive Architectures (JEPA)
 Operates like Ollama for non-generative representation learning models
 (I-JEPA, V-JEPA). Serves native desktop GUI, CLI, embedded web UI, and REST/SSE APIs.
EOF

# Debian post-installation script
cat <<'EOF' > "${BUILD_DIR}/DEBIAN/postinst"
#!/bin/sh
set -e
echo "Checking video group membership for camera access..."
if ! getent group video >/dev/null; then
    groupadd -r video || true
fi

echo "jepctl installation completed."
echo "To start the user daemon, execute:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now jepctl"
exit 0
EOF
chmod 755 "${BUILD_DIR}/DEBIAN/postinst"

# Build .deb package if dpkg-deb is present
if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb --build "${BUILD_DIR}" "target/${PKG_NAME}.deb"
    echo "Built target/${PKG_NAME}.deb successfully."
fi

# Build standalone tarball
TARBALL="target/jepctl-${VERSION}-linux-${ARCH}.tar.gz"
tar -czf "${TARBALL}" -C "${BUILD_DIR}/usr/bin" jepctl
echo "Built standalone tarball ${TARBALL} successfully."
