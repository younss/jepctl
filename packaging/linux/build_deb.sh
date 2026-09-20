#!/usr/bin/env bash
set -euo pipefail

# Linux Debian (.deb) Package and Tarball Generator for JEPA
VERSION="0.1.0"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo "amd64")"
PKG_NAME="jepa_${VERSION}_${ARCH}"
BUILD_DIR="target/packaging_linux/${PKG_NAME}"

echo "Building release binary for Linux..."
cargo build --release

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/DEBIAN"
mkdir -p "${BUILD_DIR}/usr/bin"
mkdir -p "${BUILD_DIR}/usr/lib/systemd/user"
mkdir -p "${BUILD_DIR}/usr/share/applications"
mkdir -p "${BUILD_DIR}/etc/udev/rules.d"

# Copy binary
cp "target/release/jepa" "${BUILD_DIR}/usr/bin/jepa"
chmod 755 "${BUILD_DIR}/usr/bin/jepa"

# Copy systemd unit
cp "packaging/linux/jepa.service" "${BUILD_DIR}/usr/lib/systemd/user/jepa.service"

# Create Desktop Application entry
cat <<'EOF' > "${BUILD_DIR}/usr/share/applications/jepa.desktop"
[Desktop Entry]
Name=JEPA
Comment=Joint-Embedding Predictive Architecture Desktop App
Exec=/usr/bin/jepa app
Terminal=false
Type=Application
Categories=Development;Science;ArtificialIntelligence;
EOF

# Udev rule for camera permissions
cat <<'EOF' > "${BUILD_DIR}/etc/udev/rules.d/99-jepa-camera.rules"
KERNEL=="video[0-9]*", GROUP="video", MODE="0660"
EOF

# Debian Control file
cat <<EOF > "${BUILD_DIR}/DEBIAN/control"
Package: jepa
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: JEPA Engineering Team <support@jepa.local>
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

echo "JEPA installation completed."
echo "To start the user daemon, execute:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user enable --now jepa"
exit 0
EOF
chmod 755 "${BUILD_DIR}/DEBIAN/postinst"

# Build .deb package if dpkg-deb is present
if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb --build "${BUILD_DIR}" "target/${PKG_NAME}.deb"
    echo "Built target/${PKG_NAME}.deb successfully."
fi

# Build standalone tarball
TARBALL="target/jepa-${VERSION}-linux-${ARCH}.tar.gz"
tar -czf "${TARBALL}" -C "${BUILD_DIR}/usr/bin" jepa
echo "Built standalone tarball ${TARBALL} successfully."
