//! Build script for jepctl runtime.
//! Handles platform entitlement verification, asset change detection,
//! and compile-time metadata injection.

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/ui/index.html");
    println!("cargo:rerun-if-changed=src/ui/app.js");
    println!("cargo:rerun-if-changed=src/ui/styles.css");
    println!("cargo:rerun-if-changed=packaging/macos/Info.plist");
    println!("cargo:rerun-if-changed=packaging/macos/com.jepctl.daemon.plist");
    println!("cargo:rerun-if-changed=packaging/linux/jepctl.service");
    println!("cargo:rerun-if-changed=packaging/windows/installer.iss");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    println!("cargo:rustc-env=JEPA_TARGET_OS={}", target_os);
    println!("cargo:rustc-env=JEPA_TARGET_ARCH={}", target_arch);

    // Platform-specific entitlement and security injection
    if target_os == "macos" {
        let plist_path = Path::new("packaging/macos/Info.plist");
        if plist_path.exists() {
            println!("cargo:rustc-env=JEPA_MACOS_PLIST={}", plist_path.display());
        }
    }
}
