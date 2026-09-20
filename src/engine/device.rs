//! Cross-platform hardware detection and telemetry for Metal, CUDA, and CPU backends.

use crate::types::{HardwareBackend, HardwareInfo};
use candle_core::Device;
use sysinfo::System;

/// Detect the optimal hardware acceleration device according to platform capabilities
/// and requested compute backend preference.
pub fn select_device(preference: Option<&str>) -> (Device, HardwareInfo) {
    #[allow(unused_variables)]
    let pref = preference.unwrap_or("auto").to_lowercase();

    // 1. Metal backend requested or auto on Apple Silicon
    #[cfg(feature = "metal")]
    {
        if pref == "metal" || pref == "auto" {
            match Device::new_metal(0) {
                Ok(dev) => {
                    let info = query_telemetry(HardwareBackend::Metal, None);
                    return (dev, info);
                }
                Err(err) => {
                    tracing::warn!("Metal acceleration unavailable: {}. Falling back.", err);
                }
            }
        }
    }

    // 2. CUDA backend requested or auto on NVIDIA systems
    #[cfg(feature = "cuda")]
    {
        if pref == "cuda" || pref == "auto" {
            match Device::new_cuda(0) {
                Ok(dev) => {
                    let info = query_telemetry(HardwareBackend::Cuda, Some("NVIDIA GPU (CUDA)"));
                    return (dev, info);
                }
                Err(err) => {
                    tracing::warn!("CUDA acceleration unavailable: {}. Falling back.", err);
                }
            }
        }
    }

    // 3. Optimized multi-threaded CPU fallback
    let dev = Device::Cpu;
    let info = query_telemetry(HardwareBackend::Cpu, None);
    (dev, info)
}

/// Query hardware memory, processor info, and thermal stats
pub fn query_telemetry(backend: HardwareBackend, device_name_override: Option<&str>) -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let memory_total_bytes = sys.total_memory();
    let memory_used_bytes = sys.used_memory();
    let memory_percent =
        if memory_total_bytes > 0 { (memory_used_bytes as f32 / memory_total_bytes as f32) * 100.0 } else { 0.0 };

    let cpu_threads = sys.cpus().len();
    let cpu_brand = sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_else(|| "Generic CPU".to_string());

    let device_name = match (backend, device_name_override) {
        (_, Some(custom)) => custom.to_string(),
        (HardwareBackend::Metal, None) => format!("Apple Silicon GPU (Metal - {})", cpu_brand),
        (HardwareBackend::Cuda, None) => "NVIDIA GPU (CUDA)".to_string(),
        (HardwareBackend::Cpu, None) => format!("CPU - {} ({} Cores)", cpu_brand, cpu_threads),
    };

    HardwareInfo {
        backend,
        device_name,
        memory_used_bytes,
        memory_total_bytes,
        memory_percent,
        temperature_celsius: None,
        cpu_threads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_the_best_compiled_backend() {
        let (_dev, info) = select_device(None);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert_eq!(info.backend, HardwareBackend::Metal);
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        assert_eq!(info.backend, HardwareBackend::Cpu);
        assert!(info.cpu_threads > 0);
        assert!(info.memory_total_bytes > 0);
    }

    #[test]
    fn explicit_cpu_preference_is_honoured() {
        let (dev, info) = select_device(Some("cpu"));
        assert!(matches!(dev, Device::Cpu));
        assert_eq!(info.backend, HardwareBackend::Cpu);
    }
}
