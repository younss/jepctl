//! Native desktop window wrapper using Tao and Wry webview.

use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

use crate::types::JepaError;

/// Launch native desktop application window loading the specified URL
pub fn launch_desktop_window(target_url: &str) -> Result<(), JepaError> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("JEPA - Joint-Embedding Predictive Architecture")
        .with_inner_size(tao::dpi::LogicalSize::new(1280.0, 860.0))
        .with_min_inner_size(tao::dpi::LogicalSize::new(960.0, 640.0))
        .build(&event_loop)
        .map_err(|e| JepaError::InvalidPayload(format!("Failed to create native desktop window: {}", e)))?;

    let _webview = WebViewBuilder::new()
        .with_url(target_url)
        .build(&window)
        .map_err(|e| JepaError::InvalidPayload(format!("Failed to initialize webview: {}", e)))?;

    tracing::info!("Native desktop window launched for {}", target_url);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                tracing::info!("Desktop window closed by user. Terminating application.");
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        }
    });
}
