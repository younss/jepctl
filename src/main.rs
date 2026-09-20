//! JEPA - Local runtime, CLI, and testbench for Joint-Embedding Predictive Architectures.

pub mod auth;
pub mod config;
pub mod desktop;
pub mod engine;
pub mod gestures;
pub mod hub;
pub mod media;
pub mod server;
pub mod types;

use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::auth::AuthManager;
use crate::config::{RuntimeConfig, DEFAULT_HOST, DEFAULT_PORT};
use crate::engine::device::select_device;
use crate::engine::EngineManager;
use crate::hub::manifest::get_verified_manifests;
use crate::hub::ModelCatalog;
use crate::media::capture::CameraSupervisor;
use crate::media::image::preprocess_image_bytes;
use crate::media::ring_buffer::RingBuffer;
use crate::server::handlers::AppState;
use crate::server::middleware::create_audit_log;
use crate::types::{CreateKeyRequest, Role};

#[derive(Parser)]
#[command(
    name = "jepa",
    about = "Local runtime for Joint-Embedding Predictive Architectures (I-JEPA, V-JEPA)",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Host network interface to bind (e.g. 127.0.0.1 or 0.0.0.0)
    #[arg(short = 'H', long, default_value = DEFAULT_HOST, global = true)]
    host: String,

    /// Network port to bind
    #[arg(short = 'p', long, default_value_t = DEFAULT_PORT, global = true)]
    port: u16,

    /// Disable Bearer authentication (strictly permitted only when bound to 127.0.0.1)
    #[arg(long, global = true)]
    no_auth: bool,

    /// Force compute device backend (auto, metal, cuda, cpu)
    #[arg(long, global = true)]
    device: Option<String>,

    /// Open native desktop application window
    #[arg(long, global = true)]
    gui: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start daemon server with REST, SSE and web testbench (default action)
    Serve(ServeArgs),

    /// Launch native desktop application window with daemon running in background
    App,

    /// Launch native desktop application window (alias for app)
    Gui,

    /// Run daemon and load a specific model into active GPU memory
    Run {
        /// Model name or repository identifier (e.g. facebook/ijepa_vith14_1k)
        model: String,
    },

    /// Pull model checkpoint weights from Hugging Face hub
    Pull {
        /// Repository ID (e.g. facebook/ijepa_vith14_1k or facebookresearch/jepa:vjepa_vitl16)
        model: String,
    },

    /// Compute latent representation vector for an image or video file
    Embed {
        /// Path to target image (PNG, JPEG, WebP) or video file
        path: PathBuf,

        /// Model name to use for embedding
        #[arg(short, long)]
        model: Option<String>,

        /// Output format: json (default) or raw
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Stream continuous embeddings from webcam or video capture
    Stream {
        /// Camera device index
        #[arg(short, long, default_value_t = 0)]
        camera: usize,

        /// Target frame rate (FPS)
        #[arg(long, default_value_t = 10)]
        fps: u64,

        /// Model name to use for stream
        #[arg(short, long)]
        model: Option<String>,
    },

    /// List all locally installed models and manifests
    Tags,

    /// List all locally installed models and manifests (alias for tags)
    List,

    /// Remove a model from local storage
    Rm {
        /// Model name to remove
        model: String,
    },

    /// Manage API access keys and RBAC tokens
    Key {
        #[command(subcommand)]
        sub: KeyCommands,
    },
}

#[derive(Args)]
struct ServeArgs {
    /// Open native desktop application window
    #[arg(long)]
    gui: bool,
}

#[derive(Subcommand)]
enum KeyCommands {
    /// Generate a new API token
    Generate {
        /// Name / description for token
        #[arg(short, long, default_value = "CLI Generated Key")]
        name: String,

        /// Role scope: inference or admin
        #[arg(short, long, default_value = "inference")]
        role: String,

        /// Days until expiration (optional)
        #[arg(short, long)]
        days: Option<i64>,
    },

    /// List active API tokens
    List,

    /// Revoke an API token by prefix
    Revoke {
        /// Token prefix (first 12 characters)
        prefix: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,jepa=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    // Security check: --no-auth is strictly restricted to 127.0.0.1 or localhost
    if cli.no_auth && cli.host != "127.0.0.1" && cli.host != "localhost" {
        eprintln!("Security violation: --no-auth is strictly forbidden when binding to external interfaces ({}).", cli.host);
        eprintln!("Authentication must remain enabled for external/LAN connections.");
        std::process::exit(1);
    }

    // Initialize configuration
    let config = Arc::new(RuntimeConfig::init(cli.host.clone(), cli.port, cli.no_auth)?);
    let settings = config.load_settings();

    // Initialize hardware device
    let requested_backend = cli.device.as_deref().unwrap_or(&settings.compute_backend);
    let (dev, hw_info) = select_device(Some(requested_backend));
    tracing::info!("Initialized hardware backend: {}", hw_info.device_name);

    // Initialize core components
    let engine = EngineManager::new(dev, hw_info);
    let catalog = Arc::new(ModelCatalog::new(config.clone()));
    let auth = AuthManager::init(&config.keys_db_path, &config.auth_token_path, cli.no_auth)?;
    let audit_log = create_audit_log();
    let ring_buffer = Arc::new(tokio::sync::RwLock::new(RingBuffer::new(16)));
    let camera_supervisor = Arc::new(CameraSupervisor::new(ring_buffer.clone()));
    let embeddings_total = Arc::new(AtomicU64::new(0));
    let gestures = Arc::new(tokio::sync::RwLock::new(crate::gestures::GestureStore::load(&config.gestures_path)));

    let app_state = AppState {
        engine: engine.clone(),
        catalog: catalog.clone(),
        auth: auth.clone(),
        audit_log,
        camera_supervisor: camera_supervisor.clone(),
        ring_buffer: ring_buffer.clone(),
        embeddings_total,
        gestures,
        start_time: Instant::now(),
        config: config.clone(),
    };

    let should_launch_gui = cli.gui
        || matches!(cli.command, Some(Commands::App) | Some(Commands::Gui))
        || match &cli.command {
            Some(Commands::Serve(args)) => args.gui,
            _ => false,
        };

    match cli.command {
        None | Some(Commands::Serve(_)) | Some(Commands::App) | Some(Commands::Gui) => {
            print_startup_banner(&config, &auth);
            if should_launch_gui {
                let host = cli.host.clone();
                let port = cli.port;
                let url = format!("http://{}:{}", host, port);

                tokio::spawn(async move {
                    if let Err(e) = server::start_daemon(app_state, &host, port).await {
                        tracing::error!("Daemon server error: {}", e);
                    }
                });

                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                desktop::launch_desktop_window(&url)?;
            } else {
                server::start_daemon(app_state, &cli.host, cli.port).await?;
            }
        }

        Some(Commands::Run { model }) => {
            print_startup_banner(&config, &auth);

            // Load specified model
            if let Some(m) = catalog.get_manifest(&model) {
                let weights = catalog.get_weights_path(&model);
                engine.load_model(m, weights.as_deref()).await?;
                println!("Loaded model '{}' into memory.", model);
            } else {
                println!("Model '{}' not found in catalog. Attempting to pull first...", model);
                let _ = catalog.clone().start_pull(model.clone());
            }

            if should_launch_gui {
                let host = cli.host.clone();
                let port = cli.port;
                let url = format!("http://{}:{}", host, port);

                tokio::spawn(async move {
                    if let Err(e) = server::start_daemon(app_state, &host, port).await {
                        tracing::error!("Daemon server error: {}", e);
                    }
                });

                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                desktop::launch_desktop_window(&url)?;
            } else {
                server::start_daemon(app_state, &cli.host, cli.port).await?;
            }
        }

        Some(Commands::Pull { model }) => {
            println!("Pulling model '{}' from Hugging Face hub...", model);
            let mut rx = catalog.start_pull(model.clone());

            while let Ok(event) = rx.recv().await {
                if let Some(err) = event.error {
                    eprintln!("\nError pulling model: {}", err);
                    std::process::exit(1);
                }

                print!(
                    "\rDownloading [status: {}] {:>5.1}% | Speed: {:>5.1} MB/s",
                    event.status, event.percentage, event.speed_mb_s
                );
                std::io::Write::flush(&mut std::io::stdout())?;

                if event.finished {
                    println!("\nSuccessfully pulled model '{}'.", model);
                    break;
                }
            }
        }

        Some(Commands::Embed { path, model, format }) => {
            if !path.exists() {
                eprintln!("File not found: {}", path.display());
                std::process::exit(1);
            }

            let model_name = model.unwrap_or_else(|| "facebook/ijepa_vitb16_1k".to_string());
            if let Some(m) = catalog.get_manifest(&model_name) {
                let weights = catalog.get_weights_path(&model_name);
                engine.load_model(m, weights.as_deref()).await?;
            } else {
                eprintln!("Model '{}' not found in catalog.", model_name);
                std::process::exit(1);
            }

            let file_bytes = std::fs::read(&path)?;
            let tensor = preprocess_image_bytes(&file_bytes, 224, 224, &engine.device)?;
            let (m_name, dim, embedding, _patches, latency_ms) = engine.embed_image(&tensor).await?;

            if format == "raw" {
                for (i, val) in embedding.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.6}", val);
                }
                println!();
            } else {
                let out = serde_json::json!({
                    "model": m_name,
                    "dimension": dim,
                    "latency_ms": latency_ms,
                    "embedding": embedding
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
            }
        }

        Some(Commands::Stream { camera, fps, model }) => {
            let model_name = model.unwrap_or_else(|| "facebookresearch/jepa:vjepa_vitl16".to_string());
            if let Some(m) = catalog.get_manifest(&model_name) {
                let weights = catalog.get_weights_path(&model_name);
                engine.load_model(m, weights.as_deref()).await?;
                println!("Loaded model '{}' for streaming.", model_name);
            }

            camera_supervisor.start(camera, fps)?;
            println!("Streaming embeddings from camera device {} at {} FPS. Press Ctrl-C to stop.", camera, fps);

            let mut frame_rx = camera_supervisor.subscribe();
            let mut frame_count: u64 = 0;

            while let Ok(_) = frame_rx.recv().await {
                frame_count += 1;
                let vid_tensor = {
                    let lock = ring_buffer.read().await;
                    lock.to_video_tensor(224, 224, &engine.device)?
                };

                let (m_name, _dim, embedding, latency_ms) = engine.embed_video(&vid_tensor).await?;
                let out = serde_json::json!({
                    "frame": frame_count,
                    "model": m_name,
                    "latency_ms": latency_ms,
                    "dim": embedding.len(),
                    "sample": &embedding[..embedding.len().min(4)]
                });
                println!("{}", serde_json::to_string(&out)?);
            }
        }

        Some(Commands::Tags) | Some(Commands::List) => {
            let models = catalog.list_installed();
            if models.is_empty() {
                println!("No installed models found in ~/.jepa/models.");
                println!("Run 'jepa pull <model>' to download a model.");
                println!("\nVerified models available:");
                for v in get_verified_manifests() {
                    println!("  - {:<40} ({} | {} dims)", v.name, v.architecture, v.embed_dim);
                }
            } else {
                println!("{:<42} {:<10} {:<10} {:<10} {:<12}", "NAME", "MODALITY", "DIMS", "PARAMS", "DISK SIZE");
                for m in models {
                    let size_mb = (m.disk_size_bytes / (1024 * 1024)).to_string() + " MB";
                    println!("{:<42} {:<10} {:<10} {:<10} {:<12}", m.name, m.modality.to_string(), m.embed_dim, m.parameter_count, size_mb);
                }
            }
        }

        Some(Commands::Rm { model }) => {
            match catalog.delete_model(&model)? {
                true => println!("Successfully removed model '{}'.", model),
                false => println!("Model '{}' was not found on disk.", model),
            }
        }

        Some(Commands::Key { sub }) => match sub {
            KeyCommands::Generate { name, role, days } => {
                let r = match role.to_lowercase().as_str() {
                    "admin" => Role::Admin,
                    _ => Role::Inference,
                };
                let req = CreateKeyRequest {
                    name,
                    role: r,
                    expire_days: days,
                };
                let res = auth.create_key(req).await?;
                println!("Generated API Key:");
                println!("  Prefix:      {}", res.key_prefix);
                println!("  Role:        {}", res.role);
                println!("  Name:        {}", res.name);
                println!("  Bearer Token (Save now - will not be displayed again):");
                println!("  {}", res.raw_token);
            }

            KeyCommands::List => {
                let keys = auth.list_keys().await;
                println!("{:<14} {:<12} {:<24} {:<26}", "PREFIX", "ROLE", "NAME", "CREATED");
                for k in keys {
                    let created = k.created_at.to_rfc3339();
                    println!("{:<14} {:<12} {:<24} {:<26}", k.key_prefix, k.role.to_string(), k.name, created);
                }
            }

            KeyCommands::Revoke { prefix } => {
                if auth.revoke_key(&prefix).await? {
                    println!("Successfully revoked key with prefix '{}'.", prefix);
                } else {
                    println!("Key prefix '{}' not found.", prefix);
                }
            }
        },
    }

    Ok(())
}

fn print_startup_banner(config: &RuntimeConfig, auth: &AuthManager) {
    println!(r#"
       ___ _____ ___   _   
      |_  |  ___| ___ \ /_\  
        | | |__ | |_/ // _ \ 
        | |  __||  __/ / _ \ 
    /\__/ / |___| |   / ___ \
    \____/\____/\_|  /_/   \_\  v0.1.0
"#);
    println!("  Joint-Embedding Predictive Architecture Local Runtime");
    println!("  Serving web testbench at: http://{}:{}", config.host, config.port);
    println!("  Root Storage: {}", config.home_dir.display());

    if auth.no_auth_enabled {
        println!("  Authentication: DISABLED (--no-auth development mode)");
    } else {
        println!("  Authentication: ACTIVE (Bearer token validation enforced)");
        if let Ok(token) = std::fs::read_to_string(&config.auth_token_path) {
            println!("  Admin API Token: {}", token.trim());
        }
    }
    println!();
}
