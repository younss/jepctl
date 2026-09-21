//! jepctl: local runtime, CLI, and testbench for Joint-Embedding Predictive Architectures.

pub mod auth;
pub mod companion;
pub mod config;
pub mod desktop;
pub mod engine;
pub mod gestures;
pub mod hub;
pub mod media;
pub mod robot;
pub mod server;
pub mod types;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::auth::AuthManager;
use crate::config::{DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, RuntimeConfig};
use crate::engine::EngineManager;
use crate::engine::device::select_device;
use crate::hub::ModelCatalog;
use crate::hub::manifest::get_verified_manifests;
use crate::media::capture::CameraSupervisor;
use crate::media::image::preprocess_image_bytes;
use crate::media::ring_buffer::RingBuffer;
use crate::server::handlers::AppState;
use crate::server::middleware::create_audit_log;
use crate::types::{CreateKeyRequest, Role};

#[derive(Parser)]
#[command(
    name = "jepctl",
    about = "jepctl: local runtime, CLI and testbench for JEPA style encoders (I-JEPA, V-JEPA 2, DINOv2, ViT, AudioMAE)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Host network interface to bind (e.g. 127.0.0.1 or 0.0.0.0). Overrides the
    /// "Allow access from other machines" setting when given.
    #[arg(short = 'H', long, global = true)]
    host: Option<String>,

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

    /// Browser origins allowed to call the API cross-site (comma separated,
    /// e.g. "http://localhost:5173,https://app.example.com"). Default: same-origin only.
    #[arg(long, global = true, env = "JEPA_CORS_ORIGINS", value_delimiter = ',')]
    cors_origins: Vec<String>,
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
        /// Hugging Face repository ID (e.g. facebook/ijepa_vith14_1k or facebook/dinov2-small)
        model: String,
    },

    /// Compute the latent representation of an image or a clip (GIF/WebP natively, MP4/WebM via ffmpeg)
    Embed {
        /// Path to an image (PNG, JPEG, WebP) or a clip (GIF, animated WebP, MP4, WebM)
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
    Tags {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },

    /// List all locally installed models and manifests (alias for tags)
    List {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },

    /// Manage few-shot gesture prototypes (same registry as the GUI and the API)
    Gestures {
        #[command(subcommand)]
        sub: GestureCommands,
    },

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
enum GestureCommands {
    /// List registered gestures (all models)
    List {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },

    /// Export gestures as a portable bundle (stdout or file)
    Export {
        /// Restrict to one model (default: every model)
        #[arg(short, long)]
        model: Option<String>,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Decision threshold to record in the bundle
        #[arg(long, default_value_t = gestures::DEFAULT_THRESHOLD)]
        threshold: f32,

        /// Runner-up margin to record in the bundle
        #[arg(long, default_value_t = gestures::DEFAULT_MARGIN)]
        margin: f32,

        /// Drop thumbnails to keep the bundle small
        #[arg(long)]
        no_thumbnails: bool,
    },

    /// Import a bundle produced by `export` (or GET /api/gestures/export)
    Import {
        /// Bundle file
        path: PathBuf,

        /// Remove existing gestures of the bundle's models first
        #[arg(long)]
        replace: bool,
    },

    /// Match an image file against the registered gestures of a model
    Match {
        /// Image file (PNG, JPEG, WebP)
        path: PathBuf,

        /// Model to embed with (default: facebook/ijepa_vith14_1k)
        #[arg(short, long)]
        model: Option<String>,

        #[arg(long, default_value_t = gestures::DEFAULT_THRESHOLD)]
        threshold: f32,

        #[arg(long, default_value_t = gestures::DEFAULT_MARGIN)]
        margin: f32,
    },

    /// Delete one gesture, or all gestures of a model with --model
    Remove {
        /// Gesture name
        name: Option<String>,

        /// Remove every gesture registered with this model
        #[arg(short, long)]
        model: Option<String>,
    },
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
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,jepctl=debug".into()))
        // Logs go to stderr so `jepctl tags --json | jq` and friends stay parseable.
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    let cli = Cli::parse();

    // Initialize configuration (creates ~/.jepctl and reads settings.json).
    let mut runtime_config = RuntimeConfig::init(DEFAULT_HOST.to_string(), cli.port, cli.no_auth)?;

    // The bind interface: an explicit `--host` always wins; otherwise the persisted
    // "Allow access from other machines" toggle chooses loopback (default) or all
    // interfaces, so a restart is all it takes to apply the toggle.
    runtime_config.host = match &cli.host {
        Some(h) => h.clone(),
        None if runtime_config.load_settings().allow_lan => "0.0.0.0".to_string(),
        None => DEFAULT_HOST.to_string(),
    };
    let host = runtime_config.host.clone();
    let is_loopback = host == "127.0.0.1" || host == "localhost";

    // Security: --no-auth is only ever allowed on loopback.
    if cli.no_auth && !is_loopback {
        eprintln!("Security violation: --no-auth is forbidden when binding an external interface ({host}).");
        eprintln!("Turn off Allow access from other machines in Settings, or keep authentication on for LAN access.");
        std::process::exit(1);
    }
    if !is_loopback {
        tracing::warn!(
            "Binding {host}: reachable from other machines on your network. Every request needs a Bearer token; keep tokens scoped and short-lived."
        );
    }

    runtime_config.cors_origins =
        cli.cors_origins.iter().map(|o| o.trim().to_string()).filter(|o| !o.is_empty()).collect();
    let config = Arc::new(runtime_config);
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

    let robot = crate::robot::RobotHandle::new(settings.robot_hardware.clone().unwrap_or_default());
    crate::robot::controller::spawn_control_loop(robot.clone());
    {
        // Virtual arm is connected from the start so the twin moves immediately.
        let mut core = robot.core.lock().await;
        let _ = core.backend.connect();
        // What was learned in earlier sessions is reloaded.
        if let Ok(text) = std::fs::read_to_string(&config.world_model_path) {
            match serde_json::from_str::<crate::robot::world_model::LatentWorldModel>(&text) {
                Ok(mut w) => {
                    w.fit();
                    tracing::info!("Loaded robot world model: {} transitions ({} dims)", w.transitions.len(), w.dim);
                    core.agent.world = w;
                }
                Err(e) => tracing::warn!("Ignoring unreadable world model: {}", e),
            }
        }
    }

    let companion = crate::companion::CompanionHandle::new();
    crate::companion::spawn_control_loop(companion.clone());
    if let Ok(text) = std::fs::read_to_string(&config.companion_path) {
        match serde_json::from_str::<crate::companion::CompanionMemory>(&text) {
            Ok(m) => {
                tracing::info!("Loaded companion memory: {} cues", m.cues.len());
                companion.core.lock().await.restore(m);
            }
            Err(e) => tracing::warn!("Ignoring unreadable companion memory: {}", e),
        }
    }

    let app_state = AppState {
        engine: engine.clone(),
        catalog: catalog.clone(),
        auth: auth.clone(),
        audit_log,
        camera_supervisor: camera_supervisor.clone(),
        ring_buffer: ring_buffer.clone(),
        embeddings_total,
        gestures: gestures.clone(),
        sounds: Arc::new(tokio::sync::RwLock::new(crate::gestures::GestureStore::load(&config.sounds_path))),
        mic: Arc::new(crate::media::mic::MicSupervisor::new()),
        companion: companion.clone(),
        camera_roi: Arc::new(tokio::sync::RwLock::new(settings.camera_roi)),
        robot: robot.clone(),
        start_time: Instant::now(),
        config: config.clone(),
    };

    crate::server::robot_handlers::spawn_camera_observer(app_state.clone());
    crate::server::companion_handlers::spawn_observers(app_state.clone());

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
                let host = host.clone();
                let port = cli.port;
                // The desktop window always talks to loopback even when the daemon
                // also listens on the LAN.
                let url = format!("http://127.0.0.1:{}", port);

                tokio::spawn(async move {
                    if let Err(e) = server::start_daemon(app_state, &host, port).await {
                        tracing::error!("Daemon server error: {}", e);
                    }
                });

                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                desktop::launch_desktop_window(&url)?;
            } else {
                server::start_daemon(app_state, &host, cli.port).await?;
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
                let host = host.clone();
                let port = cli.port;
                // The desktop window always talks to loopback even when the daemon
                // also listens on the LAN.
                let url = format!("http://127.0.0.1:{}", port);

                tokio::spawn(async move {
                    if let Err(e) = server::start_daemon(app_state, &host, port).await {
                        tracing::error!("Daemon server error: {}", e);
                    }
                });

                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                desktop::launch_desktop_window(&url)?;
            } else {
                server::start_daemon(app_state, &host, cli.port).await?;
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

            let model_name = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
            if let Some(m) = catalog.get_manifest(&model_name) {
                let weights = catalog.get_weights_path(&model_name);
                engine.load_model(m, weights.as_deref()).await?;
            } else {
                eprintln!("Model '{}' not found in catalog.", model_name);
                std::process::exit(1);
            }

            let file_bytes = std::fs::read(&path)?;
            if crate::media::audio::sniff_audio_format(&file_bytes).is_some() {
                let clip = crate::media::audio::decode_audio_path(&path)?;
                let (m, d, e, _p, lat) = engine.embed_audio(&clip).await?;
                eprintln!(
                    "Decoded {:.2} s of audio @ {} Hz for {}.",
                    clip.samples.len() as f32 / clip.sample_rate as f32,
                    clip.sample_rate,
                    m
                );
                print_embedding(&format, &m, d, &e, lat)?;
                return Ok(());
            }
            let media = crate::media::image::sniff_media_format(&file_bytes)?;
            let animated_webp = media == "webp" && crate::media::video::decode_clip_bytes(&file_bytes, 1, 1.0).is_ok();
            let (m_name, dim, embedding, latency_ms) = if matches!(media, "gif" | "mp4" | "webm") || animated_webp {
                let frames =
                    crate::media::video::decode_clip_path(&path, crate::media::video::MAX_DECODED_FRAMES, 8.0)?;
                let (m, d, e, _p, lat, used) = engine.embed_frames(&frames).await?;
                eprintln!("Decoded {} frame(s), sampled {} for {}.", frames.len(), used, m);
                (m, d, e, lat)
            } else {
                let prep = engine.preprocessing().await;
                let tensor = preprocess_image_bytes(&file_bytes, &prep, &engine.device)?;
                let (m, d, e, _p, lat) = engine.embed_image(&tensor).await?;
                (m, d, e, lat)
            };

            print_embedding(&format, &m_name, dim, &embedding, latency_ms)?;
        }

        Some(Commands::Stream { camera, fps, model }) => {
            let model_name = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
            let Some(m) = catalog.get_manifest(&model_name) else {
                eprintln!("Model '{}' not found in catalog.", model_name);
                std::process::exit(1);
            };
            let weights = catalog.get_weights_path(&model_name);
            engine.load_model(m, weights.as_deref()).await?;
            let prep = engine.preprocessing().await;
            println!("Loaded model '{}' for streaming.", model_name);

            camera_supervisor.start(camera, fps)?;
            println!("Streaming embeddings from camera device {} at {} FPS. Press Ctrl-C to stop.", camera, fps);

            let mut frame_rx = camera_supervisor.subscribe();
            let mut frame_count: u64 = 0;

            while frame_rx.recv().await.is_ok() {
                frame_count += 1;
                let vid_tensor = {
                    let lock = ring_buffer.read().await;
                    lock.to_video_tensor(&prep, settings.camera_roi.as_ref(), &engine.device)?
                };

                let (m_name, _dim, embedding, _patches, latency_ms) = engine.embed_video(&vid_tensor).await?;
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

        Some(Commands::Tags { json }) | Some(Commands::List { json }) => {
            let models = catalog.list_installed();
            if json {
                println!("{}", serde_json::to_string_pretty(&models)?);
            } else if models.is_empty() {
                println!("No installed models found in ~/.jepctl/models.");
                println!("Run 'jepctl pull <model>' to download a model.");
                println!("\nVerified models available:");
                for v in get_verified_manifests() {
                    println!("  - {:<40} ({} | {} dims)", v.name, v.architecture, v.embed_dim);
                }
            } else {
                println!("{:<42} {:<10} {:<10} {:<10} {:<12}", "NAME", "MODALITY", "DIMS", "PARAMS", "DISK SIZE");
                for m in models {
                    let size_mb = (m.disk_size_bytes / (1024 * 1024)).to_string() + " MB";
                    println!(
                        "{:<42} {:<10} {:<10} {:<10} {:<12}",
                        m.name,
                        m.modality.to_string(),
                        m.embed_dim,
                        m.parameter_count,
                        size_mb
                    );
                }
            }
        }

        Some(Commands::Rm { model }) => match catalog.delete_model(&model)? {
            true => println!("Successfully removed model '{}'.", model),
            false => println!("Model '{}' was not found on disk.", model),
        },

        Some(Commands::Gestures { sub }) => {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            match sub {
                GestureCommands::List { json } => {
                    let store = gestures.read().await;
                    let mut all: Vec<&gestures::RegisteredGesture> = store.gestures.values().collect();
                    all.sort_by_key(|g| (g.model_name.clone(), g.created_at));
                    if json {
                        let items: Vec<serde_json::Value> = all
                            .iter()
                            .map(|g| {
                                serde_json::json!({
                                    "name": g.name, "model_name": g.model_name, "dimension": g.dimension,
                                    "sample_count": g.samples.len(), "is_neutral": g.is_neutral,
                                    "created_at": g.created_at, "updated_at": g.updated_at
                                })
                            })
                            .collect();
                        println!("{}", serde_json::to_string_pretty(&items)?);
                    } else if all.is_empty() {
                        println!("No gestures registered. Use the GUI (Gestures tab) or POST /api/gestures.");
                    } else {
                        println!("{:<24} {:<40} {:<6} {:<8} NEUTRAL", "NAME", "MODEL", "DIMS", "SAMPLES");
                        for g in all {
                            println!(
                                "{:<24} {:<40} {:<6} {:<8} {}",
                                g.name,
                                g.model_name,
                                g.dimension,
                                g.samples.len(),
                                if g.is_neutral { "yes" } else { "" }
                            );
                        }
                    }
                }

                GestureCommands::Export { model, output, threshold, margin, no_thumbnails } => {
                    let store = gestures.read().await;
                    let bundle = store.export(model.as_deref(), threshold, margin, !no_thumbnails, now);
                    let json = serde_json::to_string_pretty(&bundle)?;
                    match output {
                        Some(path) => {
                            std::fs::write(&path, json)?;
                            eprintln!("Exported {} gesture(s) to {}", bundle.gestures.len(), path.display());
                        }
                        None => println!("{json}"),
                    }
                }

                GestureCommands::Import { path, replace } => {
                    let text = std::fs::read_to_string(&path)?;
                    let bundle: gestures::GestureBundle = serde_json::from_str(&text)?;
                    let mut store = gestures.write().await;
                    let report = store.import(bundle, replace)?;
                    store.save()?;
                    println!(
                        "Imported {} gesture(s) for {:?} (removed {}).",
                        report.imported, report.models, report.removed
                    );
                }

                GestureCommands::Match { path, model, threshold, margin } => {
                    let model_name = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
                    let Some(m) = catalog.get_manifest(&model_name) else {
                        eprintln!("Model '{}' not found in catalog.", model_name);
                        std::process::exit(2);
                    };
                    let weights = catalog.get_weights_path(&model_name);
                    engine.load_model(m, weights.as_deref()).await?;
                    let prep = engine.preprocessing().await;
                    let bytes = std::fs::read(&path)?;
                    let tensor = preprocess_image_bytes(&bytes, &prep, &engine.device)?;
                    let (_m, _d, embedding, patches, _lat) = engine.embed_image(&tensor).await?;
                    let store = gestures.read().await;
                    let registered = store.for_model(&model_name);
                    if registered.is_empty() {
                        eprintln!("No gestures registered for '{}'.", model_name);
                        std::process::exit(2);
                    }
                    let result =
                        gestures::match_gestures(&embedding, patches.as_deref(), &registered, threshold, margin);
                    println!("{}", serde_json::to_string_pretty(&result)?);
                    if !result.detected {
                        std::process::exit(1);
                    }
                }

                GestureCommands::Remove { name, model } => {
                    let mut store = gestures.write().await;
                    let removed = match (name, model) {
                        (Some(n), _) => usize::from(store.remove(&n).is_some()),
                        (None, Some(m)) => {
                            let before = store.gestures.len();
                            store.gestures.retain(|_, g| g.model_name != m);
                            before - store.gestures.len()
                        }
                        (None, None) => {
                            eprintln!("Give a gesture name or --model <name>.");
                            std::process::exit(2);
                        }
                    };
                    store.save()?;
                    println!("Removed {} gesture(s).", removed);
                }
            }
        }

        Some(Commands::Key { sub }) => match sub {
            KeyCommands::Generate { name, role, days } => {
                let r = match role.to_lowercase().as_str() {
                    "admin" => Role::Admin,
                    _ => Role::Inference,
                };
                let req = CreateKeyRequest { name, role: r, expire_days: days };
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

/// `jepctl embed` output: `raw` comma-separated floats, or JSON.
fn print_embedding(
    format: &str,
    model: &str,
    dim: usize,
    embedding: &[f32],
    latency_ms: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    if format == "raw" {
        let line: Vec<String> = embedding.iter().map(|v| format!("{v:.6}")).collect();
        println!("{}", line.join(", "));
    } else {
        let out =
            serde_json::json!({ "model": model, "dimension": dim, "latency_ms": latency_ms, "embedding": embedding });
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    Ok(())
}

fn print_startup_banner(config: &RuntimeConfig, auth: &AuthManager) {
    println!(
        r#"
        _                 _   _ 
       (_) ___ _ __   ___| |_| |
       | |/ _ \ '_ \ / __| __| |
       | |  __/ |_) | (__| |_| |
      _/ |\___| .__/ \___|\__|_|
     |__/     |_|          v{version}
"#,
        version = env!("CARGO_PKG_VERSION")
    );
    println!("  jepctl: local runtime for JEPA style encoders");
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
