# Security policy

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private advisory form:
https://github.com/younss/jepctl/security/advisories/new

You will get an acknowledgement within a few days. Fixes are released as patch versions and
credited unless you prefer otherwise.

## Threat model (what the daemon protects against)

`jepctl serve` is a local service that can read your camera. The defaults are chosen so that
**a web page open in your browser cannot use it**:

- Binds to `127.0.0.1` only. `--host 0.0.0.0` requires authentication (`--no-auth` is refused).
- Bearer tokens (`~/.jepctl/auth.token`, plus `jepctl key …` scoped tokens) with constant-time comparison; roles `admin` and `inference`. `keys.json` holds SHA-256 digests only (files from before 0.3.1 are migrated on first start); the admin token itself lives in `auth.token` (mode 0600) because the desktop testbench needs it to bootstrap.
- **No CORS headers by default.** Cross-origin pages cannot read responses. Opt in per origin with `--cors-origins`.
- `/api/auth/token` (testbench bootstrap) answers only same-origin requests (`Sec-Fetch-Site`) on loopback.
- Camera control, camera frames, ring buffer, microphone, embeddings stream, gesture and sound data, robot and companion control require an `inference` token. The SSE stream and the WebSockets accept `?token=` because `EventSource` and `WebSocket` cannot set headers: do not put those URLs in logs.
- The daemon can also **listen** (microphone) and **move** things (robot backends). Both are off until an authenticated client starts them; the physical arm additionally holds every command behind the safety gate until approved, and the E-stop can only be reset by an admin.
- Model files are confined to `~/.jepctl/models` (path-traversal checks), uploads are format-sniffed and size-capped (20 MB images, 200 MB request body).

Known gaps (help welcome): `GET /api/status`, `/api/tags`, `/api/cameras` and `/api/settings` are readable without a token (no sensitive content, but they reveal that the service exists); there is no rate limiting or brute-force lockout (tokens are 256-bit random, so guessing is not practical, but a flood is not throttled); no TLS (put a reverse proxy in front for anything beyond loopback); file permissions are enforced on Unix only; there has been no external audit.

## Can the daemon be used to attack the machine it runs on?

What an attacker who can talk to the API (a valid token, or `--no-auth` on loopback) can and cannot do:

- **No shell, no code execution.** The daemon never invokes a shell. The only subprocess is `ffmpeg` (when installed), spawned with a fixed argument list on a temporary file the daemon wrote itself; nothing from a request reaches a command line. Model checkpoints are safetensors: parsed, memory-mapped, never executed (no pickle). Jepafiles are JSON with validated fields.
- **File writes are confined.** Everything the API writes lives under `~/.jepctl` (models, keys, settings, gestures, sounds, world model, companion memory) and temporary media under the OS temp dir. Model identifiers are checked for `..`, absolute paths and prefixes, canonicalized against the models directory, and an identifier that resolves to the models directory itself is refused (so `DELETE /api/models/` cannot wipe the library).
- **Decoders are bounded.** Images and animation frames are capped at 8192 x 8192 pixels and 256 MB of decoder allocations; animations stop at 512 frames or 256 million pixels; uploads at 20 MB (images) / 200 MB (body). A decompression bomb is refused before it is allocated. The audio decoder is our own WAV parser with a sample-count check; MP3/FLAC/OGG go through `ffmpeg`.
- **Memory safety.** Pure Rust; the only `unsafe` is the memory map of a safetensors file. A memory-safety bug would have to be in a dependency (`image`, `candle`, `tokio`, `cpal`, `nokhwa`) and is not reachable through anything the daemon does deliberately.
- **Privileges.** The process runs as the user who started it. It never asks for root and installs nothing; camera and microphone access go through the OS permission prompts.
- **Admin-only capabilities that touch the OS**: the serial backend opens the device path from `robot_hardware.port` in settings (an admin can point it at any device node; the open is a plain serial open, no writes to other files) and `jepctl pull` fetches `https://huggingface.co/<repo>/resolve/main/model.safetensors` over HTTPS only.

So the realistic worst case for a compromised inference token is resource exhaustion (memory or GPU time from a flood of embeddings; there is no rate limiting) and misuse of the camera and microphone, not control of the machine. Keep tokens scoped and short-lived for that reason.

## Model weights

`jepctl pull` downloads `model.safetensors` from Hugging Face over HTTPS into a temporary file and renames it atomically. Safetensors are memory-mapped and never executed. Checksums are not verified yet.
