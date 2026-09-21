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

## Model weights

`jepctl pull` downloads `model.safetensors` from Hugging Face over HTTPS into a temporary file and renames it atomically. Safetensors are memory-mapped and never executed. Checksums are not verified yet.
