# Changelog

All notable changes to `open-keypool` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-11-20

### Fixed
- Fixed `KeyPoolAuth` with `httpx.AsyncClient` by reading streaming response bodies before parsing them.
- Ensured lifecycle callbacks fire for direct state changes and direct pool exhaustion, outside pool locks.
- Clarified that `max_retries` controls attempts made by `call()` and `KeyPoolAuth`.

## [0.4.0] - 2026-11-20

### Added
- Added transient 408/5xx cooldowns, optional recovery waiting, safe callbacks, and masked logging.
- Added `KeyPoolAuth` for sync and async `httpx` clients.
- Added OpenRouter, Cerebras, Mistral, and Anthropic provider presets.
- Added release, contribution, issue-template, and Dependabot configuration.

### Changed (behavior)
- Client errors other than documented authentication and rate-limit signals now stay ACTIVE, do not increment `failure_count`, and are returned immediately by `call()`.
- 408 and 5xx responses use the new five-second transient cooldown by default; `Retry-After` still wins.
- Added a `Typing :: Typed` package marker and bumped the package version to 0.4.0.

### Fixed
- Prevented raw API keys from appearing in record representations, status snapshots, exceptions, callbacks, and logs.
- Kept constructor options consistent across all sync and async secret loaders.

## [0.3.0] - 2026-09-25

### Added
- **Async Support**: Added `AsyncKeyPool` class supporting `async get_key()`, `async_get_key()`, `async handle_response()`, `async call()`, and async Doppler loading via `httpx.AsyncClient`.
- **High-level `pool.call()`**: Added `pool.call(fn, *args, **kwargs)` (sync & async) to execute request callables with automatic key selection, response handling, and retries.
- **Provider Presets**: Added `provider` parameter (`"groq"`, `"openai"`, `"gemini"`, `"together"`) to pre-configure rate-limit headers (e.g., `x-ratelimit-reset-*`, `retry-after-ms`) and provider-specific error body structures.
- **Cloud Secrets Loaders**: Added `KeyPool.from_aws_secrets()` (boto3) and `KeyPool.from_gcp_secrets()` (google-cloud-secret-manager) classmethods.
- **FastAPI Middleware Example**: Added `examples/fastapi_middleware.py`.
- **CI Workflow**: Added GitHub Actions workflow `.github/workflows/test.yml` running pytest across Python 3.9–3.13.

### Fixed
- **Doppler Cache Key**: Included `token` in `from_doppler()`'s in-memory cache key to prevent stale cache hits upon token rotation.
- **Status Key Collisions**: Fixed `status()` key collisions by appending disambiguating suffixes (e.g. `sk-abc...1234#2`) when multiple keys mask to the same representation.
- **Optional `python-dotenv`**: Moved `python-dotenv` from core dependencies into an `env` optional extra (`pip install open-keypool[env]`) with lazy import error messaging inside `from_env()`.
- **Documentation Link**: Updated documentation URL to `https://tusharneje.in/projects/open-keypool/`.

---

## [0.2.1] - 2026-08-25

### Fixed
- Updated PyPI package metadata and README documentation links for canonical PyPI and GitHub usage.

---

## [0.2.0] - 2026-08-25

### Added
- Classmethods `KeyPool.from_env(suffix, env_file=None)` and `KeyPool.from_json(path, suffix=None)` for loading API keys with suffix filtering.
- Thread-safe key additions and removals dynamically at runtime.
