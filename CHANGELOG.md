# Changelog

All notable changes to `open-keypool` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
