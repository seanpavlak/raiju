# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-05-12

### Added

- Optional **inference configuration** on `Raiju`: `OllamaConfig`, `OpenRouterConfig`, and `InferenceSettings`, plus `with_inference()` for builder workflows. Configuration only (no HTTP); intended for future enrichment execution hooks.

### Changed

- `OpenRouterConfig` emits a `UserWarning` when `api_key` is omitted, documenting that `resolved_api_key()` reads from the configured environment variable (default `OPENROUTER_API_KEY`).

### Fixed

- Stopped tracking Python bytecode under `raiju/__pycache__/` and `tests/__pycache__/`, and expanded `.gitignore` so those paths stay out of version control.

## [0.1.1] - 2026-03-01

### Added

- CHANGELOG, PyPI publish workflow, Dependabot, and maintainer docs.

## [0.1.0] - 2026-03-01

### Added

- Initial release.
- `Raiju` wrapper around `SparkSession` with full API delegation via `__getattr__`.
- Builder support: `Raiju.builder.appName(...).master(...).getOrCreate()`.
- Wrap an existing session: `Raiju(existing_spark_session)`.

[Unreleased]: https://github.com/seanpavlak/raiju/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.2
[0.1.1]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.1
[0.1.0]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.0
