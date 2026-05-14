# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-05-13

### Added

- **Weft** — LLM-assisted canonical column naming (`weft_dataframe`, `Raiju.weft`, `resolve_weft_mappings`): bounded column evidence, one driver HTTP call, **Pydantic** `WeftResponse` / `WeftColumnMapping`, confidence and collision guardrails, optional Spark casts in a **single** `select`, optional struct output (`raiju.weft_types`).
- **`raiju.inference.chat`** — `inference_chat` (Ollama + OpenRouter), `parse_llm_json_object`, and `truncate_llm_text`; re-exported from `raiju` and `raiju.inference`. Used by Weft and profile LLM enrichment.
- **Profile LLM enrichment** — when `ProfileOptions.inference_enrichment=True` and `inference=InferenceSettings`, bounded aggregates plus capped samples are sent to the model; reply validated as `ProfileEnrichmentResponse` (fail-soft with warnings on bad output).

### Changed

- Profile and Weft LLM calls share the same HTTP and token-accounting path (`inference_chat`).

### Fixed

- `parse_llm_json_object` returns only JSON **objects** (`dict`); top-level arrays or other values yield `None` so callers do not crash on `.get(...)`.
- `inference_chat` validates `provider` (`auto` / `ollama` / `openrouter`, case-insensitive) and requires positive `http_timeout_s`.
- `truncate_llm_text` handles non-positive `max_len` safely.

### Documentation

- README and ROADMAP updated for Weft, chat helpers, and shipped baseline; `requirements.txt` lists full runtime deps aligned with `pyproject.toml`.

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

[Unreleased]: https://github.com/seanpavlak/raiju/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.3
[0.1.2]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.2
[0.1.1]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.1
[0.1.0]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.0
