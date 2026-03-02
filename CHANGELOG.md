# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-01

### Added

- CHANGELOG, PyPI publish workflow, Dependabot, and maintainer docs.

## [0.1.0] - 2026-03-01

### Added

- Initial release.
- `Raiju` wrapper around `SparkSession` with full API delegation via `__getattr__`.
- Builder support: `Raiju.builder.appName(...).master(...).getOrCreate()`.
- Wrap an existing session: `Raiju(existing_spark_session)`.

[Unreleased]: https://github.com/seanpavlak/raiju/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.1
[0.1.0]: https://github.com/seanpavlak/raiju/releases/tag/v0.1.0
