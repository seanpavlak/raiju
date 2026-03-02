# Raiju

[![Documentation](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/seanpavlak/raiju/refs/heads/main/assets/badge.json)](https://raiju.sh)
[![PyPI version](https://img.shields.io/pypi/v/raiju.svg)](https://pypi.org/project/raiju/)
[![PyPI license](https://img.shields.io/pypi/l/raiju.svg)](https://pypi.org/project/raiju/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/raiju.svg)](https://pypi.org/project/raiju/)
[![CI](https://github.com/seanpavlak/raiju/workflows/CI/badge.svg)](https://github.com/seanpavlak/raiju/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**Docs**](https://raiju.sh)

**PySpark, unleashed.** A thin wrapper around PySpark that exposes the full `SparkSession` API through a single, delegation-based interface, so you get all of Spark’s power with a minimal, extension-ready layer.

- **Full PySpark surface:** Every `SparkSession` API is available through Raiju via delegation. Nothing is reimplemented or locked in; new and future PySpark APIs work automatically.
- **Drop-in entry point:** Use `Raiju` instead of `SparkSession`. Same builder, same methods, same DataFrames.
- **Works everywhere:** Create a new session with the builder or wrap an existing one (e.g. `spark` in Databricks).
- **Extension-ready:** A minimal foundation you can build on without forking PySpark.
- **Single dependency:** PySpark 4.0+; no extra runtime deps.

**Raiju** (雷獣, *raijū*) is a creature from Japanese folklore: a lightning beast and companion of the thunder god Raijin. The name means “thunder animal”: 雷 (*rai*, thunder) + 獣 (*jū*, beast). This project borrows it because it wraps **PySpark**, your engine for lightning-fast, distributed data, in a thin, flexible layer. All of Spark’s power is still there; Raiju is the interface that carries it.

Raiju aims to give you one clear entry point for PySpark while staying invisible: same APIs, same types, zero lock-in. You can adopt it gradually (e.g. wrap an existing `SparkSession` in Databricks) or use it as the base for your own extensions.

## Table of Contents

1. [Getting Started](#getting-started)
1. [Usage](#usage)
1. [How It Works](#how-it-works)
1. [Development](#development)
1. [Support](#support)
1. [Show Your Support](#show-your-support)
1. [Code of Conduct](#code-of-conduct)
1. [Contributing](#contributing)
1. [License](#license)
1. [Security](#security)

## Getting Started

### Installation

Raiju is available as [`raiju`](https://pypi.org/project/raiju/) on PyPI (or install from source).

Invoke or install with **uv** (recommended), **pip**, or **pipx**:

```shell
# With uv.
uv add raiju                    # Add to your project.
uv tool install raiju@latest    # Or install globally.

# With pip.
pip install raiju

# With pipx.
pipx install raiju
```

From a local clone:

```shell
pip install -e .
```

For development (linting, formatting, tests):

```shell
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, PySpark 4.0+.

### Usage

Create a session with the builder:

```python
from raiju import Raiju

raiju = Raiju.builder.appName("my_app").master("local[*]").getOrCreate()
```

Or wrap an existing session (e.g. in Databricks):

```python
from raiju import Raiju

raiju = Raiju(spark)
```

Use it like PySpark: SQL, DataFrame API, read, catalog, config. Everything is delegated:

```python
# SQL
df = raiju.sql("SELECT 1 AS one")

# DataFrame API
df = raiju.range(10).filter("id > 5")

# Read data
df = raiju.read.csv("path/to/file.csv", header=True)

# Catalog, UDFs, config
raiju.catalog.listTables()
raiju.conf.set("key", "value")
```

Any attribute or method on `SparkSession` is available on your `Raiju` instance; returned objects (DataFrames, etc.) are standard PySpark types.

## How It Works

- **No hardcoded API:** `Raiju` and its builder use `__getattr__` to forward to the real `SparkSession` (and `SparkSession.builder`). New PySpark methods and options work without changes to Raiju.
- **Single entry point:** You get a `Raiju` instance; `.read`, `.sql`, `.range`, and everything else behave as in PySpark.
- **Thin wrapper:** This layer is the base; you can add behavior on top without reimplementing Spark.

## Development

```shell
pip install -e ".[dev]"
ruff check raiju/ tests/
ruff format raiju/ tests/
pytest tests/ -v
```

## Support

Having trouble? Open an [issue](https://github.com/seanpavlak/raiju/issues) on GitHub.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

## Show Your Support

If you're using Raiju, consider adding the Raiju badge to your project's `README.md`:

```md
[![Raiju](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/seanpavlak/raiju/refs/heads/main/assets/badge.json)](https://github.com/seanpavlak/raiju)
```

## License

This repository is licensed under the [MIT License](LICENSE).

## Security

To report a security concern or vulnerability, see [SECURITY.md](SECURITY.md).
