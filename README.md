<p align="center">
  <img src="assets/icon.png" width="120" alt="Raiju" />
</p>

<h1 align="center">Raiju</h1>
<p align="center"><strong>PySpark, unleashed.</strong></p>

<p align="center">
  <a href="https://raiju.sh"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/seanpavlak/raiju/refs/heads/main/assets/badge.json" alt="Documentation" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
</p>

---

## What is Raiju?

**Raiju** (雷獣, *raijū*) is a creature from Japanese folklore, a lightning beast and companion of the thunder god Raijin. It’s said to take the form of a wolf, cat, weasel, or in this case a fox wrapped in lightning, and to descend with storms. The name means “thunder animal”: 雷 (*rai*, thunder) + 獣 (*jū*, beast).

This project borrows the name because it wraps **PySpark**, your engine for lightning-fast, distributed data, in a thin, flexible layer. All of Spark’s power is still there; Raiju is the interface that carries it.

---

## Features

- **Full PySpark surface:** Every `SparkSession` API is available through Raiju via delegation. Nothing is reimplemented or locked in; new and future PySpark APIs work automatically.
- **Drop-in entry point:** Use `Raiju` instead of `SparkSession`. Same builder, same methods, same DataFrames.
- **Works everywhere:** Create a new session with the builder or wrap an existing one (e.g. `spark` in Databricks).
- **Extension-ready:** A minimal foundation you can build on without forking PySpark.

## Requirements

- **Python** 3.9+
- **PySpark** 4.0+

## Installation

```bash
pip install -e .
```

Or from requirements only:

```bash
pip install -r requirements.txt
```

For development (linting, formatting, tests):

```bash
pip install -e ".[dev]"
```

## Quick Start

**Create a session (builder):**

```python
from raiju import Raiju

raiju = Raiju.builder.appName("my_app").master("local[*]").getOrCreate()
```

**Or wrap an existing session** (e.g. in Databricks):

```python
from raiju import Raiju

raiju = Raiju(spark)
```

**Use it like PySpark:**

```python
# SQL
df = raiju.sql("SELECT 1 as one")

# DataFrame API
df = raiju.range(10).filter("id > 5")

# Read data
df = raiju.read.csv("path/to/file.csv", header=True)

# Catalog, UDFs, config; everything is forwarded
raiju.catalog.listTables()
raiju.conf.set("key", "value")
```

Any attribute or method on `SparkSession` is available on your `Raiju` instance; it’s all delegated, so you get the full API for free.

## Development

```bash
pip install -e ".[dev]"
ruff check raiju/ tests/
ruff format raiju/ tests/
pytest tests/ -v
```

## How It Works

- **No hardcoded API:** `Raiju` and its builder use `__getattr__` to forward to the real `SparkSession` (and `SparkSession.builder`). New PySpark methods and options work without changes to Raiju.
- **Single entry point:** You get a `Raiju` instance; `.read`, `.sql`, `.range`, and everything else behave as in PySpark. Returned objects (DataFrames, etc.) are standard PySpark types.
- **Thin wrapper:** This layer is the base; you can add behavior on top without reimplementing Spark.

## License

MIT License. See [LICENSE](LICENSE).
