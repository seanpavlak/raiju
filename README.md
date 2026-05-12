# Raiju

[![Documentation](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/seanpavlak/raiju/main/assets/badge.json)](https://raiju.sh)
[![Release v0.1.1](https://img.shields.io/badge/release-v0.1.1-blue.svg)](https://github.com/seanpavlak/raiju/releases)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.1-blue.svg)](https://pypi.org/project/raiju/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/seanpavlak/raiju/actions/workflows/ci.yml/badge.svg)](https://github.com/seanpavlak/raiju/actions)
[![codecov](https://codecov.io/gh/seanpavlak/raiju/graph/badge.svg)](https://codecov.io/gh/seanpavlak/raiju)
[![Discussions](https://img.shields.io/github/discussions/seanpavlak/raiju)](https://github.com/seanpavlak/raiju/discussions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[**Docs**](https://raiju.sh)

Raiju is a **distributed PySpark execution framework** aimed at teams who need **maintainable orchestration** for data engineering workflows that stretch beyond simple columnar transforms—while staying **Spark-native** on the cluster.

In one line: *Raiju is built on PySpark to simplify complex transformation orchestration, scalable UDF-style workflows, and (as the project grows) integrated local and remote inference for operational enrichment—without replacing Spark.*

**Raiju** (雷獣, *raijū*) is a creature from Japanese folklore: a lightning beast and companion of the thunder god Raijin. The name fits a layer that rides **PySpark**, your engine for distributed data processing.

## Table of contents

1. [Why Raiju exists](#why-raiju-exists)
1. [Core design goals](#core-design-goals)
1. [Architecture overview](#architecture-overview)
1. [Integrated inference workflows (direction)](#integrated-inference-workflows-direction)
1. [Example workloads](#example-workloads)
1. [What you get today](#what-you-get-today)
1. [Roadmap](#roadmap)
1. [Getting started](#getting-started)
1. [Usage](#usage)
1. [How it works](#how-it-works)
1. [Development](#development)
1. [Changelog](#changelog)
1. [Support](#support)
1. [Show your support](#show-your-support)
1. [Code of conduct](#code-of-conduct)
1. [Contributing](#contributing)
1. [License](#license)
1. [Security](#security)

## Why Raiju exists

Many **distributed data workflows** are hard to keep healthy when:

- transformations need **cross-record** or **multi-field** reasoning
- logic becomes **deeply procedural** across many steps
- **orchestration** spans several enrichment or validation stages
- you need **dynamic** execution choices without scattering `if` trees through jobs
- **column-only** pipelines are hard to read or refactor
- **UDF-heavy** pipelines become brittle operationally

PySpark already gives you distributed compute. What teams often lack is a **clear, composable layer** for orchestration and advanced workflows—without giving up executors, partitions, and the rest of the Spark programming model. Raiju is meant to grow into that layer: **higher-level workflow composition** on top of **unchanged Spark execution**.

## Core design goals

- **Distributed-first:** Spark executors and cluster semantics stay central; Raiju coordinates and composes, it does not pretend compute is “local-first.”
- **Workflow composition:** Modular pipelines and reusable building blocks instead of one-off scripts.
- **Operational flexibility:** Room for local inference, remote providers, and hybrid patterns where privacy, cost, or latency demand it.
- **Developer ergonomics:** Less bespoke glue for complex jobs; clearer boundaries between stages.
- **Spark compatibility:** Same `DataFrame` types, same session APIs, same deployment story (including Databricks and on-prem clusters).

## Architecture overview

**Target shape:** Raiju sits as an **orchestration and execution abstraction** above Spark-native processing—handling transformation composition, coordination patterns, enrichment flows, and (optionally) inference calls—while **work still runs on Spark executors**.

**Today:** the published library is a **thin, delegation-based `SparkSession` entry point** (see [What you get today](#what-you-get-today)). That is intentional: a **stable compatibility surface** before higher-level APIs land. The layering story above is where the project is headed; see [ROADMAP.md](ROADMAP.md) for concrete backlog items (diagrams, retries, partitioning docs, inference interfaces, benchmarks).

## Integrated inference workflows (direction)

Optional support for **LLM-assisted steps inside data pipelines** is part of the vision—framed as **operational enrichment**, not a separate “agent platform.”

Planned execution styles to document and implement over time:

- **Local inference** (for example via [Ollama](https://ollama.com/)) for low-latency or air-gapped settings
- **Remote providers** (for example OpenRouter-compatible HTTP APIs) when external models are acceptable
- **Hybrid** routing by policy (cost, privacy, SLA)

Example workload types (all squarely “data engineering”):

- semantic enrichment and tagging
- entity normalization and fuzzy classification
- metadata generation and schema hints
- human-in-the-loop **review assistance** as a batch step
- contextual transforms where a model proposes a value validated by rules

Language in the project intentionally stays **grounded**: orchestration, enrichment, inference **hooks**—not hype around autonomy or “cognitive” stacks.

## Example workloads

Raiju is aimed at teams building **operational data systems** where jobs look like:

- large-scale semantic enrichment
- complex **mapPartitions** / UDF orchestration
- fuzzy entity resolution and deduplication
- metadata standardization across sources
- multi-stage enrichment with checkpoints
- operational anomaly or triage classification
- hybrid **rules + model** scoring in batch

## What you get today

**Release v0.1.x** ships a **single, extension-ready entry point** over PySpark:

- **Full PySpark surface:** `Raiju` forwards the entire `SparkSession` API via delegation—no duplicated method lists; new PySpark APIs keep working as PySpark evolves.
- **Drop-in usage:** `Raiju.builder...getOrCreate()` or `Raiju(spark)` when you already have a session (for example in Databricks).
- **Minimal dependency:** PySpark 4.0+ only; no extra runtime packages yet.

Higher-level orchestration, inference integrations, and operational guides are **on the roadmap** ([ROADMAP.md](ROADMAP.md)), not implied as shipped features.

## Roadmap

See **[ROADMAP.md](ROADMAP.md)** for a structured backlog: execution and DAG diagrams, failure handling and retry semantics, partitioning and serialization notes, benchmarks, orchestration APIs, optional inference backends, and hardening for production pipelines.

## Getting started

### Installation

Raiju is published as [`raiju`](https://pypi.org/project/raiju/) on PyPI.

With **uv** (recommended), **pip**, or **pipx**:

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

Or wrap an existing session (for example in Databricks):

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

Returned objects are standard PySpark types.

## How it works

- **No hardcoded API surface:** `Raiju` and its builder use `__getattr__` to forward to the real `SparkSession` (and `SparkSession.builder`).
- **Single entry point:** You hold a `Raiju` instance; `.read`, `.sql`, `.range`, and the rest behave as in PySpark.
- **Thin foundation:** This layer is the base for future orchestration and enrichment utilities without forking PySpark.

## Development

```shell
pip install -e ".[dev]"
ruff check raiju/ tests/
ruff format raiju/ tests/
pytest tests/ -v
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Support

Having trouble? Open an [issue](https://github.com/seanpavlak/raiju/issues) on GitHub.

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

## Show your support

If you are using Raiju, consider adding the Raiju badge to your project’s `README.md`:

```md
[![Raiju](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/seanpavlak/raiju/main/assets/badge.json)](https://github.com/seanpavlak/raiju)
```

## License

This repository is licensed under the [MIT License](LICENSE).

## Security

To report a security concern or vulnerability, see [SECURITY.md](SECURITY.md).
