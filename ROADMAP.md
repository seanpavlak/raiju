# Raiju roadmap and backlog

This document turns the project’s engineering positioning into concrete work. Items are ideation and prioritization targets—not commitments. See [README.md](README.md) for how Raiju is framed for users and contributors.

## North star

Raiju should read as **orchestration and execution infrastructure for complex distributed PySpark workflows**, not as an experimental “AI product.” Optional inference is a **workflow utility**, not a standalone agent platform.

---

## Documentation and credibility (high leverage)

| ID | Item | Notes |
|----|------|--------|
| D1 | **Execution overview diagram** | Spark driver → Raiju coordination (future) → executors; what runs where. |
| D2 | **DAG / stage diagram** | How a Raiju-orchestrated pipeline maps to Spark jobs, stages, and tasks once orchestration exists. |
| D3 | **Executor data flow** | Partition travel, shuffle boundaries, where UDFs execute. |
| D4 | **Local vs remote inference architecture** | Ollama vs OpenRouter (or similar); privacy, cost, and network boundaries. |
| D5 | **Failure handling guide** | Task failures, stage retries, executor loss, speculative execution interaction. |
| D6 | **Retry semantics** | Idempotency expectations, which layers retry, alignment with Spark’s task retries. |
| D7 | **Partitioning behavior** | When repartition/coalesce is required, skew, partition count guidance for enrichment stages. |
| D8 | **Serialization strategy** | Closure vs broadcast vs DataFrame lineage; Python worker and pickle considerations for UDF-heavy paths. |
| D9 | **Benchmark examples** | Reproducible scripts and numbers (throughput, cost per row, cluster size)—even for the current thin layer, baseline “plain PySpark vs Raiju wrapper” overhead. |

## Core framework (orchestration and composition)

| ID | Item | Notes |
|----|------|--------|
| O1 | **Pipeline / step abstraction** | Named stages, inputs/outputs, composable graphs without abandoning DataFrames. |
| O2 | **Explicit execution context** | Config, run IDs, logging hooks, correlation across stages. |
| O3 | **Stage boundaries and Spark action policy** | Document and optionally enforce where materialization happens to avoid surprise full scans. |
| O4 | **Reusable transformation modules** | Conventions for packaging UDFs + SQL + Python transforms as importable units. |
| O5 | **Testing utilities** | Local Spark fixtures, small golden datasets, property-style tests for transforms. |

## UDF and enrichment workflows

| ID | Item | Notes |
|----|------|--------|
| U1 | **Scalable UDF patterns** | MapPartitions, iterator UDFs, batching, rate limits for external calls. |
| U2 | **Enrichment pipeline template** | Multi-stage pattern: extract → normalize → join → score → persist. |
| U3 | **Cross-record patterns (careful)** | Windowed or grouped operations without hiding shuffle cost. |
| U4 | **Operational metadata** | Lineage tags, stage timing, row counts emitted for observability. |

## Integrated inference (optional, grounded)

| ID | Item | Notes |
|----|------|--------|
| I0 | **Session-scoped inference settings** | **Done (initial):** `InferenceSettings` / `OllamaConfig` / `OpenRouterConfig` on `Raiju(..., inference=...)` and `with_inference()`; no I/O. |
| I1 | **Pluggable inference backend** | Interface + reference implementations (e.g. HTTP to Ollama, OpenRouter-compatible client). |
| I2 | **Hybrid execution policy** | Per-row, per-partition, or sampled routing between local and remote. |
| I3 | **Backpressure and quotas** | Concurrency caps, timeouts, circuit breaking for remote APIs. |
| I4 | **Example workloads only** | Semantic enrichment, fuzzy classification, entity normalization—documented as data engineering tasks, not “agents.” |

## Hardening and operations

| ID | Item | Notes |
|----|------|--------|
| H1 | **Structured error types** | Clear distinction between user transform errors, Spark failures, and external service failures. |
| H2 | **Determinism and seeding** | Where randomness appears (sampling, LLM temperature) and how runs are reproducible when needed. |
| H3 | **Databricks / cluster notes** | Init patterns, cluster libraries, compatibility matrix for PySpark versions. |

## Ecosystem and packaging

| ID | Item | Notes |
|----|------|--------|
| E1 | **Optional extras** | `pip install raiju[inference]` (or similar) so core stays lean. |
| E2 | **Versioning policy** | Spark minor versions, Python floor, deprecation process. |
| E3 | **Changelog discipline** | User-visible behavior and performance called out per release. |

## What we avoid in messaging and API naming

Per project positioning: avoid hype terms that read as “AI product” rather than infrastructure (e.g. “agentic,” “autonomous,” “AI-native,” “next-gen”). Prefer **orchestration, execution, enrichment, UDF, pipeline, partition, executor**.

---

## Current baseline (v0.1.2)

The shipped library is a **SparkSession-compatible entry point** with full API delegation. Roadmap items above describe the direction for higher-level orchestration and optional inference while **preserving Spark-native execution**.

Contributors: pick an item, open a discussion or draft PR with a minimal vertical slice (docs + tests, or a small API with examples).
