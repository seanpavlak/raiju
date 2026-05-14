"""Driver-side LLM enrichment for :func:`raiju.profile.profile_dataframe`.

Uses tiny bounded samples plus precomputed aggregates — never ships whole tables.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

from pydantic import ValidationError

from raiju.inference.chat import (
    inference_chat,
    parse_llm_json_object,
    truncate_llm_text,
)
from raiju.inference.llm_schemas import (
    LLMTokenUsage,
    ProfileEnrichmentResponse,
)
from raiju.inference.settings import InferenceSettings


def _collect_samples_batch(
    df: Any,
    names: list[str],
    *,
    scan_limit: int,
    max_distinct_values: int,
    max_value_chars: int,
) -> dict[str, list[Any]]:
    """Single bounded ``collect`` for all columns (one Spark action)."""
    if not names:
        return {}
    rows = df.select(*names).limit(int(scan_limit)).collect()
    results: dict[str, list[Any]] = {n: [] for n in names}
    seen: dict[str, set[str]] = {n: set() for n in names}
    for row in rows:
        rd = row.asDict(recursive=True) if hasattr(row, "asDict") else {}
        for n in names:
            if len(results[n]) >= max_distinct_values:
                continue
            v = rd.get(n)
            if v is None:
                continue
            if isinstance(v, bytes):
                key = v[:200].hex()
            else:
                key = repr(v)[:500]
            if key in seen[n]:
                continue
            seen[n].add(key)
            if isinstance(v, str):
                results[n].append(truncate_llm_text(v, max_value_chars))
            elif isinstance(v, bytes):
                results[n].append(
                    truncate_llm_text(
                        v.decode("utf-8", errors="replace"), max_value_chars
                    )
                )
            else:
                results[n].append(truncate_llm_text(str(v), max_value_chars))
    return results


def _collect_column_samples(
    df: Any,
    col: str,
    *,
    scan_limit: int,
    max_distinct_values: int,
    max_value_chars: int,
) -> list[Any]:
    """Per-column samples (prefer :func:`_collect_samples_batch` in hot paths)."""
    d = _collect_samples_batch(
        df,
        [col],
        scan_limit=scan_limit,
        max_distinct_values=max_distinct_values,
        max_value_chars=max_value_chars,
    )
    return d.get(col, [])


def _pick_enrichment_columns(
    fields: list[Any],
    by_column: dict[str, dict[str, Any]],
    *,
    max_columns: int,
) -> list[str]:
    """Prefer string/temporal/boolean; then low-cardinality numerics."""
    scored: list[tuple[int, str]] = []
    for f in fields:
        name = f.name
        m = by_column.get(name) or {}
        cat = str(m.get("category") or "other")
        priority = 99
        if cat == "string":
            priority = 0
        elif cat == "temporal":
            priority = 1
        elif cat == "boolean":
            priority = 2
        elif cat == "numeric":
            ad = m.get("approx_distinct")
            try:
                adi = int(ad) if ad is not None else 10**9
            except (TypeError, ValueError):
                adi = 10**9
            priority = 10 if adi <= 64 else 30
        else:
            priority = 40
        scored.append((priority, name))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [n for _, n in scored[: int(max_columns)]]


_SYSTEM_PROMPT = (
    "You are a senior data engineer assisting with DataFrame profiling.\n"
    "You receive ONLY aggregate statistics and a small list of sample cell "
    "values per column (never full tables).\n"
    "Reply with a single JSON object (no markdown) matching this shape:\n"
    "{\n"
    '  "columns": [\n'
    "    {\n"
    '      "column": "<exact column name>",\n'
    '      "human_summary": "<one sentence or null>",\n'
    '      "semantic_classification": "<short label or null>",\n'
    '      "suggested_validation_regex": "<PCRE-style regex or null>",\n'
    '      "java_simple_date_format": "<e.g. yyyy-MM-dd or null>",\n'
    '      "python_strptime_directive": "<e.g. %Y-%m-%d or null>",\n'
    '      "suggested_cast_or_parse": "<e.g. to_date or null>",\n'
    '      "pii_likelihood": "<low|medium|high|null>",\n'
    '      "quality_flags": ["<short bullet>", "..."],\n'
    '      "notes": "<optional caveats or null>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Use null when unsure. Regex must be conservative — prefer failing closed "
    "over false positives.\n"
    "If samples are empty, still use aggregates only."
)


def attach_profile_llm_enrichment(
    df: Any,
    fields: list[Any],
    out: dict[str, Any],
    settings: InferenceSettings,
    *,
    provider: str,
    http_timeout_s: float,
    max_columns: int,
    sample_scan_limit: int,
    max_sample_values: int,
    max_value_chars: int,
) -> None:
    """Mutates ``out`` in place: sets ``out['llm_enrichment']`` and per-column ``llm``.

    On failure, warns and sets structured nulls.
    """
    by_column = out.get("columns")
    if not isinstance(by_column, dict):
        return

    names = _pick_enrichment_columns(
        fields,
        by_column,
        max_columns=max_columns,
    )
    if not names:
        out["llm_enrichment"] = {"status": "skipped", "reason": "no_columns"}
        return

    out["approximate_spark_actions"] = int(out.get("approximate_spark_actions", 1)) + 1

    samples_by_col = _collect_samples_batch(
        df,
        names,
        scan_limit=sample_scan_limit,
        max_distinct_values=max_sample_values,
        max_value_chars=max_value_chars,
    )

    evidence_columns: list[dict[str, Any]] = []
    for n in names:
        m = by_column.get(n) or {}
        samples = samples_by_col.get(n, [])
        agg = {k: v for k, v in m.items() if k != "llm"}
        evidence_columns.append(
            {
                "column": n,
                "aggregates": agg,
                "sample_values": samples,
            }
        )

    user_payload = {
        "instruction": (
            "Infer formats, validation regexes, and semantic meaning from "
            "aggregates and samples only."
        ),
        "columns": evidence_columns,
    }
    user_text = json.dumps(user_payload, default=str)

    text = ""
    used = ""
    usage: LLMTokenUsage | None = None
    t0 = time.perf_counter()
    try:
        used, text, usage = inference_chat(
            settings,
            system=_SYSTEM_PROMPT,
            user=user_text,
            provider=provider,
            http_timeout_s=http_timeout_s,
            purpose="profile enrichment",
        )
    except Exception as e:  # noqa: BLE001 — network, JSON, model quirks
        warnings.warn(
            f"Raiju profile LLM enrichment failed ({used or provider}): {e}",
            UserWarning,
            stacklevel=2,
        )
        out["llm_enrichment"] = {
            "status": "failed",
            "error": str(e),
            "provider_attempted": provider,
        }
        for n in names:
            if isinstance(by_column.get(n), dict):
                by_column[n]["llm"] = None
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if usage is not None:
        usage.warn_if_known()
        out["llm_token_usage"] = usage.model_dump(mode="json", exclude_none=True)

    parsed = parse_llm_json_object(text)
    if not parsed or not isinstance(parsed.get("columns"), list):
        warnings.warn(
            "Raiju profile LLM enrichment returned unparseable JSON; "
            "setting llm fields to null.",
            UserWarning,
            stacklevel=2,
        )
        out["llm_enrichment"] = {
            "status": "failed",
            "error": "unparseable_model_output",
            "raw_response_excerpt": truncate_llm_text(text, 800),
            "provider": used,
            "elapsed_ms": round(elapsed_ms, 2),
            "token_usage": out.get("llm_token_usage"),
        }
        for n in names:
            if isinstance(by_column.get(n), dict):
                by_column[n]["llm"] = None
        return

    try:
        prof = ProfileEnrichmentResponse.model_validate(parsed)
    except ValidationError as e:
        warnings.warn(
            f"Raiju profile LLM JSON failed Pydantic validation: {e}",
            UserWarning,
            stacklevel=2,
        )
        out["llm_enrichment"] = {
            "status": "failed",
            "error": "pydantic_validation_error",
            "detail": e.errors(),
            "provider": used,
            "elapsed_ms": round(elapsed_ms, 2),
            "token_usage": out.get("llm_token_usage"),
        }
        for n in names:
            if isinstance(by_column.get(n), dict):
                by_column[n]["llm"] = None
        return

    by_name = {c.column: c for c in prof.columns if c.column in names}

    for n in names:
        col = by_name.get(n)
        if isinstance(by_column.get(n), dict):
            if col is None:
                by_column[n]["llm"] = None
            else:
                by_column[n]["llm"] = col.model_dump(
                    mode="json",
                    exclude={"column"},
                    exclude_none=True,
                )

    out["llm_enrichment"] = {
        "status": "ok",
        "provider": used,
        "elapsed_ms": round(elapsed_ms, 2),
        "columns_enriched": list(names),
        "token_usage": out.get("llm_token_usage"),
    }
