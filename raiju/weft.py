"""Weft — LLM-assisted canonical column naming for PySpark DataFrames.

Maps messy source column names onto a developer-provided canonical schema using
a single bounded LLM call plus Pydantic validation and confidence guardrails.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from raiju.inference.llm_schemas import LLMTokenUsage, WeftResponse
from raiju.inference.profile_enrichment import (
    _extract_json_object,
    _ollama_chat,
    _openrouter_chat,
)
from raiju.inference.settings import InferenceSettings
from raiju.inference.token_count import build_llm_token_usage
from raiju.profile import _type_name
from raiju.weft_types import weft_canonical_select

_WEFT_SYSTEM = (
    "You are a senior data engineer mapping tabular columns to a canonical schema.\n"
    "You receive column names, Spark data types, bounded sample statistics, and "
    "sample cell values only — never full tables.\n"
    "Reply with a single JSON object (no markdown fences). Each mapping entry must "
    "include typing so downstream Spark casts are safe and predictable.\n"
    "JSON shape:\n"
    "{\n"
    '  "mappings": [\n'
    "    {\n"
    '      "source_column": "<exact input column name>",\n'
    '      "target_column": "<one of allowed_target_columns or null>",\n'
    '      "confidence": <number from 0 to 1>,\n'
    '      "reason": "<one short sentence>",\n'
    '      "action": "map" | "ignore" | "needs_review",\n'
    '      "target_spark_type": "string" | "boolean" | "byte" | "short" | "int" | '
    '"long" | "float" | "double" | "decimal" | "date" | "timestamp" | '
    '"timestamp_ntz",\n'
    '      "nullable": <true|false>,\n'
    '      "decimal_precision": <1-38 or null>,\n'
    '      "decimal_scale": <0-18 or null>,\n'
    '      "temporal_parse_strategy": "native" | "spark_formats" | "python_dateutil",\n'
    '      "spark_timestamp_formats": ["<Spark SimpleDateFormat patterns>", ...],\n'
    '      "python_dateutil_fuzzy": <true|false>\n'
    "    }\n"
    "  ],\n"
    '  "unmapped_columns": ["..."],\n'
    '  "ambiguous_columns": ["..."],\n'
    '  "notes": ["..."]\n'
    "}\n"
    "Typing rules:\n"
    '- For action "ignore" or "needs_review", still fill target_spark_type (e.g. '
    '"string") and nullable=true; target_column may be null.\n'
    "- For currency or money strings use decimal with realistic precision/scale "
    "(e.g. precision=19 scale=4).\n"
    "- For epoch numbers use long + temporal_parse_strategy native.\n"
    '- temporal_parse_strategy "native": trust physical Spark types when already '
    "date/timestamp; for STRING dates try spark_formats instead.\n"
    '- Use "spark_formats" with 1–6 patterns when dates are messy but structured '
    '(e.g. "MM/dd/yyyy", "yyyy-MM-dd").\n'
    '- Use "python_dateutil" only when text is highly irregular; set '
    "python_dateutil_fuzzy true only if needed (slower UDF path).\n"
    "- Match target_spark_type to the canonical field semantics described.\n"
    "Mapping rules:\n"
    '- Include exactly one "mappings" entry per input column in "columns".\n'
    '- Use action "map" only when target_column is one of allowed_target_columns.\n'
    '- Use "ignore" for junk columns (target_column null).\n'
    '- Use "needs_review" when unsure; set target_column to best guess or null.\n'
)


def _trunc_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _scan_column_evidence(
    df: Any,
    *,
    scan_limit: int,
    max_distinct_samples: int,
    max_value_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    """One bounded collect; returns evidence list and rows scanned."""
    fields = list(df.schema.fields)
    names = [f.name for f in fields]
    if not names:
        return [], 0
    lim = max(1, int(scan_limit))
    rows = df.select(*names).limit(lim).collect()
    n = len(rows)

    out: list[dict[str, Any]] = []
    for f in fields:
        col = f.name
        samples: list[str] = []
        seen: set[str] = set()
        nulls = 0
        distinct_keys: set[str] = set()
        for row in rows:
            rd = row.asDict(recursive=True) if hasattr(row, "asDict") else {}
            v = rd.get(col)
            if v is None:
                nulls += 1
                continue
            if isinstance(v, bytes):
                key = v[:200].hex()
                text = v.decode("utf-8", errors="replace")
            else:
                key = repr(v)[:500]
                text = str(v)
            distinct_keys.add(key)
            if key in seen or len(samples) >= max_distinct_samples:
                continue
            seen.add(key)
            samples.append(_trunc_str(text, max_value_chars))
        tn = _type_name(f.dataType)
        out.append(
            {
                "column": col,
                "dtype": tn,
                "sample_row_count": n,
                "sample_null_count": nulls,
                "sample_distinct_non_null": len(distinct_keys),
                "sample_values": samples,
            }
        )
    return out, n


def _run_weft_llm(
    settings: InferenceSettings,
    user_text: str,
    *,
    provider: str,
    http_timeout_s: float,
) -> tuple[str, str, LLMTokenUsage | None]:
    """Returns (provider_label, assistant_text, token_usage)."""
    model_ollama = "llama3.2"
    model_or = "openai/gpt-4o-mini"
    if settings.ollama and settings.ollama.default_model:
        model_ollama = settings.ollama.default_model
    if settings.openrouter and settings.openrouter.default_model:
        model_or = settings.openrouter.default_model

    text = ""
    used = ""
    usage: LLMTokenUsage | None = None
    use_ollama = False
    use_openrouter = False
    if provider == "ollama":
        use_ollama = bool(settings.ollama)
    elif provider == "openrouter":
        use_openrouter = bool(settings.openrouter)
    else:
        if settings.ollama:
            use_ollama = True
        elif settings.openrouter:
            use_openrouter = True

    if use_ollama:
        if not settings.ollama:
            raise RuntimeError(
                "Ollama selected but not configured on InferenceSettings"
            )
        text, raw = _ollama_chat(
            settings.ollama.resolved_base_url(),
            model_ollama,
            _WEFT_SYSTEM,
            user_text,
            http_timeout_s,
        )
        used = f"ollama:{model_ollama}"
        usage = build_llm_token_usage(
            provider="ollama",
            model=model_ollama,
            system=_WEFT_SYSTEM,
            user=user_text,
            assistant=text,
            raw_api_response=raw,
        )
    elif use_openrouter:
        if not settings.openrouter:
            raise RuntimeError(
                "OpenRouter selected but not configured on InferenceSettings"
            )
        key = settings.openrouter.resolved_api_key()
        if not key:
            raise RuntimeError("OpenRouter API key missing")
        text, raw = _openrouter_chat(
            settings.openrouter.resolved_base_url(),
            model_or,
            key,
            _WEFT_SYSTEM,
            user_text,
            http_timeout_s,
        )
        used = f"openrouter:{model_or}"
        usage = build_llm_token_usage(
            provider="openrouter",
            model=model_or,
            system=_WEFT_SYSTEM,
            user=user_text,
            assistant=text,
            raw_api_response=raw,
        )
    else:
        raise RuntimeError(
            f"No inference backend available for weft (provider={provider!r})"
        )
    return used, text, usage


def _validate_llm_targets(
    resp: WeftResponse,
    allowed: set[str],
) -> list[str]:
    """Return human-readable issues for invalid map targets (does not mutate)."""
    issues: list[str] = []
    for m in resp.mappings:
        if m.action == "map" and m.target_column not in allowed:
            issues.append(
                f"{m.source_column!r}: map target {m.target_column!r} not in schema"
            )
        if (
            m.target_column is not None
            and m.target_column not in allowed
            and m.action != "ignore"
        ):
            if m.action == "needs_review":
                pass  # nullable / out-of-schema guess ok for review
            elif m.action == "map":
                pass  # already captured
            else:
                issues.append(f"{m.source_column!r}: unexpected target outside schema")
    return issues


def resolve_weft_mappings(
    resp: WeftResponse,
    source_columns: list[str],
    allowed_targets: set[str],
    *,
    min_confidence: float,
    require_review_below: float,
    allow_unmapped: bool,
    allow_many_to_one: bool,
) -> dict[str, Any]:
    """Turn a validated LLM :class:`WeftResponse` into rename decisions and report.

    Returns a dict with keys: ``accepted_mappings``, ``accepted_specs``,
    ``needs_review``, ``review_suggested``, ``ignored_columns``,
    ``confidence_scores``, ``unresolved_sources``, ``target_collisions``.
    """
    if not (
        0.0 <= float(min_confidence) <= 1.0
        and 0.0 <= float(require_review_below) <= 1.0
    ):
        raise ValueError("confidence bounds must be in [0, 1]")
    if float(min_confidence) > float(require_review_below):
        raise ValueError("min_confidence must be <= require_review_below")

    src_set = set(source_columns)
    by_src = {m.source_column: m for m in resp.mappings}
    if set(by_src.keys()) != src_set:
        missing = sorted(src_set - set(by_src.keys()))
        extra = sorted(set(by_src.keys()) - src_set)
        raise ValueError(
            "weft mappings must cover exactly DataFrame columns; "
            f"missing={missing!r} extra={extra!r}"
        )

    issues = _validate_llm_targets(resp, allowed_targets)
    if issues:
        raise ValueError("weft LLM produced invalid map targets: " + "; ".join(issues))

    confidence_scores = {m.source_column: float(m.confidence) for m in resp.mappings}

    # Proposed map edges: source -> target (before collision pruning)
    proposed: dict[str, str] = {}
    ignored_columns: list[str] = []
    needs_review: dict[str, list[str]] = {}
    review_suggested: dict[str, str] = {}

    for name in source_columns:
        m = by_src[name]
        if m.action == "ignore":
            ignored_columns.append(name)
            continue
        if m.action == "needs_review":
            cand = [m.target_column] if m.target_column else []
            needs_review[name] = [c for c in cand if c]
            continue
        # map
        tgt = m.target_column
        assert tgt is not None
        if tgt not in allowed_targets:
            needs_review.setdefault(name, []).append(str(tgt))
            continue
        if float(m.confidence) < float(min_confidence):
            continue
        proposed[name] = tgt
        if float(m.confidence) < float(require_review_below):
            review_suggested[name] = tgt

    # Target collisions among proposed
    target_to_sources: dict[str, list[str]] = {}
    for s, t in proposed.items():
        target_to_sources.setdefault(t, []).append(s)
    target_collisions = {t: ss for t, ss in target_to_sources.items() if len(ss) > 1}
    if target_collisions:
        if not allow_many_to_one:
            for t, ss in target_collisions.items():
                for s in ss:
                    needs_review.setdefault(s, []).append(t)
            for s in {x for ss in target_collisions.values() for x in ss}:
                proposed.pop(s, None)
                review_suggested.pop(s, None)
        else:
            for t, ss in target_collisions.items():
                best = max(ss, key=lambda s: float(by_src[s].confidence))
                for s in ss:
                    if s != best:
                        needs_review.setdefault(s, []).append(t)
                        proposed.pop(s, None)
                        review_suggested.pop(s, None)

    accepted_mappings = dict(proposed)
    accepted_specs = {src: by_src[src] for src in accepted_mappings}

    unresolved_sources = [
        c
        for c in source_columns
        if c not in accepted_mappings and c not in ignored_columns
    ]

    if not allow_unmapped and unresolved_sources:
        raise ValueError(
            "weft allow_unmapped=False but columns remain unresolved after guardrails: "
            f"{unresolved_sources!r}"
        )

    return {
        "accepted_mappings": accepted_mappings,
        "accepted_specs": accepted_specs,
        "needs_review": {k: v for k, v in needs_review.items() if v},
        "review_suggested": review_suggested,
        "ignored_columns": ignored_columns,
        "confidence_scores": confidence_scores,
        "unresolved_sources": unresolved_sources,
        "target_collisions": target_collisions,
        "model_notes": list(resp.notes),
    }


def _source_spark_types(df: Any) -> dict[str, str]:
    return {_f.name: _type_name(_f.dataType) for _f in df.schema.fields}


def weft_dataframe(
    df: Any,
    structure: Mapping[str, str],
    inference: InferenceSettings,
    *,
    prior_mappings: Mapping[str, str] | None = None,
    min_confidence: float = 0.85,
    require_review_below: float = 0.95,
    allow_unmapped: bool = False,
    allow_many_to_one: bool = False,
    provider: str = "auto",
    http_timeout_s: float = 120.0,
    sample_scan_limit: int = 120,
    max_sample_values: int = 14,
    max_value_chars: int = 280,
    apply_typing: bool = True,
    output: str = "flat",
    struct_name: str = "weft",
    keep_extra_columns: bool = False,
    emit_weft_warnings: bool = True,
    return_report: bool = False,
) -> Any | tuple[Any, dict[str, Any]]:
    """Map ``df`` columns onto ``structure`` using one bounded LLM call, then a
    **single** Spark ``select`` for renames, casts, and canonical column order.

    Parameters
    ----------
    df
        PySpark ``DataFrame``.
    structure
        Canonical field names (keys) to natural-language descriptions (values).
        Column order in the output follows this mapping's iteration order.
    inference
        :class:`~raiju.inference.InferenceSettings` with Ollama and/or OpenRouter.
    prior_mappings
        Optional accepted source→target hints from earlier runs.
    min_confidence
        Below this, a proposed ``map`` is not applied (column stays unresolved
        unless ignored).
    require_review_below
        Applied ``map`` decisions in ``[min_confidence, require_review_below)``
        appear under ``report['review_suggested']``.
    allow_unmapped
        When ``False``, raises if any source column is neither accepted nor
        explicitly ignored after guardrails.
    allow_many_to_one
        When ``True``, multiple sources may rename to the same target; otherwise
        colliding proposals are withheld and listed under ``needs_review``.
    provider
        ``\"auto\"`` (prefer Ollama if configured), ``\"ollama\"``, or
        ``\"openrouter\"``.
    apply_typing
        When ``True`` (default), coerce each accepted column toward the LLM's
        ``target_spark_type`` using Spark-native casts and bounded
        ``try_to_timestamp`` patterns; ``python_dateutil`` uses a Python UDF only
        where the model requests it.
    output
        ``\"flat\"`` — one column per canonical key (default). ``\"struct\"`` —
        nest those columns under ``struct_name`` as a struct (still one scan).
    struct_name
        Top-level column name when ``output=\"struct\"``.
    keep_extra_columns
        When ``True``, append unmapped non-ignored source columns after the
        canonical block (still a single ``select``).
    emit_weft_warnings
        When ``True``, emit :class:`~raiju.inference.llm_schemas.WeftWarning` for
        missing canonical slots, dateutil UDF use, etc.
    return_report
        If ``True``, return ``(dataframe, report_dict)``.

    Returns
    -------
    DataFrame or (DataFrame, report)
        Canonicalized frame; report includes mappings, typing, advisories, and
        optional ``llm_token_usage``.
    """
    if not isinstance(inference, InferenceSettings):
        raise TypeError("inference must be InferenceSettings")
    if not structure:
        raise ValueError("structure must be a non-empty mapping")
    if output not in ("flat", "struct"):
        raise ValueError('output must be "flat" or "struct"')

    allowed = set(structure.keys())
    source_columns = list(df.columns)
    evidence, _n = _scan_column_evidence(
        df,
        scan_limit=sample_scan_limit,
        max_distinct_samples=max_sample_values,
        max_value_chars=max_value_chars,
    )

    user_payload: dict[str, Any] = {
        "allowed_target_columns": sorted(allowed),
        "canonical_field_descriptions": dict(structure),
        "columns": evidence,
    }
    if prior_mappings:
        user_payload["prior_accepted_mappings"] = dict(prior_mappings)

    user_text = json.dumps(user_payload, default=str)
    t0 = time.perf_counter()
    used, text, usage = _run_weft_llm(
        inference,
        user_text,
        provider=provider,
        http_timeout_s=float(http_timeout_s),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if usage is not None:
        usage.warn_if_known()

    parsed = _extract_json_object(text)
    if not parsed or not isinstance(parsed.get("mappings"), list):
        raise ValueError(
            "weft model returned unparseable JSON (expected object with mappings list)"
        )

    try:
        resp = WeftResponse.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"weft model JSON failed validation: {e}") from e

    resolved = resolve_weft_mappings(
        resp,
        source_columns,
        allowed,
        min_confidence=min_confidence,
        require_review_below=require_review_below,
        allow_unmapped=allow_unmapped,
        allow_many_to_one=allow_many_to_one,
    )

    struct_schema_simple: str | None
    out_df, _struct_type, advisory = weft_canonical_select(
        df,
        dict(structure),
        resolved["accepted_mappings"],
        resolved["accepted_specs"],
        _source_spark_types(df),
        apply_typing=apply_typing,
        output=output,
        struct_name=struct_name,
        keep_extra_columns=keep_extra_columns,
        emit_warnings=emit_weft_warnings,
        ignored_sources=frozenset(resolved["ignored_columns"]),
    )
    if _struct_type is not None:
        struct_schema_simple = _struct_type.simpleString()
    else:
        struct_schema_simple = None

    typing_applied = {
        tgt: resolved["accepted_specs"][src].target_spark_type
        for src, tgt in resolved["accepted_mappings"].items()
    }
    nullability_applied = {
        tgt: resolved["accepted_specs"][src].nullable
        for src, tgt in resolved["accepted_mappings"].items()
    }

    report: dict[str, Any] = {
        "accepted_mappings": resolved["accepted_mappings"],
        "accepted_specs": {
            src: spec.model_dump(mode="json", exclude_none=True)
            for src, spec in resolved["accepted_specs"].items()
        },
        "typing_applied": typing_applied,
        "nullability_applied": nullability_applied,
        "needs_review": resolved["needs_review"],
        "review_suggested": resolved["review_suggested"],
        "ignored_columns": resolved["ignored_columns"],
        "confidence_scores": resolved["confidence_scores"],
        "unresolved_sources": resolved["unresolved_sources"],
        "target_collisions": resolved["target_collisions"],
        "model_notes": resolved["model_notes"],
        "model_unmapped_columns": list(resp.unmapped_columns),
        "model_ambiguous_columns": list(resp.ambiguous_columns),
        "weft_advisory": advisory,
        "struct_schema_simple": struct_schema_simple,
        "output": output,
        "struct_name": struct_name if output == "struct" else None,
        "provider": used,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    if usage is not None:
        report["llm_token_usage"] = usage.model_dump(mode="json", exclude_none=True)

    if return_report:
        return out_df, report
    return out_df
