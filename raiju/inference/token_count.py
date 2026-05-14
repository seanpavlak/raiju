"""Token counting for inference calls using ``tiktoken`` (OpenAI-compatible BPE).

Counts are **estimates** when the serving model is not an exact OpenAI model
name: we pick the closest ``tiktoken.encoding_for_model`` match, then fall
back to ``cl100k_base``. Provider ``usage`` fields are preserved under
``raw_usage["api"]`` for audit.
"""

from __future__ import annotations

from typing import Any, Literal

import tiktoken

from raiju.inference.llm_schemas import LLMTokenUsage

__all__ = ["build_llm_token_usage"]


def _encoding_for_model_hint(model: str) -> tuple[Any, str]:
    """Return (encoding, label); label is the tiktoken encoding name or fallback."""
    m = (model or "").strip()
    if not m:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc, "cl100k_base"
    candidates = [m, m.split("/")[-1]]
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            enc = tiktoken.encoding_for_model(c)
            return enc, enc.name
        except KeyError:
            continue
    enc = tiktoken.get_encoding("cl100k_base")
    return enc, "cl100k_base"


def _openrouter_api_usage(body: dict[str, Any]) -> dict[str, Any]:
    raw = body.get("usage")
    return dict(raw) if isinstance(raw, dict) else {}


def _ollama_api_usage(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("prompt_eval_count", "eval_count", "total_duration", "load_duration"):
        if k in body:
            out[k] = body[k]
    return out


def build_llm_token_usage(
    *,
    provider: Literal["ollama", "openrouter"],
    model: str,
    system: str,
    user: str,
    assistant: str,
    raw_api_response: dict[str, Any],
) -> LLMTokenUsage:
    enc, enc_label = _encoding_for_model_hint(model)
    prompt_tokens = len(enc.encode(system)) + len(enc.encode(user))
    completion_tokens = len(enc.encode(assistant)) if assistant else 0
    total_tokens = prompt_tokens + completion_tokens

    api_blob: dict[str, Any]
    if provider == "openrouter":
        api_blob = _openrouter_api_usage(raw_api_response)
    else:
        api_blob = _ollama_api_usage(raw_api_response)

    raw_usage: dict[str, Any] = {
        "tiktoken": {
            "encoding": enc_label,
            "model_hint": model,
        },
        "api": api_blob,
    }
    return LLMTokenUsage(
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
    )
