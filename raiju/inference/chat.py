"""Ollama / OpenRouter chat completion and helpers for LLM payloads and replies."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from raiju.inference.llm_schemas import LLMTokenUsage
from raiju.inference.settings import InferenceSettings
from raiju.inference.token_count import build_llm_token_usage

__all__ = [
    "inference_chat",
    "parse_llm_json_object",
    "truncate_llm_text",
]

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def truncate_llm_text(s: str, max_len: int) -> str:
    """Ellipsize a string for prompts, logs, or excerpts (UTF-8 safe slice)."""
    ml = int(max_len)
    if ml <= 0:
        return ""
    if ml <= 3:
        return s[:ml]
    if len(s) <= ml:
        return s
    return s[: ml - 3] + "..."


def _http_json_post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _ollama_chat(
    base: str,
    model: str,
    system: str,
    user: str,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    url = f"{base.rstrip('/')}/api/chat"
    j = _http_json_post(
        url,
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        {},
        timeout,
    )
    msg = j.get("message") or {}
    return str(msg.get("content") or ""), j


def _openrouter_chat(
    base: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    url = f"{base.rstrip('/')}/chat/completions"
    j = _http_json_post(
        url,
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        {
            "Authorization": f"Bearer {api_key}",
        },
        timeout,
    )
    choices = j.get("choices") or []
    if not choices:
        return "", j
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or ""), j


def inference_chat(
    settings: InferenceSettings,
    *,
    system: str,
    user: str,
    provider: str,
    http_timeout_s: float,
    purpose: str = "inference",
) -> tuple[str, str, LLMTokenUsage | None]:
    """Run one Ollama or OpenRouter chat completion.

    Returns ``(provider_label, assistant_text, token_usage)`` where *provider_label*
    is like ``\"ollama:llama3.2\"`` or ``\"openrouter:openai/gpt-4o-mini\"``.

    Parameters
    ----------
    settings
        :class:`~raiju.inference.InferenceSettings` with at least one backend.
    system, user
        Prompt strings (typically system rules and a JSON user payload).
    provider
        ``\"auto\"`` (prefer Ollama when configured), ``\"ollama\"``, or
        ``\"openrouter\"`` (case-insensitive; surrounding whitespace stripped).
    http_timeout_s
        Per-request HTTP timeout in seconds (must be positive).
    purpose
        Short label for error messages if no backend is available.
    """
    p = str(provider).strip().lower()
    if p not in ("auto", "ollama", "openrouter"):
        raise ValueError(
            f"provider must be one of: auto, ollama, openrouter (got {provider!r})"
        )
    provider = p
    if float(http_timeout_s) <= 0:
        raise ValueError("http_timeout_s must be positive")

    model_ollama = "llama3.2"
    model_or = "openai/gpt-4o-mini"
    if settings.ollama and settings.ollama.default_model:
        model_ollama = settings.ollama.default_model
    if settings.openrouter and settings.openrouter.default_model:
        model_or = settings.openrouter.default_model

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
            system,
            user,
            http_timeout_s,
        )
        used = f"ollama:{model_ollama}"
        usage = build_llm_token_usage(
            provider="ollama",
            model=model_ollama,
            system=system,
            user=user,
            assistant=text,
            raw_api_response=raw,
        )
        return used, text, usage
    if use_openrouter:
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
            system,
            user,
            http_timeout_s,
        )
        used = f"openrouter:{model_or}"
        usage = build_llm_token_usage(
            provider="openrouter",
            model=model_or,
            system=system,
            user=user,
            assistant=text,
            raw_api_response=raw,
        )
        return used, text, usage
    raise RuntimeError(
        f"No inference backend available for {purpose} (provider={provider!r})"
    )


def parse_llm_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON **object** from model text (optional ```json fences).

    Returns ``None`` if the payload parses to a non-object (e.g. bare JSON array).
    """
    t = text.strip()
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            inner = json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(inner, dict):
            return inner
    return None
