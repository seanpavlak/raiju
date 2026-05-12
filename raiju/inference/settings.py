"""Inference provider settings attached at Raiju initialization (no I/O)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from urllib.parse import urlparse


def _normalize_http_base(url: str) -> str:
    stripped = url.strip().rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http(s): {url!r}")
    if not parsed.netloc:
        raise ValueError(f"base_url must include a host: {url!r}")
    return stripped


@dataclass(frozen=True)
class OllamaConfig:
    """
    Local Ollama HTTP API (default port 11434).

    Used for future partition-local or driver-coordinated inference calls.
    """

    base_url: str = "http://127.0.0.1:11434"
    default_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_http_base(self.base_url))

    def resolved_base_url(self) -> str:
        """Base URL without trailing slash, suitable for joining API paths."""
        return self.base_url.rstrip("/")


@dataclass(frozen=True)
class OpenRouterConfig:
    """
    OpenRouter-compatible HTTP API (OpenAI-style paths under base_url).

    API key may be supplied explicitly or read from ``api_key_env_var``.
    If ``api_key`` is ``None``, constructing this config emits a ``UserWarning``
    that ``resolved_api_key()`` will read from that environment variable.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str | None = None
    api_key: str | None = None
    api_key_env_var: str = "OPENROUTER_API_KEY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_http_base(self.base_url))
        if not (self.api_key_env_var and str(self.api_key_env_var).strip()):
            raise ValueError("api_key_env_var must be a non-empty string")
        if self.api_key is None:
            warnings.warn(
                "OpenRouterConfig.api_key is None; resolved_api_key() will read from "
                f"the environment variable {self.api_key_env_var!r}. "
                "Pass api_key=... to silence this warning.",
                UserWarning,
                stacklevel=3,
            )

    def resolved_base_url(self) -> str:
        return self.base_url.rstrip("/")

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env_var)


@dataclass(frozen=True)
class InferenceSettings:
    """
    Bundle of optional inference backends for future Raiju execution hooks.

    No network requests are made when constructing or attaching these settings.
    """

    ollama: OllamaConfig | None = None
    openrouter: OpenRouterConfig | None = None

    def __post_init__(self) -> None:
        if self.ollama is None and self.openrouter is None:
            raise ValueError(
                "InferenceSettings requires at least one of: ollama, openrouter"
            )
