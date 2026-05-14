"""Pydantic models for LLM responses and token accounting (Raiju inference)."""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "LLMTokenUsage",
    "ProfileEnrichmentColumn",
    "ProfileEnrichmentResponse",
    "RaijuLLMUsageWarning",
]


class RaijuLLMUsageWarning(UserWarning):
    """Emitted after a successful LLM HTTP call with best-effort token counts."""


class LLMTokenUsage(BaseModel):
    """Normalized token accounting across providers."""

    model_config = ConfigDict(extra="allow")

    provider: Literal["ollama", "openrouter"]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_usage: dict[str, Any] = Field(default_factory=dict)

    def warn_if_known(self) -> None:
        """Surface totals in the warning stream (notebooks / CI pick this up)."""
        if self.total_tokens is not None:
            msg = (
                "Raiju LLM token usage (tiktoken estimate; "
                f"{self.provider}): total={self.total_tokens}"
            )
            if self.prompt_tokens is not None or self.completion_tokens is not None:
                msg += (
                    f" (prompt={self.prompt_tokens}, "
                    f"completion={self.completion_tokens})"
                )
            warnings.warn(msg, RaijuLLMUsageWarning, stacklevel=3)
        else:
            warnings.warn(
                f"Raiju LLM call completed ({self.provider}) but token totals "
                "were unavailable.",
                RaijuLLMUsageWarning,
                stacklevel=3,
            )


class ProfileEnrichmentColumn(BaseModel):
    """Strict shape for one column in the profiling LLM JSON payload."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    column: str
    human_summary: str | None = None
    semantic_classification: str | None = None
    suggested_validation_regex: str | None = None
    java_simple_date_format: str | None = None
    python_strptime_directive: str | None = None
    suggested_cast_or_parse: str | None = None
    pii_likelihood: str | None = None
    quality_flags: list[str] | None = None
    notes: str | None = None

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _coerce_flags(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return [str(x) for x in v[:12]]
        return None

    @field_validator("suggested_validation_regex", mode="after")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        try:
            re.compile(v)
        except re.error:
            warnings.warn(
                f"LLM suggested invalid regex; dropping: {v!r}",
                UserWarning,
                stacklevel=4,
            )
            return None
        return v


class ProfileEnrichmentResponse(BaseModel):
    """Root object the profiling LLM must return."""

    model_config = ConfigDict(extra="ignore")

    columns: list[ProfileEnrichmentColumn]
