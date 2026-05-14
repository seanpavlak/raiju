"""Pydantic models for LLM responses and token accounting (Raiju inference)."""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LLMTokenUsage",
    "ProfileEnrichmentColumn",
    "ProfileEnrichmentResponse",
    "RaijuLLMUsageWarning",
    "WeftColumnMapping",
    "WeftResponse",
    "WeftWarning",
]


class RaijuLLMUsageWarning(UserWarning):
    """Emitted after a successful LLM HTTP call with best-effort token counts."""


class WeftWarning(UserWarning):
    """Advisory for casts, missing canonical slots, fuzzy date UDFs, and coercion."""


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


class WeftColumnMapping(BaseModel):
    """One source column decision from the Weft schema-mapping LLM."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source_column: str
    target_column: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str
    action: Literal["map", "ignore", "needs_review"]
    # Typing / nullability (used when action is "map" after guardrails accept the map)
    target_spark_type: Literal[
        "string",
        "boolean",
        "byte",
        "short",
        "int",
        "long",
        "float",
        "double",
        "decimal",
        "date",
        "timestamp",
        "timestamp_ntz",
    ] = "string"
    nullable: bool = True
    decimal_precision: int | None = Field(default=None, ge=1, le=38)
    decimal_scale: int | None = Field(default=None, ge=0, le=18)
    temporal_parse_strategy: Literal[
        "native",
        "spark_formats",
        "python_dateutil",
    ] = "native"
    spark_timestamp_formats: list[str] = Field(default_factory=list)
    python_dateutil_fuzzy: bool = False

    @field_validator("spark_timestamp_formats", mode="before")
    @classmethod
    def _cap_ts_formats(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x)[:120] for x in v[:12]]
        return []

    @model_validator(mode="after")
    def _action_target_consistency(self) -> WeftColumnMapping:
        if self.action == "map":
            if self.target_column is None or not str(self.target_column).strip():
                raise ValueError("action 'map' requires a non-empty target_column")
        return self

    @model_validator(mode="after")
    def _decimal_fields_when_needed(self) -> WeftColumnMapping:
        if self.action != "map":
            return self
        if self.target_spark_type != "decimal":
            return self
        if self.decimal_precision is None and self.decimal_scale is not None:
            raise ValueError(
                "decimal_scale requires decimal_precision for weft mapping"
            )
        return self


class WeftResponse(BaseModel):
    """Structured reply the Weft LLM must return (validated before rename)."""

    model_config = ConfigDict(extra="ignore")

    mappings: list[WeftColumnMapping]
    unmapped_columns: list[str] = Field(default_factory=list)
    ambiguous_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_mapping_sources(self) -> WeftResponse:
        seen: set[str] = set()
        for m in self.mappings:
            if m.source_column in seen:
                raise ValueError(
                    f"duplicate mappings.source_column: {m.source_column!r}"
                )
            seen.add(m.source_column)
        return self

    @field_validator("unmapped_columns", "ambiguous_columns", "notes", mode="before")
    @classmethod
    def _string_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v[:256]]
        return []
