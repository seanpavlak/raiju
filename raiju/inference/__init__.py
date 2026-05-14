"""Optional inference configuration for Raiju (Ollama / OpenRouter).

Settings attach without I/O. :mod:`raiju.inference.chat` performs bounded
driver-side completions; :mod:`raiju.inference.profile_enrichment` uses it when
profiling with ``inference_enrichment=True``.
"""

from raiju.inference.chat import (
    inference_chat,
    parse_llm_json_object,
    truncate_llm_text,
)
from raiju.inference.llm_schemas import (
    LLMTokenUsage,
    ProfileEnrichmentColumn,
    ProfileEnrichmentResponse,
    RaijuLLMUsageWarning,
    WeftColumnMapping,
    WeftResponse,
    WeftWarning,
)
from raiju.inference.settings import InferenceSettings, OllamaConfig, OpenRouterConfig
from raiju.inference.token_count import build_llm_token_usage

__all__ = [
    "InferenceSettings",
    "LLMTokenUsage",
    "OllamaConfig",
    "OpenRouterConfig",
    "ProfileEnrichmentColumn",
    "ProfileEnrichmentResponse",
    "RaijuLLMUsageWarning",
    "WeftColumnMapping",
    "WeftResponse",
    "WeftWarning",
    "build_llm_token_usage",
    "inference_chat",
    "parse_llm_json_object",
    "truncate_llm_text",
]
