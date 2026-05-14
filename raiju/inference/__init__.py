"""Optional inference configuration for Raiju (Ollama / OpenRouter).

Settings attach without I/O. :mod:`raiju.inference.profile_enrichment` performs
bounded driver-side LLM calls when profiling with ``inference_enrichment=True``.
"""

from raiju.inference.llm_schemas import (
    LLMTokenUsage,
    ProfileEnrichmentColumn,
    ProfileEnrichmentResponse,
    RaijuLLMUsageWarning,
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
    "build_llm_token_usage",
]
