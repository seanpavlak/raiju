"""Raiju — distributed PySpark execution utilities with a Spark-native entry point.

The public `Raiju` type wraps `SparkSession` and forwards all attributes and
methods so PySpark APIs stay available without a duplicated surface area.
"""

from raiju.inference import (
    InferenceSettings,
    LLMTokenUsage,
    OllamaConfig,
    OpenRouterConfig,
    ProfileEnrichmentColumn,
    ProfileEnrichmentResponse,
    RaijuLLMUsageWarning,
    build_llm_token_usage,
)
from raiju.joins import BroadcastJoinPolicy, weave
from raiju.profile import ProfileOptions, profile_dataframe, profile_to_describe_rows
from raiju.session import Raiju

__all__ = [
    "BroadcastJoinPolicy",
    "InferenceSettings",
    "LLMTokenUsage",
    "OllamaConfig",
    "OpenRouterConfig",
    "ProfileEnrichmentColumn",
    "ProfileEnrichmentResponse",
    "ProfileOptions",
    "Raiju",
    "RaijuLLMUsageWarning",
    "build_llm_token_usage",
    "profile_dataframe",
    "profile_to_describe_rows",
    "weave",
]
