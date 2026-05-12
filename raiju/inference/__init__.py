"""
Optional inference configuration for future enrichment execution.

Holds provider endpoints, default models, and OpenRouter credential resolution
without performing network I/O at import time.
"""

from raiju.inference.settings import InferenceSettings, OllamaConfig, OpenRouterConfig

__all__ = ["InferenceSettings", "OllamaConfig", "OpenRouterConfig"]
