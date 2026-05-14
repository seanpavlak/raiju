"""Guards on :mod:`raiju.inference.chat` (no live HTTP)."""

from typing import Any

import pytest
from raiju.inference.chat import inference_chat, parse_llm_json_object
from raiju.inference.settings import InferenceSettings, OllamaConfig


def test_parse_llm_json_object_rejects_top_level_array():
    assert parse_llm_json_object('["not", "an", "object"]') is None


def test_inference_chat_rejects_unknown_provider():
    s = InferenceSettings(ollama=OllamaConfig())
    with pytest.raises(ValueError, match="provider must be one of"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="azure",
            http_timeout_s=30.0,
        )


def test_inference_chat_rejects_non_positive_timeout():
    s = InferenceSettings(ollama=OllamaConfig())
    with pytest.raises(ValueError, match="http_timeout_s must be positive"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="auto",
            http_timeout_s=0.0,
        )


def test_inference_chat_accepts_spaced_auto(monkeypatch):
    called: dict[str, Any] = {}

    def fake_ollama(base, model, system, user, timeout):
        called["provider_arg"] = (base, model, timeout)
        return ("ok", {"message": {"content": "ok"}})

    monkeypatch.setattr("raiju.inference.chat._ollama_chat", fake_ollama)
    s = InferenceSettings(ollama=OllamaConfig())
    used, text, _usage = inference_chat(
        s,
        system="sys",
        user="usr",
        provider="  AUTO  ",
        http_timeout_s=5.0,
    )
    assert text == "ok"
    assert used.startswith("ollama:")
    assert called["provider_arg"][2] == 5.0
