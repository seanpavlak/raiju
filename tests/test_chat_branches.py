"""Branch coverage for :mod:`raiju.inference.chat` (mocked HTTP)."""

import json
import warnings
from unittest.mock import patch

import pytest
from raiju.inference.chat import (
    inference_chat,
    parse_llm_json_object,
    truncate_llm_text,
)
from raiju.inference.settings import InferenceSettings, OllamaConfig, OpenRouterConfig


def test_truncate_llm_text_nonpositive_and_short():
    assert truncate_llm_text("abc", 0) == ""
    assert truncate_llm_text("abc", 2) == "ab"
    assert truncate_llm_text("hello", 10) == "hello"


def test_parse_llm_json_object_brace_substring():
    t = 'prefix {"a": 1} suffix'
    assert parse_llm_json_object(t) == {"a": 1}


def test_parse_llm_json_object_invalid_json_returns_none():
    assert parse_llm_json_object("not json at all {{{") is None


def test_parse_llm_json_object_non_object_array_returns_none():
    assert parse_llm_json_object("[1, 2]") is None


def test_parse_llm_json_object_brace_inner_invalid_returns_none():
    assert parse_llm_json_object('xx {"a":} yy') is None


def test_inference_chat_invalid_provider():
    s = InferenceSettings(ollama=OllamaConfig())
    with pytest.raises(ValueError, match="provider"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="localai",
            http_timeout_s=1.0,
        )


def test_inference_chat_non_positive_timeout():
    s = InferenceSettings(ollama=OllamaConfig())
    with pytest.raises(ValueError, match="http_timeout_s"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="ollama",
            http_timeout_s=0.0,
        )


def test_inference_chat_openrouter_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = OpenRouterConfig(
            api_key=None,
            default_model="x/y",
            api_key_env_var="RAIJU_OR_KEY_ABSENT_TEST",
        )
    monkeypatch.delenv("RAIJU_OR_KEY_ABSENT_TEST", raising=False)
    s = InferenceSettings(openrouter=cfg)
    with pytest.raises(RuntimeError, match="API key"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="openrouter",
            http_timeout_s=1.0,
        )


def test_inference_chat_no_backend_when_openrouter_requested_but_only_ollama():
    s = InferenceSettings(ollama=OllamaConfig())
    with pytest.raises(RuntimeError, match="No inference backend"):
        inference_chat(
            s,
            system="s",
            user="u",
            provider="openrouter",
            http_timeout_s=1.0,
        )


def test_http_json_post_parses_json_body(monkeypatch):
    from raiju.inference import chat as chat_mod

    class _Resp:
        def read(self):
            return json.dumps({"ok": True, "n": 3}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        assert timeout == 2.5
        assert req.get_method() == "POST"
        return _Resp()

    monkeypatch.setattr(chat_mod.urllib.request, "urlopen", fake_urlopen)
    out = chat_mod._http_json_post(
        "https://example.com/x",
        {"a": 1},
        {"X-Test": "1"},
        2.5,
    )
    assert out == {"ok": True, "n": 3}


def test_inference_chat_ollama_branch(monkeypatch):
    raw = {"message": {"content": "model reply"}, "prompt_eval_count": 1}

    def fake_ollama(base, model, system, user, timeout):
        assert "llama" in model or model
        return "model reply", raw

    monkeypatch.setattr("raiju.inference.chat._ollama_chat", fake_ollama)
    s = InferenceSettings(ollama=OllamaConfig(default_model="llama3.2"))
    used, text, usage = inference_chat(
        s,
        system="sys",
        user="usr",
        provider="ollama",
        http_timeout_s=10.0,
    )
    assert text == "model reply"
    assert used.startswith("ollama:")
    assert usage is not None and usage.provider == "ollama"


def test_inference_chat_openrouter_branch(monkeypatch):
    raw = {
        "choices": [{"message": {"content": "or reply"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }

    def fake_or(base, model, key, system, user, timeout):
        assert key == "sk"
        return "or reply", raw

    monkeypatch.setattr("raiju.inference.chat._openrouter_chat", fake_or)
    s = InferenceSettings(
        openrouter=OpenRouterConfig(
            api_key="sk",
            default_model="openai/gpt-4o-mini",
        )
    )
    used, text, usage = inference_chat(
        s,
        system="s",
        user="u",
        provider="openrouter",
        http_timeout_s=10.0,
    )
    assert text == "or reply"
    assert used.startswith("openrouter:")
    assert usage is not None and usage.provider == "openrouter"


def test_inference_chat_openrouter_empty_choices(monkeypatch):
    def fake_or(base, model, key, system, user, timeout):
        return "", {"choices": []}

    monkeypatch.setattr("raiju.inference.chat._openrouter_chat", fake_or)
    s = InferenceSettings(
        openrouter=OpenRouterConfig(api_key="k", default_model="openai/gpt-4o-mini")
    )
    used, text, _usage = inference_chat(
        s, system="s", user="u", provider="openrouter", http_timeout_s=5.0
    )
    assert text == ""
    assert used.startswith("openrouter:")


def test_openrouter_chat_parses_message_from_first_choice():
    from raiju.inference import chat as chat_mod

    with patch.object(chat_mod, "_http_json_post", return_value={"choices": []}):
        text, raw = chat_mod._openrouter_chat(
            "https://example.com/api/v1",
            "m",
            "k",
            "sys",
            "usr",
            5.0,
        )
    assert text == ""
    assert raw == {"choices": []}


def test_ollama_chat_builds_request(monkeypatch):
    from raiju.inference import chat as chat_mod

    captured: dict = {}

    def capture_post(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "hi"}}

    monkeypatch.setattr(chat_mod, "_http_json_post", capture_post)
    text, raw = chat_mod._ollama_chat(
        "http://127.0.0.1:11434",
        "mistral",
        "system text",
        "user text",
        3.0,
    )
    assert text == "hi"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["model"] == "mistral"
    assert raw["message"]["content"] == "hi"
