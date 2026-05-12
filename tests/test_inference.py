"""Tests for inference settings and Raiju attachment."""

from unittest.mock import MagicMock

import pytest
from raiju import InferenceSettings, OllamaConfig, OpenRouterConfig, Raiju


class TestInferenceSettings:
    def test_requires_at_least_one_backend(self):
        with pytest.raises(ValueError, match="at least one"):
            InferenceSettings()

    def test_ollama_only(self):
        s = InferenceSettings(ollama=OllamaConfig(default_model="llama3.2"))
        assert s.ollama is not None
        assert s.ollama.default_model == "llama3.2"
        assert s.openrouter is None

    def test_openrouter_only(self):
        s = InferenceSettings(
            openrouter=OpenRouterConfig(api_key="sk-test", default_model="x/y")
        )
        assert s.openrouter is not None
        assert s.openrouter.resolved_api_key() == "sk-test"

    def test_openrouter_resolves_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        with pytest.warns(UserWarning, match="OPENROUTER_API_KEY"):
            s = InferenceSettings(openrouter=OpenRouterConfig())
        assert s.openrouter is not None
        assert s.openrouter.resolved_api_key() == "from-env"

    def test_openrouter_warns_custom_env_var_name(self, monkeypatch):
        monkeypatch.setenv("MY_OR_KEY", "secret")
        with pytest.warns(UserWarning, match="MY_OR_KEY"):
            cfg = OpenRouterConfig(api_key_env_var="MY_OR_KEY")
        assert cfg.resolved_api_key() == "secret"

    def test_openrouter_explicit_key_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        s = InferenceSettings(
            openrouter=OpenRouterConfig(api_key="explicit"),
        )
        assert s.openrouter is not None
        assert s.openrouter.resolved_api_key() == "explicit"

    def test_invalid_base_url_scheme(self):
        with pytest.raises(ValueError, match="http"):
            OllamaConfig(base_url="ftp://127.0.0.1:11434")

    def test_normalized_urls(self):
        o = OllamaConfig(base_url="http://127.0.0.1:11434/")
        assert o.resolved_base_url() == "http://127.0.0.1:11434"


class TestRaijuInference:
    def test_default_inference_is_none(self):
        spark = MagicMock()
        r = Raiju(spark)
        assert r.inference is None

    def test_constructor_attaches_inference(self):
        spark = MagicMock()
        inf = InferenceSettings(
            ollama=OllamaConfig(default_model="m"),
            openrouter=OpenRouterConfig(api_key="k"),
        )
        r = Raiju(spark, inference=inf)
        assert r.inference is inf
        assert r.inference.ollama is not None
        assert r.inference.ollama.default_model == "m"

    def test_delegation_unchanged_with_inference(self):
        spark = MagicMock()
        spark.sql.return_value = "ok"
        inf = InferenceSettings(ollama=OllamaConfig())
        r = Raiju(spark, inference=inf)
        assert r.sql("SELECT 1") == "ok"

    def test_setattr_does_not_touch_inference_storage(self):
        spark = MagicMock()
        inf = InferenceSettings(ollama=OllamaConfig())
        r = Raiju(spark, inference=inf)
        r.other = 1
        assert r.inference is inf
        assert getattr(spark, "other") == 1

    def test_with_inference_same_spark_new_wrapper(self):
        spark = MagicMock()
        r0 = Raiju(spark)
        inf = InferenceSettings(ollama=OllamaConfig())
        r1 = r0.with_inference(inf)
        assert r1 is not r0
        assert r1._spark is spark
        assert r1.inference is inf
        assert r0.inference is None

    def test_with_inference_type_error(self):
        spark = MagicMock()
        with pytest.raises(TypeError, match="InferenceSettings"):
            Raiju(spark).with_inference("not-settings")  # type: ignore[arg-type]
