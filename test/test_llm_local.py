"""Local mlx-lm backend with generation mocked: markdown stripped, empty output raises a clear error."""
import pytest

from voicebot.llm import mlx_lm


def test_reply_cleans_markdown(monkeypatch):
    monkeypatch.setattr(mlx_lm, "stream_reply", lambda s, h, u: iter([("**Hi", None), ("**Hi there!", None)]))
    assert mlx_lm.reply("sys", [], "hello") == "Hi there!"


def test_empty_reply_raises(monkeypatch):
    monkeypatch.setattr(mlx_lm, "stream_reply", lambda s, h, u: iter([("  ", None)]))
    with pytest.raises(RuntimeError, match="empty"):
        mlx_lm.reply("sys", [], "hello")


def test_default_model_is_lfm():
    assert mlx_lm.DEFAULT_MODEL == "mlx-community/LFM2.5-1.2B-Instruct-4bit" and mlx_lm.INFO["model"] == mlx_lm.DEFAULT_MODEL


def test_complete_is_raw(monkeypatch):
    monkeypatch.setattr(mlx_lm, "stream_reply", lambda s, h, u: iter([("{\"new_facts\"", None), (' {"new_facts": []} ', None)]))
    assert mlx_lm.complete("sys", "text") == '{"new_facts": []}'
    monkeypatch.setattr(mlx_lm, "stream_reply", lambda s, h, u: iter([("", None)]))
    with pytest.raises(RuntimeError, match="empty"):
        mlx_lm.complete("sys", "text")
