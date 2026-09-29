"""Tests for the OpenRouter LLM module (mocked; one live test needs OPENROUTER_API_KEY)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from voicebot import config
from voicebot.llm import openrouter


def _fake_client(content="Hello **there**!"):
    client = MagicMock()
    msg = SimpleNamespace(content=content)
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return client


def test_reply_builds_messages_and_cleans(monkeypatch):
    client = _fake_client()
    monkeypatch.setattr(openrouter, "_client", lambda: client)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    out = openrouter.reply("sys", history, "how are you")
    assert out == "Hello there!"
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[-1]["content"] == "how are you"
    assert len(history) == 2  # not mutated


def test_complete(monkeypatch):
    client = _fake_client("fact")
    monkeypatch.setattr(openrouter, "_client", lambda: client)
    assert openrouter.complete("sys", "text") == "fact"


def test_missing_key(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter.reply("s", [], "hi")


def test_bad_model_error(monkeypatch):
    client = MagicMock()
    resp = httpx.Response(404, request=httpx.Request("POST", "http://x"))
    client.chat.completions.create.side_effect = openai.NotFoundError("nf", response=resp, body=None)
    monkeypatch.setattr(openrouter, "_client", lambda: client)
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        openrouter.reply("s", [], "hi")


@pytest.mark.skipif(not config.OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set")
def test_live_reply():
    out = openrouter.reply(config.BASE_SYSTEM_PROMPT, [], "Say hello in one short sentence.")
    assert out


def test_complete_keeps_json_underscores(monkeypatch):
    """complete() must not strip markdown chars, or JSON keys like new_facts break."""
    from types import SimpleNamespace
    from voicebot.llm import openrouter
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"new_facts": []}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: resp)))
    monkeypatch.setattr(openrouter, "_client", lambda: client)
    assert "new_facts" in openrouter.complete("s", "u")
