"""LLM dispatcher (mocked): --llm_model picks the implementation and installs deps when missing; emoji are stripped."""
import sys
from unittest.mock import MagicMock

import pytest

from voicebot import llm
from voicebot.llm import mlx_lm, openrouter


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    llm._current = None


def test_unknown_backend():
    with pytest.raises(ValueError, match="remote-llm"):
        llm.select_backend("nope")


def test_remote_llm_selected_and_routes(monkeypatch):
    assert llm.select_backend("remote-llm") is openrouter
    monkeypatch.setattr(openrouter, "reply", lambda s, h, u: "r")
    monkeypatch.setattr(openrouter, "complete", lambda s, u: "c")
    assert llm.reply("s", [], "u") == "r" and llm.complete("s", "u") == "c"


def _missing(monkeypatch):
    real_import = llm.importlib.import_module
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    monkeypatch.setattr(llm.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()) if n == "mlx_lm" else real_import(n))


def test_local_llm_installs_when_missing_then_loads(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(llm.subprocess, "run", run)
    monkeypatch.setattr(mlx_lm, "load", MagicMock())
    _missing(monkeypatch)
    assert llm.select_backend("local-llm") is mlx_lm
    assert run.call_args[0][0] == ["uv", "sync", "--inexact", "--group", "local-llm"]
    mlx_lm.load.assert_called_once()


def test_local_install_failure_is_clear(monkeypatch):
    monkeypatch.setattr(llm.subprocess, "run", MagicMock(side_effect=FileNotFoundError("uv")))
    _missing(monkeypatch)
    with pytest.raises(RuntimeError, match="remote-llm"):
        llm.select_backend("local-llm")


@pytest.mark.parametrize("raw,clean", [
    ("Hi there! \U0001F60A", "Hi there!"),
    ("Great \U0001F44D\U0001F3FD news ❤️ today", "Great news today"),
    ("Hello \U0001F468‍\U0001F469‍\U0001F467 friend", "Hello friend"),
    ("Don't worry, Zoë — it’s “fine”; café, naïve (ok)?", "Don't worry, Zoë — it’s “fine”; café, naïve (ok)?"),
])
def test_clean_strips_emoji_keeps_text(raw, clean):
    assert openrouter._clean(raw) == clean
