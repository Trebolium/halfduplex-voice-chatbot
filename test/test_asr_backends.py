"""ASR dispatcher + local backend (mocked): --asr_model picks the implementation and installs deps when missing."""
import sys
import types
from unittest.mock import MagicMock

import pytest

from voicebot import asr
from voicebot.asr import mlx_whisper


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    asr._current = None


def test_unknown_backend():
    with pytest.raises(ValueError, match="remote-whisper"):
        asr.select_backend("nope")


def test_remote_whisper_selected():
    assert asr.select_backend("remote-whisper").INFO["location"] == "cloud"


def test_local_whisper_installs_when_missing_then_loads(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(asr.subprocess, "run", run)
    monkeypatch.setattr(mlx_whisper, "load", MagicMock())
    monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)
    real_import = asr.importlib.import_module
    monkeypatch.setattr(asr.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()) if n == "mlx_whisper" else real_import(n))
    asr.select_backend("local-whisper")
    assert run.call_args[0][0] == ["uv", "sync", "--inexact", "--group", "local-asr"]
    mlx_whisper.load.assert_called_once()


def test_local_install_failure_is_clear(monkeypatch):
    monkeypatch.setattr(asr.subprocess, "run", MagicMock(side_effect=FileNotFoundError("uv")))
    real_import = asr.importlib.import_module
    monkeypatch.setattr(asr.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()) if n == "mlx_whisper" else real_import(n))
    with pytest.raises(RuntimeError, match="remote-whisper"):
        asr.select_backend("local-whisper")


def test_local_transcribe_mocked(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(transcribe=lambda path, **kw: {"text": "  hi there "})
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    assert mlx_whisper.transcribe(tmp_path / "a.wav") == "hi there"
