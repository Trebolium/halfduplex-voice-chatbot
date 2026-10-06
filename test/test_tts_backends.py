"""TTS dispatcher (mocked): --tts_model picks the implementation, installs deps when missing, routes synthesize()."""
import sys
from unittest.mock import MagicMock

import pytest

from voicebot import tts
from voicebot.tts import edge_tts, piper


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    tts._current = None


def test_unknown_backend():
    with pytest.raises(ValueError, match="piper"):
        tts.select_backend("nope")


def test_default_is_piper_and_routes(monkeypatch):
    monkeypatch.setattr(piper, "load", MagicMock())
    monkeypatch.setattr(piper, "synthesize", lambda t: f"piper:{t}")
    assert tts.DEFAULT == "piper" and tts.current() is piper
    assert tts.synthesize("hi") == "piper:hi"


def test_edge_tts_selected_without_install(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(tts.subprocess, "run", run)
    assert tts.select_backend("edge-tts") is edge_tts
    run.assert_not_called()


def _missing(monkeypatch):
    real_import = tts.importlib.import_module
    monkeypatch.delitem(sys.modules, "piper", raising=False)
    monkeypatch.setattr(tts.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError()) if n == "piper" else real_import(n))


def test_local_installs_when_missing_then_loads(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(tts.subprocess, "run", run)
    monkeypatch.setattr(piper, "load", MagicMock())
    _missing(monkeypatch)
    tts.select_backend("piper")
    assert run.call_args[0][0] == ["uv", "sync", "--inexact", "--group", "local-tts"]
    piper.load.assert_called_once()


def test_install_failure_is_clear(monkeypatch):
    monkeypatch.setattr(tts.subprocess, "run", MagicMock(side_effect=FileNotFoundError("uv")))
    _missing(monkeypatch)
    with pytest.raises(RuntimeError, match="edge-tts"):
        tts.select_backend("piper")


def test_piper_downloads_voice_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(piper, "MODEL", tmp_path / "v.onnx")
    monkeypatch.setattr(piper, "_voice", None)
    called = MagicMock()
    monkeypatch.setattr(piper, "_download", called)
    monkeypatch.setitem(sys.modules, "piper", MagicMock(PiperVoice=MagicMock()))
    piper.load()
    called.assert_called_once()


def test_piper_download_failure_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(piper, "MODEL", tmp_path / "v.onnx")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=OSError("offline")))
    with pytest.raises(RuntimeError, match="Check your connection"):
        piper._download()
    assert not (tmp_path / "v.onnx").exists()


def test_piper_fetch_resumes_truncated_stream(monkeypatch, tmp_path):
    """First response ends early; the Range retry completes the file."""
    import io
    import urllib.request
    data, seen = b"0123456789", []

    def fake_urlopen(req):
        rng = req.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        seen.append(start)
        body = data[start:] if start else data[:4]  # first call: truncated
        r = io.BytesIO(body); r.headers = {"Content-Length": str(len(data) - start)}
        r.__enter__ = lambda self=r: self; r.__exit__ = lambda *a: None
        return r
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    piper._fetch("http://x/f", tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == data and seen == [0, 4]
