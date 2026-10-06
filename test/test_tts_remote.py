"""Remote TTS backends with the network mocked: files are written, errors say how to fix them."""
from unittest.mock import MagicMock

import pytest

from voicebot import config
from voicebot.tts import elevenlabs, google_tts, groq_orpheus


def _resp(code=200, body=b"audio", text=""):
    return MagicMock(status_code=code, content=body, text=text)


def test_groq_orpheus_writes_wav(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k"); monkeypatch.setattr(groq_orpheus, "ARTEFACTS_DIR", tmp_path)
    post = MagicMock(return_value=_resp())
    monkeypatch.setattr(groq_orpheus.httpx, "post", post)
    p = groq_orpheus.synthesize("hi")
    assert p.read_bytes() == b"audio" and post.call_args.kwargs["json"]["model"] == groq_orpheus.MODEL


def test_groq_orpheus_terms_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k"); monkeypatch.setattr(groq_orpheus, "ARTEFACTS_DIR", tmp_path)
    monkeypatch.setattr(groq_orpheus.httpx, "post", MagicMock(return_value=_resp(400, text="requires terms acceptance")))
    with pytest.raises(RuntimeError, match="console.groq.com"):
        groq_orpheus.synthesize("hi")


def test_elevenlabs_writes_mp3_and_uses_voice_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k"); monkeypatch.setattr(config, "TTS_VOICE", "myvoice")
    monkeypatch.setattr(elevenlabs, "ARTEFACTS_DIR", tmp_path)
    post = MagicMock(return_value=_resp())
    monkeypatch.setattr(elevenlabs.httpx, "post", post)
    assert elevenlabs.synthesize("hi").suffix == ".mp3" and "myvoice" in post.call_args[0][0]


def test_elevenlabs_error_is_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k"); monkeypatch.setattr(elevenlabs, "ARTEFACTS_DIR", tmp_path)
    monkeypatch.setattr(elevenlabs.httpx, "post", MagicMock(return_value=_resp(401, text="bad key")))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        elevenlabs.synthesize("hi")


def test_missing_keys():
    for mod, key in ((groq_orpheus, "GROQ_API_KEY"), (elevenlabs, "ELEVENLABS_API_KEY")):
        old = getattr(config, key); setattr(config, key, "")
        try:
            with pytest.raises(RuntimeError, match=key):
                mod.synthesize("hi")
        finally:
            setattr(config, key, old)


def test_gtts_error_is_clear(monkeypatch, tmp_path):
    import gtts
    monkeypatch.setattr(google_tts, "ARTEFACTS_DIR", tmp_path)
    monkeypatch.setattr(gtts, "gTTS", MagicMock(side_effect=OSError("offline")))
    with pytest.raises(RuntimeError, match="edge-tts"):
        google_tts.synthesize("hi")


def test_elevenlabs_retries_once_on_dropped_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k"); monkeypatch.setattr(elevenlabs, "ARTEFACTS_DIR", tmp_path)
    post = MagicMock(side_effect=[elevenlabs.httpx.ConnectError("ssl eof"), _resp()])
    monkeypatch.setattr(elevenlabs.httpx, "post", post)
    assert elevenlabs.synthesize("hi").read_bytes() == b"audio" and post.call_count == 2
    post.side_effect = elevenlabs.httpx.ConnectError("down")
    with pytest.raises(RuntimeError, match="internet connection"):
        elevenlabs.synthesize("hi")
