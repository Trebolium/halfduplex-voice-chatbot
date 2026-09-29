"""Tests for VAD trimming and Groq ASR (mocked, plus optional live)."""
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicebot import config
from voicebot.asr import groq_asr, vad

SR = config.SAMPLE_RATE


def _wav(path: Path, samples: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.astype(np.int16).tobytes())
    return path


def test_silence_returns_none(tmp_path, monkeypatch):
    """Pure silence yields no speech."""
    monkeypatch.setattr(vad, "ARTEFACTS_DIR", tmp_path)
    assert vad.trim_speech(_wav(tmp_path / "s.wav", np.zeros(SR * 2))) is None


def test_short_burst_returns_none(tmp_path, monkeypatch):
    """A brief burst (60ms; webrtcvad hangover stretches a 200ms one past the minimum) is dropped."""
    monkeypatch.setattr(vad, "ARTEFACTS_DIR", tmp_path)
    x = np.zeros(SR * 2)
    x[SR:SR + int(SR * 0.06)] = np.random.default_rng(0).normal(0, 8000, int(SR * 0.06))
    assert vad.trim_speech(_wav(tmp_path / "b.wav", x)) is None


def test_output_never_longer(tmp_path, monkeypatch):
    """Noisy long segment: if kept, output is a valid wav no longer than input."""
    monkeypatch.setattr(vad, "ARTEFACTS_DIR", tmp_path)
    x = np.zeros(SR * 3)
    x[SR:2 * SR] = np.random.default_rng(1).normal(0, 8000, SR)
    out = vad.trim_speech(_wav(tmp_path / "n.wav", x))
    if out is not None:
        assert out.name == "n_trimmed.wav"
        with wave.open(str(out)) as w:
            assert w.getnframes() <= 3 * SR


def test_transcribe_mocked(tmp_path, monkeypatch):
    """Mocked Groq client returns stripped text."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake")
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text="  hello there ")
    with patch.object(groq_asr, "Groq", return_value=client):
        assert groq_asr.transcribe(_wav(tmp_path / "a.wav", np.zeros(SR))) == "hello there"
    assert client.audio.transcriptions.create.call_args.kwargs["language"] == "en"


def test_transcribe_error_wrapped(tmp_path, monkeypatch):
    """API errors become clear RuntimeErrors."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake")
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = ConnectionError("boom")
    with patch.object(groq_asr, "Groq", return_value=client):
        with pytest.raises(RuntimeError, match="Check GROQ_API_KEY"):
            groq_asr.transcribe(_wav(tmp_path / "a.wav", np.zeros(SR)))


def test_missing_key(tmp_path, monkeypatch):
    """Missing key gives a clear error."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        groq_asr.transcribe(tmp_path / "x.wav")


@pytest.mark.skipif(not config.GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_transcribe_live(tmp_path):
    """Live Groq call on silence; only checks it returns a string."""
    assert isinstance(groq_asr.transcribe(_wav(tmp_path / "l.wav", np.zeros(SR))), str)
