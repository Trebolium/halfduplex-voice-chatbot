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


def test_noise_returns_none(tmp_path, monkeypatch):
    """White noise is not speech (the usual Whisper hallucination trigger)."""
    monkeypatch.setattr(vad, "ARTEFACTS_DIR", tmp_path)
    x = np.random.default_rng(1).normal(0, 3000, SR * 2)
    assert vad.trim_speech(_wav(tmp_path / "n.wav", x)) is None


def test_real_speech_kept_and_trimmed(tmp_path, monkeypatch):
    """Real speech padded with silence keeps the speech and drops most of the silence."""
    import soundfile as sf
    monkeypatch.setattr(vad, "ARTEFACTS_DIR", tmp_path)
    speech, sr = sf.read("test/test_data/002.wav", dtype="int16")
    assert sr == SR
    padded = np.concatenate([np.zeros(SR * 3, dtype=np.int16), speech, np.zeros(SR * 3, dtype=np.int16)])
    out = vad.trim_speech(_wav(tmp_path / "p.wav", padded))
    assert out is not None and out.name == "p_trimmed.wav"
    assert 0.8 * len(speech) < sf.info(str(out)).frames < len(padded) * 0.7


def test_is_speech_frames():
    """Live per-frame detector: silence is False."""
    vad.reset()
    assert vad.is_speech(np.zeros(vad.FRAME_SAMPLES, dtype=np.int16)) is False


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
