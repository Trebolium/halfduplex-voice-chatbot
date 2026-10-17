"""--test_start/--test_end flag validation and a mocked slice run."""
import pytest

from voicebot import main, pipeline_test as pt

WAV, TXT = "test/test_data/002.wav", "test/test_data/002.txt"


def args(*a):
    return main.parse_args(list(a))


def test_no_test_flags_is_live_app():
    assert pt.validate(args()) is None


@pytest.mark.parametrize("flags,expected", [
    (["--test_start", "asr", "--test_end", "llm", "--test_audio", WAV], ("ASR", "LLM")),
    (["--test_start", "VAD", "--test_audio", WAV], ("VAD", "TTS")),
    (["--test_end", "ASR", "--test_audio", WAV], ("VAD", "ASR")),
    (["--test_start", "LLM", "--test_end", "TTS", "--test_text", TXT], ("LLM", "TTS")),
    (["--test_start", "TTS", "--test_text", TXT], ("TTS", "TTS")),
])
def test_valid_combos(flags, expected):
    assert pt.validate(args(*flags)) == expected


@pytest.mark.parametrize("flags,msg", [
    (["--test_start", "ASR", "--test_end", "LLM", "--test_text", TXT], "needs audio"),
    (["--test_start", "ASR", "--test_end", "LLM"], "needs --test_audio"),
    (["--test_start", "LLM", "--test_audio", WAV], "takes text"),
    (["--test_start", "LLM"], "needs --test_text"),
    (["--test_start", "TTS", "--test_end", "ASR", "--test_text", TXT], "comes after"),
    (["--test_audio", WAV], "only apply to test runs"),
    (["--test_start", "VAD", "--test_audio", "nope.wav"], "not found"),
])
def test_invalid_combos(flags, msg):
    with pytest.raises(Exception, match=msg):
        pt.validate(args(*flags))


def test_run_asr_to_llm_only_calls_those(monkeypatch):
    calls = []
    monkeypatch.setattr(pt.asr, "select_backend", lambda n: calls.append("sel-asr"))
    monkeypatch.setattr(pt.llm, "select_backend", lambda n: calls.append("sel-llm"))
    monkeypatch.setattr(pt.tts, "select_backend", lambda n: calls.append("sel-tts"))
    monkeypatch.setattr(pt.vad, "reset", lambda: calls.append("sel-vad"))
    monkeypatch.setattr(pt, "transcribe", lambda p: calls.append("asr") or "hello")
    monkeypatch.setattr(pt, "reply", lambda s, h, t: calls.append("llm") or "hi")
    monkeypatch.setattr(pt, "synthesize", lambda t: calls.append("tts"))
    pt.run("ASR", "LLM", args("--test_start", "ASR", "--test_end", "LLM", "--test_audio", WAV), {"asr": "a", "llm": "l", "tts": "t"})
    assert calls == ["sel-asr", "sel-llm", "asr", "llm"]


def test_run_to_tts_does_not_play(monkeypatch):
    monkeypatch.setattr(pt.tts, "select_backend", lambda n: None)
    monkeypatch.setattr(pt, "synthesize", lambda t: "out.mp3")
    assert not hasattr(pt, "play")  # test runs never play audio
    pt.run("TTS", "TTS", args("--test_start", "TTS", "--test_text", TXT), {"asr": "a", "llm": "l", "tts": "t"})
