"""StreamingSession with fake ASR/LLM: speculative reply at a pause, discard on resumed speech, fallback, failures."""
import time

import numpy as np
import soundfile as sf

from voicebot import config
from voicebot.asr.streaming import StreamingSession

FRAME = np.ones(512, np.int16)
WORDS = "hello there my name is margaret and i like roses".split()


def fake_asr(path):
    """Transcript grows with the audio: one word per ~0.25s, like a re-transcription of the audio so far."""
    secs = sf.info(str(path)).duration
    return " ".join(WORDS[:max(1, int(secs / 0.25))]).capitalize() + "."


def make(monkeypatch, tmp_path, transcribe=fake_asr, replies=None):
    monkeypatch.setattr(config, "ARTEFACTS_DIR", tmp_path)
    replies = replies if replies is not None else []
    s = StreamingSession("sys", [{"role": "user", "content": "earlier"}], transcribe=transcribe,
                         reply=lambda sp, h, t: replies.append((t, list(h))) or f"reply to {t}",
                         every_s=0.5, pause_s=0.3, window=2)
    return s, replies


def wait_idle(s):
    while s.pending:
        time.sleep(0.01)


def speak(s, secs, speech=True):
    for _ in range(int(secs / (512 / 16000))):
        s.feed(FRAME, speech)
        time.sleep(0.001)


def test_speculative_reply_used_at_endpoint(monkeypatch, tmp_path):
    s, replies = make(monkeypatch, tmp_path)
    speak(s, 1.5); speak(s, 1.0, speech=False)  # speech, then the end-of-turn silence
    res = s.finish()
    assert res["path"] == "speculative" and res["speculations"] == 1 and res["partials"] >= 2
    assert res["reply"] == f"reply to {res['text']}" and res["text"].startswith("Hello there")
    assert replies[0][1] == [{"role": "user", "content": "earlier"}]  # history passed through, not mutated


def test_resumed_speech_discards_speculation(monkeypatch, tmp_path):
    s, replies = make(monkeypatch, tmp_path)
    speak(s, 1.0); speak(s, 0.5, speech=False)  # pause -> speculation
    wait_idle(s)
    speak(s, 0.6); speak(s, 1.0, speech=False)  # user keeps talking, then really stops
    res = s.finish()
    assert res["stale_speculations"] == 1 and res["speculations"] == 2 and res["path"] == "speculative"
    assert len(res["text"].split()) > len(replies[0][0].split())  # final transcript covers the extra speech


def test_failing_partials_fall_back(monkeypatch, tmp_path):
    calls = []

    def flaky(path):
        calls.append(path.name)
        if "partial" in path.name:
            raise RuntimeError("rate limited")
        return "Final words."
    s, _ = make(monkeypatch, tmp_path, transcribe=flaky)
    speak(s, 0.8); speak(s, 1.0, speech=False)
    res = s.finish()
    assert res["path"] == "fallback" and res["text"] == "Final words." and res["partial_failures"] >= 1
    assert res["reply"] == "reply to Final words."


def test_stale_speculation_stops_generating(monkeypatch, tmp_path):
    """With a token-streaming LLM, resumed speech stops the speculative generation at the next token."""
    monkeypatch.setattr(config, "ARTEFACTS_DIR", tmp_path)
    tokens_made = []

    def slow_stream(sp, h, text):
        out = ""
        for w in ["one", "two", "three", "four", "five", "six"]:
            time.sleep(0.05); out += w + " "; tokens_made.append(w)
            yield out, None
    s = StreamingSession("sys", [], transcribe=fake_asr, reply=lambda *a: "batch reply", stream_reply=slow_stream,
                         every_s=0.5, pause_s=0.3, window=2)
    speak(s, 0.8); speak(s, 0.4, speech=False)  # pause -> speculative generation starts
    while not s.stats["speculations"]:
        time.sleep(0.01)
    speak(s, 0.1)  # user resumes right away
    wait_idle(s)
    assert s.stats["cancelled_generations"] == 1 and len(tokens_made) < 6
    speak(s, 1.0, speech=False)
    res = s.finish()
    assert res["path"] == "speculative" and res["reply"] == "one two three four five six"
