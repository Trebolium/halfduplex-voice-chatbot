"""Loop test with every stage mocked: one turn runs end to end, silence ends the session."""
from pathlib import Path

from voicebot import main


def test_run_turn_and_end(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "record", lambda wait_for_space: Path("a.wav") if not calls else None)
    monkeypatch.setattr(main, "trim_speech", lambda p: p)
    monkeypatch.setattr(main, "transcribe", lambda p: "hello")
    monkeypatch.setattr(main, "reply", lambda s, h, t: "hi there")
    monkeypatch.setattr(main, "synthesize", lambda t: Path("r.mp3"))
    monkeypatch.setattr(main, "play", lambda p: calls.append(p))
    history = []
    assert main.run_turn("sys", history, first=True) is True
    assert history[-1] == {"role": "assistant", "content": "hi there"}
    assert main.run_turn("sys", history, first=False) is False
