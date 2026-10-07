from voicebot.config import MAX_RECORD_SECS, SILENCE_END_SECS
from voicebot.input.recorder import FRAME_MS, endpoint_state


def _run(pattern):
    """Feed a list of is_speech bools; return frame count at which recording ends (or None)."""
    s = 0
    for i, sp in enumerate(pattern, 1):
        s, done = endpoint_state(sp, s, i)
        if done:
            return i
    return None


def test_ends_after_silence():
    n = -(-int(SILENCE_END_SECS * 1000) // FRAME_MS)
    assert _run([True] * 10 + [False] * n) == 10 + n


def test_speech_resets_silence():
    n = -(-int(SILENCE_END_SECS * 1000) // FRAME_MS)
    assert _run([True] * 5 + [False] * (n - 1) + [True] + [False] * (n - 1)) is None


def test_max_cap():
    n = int(MAX_RECORD_SECS * 1000 / FRAME_MS)
    assert _run([True] * (n + 5)) == n


def test_untrusted_keyboard_falls_back_to_enter(monkeypatch, capsys):
    """Without macOS keyboard permission we must not hang on pynput: ask for ENTER and explain how to fix it."""
    from voicebot.input import recorder
    monkeypatch.setattr(recorder, "_keyboard_trusted", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    recorder._wait_for_space()
    assert "Accessibility" in capsys.readouterr().out
