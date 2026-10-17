import json

import pytest

from voicebot import config
from voicebot.llm import openrouter
from voicebot.memory import store


@pytest.fixture(autouse=True)
def tmp_users(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USERS_DIR", tmp_path)


def test_new_user_prompt_is_first_meeting():
    u = store.load("Mary Jane")
    assert u["sessions"] == 0 and u["facts"] == []
    p = store.build_system_prompt(u)
    assert config.BASE_SYSTEM_PROMPT in p and "first meeting" in p


def test_end_session_saves_and_returning_user(monkeypatch):
    reply = '```json\n{"new_facts": ["Has a cat named Tom"], "summary": "Talked about pets."}\n```'
    monkeypatch.setattr(openrouter, "complete", lambda s, t: reply)
    u = store.load("Bob")
    store.end_session(u, [{"role": "user", "content": "I have a cat"}])
    u2 = store.load("Bob")
    assert u2["sessions"] == 1 and u2["facts"] == ["Has a cat named Tom"] and u2["last_seen"]
    p = store.build_system_prompt(u2)
    assert "Has a cat named Tom" in p and "Talked about pets." in p and "first meeting" not in p


def test_facts_deduped(monkeypatch):
    monkeypatch.setattr(openrouter, "complete", lambda s, t: json.dumps({"new_facts": ["likes tea", "Likes Tea"], "summary": "x"}))
    u = store.load("Ann")
    store.end_session(u, [{"role": "user", "content": "hi"}])
    assert u["facts"] == ["likes tea"]


def test_bad_json_keeps_old_memory(monkeypatch):
    monkeypatch.setattr(openrouter, "complete", lambda s, t: "not json at all")
    u = store.load("Sam")
    u["facts"], u["summary"] = ["old fact"], "old summary"
    store.end_session(u, [{"role": "user", "content": "hi"}])
    assert u["facts"] == ["old fact"] and u["summary"] == "old summary"


def test_empty_history_noop():
    u = store.load("Zed")
    store.end_session(u, [])
    assert u["sessions"] == 0


def test_log_turn_persists():
    u = store.load("Turn")
    store.log_turn(u, "hi", "hello!", {"asr": 0.12345, "llm": 1.5}, {"llm": {"model": "m"}})
    store.add_latency(u, "tts", 0.4)
    turn = store.load("Turn")["cascading conversation"][0]
    assert turn == {"user": "hi", "llm": "hello!", "latency_s": {"asr": 0.123, "llm": 1.5, "tts": 0.4}, "setup": {"llm": {"model": "m"}}}


def test_user_files_live_in_own_folder_and_archive_turn(tmp_path):
    u = store.load("Mary Jane")
    assert (tmp_path / "mary_jane" / "mary_jane.json").exists()
    src = tmp_path / "src"; src.mkdir()
    for n in ("input_1.wav", "input_1_trimmed.wav", "reply_1.mp3"):
        (src / n).write_bytes(b"x")
    store.archive_turn(u, "hello", "hi there", src / "input_1.wav", src / "input_1_trimmed.wav", src / "reply_1.mp3")
    base = tmp_path / "mary_jane"
    assert sorted(p.name for p in (base / "audio").iterdir()) == ["input_1.wav", "input_1_trimmed.wav", "reply_1.mp3"]
    assert (base / "text" / "input_1.txt").read_text() == "hello" and (base / "text" / "reply_1.txt").read_text() == "hi there"
    assert not list(src.iterdir())
