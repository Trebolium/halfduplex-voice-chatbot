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
