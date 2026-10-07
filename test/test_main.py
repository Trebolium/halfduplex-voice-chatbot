"""Loop test with every stage mocked: one turn runs end to end, silence ends the session."""
from pathlib import Path

from voicebot import main


def test_run_turn_and_end(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "record", lambda wait_for_space: Path("a.wav") if not calls else None)
    monkeypatch.setattr(main.tts, "current", lambda: main.tts.edge_tts)  # don't load a real TTS model in a unit test
    monkeypatch.setattr(main, "trim_speech", lambda p: p)
    monkeypatch.setattr(main, "transcribe", lambda p: "hello")
    monkeypatch.setattr(main, "reply", lambda s, h, t: "hi there")
    monkeypatch.setattr(main, "synthesize", lambda t: Path("r.mp3"))
    monkeypatch.setattr(main, "play", lambda p: calls.append(p))
    history = []
    user = {"name": "T"}
    monkeypatch.setattr(main.store, "log_turn", lambda u, t, a, lat=None, setup=None: u.setdefault("c", []).append((t, a, sorted(lat), sorted(setup))))
    monkeypatch.setattr(main.store, "add_latency", lambda u, stage, secs: u.__setitem__(stage + "_lat", secs))
    assert main.run_turn("sys", history, first=True, user=user) is True
    assert user["c"] == [("hello", "hi there", ["asr", "llm", "vad"], ["asr", "config", "llm", "tts", "vad"])]
    assert user["tts_lat"] > 0
    assert history[-1] == {"role": "assistant", "content": "hi there"}
    assert main.run_turn("sys", history, first=False) is False


def test_no_speech_skips_asr(monkeypatch):
    """If VAD finds nothing, ASR is never called (avoids hallucinated text) and the loop keeps listening."""
    monkeypatch.setattr(main, "record", lambda wait_for_space: Path("a.wav"))
    monkeypatch.setattr(main, "trim_speech", lambda p: None)
    monkeypatch.setattr(main, "transcribe", lambda p: (_ for _ in ()).throw(AssertionError("ASR should not run")))
    t = {}
    assert main.run_turn("sys", [], first=True, timings=t) is True
    assert t["retry"] is True  # main loop keeps asking for SPACE after a failed first take


# --- --mode local|remote ---
import pytest

from voicebot import config


@pytest.fixture(autouse=True)
def restore_config():
    """configure() mutates config for remote overrides; undo it so other tests see the real .env values."""
    keys = ["ASR_MODEL", "LLM_MODEL", "LLM_PROVIDER", "TTS_VOICE", "GROQ_API_KEY", "OPENROUTER_API_KEY", "ELEVENLABS_API_KEY"]
    saved = {k: getattr(config, k) for k in keys}
    yield
    for k, v in saved.items():
        setattr(config, k, v)


def test_default_mode_is_local(monkeypatch):
    monkeypatch.delenv("MODE", raising=False)
    args = main.parse_args([])
    assert args.mode == "local"
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(main.platform, "machine", lambda: "arm64")
    assert main.configure(args) == {"asr": "local-whisper", "llm": "local-llm", "tts": "piper"}


def test_local_mode_ignores_remote_flags(monkeypatch, capsys):
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(main.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config, "LLM_MODEL", "orig")
    names = main.configure(main.parse_args(["--llm_model", "x/y", "--tts_voice", "v"]))
    assert names["llm"] == "local-llm" and config.LLM_MODEL == "orig"
    assert "ignored in local mode" in capsys.readouterr().out


def test_local_mode_needs_apple_silicon(monkeypatch):
    monkeypatch.setattr(main.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="--mode remote"):
        main.configure(main.parse_args([]))


def test_remote_mode_applies_overrides(monkeypatch):
    for k, v in {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "ASR_MODEL": "a", "LLM_MODEL": "m", "LLM_PROVIDER": "", "TTS_VOICE": "v"}.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.delenv("REMOTE_TTS", raising=False)
    args = main.parse_args(["--mode", "remote", "--asr_model", "whisper-large-v3", "--llm_model", "google/gemini-3.8-flash",
                            "--llm_provider", "google-vertex", "--tts_voice", "en-GB-SoniaNeural"])
    assert main.configure(args) == {"asr": "remote-whisper", "llm": "remote-llm", "tts": main.tts.REMOTE_DEFAULT}
    assert (config.ASR_MODEL, config.LLM_MODEL, config.LLM_PROVIDER, config.TTS_VOICE) == (
        "whisper-large-v3", "google/gemini-3.8-flash", "google-vertex", "en-GB-SoniaNeural")


def test_remote_mode_without_defaults_keeps_env(monkeypatch):
    for k, v in {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "LLM_MODEL": "from-env"}.items():
        monkeypatch.setattr(config, k, v)
    main.configure(main.parse_args(["--mode", "remote"]))
    assert config.LLM_MODEL == "from-env"


@pytest.mark.parametrize("missing", ["GROQ_API_KEY", "OPENROUTER_API_KEY"])
def test_remote_mode_missing_key_is_clear(monkeypatch, missing):
    monkeypatch.setattr(config, "GROQ_API_KEY", "g"); monkeypatch.setattr(config, "OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(config, missing, "")
    with pytest.raises(RuntimeError, match=missing):
        main.configure(main.parse_args(["--mode", "remote"]))


def test_remote_new_model_drops_env_provider_pin(monkeypatch, capsys):
    for k, v in {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "LLM_PROVIDER": "deepinfra/turbo"}.items():
        monkeypatch.setattr(config, k, v)
    main.configure(main.parse_args(["--mode", "remote", "--llm_model", "google/gemini-2.5-flash"]))
    assert config.LLM_PROVIDER == "" and "dropping the LLM_PROVIDER pin" in capsys.readouterr().out
    main.configure(main.parse_args(["--mode", "remote", "--llm_model", "m", "--llm_provider", "together"]))
    assert config.LLM_PROVIDER == "together"


def _mac(monkeypatch):
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(main.platform, "machine", lambda: "arm64")


def test_tts_model_works_in_local_mode(monkeypatch):
    _mac(monkeypatch)
    assert main.configure(main.parse_args(["--tts_model", "kokoro"]))["tts"] == "kokoro"


def test_tts_model_works_in_remote_mode(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setattr(config, k, "x")
    assert main.configure(main.parse_args(["--mode", "remote", "--tts_model", "edge-tts"]))["tts"] == "edge-tts"


def test_tts_model_wrong_mode_is_clear(monkeypatch):
    _mac(monkeypatch)
    with pytest.raises(ValueError, match="--mode remote"):
        main.configure(main.parse_args(["--tts_model", "gtts"]))
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setattr(config, k, "x")
    with pytest.raises(ValueError, match="--mode local"):
        main.configure(main.parse_args(["--mode", "remote", "--tts_model", "piper"]))


def test_elevenlabs_needs_key(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setattr(config, k, "x")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        main.configure(main.parse_args(["--mode", "remote", "--tts_model", "elevenlabs"]))
