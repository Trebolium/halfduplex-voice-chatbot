"""Master eval: dataset pairing, WER, arg parsing, and a mocked end-to-end run writing config + metrics."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicebot import config, evaluation as ev


def wav(path, secs):
    sf.write(str(path), np.zeros(int(16000 * secs), np.float32), 16000)


def test_wer():
    assert ev.word_errors("Hello, world!", "hello world") == (0, 2)
    assert ev.word_errors("the cat sat", "the bat sat down") == (2, 3)  # 1 sub + 1 insertion
    assert ev.word_errors("a b c", "") == (3, 3)
    # Whisper normalizer: contractions, spelled numbers and UK/US spelling are not errors
    assert ev.word_errors("i think it is very important", "I think it's very important.") == (0, 6)
    assert ev.word_errors("the reports for two thousand and six", "The reports for 2006") == (0, 4)
    assert ev.word_errors("organisations of colour", "organizations of color") == (0, 3)


def test_env_dataset_pairs():
    if not config.TEST_AUDIO_PATH or not Path(config.TEST_AUDIO_PATH).is_dir():
        pytest.skip("TEST_AUDIO_PATH in .env is not set or missing")
    pts = ev.find_datapoints(Path(config.TEST_AUDIO_PATH))  # searches subfolders too
    assert pts and all(a.stem == t.stem and a.parent == t.parent and a.is_absolute() for a, t in pts)


def test_bucket():
    assert [ev.bucket(s) for s in (1, 5, 29.9, 30, 120)] == ["short <5s", "medium 5-15s", "long 15-30s", "longform >30s", "longform >30s"]


def test_unpaired_and_missing(tmp_path):
    (tmp_path / "a.wav").write_bytes(b""); (tmp_path / "b.txt").write_text("x")
    with pytest.raises(ValueError, match="Unpaired"):
        ev.find_datapoints(tmp_path)
    with pytest.raises(FileNotFoundError, match="Test data not found.*TEST_AUDIO_PATH"):
        ev.find_datapoints(tmp_path / "nope")


def test_parse_args(monkeypatch):
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", "from_env/")
    args, d = ev.parse_args(["--mode", "remote"])
    assert d == Path("from_env/").resolve() and args.mode == "remote"
    with pytest.raises(SystemExit):
        ev.parse_args(["--data_dir", "x/"])  # removed: the dataset comes only from .env
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", "")
    with pytest.raises(FileNotFoundError, match="Test data not found: TEST_AUDIO_PATH is not set"):
        ev.parse_args([])


def test_mocked_run(tmp_path, monkeypatch):
    data = tmp_path / "data"; (data / "setA").mkdir(parents=True)
    for d, n, secs in ((data, "one", 2), (data / "setA", "two", 40)):
        wav(d / f"{n}.wav", secs); (d / f"{n}.txt").write_text("hello there friend")
    wav(tmp_path / "out.wav", 1)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ev.main, "configure", lambda a: {"asr": "a", "llm": "l", "tts": "t"})
    for mod in (ev.asr, ev.llm, ev.tts):
        monkeypatch.setattr(mod, "select_backend", lambda n: None)
        monkeypatch.setattr(mod, "current", lambda: type("M", (), {"INFO": {}}))
    monkeypatch.setattr(ev.vad, "reset", lambda: None)
    monkeypatch.setattr(ev.vad, "trim_speech", lambda p: p)
    monkeypatch.setattr(ev.asr, "transcribe", lambda p: "hello their friend")
    monkeypatch.setattr(ev.llm, "reply", lambda s, h, t: "hi!")
    monkeypatch.setattr(ev.tts, "synthesize", lambda t: tmp_path / "out.wav")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", "data")  # relative path, like ./test/test_data/
    out = ev.run([])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["commit_sha"] and cfg["datetime"] and len(cfg["datapoints"]) == 2
    m = json.loads((out / "metrics.json").read_text())
    s = m["summary"]
    assert s["n_ok"] == 2 and s["overall"]["wer_corpus"] == pytest.approx(1 / 3, abs=1e-3)
    assert set(s["by_dataset"]) == {"root", "setA"} and set(s["by_bucket"]) == {"short <5s", "longform >30s"}
    assert set(s["overall"]["latency_s"]) == {"vad", "asr", "llm", "tts", "total"} and "p90" in s["overall"]["latency_s"]["asr"]
    row = m["datapoints"][1]
    assert row["audio_s"] == 40 and row["tts_audio_s"] == 1 and row["rtf"]["asr"] == pytest.approx(row["latency_s"]["asr"] / 40, abs=1e-3)


def test_clips_near(tmp_path, monkeypatch):
    for n, secs in (("a", 2), ("b", 11), ("c", 40)):
        wav(tmp_path / f"{n}.wav", secs); (tmp_path / f"{n}.txt").write_text("x")
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", str(tmp_path))
    got = ev.clips_near({"short": 3, "long": 12})
    assert got["short"][0].stem == "a" and got["long"][0].stem == "b" and got["long"][1].suffix == ".txt"
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", "")
    with pytest.raises(FileNotFoundError, match="Test data not found"):
        ev.clips_near({"short": 3})


def test_mocked_streaming_run(tmp_path, monkeypatch):
    """In streaming mode the eval takes text/reply/waits from stream_clip and records the stream stats."""
    data = tmp_path / "data"; data.mkdir()
    wav(data / "one.wav", 3); (data / "one.txt").write_text("hello there friend")
    wav(tmp_path / "out.wav", 1)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", str(data))
    monkeypatch.setattr(config, "STREAMING", True)
    monkeypatch.setattr(ev.main, "configure", lambda a: {"asr": "a", "llm": "l", "tts": "t"})
    for mod in (ev.asr, ev.llm, ev.tts):
        monkeypatch.setattr(mod, "select_backend", lambda n: None)
        monkeypatch.setattr(mod, "current", lambda: type("M", (), {"INFO": {}}))
    monkeypatch.setattr(ev.vad, "reset", lambda: None)
    monkeypatch.setattr(ev, "stream_clip", lambda a, sp: {"text": "hello there friend", "reply": "hi", "asr_wait_s": 0.0,
                                                          "llm_wait_s": 0.4, "path": "speculative", "partials": 3,
                                                          "partial_failures": 0, "retractions": 0, "speculations": 1, "stale_speculations": 0})
    monkeypatch.setattr(ev.tts, "synthesize", lambda t: tmp_path / "out.wav")
    out = ev.run([])
    row = json.loads((out / "metrics.json").read_text())["datapoints"][0]
    assert row["wer"] == 0 and row["latency_s"]["llm"] == 0.4 and row["latency_s"]["vad"] == 0 and row["stream"]["path"] == "speculative"
    assert "streamed" in json.loads((out / "config.json").read_text())["asr_input"]


def test_interrupted_run_creates_no_results_folder(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    wav(data / "one.wav", 2); (data / "one.txt").write_text("hello there friend")
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ev.main, "configure", lambda a: {"asr": "a", "llm": "l", "tts": "t"})
    for mod in (ev.asr, ev.llm, ev.tts):
        monkeypatch.setattr(mod, "select_backend", lambda n: None)
        monkeypatch.setattr(mod, "current", lambda: type("M", (), {"INFO": {}}))
    monkeypatch.setattr(ev.vad, "reset", lambda: None)
    monkeypatch.setattr(ev.vad, "trim_speech", lambda p: p)
    monkeypatch.setattr(ev.asr, "transcribe", lambda p: "hello")
    monkeypatch.setattr(ev.llm, "reply", lambda s, h, t: "hi!")
    def boom(t): raise KeyboardInterrupt  # Ctrl-C mid-run
    monkeypatch.setattr(ev.tts, "synthesize", boom)
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", str(data))
    with pytest.raises(KeyboardInterrupt):
        ev.run([])
    assert not (tmp_path / "results").exists()


def test_note_in_config_and_folder_name(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    wav(data / "one.wav", 2); (data / "one.txt").write_text("hello there friend")
    wav(tmp_path / "out.wav", 1)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ev.main, "configure", lambda a: {"asr": "a", "llm": "l", "tts": "t"})
    for mod in (ev.asr, ev.llm, ev.tts):
        monkeypatch.setattr(mod, "select_backend", lambda n: None)
        monkeypatch.setattr(mod, "current", lambda: type("M", (), {"INFO": {}}))
    monkeypatch.setattr(ev.vad, "reset", lambda: None)
    monkeypatch.setattr(ev.vad, "trim_speech", lambda p: p)
    monkeypatch.setattr(ev.asr, "transcribe", lambda p: "hello")
    monkeypatch.setattr(ev.llm, "reply", lambda s, h, t: "hi!")
    monkeypatch.setattr(ev.tts, "synthesize", lambda t: tmp_path / "out.wav")
    monkeypatch.setattr(config, "TEST_AUDIO_PATH", str(data))
    out = ev.run(["--note", "Streaming: reload fix!"])
    assert out.name.endswith("_streaming-reload-fix")
    assert json.loads((out / "config.json").read_text())["note"] == "Streaming: reload fix!"
