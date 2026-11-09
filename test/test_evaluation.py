"""Master eval: dataset pairing, WER, arg parsing, and a mocked end-to-end run writing config + metrics."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicebot import evaluation as ev


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


def test_default_dataset_pairs():
    pts = ev.find_datapoints(Path(ev.DEFAULT_DATA_DIR))  # searches subfolders (libritts/, voxpopuli/) too
    assert len(pts) >= 27 and all(a.stem == t.stem and a.parent == t.parent and a.is_absolute() for a, t in pts)


def test_bucket():
    assert [ev.bucket(s) for s in (1, 5, 29.9, 30, 120)] == ["short <5s", "medium 5-15s", "long 15-30s", "longform >30s", "longform >30s"]


def test_unpaired_and_missing(tmp_path):
    (tmp_path / "a.wav").write_bytes(b""); (tmp_path / "b.txt").write_text("x")
    with pytest.raises(ValueError, match="Unpaired"):
        ev.find_datapoints(tmp_path)
    with pytest.raises(FileNotFoundError, match="--data_dir"):
        ev.find_datapoints(tmp_path / "nope")


def test_parse_args():
    args, d = ev.parse_args(["--data_dir", "x/", "--mode", "remote"])
    assert d == Path("x/").resolve() and args.mode == "remote"
    assert ev.parse_args([])[1] == Path(ev.DEFAULT_DATA_DIR).resolve()


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
    out = ev.run(["--data_dir", "data"])  # relative, like the default ./test/test_data/
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["commit_sha"] and cfg["datetime"] and len(cfg["datapoints"]) == 2
    m = json.loads((out / "metrics.json").read_text())
    s = m["summary"]
    assert s["n_ok"] == 2 and s["overall"]["wer_corpus"] == pytest.approx(1 / 3, abs=1e-3)
    assert set(s["by_dataset"]) == {"root", "setA"} and set(s["by_bucket"]) == {"short <5s", "longform >30s"}
    assert set(s["overall"]["latency_s"]) == {"vad", "asr", "llm", "tts", "total"} and "p90" in s["overall"]["latency_s"]["asr"]
    row = m["datapoints"][1]
    assert row["audio_s"] == 40 and row["tts_audio_s"] == 1 and row["rtf"]["asr"] == pytest.approx(row["latency_s"]["asr"] / 40, abs=1e-3)
