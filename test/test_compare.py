"""voicebot-compare: flag errors, missing runs, config diff and metric deltas on fake runs."""
import json

import pytest

from voicebot import compare


def _group(wer, total):
    lat = {s: {"p50": total / 5, "p90": total / 4, "mean": total / 5} for s in ["vad", "asr", "llm", "tts"]}
    lat["total"] = {"p50": total, "p90": total * 1.5, "mean": total}
    return {"n": 2, "wer_corpus": wer, "rtf": {"asr": {"p50": 0.02}}, "latency_s": lat}


def _run(root, name, mode, wer, total, clips=("a.wav", "b.wav")):
    d = root / name; d.mkdir()
    (d / "config.json").write_text(json.dumps({"datetime": "t", "commit_sha": "abc1234", "dirty": False, "mode": mode,
                                               "setup": {"asr": {"model": f"{mode}-asr"}}, "datapoints": list(clips)}))
    g = _group(wer, total)
    (d / "metrics.json").write_text(json.dumps({"summary": {"n_ok": 2, "n_failed": 0, "overall": g,
                                                            "by_dataset": {"libritts": g}, "by_bucket": {"short <5s": g}}}))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESULTS_DIR", tmp_path)
    _run(tmp_path, "run_local", "local", 0.08, 4.0)
    _run(tmp_path, "run_remote", "remote", 0.04, 7.0)
    return tmp_path


def test_no_flag_is_informative(results):
    with pytest.raises(SystemExit, match="No runs to compare: pass --compare_results.*run_local, run_remote"):
        compare.parse_args([])


def test_needs_two_runs(results):
    with pytest.raises(SystemExit, match="at least two runs"):
        compare.parse_args(["--compare_results", "run_local"])


def test_missing_run(results):
    with pytest.raises(SystemExit, match="'nope' not found.*Available: run_local, run_remote"):
        compare.compare(["run_local", "nope"])


def test_report(results):
    out = compare.compare(["run_local", "evaluate/results/run_remote"])  # path form also accepted
    assert "mode:" in out and "setup.asr.model:" in out and "WARNING" not in out
    assert "-4.0pt" in out and "+3.000" in out  # WER delta in points, total p50 delta in seconds
    assert "[libritts]" in out and "[short <5s]" in out


def test_different_clips_warns(results):
    _run(results, "run_other", "local", 0.1, 4.0, clips=("x.wav",))
    assert "WARNING: the runs used different clips" in compare.compare(["run_local", "run_other"])


def test_legacy_format_rejected(results):
    d = results / "old"; d.mkdir()
    (d / "config.json").write_text("{}"); (d / "metrics.json").write_text(json.dumps({"summary": {"n_ok": 1, "wer_corpus": 0.1}}))
    with pytest.raises(SystemExit, match="older metrics.json format"):
        compare.compare(["run_local", "old"])


def test_cli_writes_report_and_charts(results, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "OUT_DIR", tmp_path / "out")
    for name in ("run_local", "run_remote"):  # per-clip rows for the ASR scaling chart
        mp = results / name / "metrics.json"; m = json.loads(mp.read_text())
        m["datapoints"] = [{"audio_s": 2, "latency_s": {"asr": 0.2}}, {"audio_s": 20, "latency_s": {"asr": 0.5}}]
        mp.write_text(json.dumps(m))
    monkeypatch.setattr("sys.argv", ["voicebot-compare", "--compare_results", "run_local", "run_remote"])
    compare.cli()
    out = tmp_path / "out" / "run_local__vs__run_remote"
    assert (out / "report.txt").read_text().startswith("Runs")
    assert sorted(p.name for p in out.glob("*.png")) == ["1_latency_breakdown.png", "2_wer_by_group.png",
                                                        "3_latency_by_group.png", "4_asr_scaling.png"]
