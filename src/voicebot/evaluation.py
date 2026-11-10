"""Master eval: runs every datapoint (audio + matching .txt) through VAD -> ASR -> LLM -> TTS, recording per-stage latency and ASR WER.

Run: uv run voicebot-eval [--note "text"] [--mode local|remote] [any voicebot model flag]; clips come from TEST_AUDIO_PATH in .env
Writes evaluate/results/<datetime>_<sha>[_<note>]/config.json + metrics.json only once the whole run has finished (nothing is created if it is interrupted).
Summaries: corpus WER, real-time factor (stage time / audio length) and p50/p90 latency, overall + per dataset + per duration bucket.
"""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from whisper_normalizer.english import EnglishTextNormalizer

from voicebot import asr, config, llm, main, setup_info, tts
from voicebot.asr import vad
from voicebot.memory import store
from voicebot.asr.streaming import StreamingSession, load_16k
from voicebot.reply_stream import join_audio
from voicebot.input.recorder import FRAME_SAMPLES, endpoint_state

RESULTS_DIR = config.ROOT / "evaluate" / "results"
AUDIO_EXTS = {".wav", ".flac", ".ogg"}  # anything soundfile can read
STAGES = ["vad", "asr", "llm", "tts"]
BUCKETS = [(0, 5, "short <5s"), (5, 15, "medium 5-15s"), (15, 30, "long 15-30s"), (30, float("inf"), "longform >30s")]


def bucket(secs: float) -> str:
    """Duration bucket label for an input clip (30s = Whisper's window, so >30s is its own regime)."""
    return next(name for lo, hi, name in BUCKETS if lo <= secs < hi)


def find_datapoints(data_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every audio file (searched recursively) with its same-named .txt in the same folder; error if either is missing."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Test data not found: folder '{data_dir}' does not exist. Fix TEST_AUDIO_PATH in .env so it points at a folder of <name>.wav + <name>.txt pairs.")
    key = lambda p: str(p.relative_to(data_dir).with_suffix(""))
    audio = {key(p): p for p in sorted(data_dir.rglob("*")) if p.suffix.lower() in AUDIO_EXTS}
    texts = {key(p): p for p in sorted(data_dir.rglob("*.txt"))}
    no_text, no_audio = sorted(audio.keys() - texts.keys()), sorted(texts.keys() - audio.keys())
    if no_text or no_audio:
        raise ValueError(f"Unpaired files in {data_dir}: audio without .txt {no_text}, .txt without audio {no_audio}. "
                         "Every datapoint needs <name>.wav and <name>.txt.")
    if not audio:
        raise ValueError(f"No audio files ({', '.join(sorted(AUDIO_EXTS))}) in {data_dir}.")
    return [(audio[k].resolve(), texts[k].resolve()) for k in sorted(audio)]


_normalizer = EnglishTextNormalizer()  # OpenAI Whisper's standard English normalizer (Open ASR Leaderboard convention)


def clips_near(targets: dict[str, float]) -> dict[str, tuple[Path, Path]]:
    """For older benchmark scripts: {label: (wav, txt)} of the TEST_AUDIO_PATH clip closest to each target length in seconds."""
    if not config.TEST_AUDIO_PATH:
        raise FileNotFoundError("Test data not found: TEST_AUDIO_PATH is not set in .env. Add TEST_AUDIO_PATH=<folder of .wav + .txt pairs>.")
    points = find_datapoints(Path(config.TEST_AUDIO_PATH).resolve())
    dur = {a: sf.info(str(a)).duration for a, _ in points}
    out = {label: min(points, key=lambda p: abs(dur[p[0]] - secs)) for label, secs in targets.items()}
    for label, (a, _) in out.items():
        print(f"[eval] {label}: {a.name} ({dur[a]:.1f}s)")
    return out


def normalize(text: str) -> list[str]:
    """Whisper EnglishTextNormalizer (case, punctuation, contractions, numbers, US/UK spelling), split into words."""
    return _normalizer(text).split()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, reference word count) - WER = errors / ref_words."""
    r, h = normalize(ref), normalize(hyp)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1], len(r)


def git_info() -> dict:
    """Latest commit SHA and whether the working tree had uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit_sha": git("rev-parse", "HEAD") or "unknown", "dirty": bool(git("status", "--porcelain"))}
    except FileNotFoundError:
        print("[eval] WARNING: git not found - commit SHA recorded as 'unknown'.")
        return {"commit_sha": "unknown", "dirty": None}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Path]:
    """Same flags as voicebot (mode + model flags) plus --note; the dataset folder comes only from TEST_AUDIO_PATH in .env."""
    if not config.TEST_AUDIO_PATH:
        raise FileNotFoundError("Test data not found: TEST_AUDIO_PATH is not set in .env. Add "
                                "TEST_AUDIO_PATH=<folder of <name>.wav + <name>.txt pairs> to .env (evaluate/sample_dataset.py can build one).")
    note_p = argparse.ArgumentParser(add_help=False)
    note_p.add_argument("--note", default="", help="free-text label stored in config.json and appended to the results folder name")
    note_args, rest = note_p.parse_known_args(argv)
    args = main.parse_args(rest)
    args.note = note_args.note.strip()
    return args, Path(config.TEST_AUDIO_PATH).resolve()  # absolute, so clips map to their dataset subfolder


def run_datapoint(audio: Path, ref_path: Path, system_prompt: str, data_dir: Path) -> dict:
    """One clip through the full pipeline; per-stage latency and RTF, transcript, WER, reply."""
    ref = ref_path.read_text().strip()
    if not ref:
        raise ValueError(f"{ref_path} is empty - it must hold the reference transcript for {audio.name}.")
    secs = sf.info(str(audio)).duration
    rel = audio.relative_to(data_dir) if audio.is_relative_to(data_dir) else audio
    t, row = {}, {"audio": str(audio), "reference_txt": str(ref_path), "reference": ref,
                  "dataset": rel.parts[0] if len(rel.parts) > 1 else "root", "audio_s": round(secs, 3), "bucket": bucket(secs)}
    print(f"[eval] {row['dataset']} / {row['bucket']} ({secs:.1f}s audio)")
    if config.STREAMING:
        res = stream_clip(audio, system_prompt)
        hyp = res["text"]
        t.update(vad=0.0, asr=res["asr_wait_s"], llm=res["llm_wait_s"], tts=res["tts_wait_s"])
        results = [r for _, r in res["stream"].audio()]  # drain: the rest of the reply finishes generating + synthesizing (not timed)
        if not results:
            raise RuntimeError(f"The LLM returned an empty reply for {audio.name}. Try re-running, or another LLM model.")
        answer, out = res["stream"].reply, join_audio([r["path"] for r in results])
        synth_s = sum(r["secs"] for r in results)
        row["stream"] = {k: res[k] for k in ("path", "partials", "partial_failures", "retractions", "speculations", "stale_speculations")}
    else:
        t0 = time.time(); trimmed = vad.trim_speech(audio); t["vad"] = time.time() - t0
        if trimmed is None:
            raise RuntimeError(f"VAD found no speech in {audio.name}. Check the clip has audible speech.")
        t0 = time.time(); hyp = asr.transcribe(trimmed); t["asr"] = time.time() - t0
    errors, n = word_errors(ref, hyp)
    row.update(transcript=hyp, word_errors=errors, ref_words=n, wer=errors / n if n else 0.0)
    print(f"[eval] WER {row['wer']:.3f} ({errors}/{n} words)")
    if not config.STREAMING:
        t0 = time.time(); answer = llm.reply(system_prompt, [], hyp); t["llm"] = time.time() - t0
        t0 = time.time(); out = tts.synthesize(answer); t["tts"] = time.time() - t0; synth_s = t["tts"]
    t["total"] = sum(t[s] for s in STAGES)
    tts_secs = sf.info(str(out)).duration  # soundfile reads wav and mp3
    rtf = {"vad": t["vad"] / secs, "asr": t["asr"] / secs, "tts": synth_s / tts_secs if tts_secs else None}
    row.update(reply=answer, reply_chars=len(answer), tts_audio=str(out), tts_audio_s=round(tts_secs, 3),
               latency_s={k: round(v, 3) for k, v in t.items()}, rtf={k: v if v is None else round(v, 4) for k, v in rtf.items()})
    print("[eval] latency: " + "  ".join(f"{k}={v:.2f}s" for k, v in t.items()) + f"  | ASR RTF {rtf['asr']:.3f}")
    return row


def stream_clip(audio: Path, system_prompt: str) -> dict:
    """Play a clip in real time (32ms frames, Silero per frame, as the mic would) into a StreamingSession, then pad with
    silence until the normal endpoint fires after the clip's end (same speech as batch mode); returns finish()."""
    pcm = load_16k(audio)
    content_frames = len(pcm) // FRAME_SAMPLES  # the endpoint may only fire after the clip ends, so mid-clip reading
    pad = np.zeros(int(config.SAMPLE_RATE * (config.SILENCE_END_SECS + 0.5)), np.int16)  # pauses >= 1s don't cut it short
    pcm = np.concatenate([pcm, pad])
    session = StreamingSession(system_prompt, [], synthesize=tts.synthesize)
    vad.reset()
    started, silence, rec, t_start = False, 0, 0, time.time()
    for i in range(len(pcm) // FRAME_SAMPLES):
        time.sleep(max(0.0, t_start + i * FRAME_SAMPLES / config.SAMPLE_RATE - time.time()))  # real-time pacing
        f = pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
        sp = vad.is_speech(f)
        if not started and not sp:
            continue
        started = True
        session.feed(f, sp)
        rec += 1
        silence, done = endpoint_state(sp, silence, rec)
        if done and i >= content_frames:
            break
    if not started:
        session.finish()
        raise RuntimeError(f"VAD found no speech in {audio.name}. Check the clip has audible speech.")
    return session.finish()


def _pct(values: list[float]) -> dict:
    """p50/p90/mean - percentiles over min/max because tail latency is what users notice."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {}
    return {"p50": round(float(np.percentile(v, 50)), 4), "p90": round(float(np.percentile(v, 90)), 4), "mean": round(float(v.mean()), 4)}


def _group(ok: list[dict]) -> dict:
    """Metrics for one set of clips: corpus WER (headline), mean per-clip WER, latency + RTF percentiles."""
    errs, words = sum(r["word_errors"] for r in ok), sum(r["ref_words"] for r in ok)
    return {"n": len(ok), "audio_s_total": round(sum(r["audio_s"] for r in ok), 1),
            "wer_corpus": round(errs / words, 4) if words else 0.0, "wer_mean_per_clip": round(float(np.mean([r["wer"] for r in ok])), 4),
            "latency_s": {s: _pct([r["latency_s"][s] for r in ok]) for s in STAGES + ["total"]},
            "rtf": {s: _pct([r["rtf"][s] for r in ok]) for s in ["vad", "asr", "tts"]}}


def summarize(rows: list[dict]) -> dict:
    """Overall, per-dataset and per-duration-bucket summaries of the successful datapoints."""
    ok = [r for r in rows if "error" not in r]
    out = {"n_ok": len(ok), "n_failed": len(rows) - len(ok)}
    if ok:
        out["overall"] = _group(ok)
        for key in ("dataset", "bucket"):
            out[f"by_{key}"] = {k: _group([r for r in ok if r[key] == k]) for k in sorted({r[key] for r in ok})}
    return out


def run(argv: list[str] | None = None) -> Path:
    """Select backends, write config.json, warm up, run every datapoint, write metrics.json. Returns the run folder."""
    started = datetime.now().astimezone()
    args, data_dir = parse_args(argv)
    points = find_datapoints(data_dir)
    print(f"[eval] {len(points)} datapoints from {data_dir.resolve()}")
    names = main.configure(args)
    print(f"[eval] mode={args.mode}: asr={names['asr']} llm={names['llm']} tts={names['tts']}")
    asr.select_backend(names["asr"]); llm.select_backend(names["llm"]); tts.select_backend(names["tts"]); vad.reset()

    git = git_info()
    slug = re.sub(r"[^a-z0-9]+", "-", args.note.lower()).strip("-")[:40]  # note -> short folder-safe suffix
    out_dir = RESULTS_DIR / f"{started:%Y%m%d_%H%M%S}_{git['commit_sha'][:7]}{'_' + slug if slug else ''}"
    system_prompt = store.build_system_prompt({"name": "friend", "sessions": 0, "facts": []})  # fresh user, as in --test runs
    cfg = {"datetime": started.isoformat(timespec="seconds"), **git, "mode": args.mode, "note": args.note, "backends": names,
           "data_dir": str(data_dir.resolve()), "datapoints": [str(a) for a, _ in points],
           "setup": setup_info.build(asr.current(), llm.current(), tts.current(), vad),
           "tts_voice": config.TTS_VOICE or "backend-default", "eval_system_prompt": system_prompt,
           "wer_normalization": "whisper EnglishTextNormalizer (whisper-normalizer package), applied to reference and transcript",
           "duration_buckets": [name for _, _, name in BUCKETS], "rtf": "stage seconds / audio seconds (input clip for vad/asr, generated reply audio for tts)",
           "asr_input": "streamed in real time; partial ASR on the audio so far, no VAD trim" if config.STREAMING else "VAD-trimmed audio (same as the live pipeline)",
           "latency_meaning": "seconds from the endpoint to the FIRST audio: asr = wait for the transcript, llm = until the first sentence exists, tts = until its audio is ready (vad=0); the rest of the reply is generated/synthesized sentence by sentence while it plays" if config.STREAMING else "processing time per stage after the endpoint", "warmup": "one untimed VAD+ASR+LLM+TTS pass on the first datapoint"}
    print(f"[eval] results will be written to {out_dir} when the run finishes (commit {git['commit_sha'][:7]}{', dirty tree' if git['dirty'] else ''})")

    print("[eval] warm-up pass (untimed) on the first datapoint...")
    try:
        run_datapoint(*points[0], system_prompt, data_dir)
    except Exception as e:
        print(f"[eval] WARNING: warm-up failed ({e}); continuing, but the first timed clip may be slow.")

    rows = []
    for i, (audio, ref) in enumerate(points, 1):
        print(f"\n[eval] >>> datapoint {i}/{len(points)}: {audio.name}")
        try:
            rows.append(run_datapoint(audio, ref, system_prompt, data_dir))
        except Exception as e:
            print(f"[eval] ERROR on {audio.name}: {e}")
            rows.append({"audio": str(audio), "reference_txt": str(ref), "error": str(e)})

    summary = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "datapoints": rows}, indent=2))
    print(f"\n[eval] done in {(datetime.now().astimezone() - started).total_seconds():.1f}s -> {out_dir}")
    if summary["n_ok"]:
        print_summary(summary)
    if summary["n_failed"]:
        print(f"[eval] {summary['n_failed']} datapoint(s) failed - see 'error' entries in metrics.json.")
    return out_dir


def print_summary(summary: dict) -> None:
    """Console table: one row per group with WER, ASR RTF and p50/p90 of each stage."""
    print(f"\n[eval] {'group':<24}{'n':>3} {'WER':>7} {'ASR RTF':>8}  " + "  ".join(f"{s + ' p50/p90':>14}" for s in STAGES + ["total"]))
    groups = [("overall", summary["overall"])] + [(k, v) for key in ("by_dataset", "by_bucket") for k, v in summary[key].items()]
    for name, g in groups:
        lat = "  ".join(f"{g['latency_s'][s]['p50']:>6.2f}/{g['latency_s'][s]['p90']:<7.2f}" for s in STAGES + ["total"])
        print(f"[eval] {name:<24}{g['n']:>3} {g['wer_corpus']:>7.3f} {g['rtf']['asr']['p50']:>8.3f}  {lat}")


def cli() -> None:
    run()


if __name__ == "__main__":
    cli()
