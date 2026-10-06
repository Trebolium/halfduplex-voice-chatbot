"""Benchmark remote/free TTS backends on test/test_data/*.txt; picks the lowest-latency one that works. Run: uv run python evaluate/tts_remote_latency.py"""
import importlib
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

BACKENDS = {"edge-tts": "voicebot.tts.edge_tts", "gtts": "voicebot.tts.google_tts",
            "groq-orpheus": "voicebot.tts.groq_orpheus", "elevenlabs": "voicebot.tts.elevenlabs"}
TEXTS = {"short": Path("test/test_data/84_121123_000008_000002.txt"), "long": Path("test/test_data/002.txt")}
RUNS = 10
PACE_S = {"groq-orpheus": 9.0}  # free tier allows 10 requests/min: wait between calls (outside the timer) so rate limits do not skew or break the run
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)

texts = {k: p.read_text().strip() for k, p in TEXTS.items()}
results, skipped = {}, {}
for name, mod_path in BACKENDS.items():
    mod = importlib.import_module(mod_path)
    print(f"\n=== {name} ===")
    pace = PACE_S.get(name, 0)
    try:
        if hasattr(mod, "load"):
            mod.load()
        mod.synthesize("Warm up.")  # connection setup, not timed
        for label, text in texts.items():
            runs = []
            for i in range(RUNS):
                time.sleep(pace)
                t0 = time.perf_counter()
                path = mod.synthesize(text)  # returns once the audio file is written
                runs.append(time.perf_counter() - t0)
                print(f"[bench] {name} | {label} | run {i + 1}/{RUNS}: {runs[-1]:.3f}s")
            results.setdefault(name, {})[label] = {"runs": runs, "audio_secs": sf.info(str(path)).duration}
    except Exception as e:
        results.pop(name, None)
        skipped[name] = str(e)
        print(f"[bench] SKIPPED {name}: {e}")

winner = min(results, key=lambda n: statistics.mean(statistics.mean(results[n][l]["runs"]) for l in texts)) if results else None
(OUT / "tts_remote_latency.json").write_text(json.dumps({"texts": texts, "results": results, "skipped": skipped, "winner": winner}, indent=2))

names = list(results)
if names:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, label in zip(axes, texts):
        means = [statistics.mean(results[n][label]["runs"]) for n in names]
        ax.bar(names, means, color="#4C78A8", alpha=0.85)
        for i, n in enumerate(names):
            ax.scatter([i] * RUNS, results[n][label]["runs"], color="k", s=10, zorder=3)
            ax.text(i, means[i], f"{means[i]:.2f}s", ha="center", va="bottom")
        ax.set_title(f"{label} text ({len(texts[label])} chars)"); ax.set_ylabel("seconds: synthesize() -> audio file written")
    fig.suptitle(f"Remote TTS latency, mean of {RUNS} runs (dots = runs). Skipped: {', '.join(skipped) or 'none'}"); fig.tight_layout()
    fig.savefig(OUT / "tts_remote_latency.png", dpi=130)

print("\n=== SUMMARY (mean / min / max seconds) ===")
for n in names:
    for l in texts:
        r = results[n][l]["runs"]
        print(f"{n:14s} {l:6s} {statistics.mean(r):.3f} / {min(r):.3f} / {max(r):.3f}  (audio {results[n][l]['audio_secs']:.1f}s)")
for n, why in skipped.items():
    print(f"{n:14s} SKIPPED: {why[:160]}")
print(f"WINNER (lowest mean latency): {winner}")
