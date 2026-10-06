"""Benchmark top-4 local TTS backends on test/test_data/*.txt (one subprocess per model). Run: uv run python evaluate/tts_top4_latency.py"""
import importlib
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

BACKENDS = {"Kokoro": "voicebot.tts.kokoro", "Pocket TTS": "voicebot.tts.pocket",
            "Piper": "voicebot.tts.piper", "Kitten TTS mini": "voicebot.tts.kitten"}
TEXTS = {"short": "test/test_data/84_121123_000008_000002.txt", "long": "test/test_data/002.txt"}
RUNS = 8
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)


def worker(mod_path: str):
    """Load one model, warm up, time RUNS syntheses per text, print JSON on the last line."""
    mod = importlib.import_module(mod_path)
    t0 = time.perf_counter(); mod.load(); load_s = time.perf_counter() - t0
    t0 = time.perf_counter(); mod.synthesize("Warm up."); first_s = time.perf_counter() - t0  # first call pays JIT costs
    res = {}
    for label, f in TEXTS.items():
        text = Path(f).read_text().strip(); runs = []
        for i in range(RUNS):
            t0 = time.perf_counter(); path = mod.synthesize(text); runs.append(time.perf_counter() - t0)
            print(f"[bench] {mod_path} | {label} | run {i + 1}/{RUNS}: {runs[-1]:.3f}s", file=sys.stderr, flush=True)
        res[label] = {"chars": len(text), "runs": runs, "audio_secs": sf.info(str(path)).duration}
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    print("RESULT " + json.dumps({"load_s": load_s, "first_synth_s": first_s, "peak_rss_mb": peak_mb, "results": res}))


def plot(data):
    """Write latency and cold-start plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(data); c = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, label in zip(axes[:2], TEXTS):
        means = [statistics.mean(data[n]["results"][label]["runs"]) for n in names]
        ax.bar(names, means, color=c, alpha=0.8)
        for i, n in enumerate(names):
            ax.scatter([i] * RUNS, data[n]["results"][label]["runs"], color="k", s=10, zorder=3)
            ax.text(i + 0.08, means[i], f"{means[i]:.2f}s", ha="left", va="bottom")
        ax.set_title(f"{label} text ({data[names[0]]['results'][label]['chars']} chars)"); ax.set_ylabel("seconds: synthesize() call -> wav written")
        ax.tick_params(axis="x", rotation=15)
    w = 0.38
    for j, label in enumerate(TEXTS):
        rtf = [statistics.mean(data[n]["results"][label]["runs"]) / data[n]["results"][label]["audio_secs"] for n in names]
        xs = [i + (j - 0.5) * w for i in range(len(names))]
        axes[2].bar(xs, rtf, w, label=label)
        for x, v in zip(xs, rtf):
            axes[2].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].axhline(1, color="r", ls="--", lw=1, label="real time"); axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_title("Real-time factor (lower is better)"); axes[2].legend()
    fig.suptitle(f"Top-4 local TTS latency, mean of {RUNS} runs (dots = runs), M2 CPU"); fig.tight_layout(); fig.savefig(OUT / "tts_top4_latency.png", dpi=130)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(names, [data[n]["load_s"] for n in names], color="#F58518", label="model load")
    ax.bar(names, [data[n]["first_synth_s"] for n in names], bottom=[data[n]["load_s"] for n in names], color="#4C78A8", label="first synth")
    for i, n in enumerate(names):
        ax.text(i, data[n]["load_s"] + data[n]["first_synth_s"], f"{data[n]['load_s'] + data[n]['first_synth_s']:.2f}s", ha="center", va="bottom")
    ax.set_ylabel("seconds"); ax.set_title("Cold start (load + first synth)"); ax.legend(); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / "tts_top4_cold_start.png", dpi=130)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1]); sys.exit()
    data = {}
    for name, mod in BACKENDS.items():
        print(f"\n=== {name}: running in subprocess ===", flush=True)
        p = subprocess.run([sys.executable, __file__, mod], stdout=subprocess.PIPE, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode or not line:
            sys.exit(f"{name} failed (exit {p.returncode}); see log above. Check `uv sync --group bench` and models/ files.")
        data[name] = json.loads(line[0][7:])
    (OUT / "tts_top4_latency.json").write_text(json.dumps(data, indent=2)); plot(data)
    print("\n=== SUMMARY (mean / min / max s) ===")
    for n, d in data.items():
        for label, r in d["results"].items():
            m = statistics.mean(r["runs"]); print(f"{n:16s} {label:6s} {m:.3f} / {min(r['runs']):.3f} / {max(r['runs']):.3f}  audio {r['audio_secs']:.1f}s RTF {m / r['audio_secs']:.2f}")
        print(f"{n:16s} load {d['load_s']:.2f}s first synth {d['first_synth_s']:.2f}s peak RSS {d['peak_rss_mb']:.0f}MB")
