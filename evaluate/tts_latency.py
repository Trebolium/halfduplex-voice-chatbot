"""Benchmark TTS backends: clip -> ASR -> LLM -> [timer: LLM reply returned -> audio file written]. Run: uv run python evaluate/tts_latency.py"""
import importlib
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from voicebot.asr.groq_asr import transcribe
from voicebot.llm.openrouter import reply
from voicebot.memory.store import build_system_prompt, load

BACKENDS = {"edge-tts (remote)": "voicebot.tts.edge_tts", "Kokoro": "voicebot.tts.kokoro",
            "Piper": "voicebot.tts.piper", "Pocket TTS": "voicebot.tts.pocket"}
CLIPS = {"short (3s clip)": "test/test_data/84_121123_000008_000002.wav", "long (12s clip)": "test/test_data/002.wav"}
RUNS = 8
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)


def audio_secs(path: Path) -> float:
    """Duration of an audio file (mp3 or wav) in seconds."""
    return sf.info(str(path)).duration


# 1. ASR + LLM once per clip so every backend receives identical text
replies = {}
for label, wav in CLIPS.items():
    print(f"\n=== {label}: ASR -> LLM ===")
    text = transcribe(Path(wav))
    replies[label] = {"asr": text, "reply": reply(build_system_prompt(load("eval_user")), [], text)}
    print(f"[bench] {label}: reply is {len(replies[label]['reply'])} chars")

# 2. Time each backend
results, cold = {}, {}
for name, mod_path in BACKENDS.items():
    mod = importlib.import_module(mod_path)
    print(f"\n=== {name}: loading + warm-up ===")
    t0 = time.perf_counter()
    if hasattr(mod, "load"):
        mod.load()
    mod.synthesize("Warm up.")  # first call pays one-off costs (JIT, connection setup)
    cold[name] = time.perf_counter() - t0
    for label, r in replies.items():
        runs = []
        for i in range(RUNS):
            t0 = time.perf_counter()          # LLM reply has just been returned
            path = mod.synthesize(r["reply"])  # returns once the audio file is written
            runs.append(time.perf_counter() - t0)
            print(f"[bench] {name} | {label} | run {i + 1}/{RUNS}: {runs[-1]:.3f}s")
        results.setdefault(name, {})[label] = {"runs": runs, "audio_secs": audio_secs(path)}

(OUT / "tts_latency.json").write_text(json.dumps({"replies": replies, "cold_start_s": cold, "results": results}, indent=2))

# 3. Plots
names, labels = list(BACKENDS), list(CLIPS)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, label in zip(axes[:2], labels):
    means = [statistics.mean(results[n][label]["runs"]) for n in names]
    ax.bar(names, means, color="#4C78A8", alpha=0.8)
    for i, n in enumerate(names):
        ax.scatter([i] * RUNS, results[n][label]["runs"], color="k", s=10, zorder=3)
        ax.text(i, means[i], f"{means[i]:.2f}s", ha="center", va="bottom")
    ax.set_title(f"{label}\nreply = {len(replies[label]['reply'])} chars"); ax.set_ylabel("seconds: LLM reply -> audio file")
    ax.tick_params(axis="x", rotation=20)
x = range(len(names)); w = 0.38
for j, label in enumerate(labels):
    rtf = [statistics.mean(results[n][label]["runs"]) / results[n][label]["audio_secs"] for n in names]
    axes[2].bar([i + (j - 0.5) * w for i in x], rtf, w, label=label)
axes[2].axhline(1, color="r", ls="--", lw=1, label="real time"); axes[2].set_xticks(list(x)); axes[2].set_xticklabels(names, rotation=20)
axes[2].set_title("Real-time factor (synth time / audio length)\nlower is better"); axes[2].legend()
fig.suptitle(f"TTS latency, mean of {RUNS} runs (dots = individual runs), Apple Silicon CPU"); fig.tight_layout()
fig.savefig(OUT / "tts_latency.png", dpi=130)

fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(names, [cold[n] for n in names], color="#F58518")
ax.set_ylabel("seconds"); ax.set_title("Cold start (model load + first synth)"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
fig.savefig(OUT / "tts_cold_start.png", dpi=130)

print("\n=== SUMMARY (mean / min / max seconds) ===")
for n in names:
    for label in labels:
        r = results[n][label]["runs"]
        print(f"{n:20s} {label:16s} {statistics.mean(r):.3f} / {min(r):.3f} / {max(r):.3f}  (audio {results[n][label]['audio_secs']:.1f}s)")
    print(f"{n:20s} cold start {cold[n]:.2f}s")
