"""Profile LLM latency: ASR transcript generated -> full reply returned. Run: uv run python evaluate/llm_latency.py"""
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voicebot import config
from voicebot.asr.groq_asr import transcribe
from voicebot.llm.openrouter import reply
from voicebot.memory.store import build_system_prompt, load

MODELS = {"Gemini 3.8 Flash": ("google/gemini-3.8-flash", ""),
          "Llama 3.3 70B (DeepInfra Turbo)": ("meta-llama/llama-3.3-70b-instruct", "deepinfra/turbo")}
CLIPS = {"short (3s clip)": "test/test_data/84_121123_000008_000002.wav", "long (12s clip)": "test/test_data/002.wav"}
RUNS = 10
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)

# Transcribe once per clip so every model gets identical input
texts = {label: transcribe(Path(wav)) for label, wav in CLIPS.items()}
system = build_system_prompt(load("eval_user"))
results = {}
for name, (model, provider) in MODELS.items():
    config.LLM_MODEL, config.LLM_PROVIDER = model, provider  # swap the LLM; reply() reads config each call
    print(f"\n=== {name} ({model}, provider={provider or 'default'}) ===")
    reply(system, [], "Hello")  # warm-up: connection setup
    for label, text in texts.items():
        runs, chars = [], []
        for i in range(RUNS):
            t0 = time.perf_counter()          # transcript has just been generated
            ans = reply(system, [], text)     # returns when the full response is complete
            runs.append(time.perf_counter() - t0); chars.append(len(ans))
            print(f"[bench] {name} | {label} | run {i + 1}/{RUNS}: {runs[-1]:.3f}s ({len(ans)} chars)")
        results.setdefault(name, {})[label] = {"runs": runs, "chars": chars, "sample": ans}
(OUT / "llm_latency.json").write_text(json.dumps({"inputs": texts, "results": results}, indent=2))

names, labels = list(MODELS), list(CLIPS)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, label in zip(axes, labels):
    means = [statistics.mean(results[n][label]["runs"]) for n in names]
    ax.bar(names, means, color=["#4C78A8", "#F58518"], alpha=0.85)
    for i, n in enumerate(names):
        ax.scatter([i] * RUNS, results[n][label]["runs"], color="k", s=10, zorder=3)
        ax.text(i, means[i], f"{means[i]:.2f}s\n~{statistics.mean(results[n][label]['chars']):.0f} chars", ha="center", va="bottom")
    ax.set_title(f"{label}\n{len(texts[label])} char transcript"); ax.set_ylabel("seconds: transcript -> full reply")
    ax.set_ylim(0, max(max(results[n][label]["runs"]) for n in names) * 1.25); ax.tick_params(axis="x", rotation=10)
fig.suptitle(f"LLM latency via OpenRouter, mean of {RUNS} runs (dots = individual runs)"); fig.tight_layout()
fig.savefig(OUT / "llm_latency.png", dpi=130)

print("\n=== SUMMARY (mean / min / max seconds) ===")
for n in names:
    for label in labels:
        r = results[n][label]
        print(f"{n:32s} {label:16s} {statistics.mean(r['runs']):.3f} / {min(r['runs']):.3f} / {max(r['runs']):.3f}  (avg {statistics.mean(r['chars']):.0f} chars)")
