"""Benchmark local mlx-lm models vs the remote LLM: ASR transcript -> full reply (+ time to first token). Run: uv run python evaluate/llm_local_latency.py"""
import gc
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voicebot import config
from voicebot.asr.groq_asr import transcribe
from voicebot.llm import mlx_lm, openrouter
from voicebot.llm.openrouter import _clean
from voicebot.memory.store import build_system_prompt, load

LOCAL = {"Llama 3.2 1B": "mlx-community/Llama-3.2-1B-Instruct-4bit", "Gemma 3 1B": "mlx-community/gemma-3-1b-it-qat-4bit",
         "LFM2.5 1.2B": "mlx-community/LFM2.5-1.2B-Instruct-4bit", "Llama 3.2 3B": "mlx-community/Llama-3.2-3B-Instruct-4bit",
         "Qwen3 4B (2507)": "mlx-community/Qwen3-4B-Instruct-2507-4bit"}
REMOTE = "Llama 3.3 70B (DeepInfra Turbo, remote)"
CLIPS = {"short (3s clip)": "test/test_data/84_121123_000008_000002.wav", "long (12s clip)": "test/test_data/002.wav"}
RUNS = 5
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)

texts = {label: transcribe(Path(wav)) for label, wav in CLIPS.items()}  # identical input for every model
system = build_system_prompt(load("eval_user"))
results, loadtime = {}, {}


def record(name, label, runs):
    results.setdefault(name, {})[label] = runs
    print(f"[bench] {name} | {label}: mean {statistics.mean(r['total'] for r in runs):.2f}s")


# remote baseline (non-streaming, so no TTFT)
config.LLM_MODEL, config.LLM_PROVIDER = "meta-llama/llama-3.3-70b-instruct", "deepinfra/turbo"
openrouter.reply(system, [], "Hello")
for label, text in texts.items():
    runs = []
    for _ in range(RUNS):
        t0 = time.perf_counter(); ans = openrouter.reply(system, [], text)
        runs.append({"total": time.perf_counter() - t0, "ttft": None, "tps": None, "text": ans})
    record(REMOTE, label, runs)

for name, repo in LOCAL.items():
    print(f"\n=== {name} ({repo}) ===")
    t0 = time.perf_counter(); mlx_lm.load(repo); loadtime[name] = time.perf_counter() - t0  # includes download on first run
    for _ in mlx_lm.stream_reply(system, [], "Hello"): pass  # warm-up (kernel compile)
    t0 = time.perf_counter(); mlx_lm.load(repo); loadtime[name] = time.perf_counter() - t0  # warm load from disk cache
    for label, text in texts.items():
        runs = []
        for _ in range(RUNS):
            t0 = time.perf_counter(); first = None
            for ans, r in mlx_lm.stream_reply(system, [], text):
                if first is None:
                    first = time.perf_counter() - t0
            runs.append({"total": time.perf_counter() - t0, "ttft": first, "tps": r.generation_tps, "text": _clean(ans)})
        record(name, label, runs)
    gc.collect()

(OUT / "llm_local_latency.json").write_text(json.dumps({"inputs": texts, "load_s": loadtime, "results": results}, indent=2))

names, labels = [REMOTE] + list(LOCAL), list(CLIPS)
short = {n: n.replace(" (DeepInfra Turbo, remote)", "\n(DeepInfra, remote)") for n in names}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, label in zip(axes[:2], labels):
    means = [statistics.mean(r["total"] for r in results[n][label]) for n in names]
    ax.bar([short[n] for n in names], means, color=["#F58518"] + ["#4C78A8"] * len(LOCAL), alpha=0.85)
    for i, n in enumerate(names):
        ax.scatter([i] * RUNS, [r["total"] for r in results[n][label]], color="k", s=10, zorder=3)
        ax.text(i, means[i], f"{means[i]:.2f}s", ha="center", va="bottom")
    ax.set_title(f"Full reply, {label}"); ax.set_ylabel("seconds: transcript -> complete reply"); ax.tick_params(axis="x", rotation=25)
loc = list(LOCAL)
w = 0.38
for j, label in enumerate(labels):
    axes[2].bar([i + (j - 0.5) * w for i in range(len(loc))], [statistics.mean(r["ttft"] for r in results[n][label]) for n in loc], w, label=label)
axes[2].set_xticks(range(len(loc))); axes[2].set_xticklabels(loc, rotation=25); axes[2].legend()
axes[2].set_title("Local models: time to first token"); axes[2].set_ylabel("seconds")
fig.suptitle(f"LLM latency: local mlx-lm (M2 Air 8GB) vs remote, mean of {RUNS} runs (dots = runs)"); fig.tight_layout()
fig.savefig(OUT / "llm_local_latency.png", dpi=130)

print("\n=== SUMMARY: total s | ttft s | tok/s | chars | warm load s ===")
for n in names:
    for label in labels:
        rs = results[n][label]
        ttft = statistics.mean(r["ttft"] for r in rs) if rs[0]["ttft"] else float("nan")
        tps = statistics.mean(r["tps"] for r in rs) if rs[0]["tps"] else float("nan")
        print(f"{n:42s} {label:16s} {statistics.mean(r['total'] for r in rs):.2f} | {ttft:.2f} | {tps:.0f} | {statistics.mean(len(r['text']) for r in rs):.0f} | {loadtime.get(n, float('nan')):.1f}")
