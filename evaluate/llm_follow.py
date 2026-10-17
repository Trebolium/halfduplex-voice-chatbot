"""Instruction-following + latency + peak memory for local mlx-lm models that fit in 8GB. Run: uv run python evaluate/llm_follow.py"""
import gc
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx

from voicebot.llm import mlx_lm
from voicebot.llm.openrouter import _clean
from voicebot.memory.store import build_system_prompt

MODELS = {"LFM2.5 1.2B (current)": "mlx-community/LFM2.5-1.2B-Instruct-4bit", "Gemma 3 1B": "mlx-community/gemma-3-1b-it-qat-4bit",
          "Llama 3.2 1B": "mlx-community/Llama-3.2-1B-Instruct-4bit", "Qwen3 1.7B": "mlx-community/Qwen3-1.7B-4bit",
          "Llama 3.2 3B": "mlx-community/Llama-3.2-3B-Instruct-4bit"}
FIRST = build_system_prompt({"name": "Nial", "sessions": 0, "facts": []})
KNOWN = build_system_prompt({"name": "Nial", "sessions": 2, "facts": ["has a dog called Biscuit", "used to be a bus driver"], "summary": "Chatted about his garden."})
# (system prompt, user text, extra check name) - each reply is also checked against the STYLE_RULES in the prompt
CASES = [(FIRST, "Hello?", "greets_by_name"), (KNOWN, "I had a rough morning, my knee is sore.", "short"),
         (KNOWN, "What do you remember about me?", "uses_memory"), (KNOWN, "Tell me about the moon landing, in detail please.", "elaborates"),
         (KNOWN, "I went for a walk by the river today.", "short"), (KNOWN, "Give me three tips for sleeping better.", "short")]
RUNS = 2
OUT = Path("evaluate/results"); OUT.mkdir(parents=True, exist_ok=True)


def checks(text: str, extra: str) -> dict:
    """Pass/fail per instruction the system prompt gives."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    c = {"plain_text": not re.search(r"[*#_`]|^\s*[-•\d]+[.)]?\s", text, re.M) and not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", text),
         "one_question": text.count("?") == 1 and text.strip().endswith("?")}
    if extra == "greets_by_name": c["greets_by_name"] = "nial" in text.lower()
    if extra == "uses_memory": c["uses_memory"] = "biscuit" in text.lower() or "bus" in text.lower() or "garden" in text.lower()
    if extra == "short": c["short"] = len(sents) <= 4
    if extra == "elaborates": c["elaborates"] = len(text) > 300
    return c


results = {}
for name, repo in MODELS.items():
    print(f"\n=== {name} ({repo}) ===")
    mlx_lm.load(repo)
    for _ in mlx_lm.stream_reply(FIRST, [], "Hello"): pass  # warm-up
    mx.reset_peak_memory()
    rows = []
    for system, user, extra in CASES:
        for _ in range(RUNS):
            t0 = time.perf_counter()
            for ans, r in mlx_lm.stream_reply(system, [], user): pass
            dt, text = time.perf_counter() - t0, _clean(ans)
            c = checks(text, extra)
            rows.append({"user": user, "text": text, "s": dt, "checks": c})
            print(f"[bench] {name} | {user[:35]!r}: {dt:.2f}s {'PASS' if all(c.values()) else 'FAIL ' + str([k for k, v in c.items() if not v])}")
    results[name] = {"rows": rows, "peak_gb": mx.get_peak_memory() / 1e9}
    gc.collect(); mx.clear_cache()

(OUT / "llm_follow.json").write_text(json.dumps(results, indent=2))
print("\n=== SUMMARY: all-checks pass % | per-check pass % | mean reply s | peak GPU mem GB ===")
for n, d in results.items():
    rows = d["rows"]
    allp = 100 * statistics.mean(all(r["checks"].values()) for r in rows)
    per = {k: 100 * statistics.mean(r["checks"][k] for r in rows if k in r["checks"]) for k in {k for r in rows for k in r["checks"]}}
    print(f"{n:24s} {allp:3.0f}% | {' '.join(f'{k}={v:.0f}' for k, v in sorted(per.items()))} | {statistics.mean(r['s'] for r in rows):.2f}s | {d['peak_gb']:.2f}GB")
