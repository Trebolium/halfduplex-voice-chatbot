"""Compare eval runs side by side: config differences, then WER / RTF / latency per group with deltas vs the first run.

Run: uv run voicebot-compare --compare_results <run_dir> <run_dir> [...]   (folder names inside evaluate/results/)
Writes report.txt + PNG charts to evaluate/result_comparisons/<run>__vs__<run>/.
"""
import argparse
import json
from pathlib import Path

from voicebot import config
from voicebot.evaluation import RESULTS_DIR, STAGES

OUT_DIR = config.ROOT / "evaluate" / "result_comparisons"
# Reference categorical palette (fixed order, validated for CVD separation); runs and stages each take slots 1..n
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # second encoding so runs never rely on colour alone
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

SKIP_CONFIG = {"datetime", "datapoints", "data_dir"}  # always differ or are checked separately


def parse_args(argv: list[str] | None = None) -> list[str]:
    """--compare_results is required and needs at least two run folder names."""
    p = argparse.ArgumentParser(prog="voicebot-compare", description="Compare metrics of evaluation runs in evaluate/results/.")
    p.add_argument("--compare_results", nargs="+", metavar="RUN_DIR",
                   help="two or more folder names in evaluate/results/ (the first is the baseline for deltas)")
    runs = p.parse_args(argv).compare_results
    available = ", ".join(sorted(d.name for d in RESULTS_DIR.glob("*") if (d / "metrics.json").exists())) or "none yet"
    if not runs:
        raise SystemExit("No runs to compare: pass --compare_results <run_dir> <run_dir> with folder names from "
                         f"evaluate/results/ (available: {available}). Create runs with `uv run voicebot-eval`.")
    if len(runs) < 2:
        raise SystemExit(f"--compare_results needs at least two runs, got 1 ({runs[0]}). Available: {available}.")
    return runs


def load(name: str) -> tuple[dict, dict]:
    """(config, metrics) for one run folder; clear error if it is missing or incomplete."""
    d = RESULTS_DIR / Path(name).name  # accept either the bare name or evaluate/results/<name>
    if not (d / "config.json").exists() or not (d / "metrics.json").exists():
        avail = ", ".join(sorted(x.name for x in RESULTS_DIR.glob("*") if (x / "metrics.json").exists())) or "none yet"
        raise SystemExit(f"Run '{name}' not found or incomplete (needs config.json + metrics.json in {d}). Available: {avail}.")
    return json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text())


def flatten(d: dict, prefix: str = "") -> dict:
    """{"setup.asr.model": ..., ...} so nested configs can be compared key by key."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def config_diff(names: list[str], configs: list[dict]) -> list[str]:
    """Lines listing every config value that differs between runs, plus a dataset-mismatch warning."""
    flat = [flatten({k: v for k, v in c.items() if k not in SKIP_CONFIG}) for c in configs]
    keys = sorted(set().union(*flat))
    lines = []
    for k in keys:
        vals = [f.get(k, "-") for f in flat]
        if len({json.dumps(v, sort_keys=True) for v in vals}) > 1:
            lines.append(f"  {k}:")
            lines += [f"      {n:<28} {str(v)[:90]}" for n, v in zip(names, vals)]
    sets = [set(c.get("datapoints", [])) for c in configs]
    if any(s != sets[0] for s in sets):
        lines.append(f"  WARNING: the runs used different clips ({', '.join(str(len(s)) for s in sets)} clips) - "
                     "WER and latency are not directly comparable.")
    return lines or ["  (none besides datetime - same settings)"]


def rows(summary: dict) -> list[tuple[str, dict]]:
    """(group name, group metrics) for overall, each dataset and each duration bucket."""
    if "overall" not in summary:
        return []
    return [("overall", summary["overall"])] + [(k, v) for key in ("by_dataset", "by_bucket") for k, v in summary.get(key, {}).items()]


METRICS = [("WER (corpus)", lambda g: g["wer_corpus"], "pct"), ("ASR RTF p50", lambda g: g["rtf"]["asr"]["p50"], "num")] + \
          [(f"{s} p50 (s)", lambda g, s=s: g["latency_s"][s]["p50"], "num") for s in STAGES + ["total"]] + \
          [("total p90 (s)", lambda g: g["latency_s"]["total"]["p90"], "num")]


def fmt(v, kind: str) -> str:
    return "-" if v is None else (f"{v * 100:.1f}%" if kind == "pct" else f"{v:.3f}")


def table(names: list[str], metrics: list[dict]) -> list[str]:
    """One block per group: metric rows x run columns, with a delta column per run vs the first (baseline)."""
    groups = {}
    for i, m in enumerate(metrics):
        for g, vals in rows(m["summary"]):
            groups.setdefault(g, [None] * len(metrics))[i] = vals
    w = max(14, max(len(n) for n in names) + 2)
    head = f"  {'metric':<16}" + "".join(f"{n:>{w}}" for n in names) + "".join(f"{'Δ ' + str(i + 1) + ' vs 1':>12}" for i in range(1, len(names)))
    out = []
    for g, per_run in groups.items():
        n = [str(v["n"]) if v else "-" for v in per_run]
        out += [f"\n[{g}]  n = {' / '.join(n)}", head]
        for label, get, kind in METRICS:
            vals = []
            for v in per_run:
                try:
                    vals.append(get(v) if v else None)
                except (KeyError, TypeError):
                    vals.append(None)
            deltas = []
            for v in vals[1:]:
                if v is None or vals[0] is None:
                    deltas.append("-")
                else:
                    d = v - vals[0]
                    deltas.append(f"{d * 100:+.1f}pt" if kind == "pct" else f"{d:+.3f}")
            out.append(f"  {label:<16}" + "".join(f"{fmt(v, kind):>{w}}" for v in vals) + "".join(f"{d:>12}" for d in deltas))
    return out


def compare(names: list[str]) -> str:
    """Build the text report for already-validated run names (cli also saves it and the charts)."""
    print(f"[compare] loading {len(names)} runs from {RESULTS_DIR}")
    loaded = [load(n) for n in names]
    configs, metrics = [c for c, _ in loaded], [m for _, m in loaded]
    legacy = [n for n, m in zip(names, metrics) if "overall" not in m["summary"]]
    if legacy:
        raise SystemExit(f"Run(s) {', '.join(legacy)} use an older metrics.json format without per-group summaries. "
                         "Re-run `uv run voicebot-eval` for that setup and compare the new run instead.")
    lines = ["Runs (run 1 is the baseline for Δ):"]
    for i, (n, c, m) in enumerate(zip(names, configs, metrics), 1):
        dirty = " (uncommitted changes)" if c.get("dirty") else ""
        note = f"  note: {c['note']}" if c.get("note") else ""
        lines.append(f"  {i}. {n}{note}  {c.get('datetime', '?')}  commit {str(c.get('commit_sha', '?'))[:7]}{dirty}  mode={c.get('mode')}  "
                     f"backends={c.get('backends')}  ok={m['summary']['n_ok']} failed={m['summary']['n_failed']}")
    lines += ["", "Config differences:"] + config_diff(names, configs)
    lines += ["", "Metrics (WER lower is better; latency in seconds, lower is better; Δ for WER in percentage points):"]
    lines += table(names, metrics)
    return "\n".join(lines)


def _style(ax, title: str, xlabel: str) -> None:
    """Recessive axes: no top/right spines, light grid on the value axis only, text in ink colours."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _groups_in_order(metrics: list[dict], configs: list[dict]) -> list[str]:
    """overall, then datasets, then duration buckets in their natural (short -> long) order."""
    s0 = metrics[0]["summary"]
    datasets = sorted({k for m in metrics for k in m["summary"].get("by_dataset", {})})
    order = configs[0].get("duration_buckets") or sorted({k for m in metrics for k in m["summary"].get("by_bucket", {})})
    buckets = [b for b in order if any(b in m["summary"].get("by_bucket", {}) for m in metrics)]
    return ["overall"] + datasets + buckets if "overall" in s0 else datasets + buckets


def _group(summary: dict, name: str) -> dict | None:
    return summary["overall"] if name == "overall" else summary.get("by_dataset", {}).get(name) or summary.get("by_bucket", {}).get(name)


def plot(names: list[str], configs: list[dict], metrics: list[dict], out: Path) -> list[Path]:
    """Render the comparison charts as PNGs into out/; returns the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    short = [f"{i}. {c.get('mode', '?')} ({n})" for i, (n, c) in enumerate(zip(names, configs), 1)]  # legend labels
    files, n = [], len(names)
    groups = _groups_in_order(metrics, configs)

    # 1) Where the time goes: stacked mean latency per stage, one bar per run (means add up to the total mean)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.7 * n), facecolor=SURFACE)
    for i, m in enumerate(metrics):
        left = 0.0
        lat = m["summary"]["overall"]["latency_s"]
        for j, stage in enumerate(STAGES):
            w = lat[stage]["mean"]
            ax.barh(i, w, left=left, height=0.55, color=SERIES[j], edgecolor=SURFACE, linewidth=2, label=stage if i == 0 else None)
            if w > 0.25:  # direct label where it fits (relief for the low-contrast slots)
                ax.text(left + w / 2, i, f"{stage}\n{w:.2f}s", ha="center", va="center", fontsize=8, color=INK)
            left += w
        ax.text(left + 0.05, i, f"{left:.2f}s", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(n), short); ax.invert_yaxis()
    _style(ax, "Where the time goes - mean seconds per stage (overall)", "seconds (mean; stages add up to the total)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    files.append(out / "1_latency_breakdown.png"); fig.tight_layout(); fig.savefig(files[-1], dpi=150); plt.close(fig)

    # 2) and 3) Grouped horizontal bars per group: WER, and total latency p50 with a p90 tick
    def grouped(metric, title, xlabel, fname, p90=None, pct=False):
        fig, ax = plt.subplots(figsize=(10, 1.2 + 0.32 * n * len(groups) + 0.3 * len(groups)), facecolor=SURFACE)
        h = 0.8 / n
        for i, m in enumerate(metrics):
            ys, vals, ticks = [], [], []
            for gi, g in enumerate(groups):
                d = _group(m["summary"], g)
                if d is None:
                    continue
                ys.append(gi - 0.4 + h * (i + 0.5)); vals.append(metric(d) * (100 if pct else 1))
                ticks.append(p90(d) if p90 else None)
            ax.barh(ys, vals, height=h * 0.85, color=SERIES[i % 8], edgecolor=SURFACE, linewidth=1, label=short[i])
            if p90:
                ax.scatter(ticks, ys, marker="|", s=120, color=INK, linewidths=2, zorder=3, label="p90" if i == 0 else None)
        ax.set_yticks(range(len(groups)), [f"{g}" for g in groups]); ax.invert_yaxis()
        for gi in range(1, len(groups)):
            ax.axhline(gi - 0.5, color=GRID, linewidth=0.6)
        _style(ax, title, xlabel)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(n + 1, 3))
        files.append(out / fname); fig.tight_layout(); fig.savefig(files[-1], dpi=150); plt.close(fig)

    grouped(lambda d: d["wer_corpus"], "ASR word error rate by group (corpus WER, lower is better)", "WER (%)",
            "2_wer_by_group.png", pct=True)
    grouped(lambda d: d["latency_s"]["total"]["p50"], "Total turn latency by group - bar = p50, tick = p90 (lower is better)",
            "seconds (VAD + ASR + LLM + TTS)", "3_latency_by_group.png", p90=lambda d: d["latency_s"]["total"]["p90"])

    # 4) ASR time vs clip length, every clip: shows fixed overhead (intercept) and per-second cost (slope) per run
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for i, m in enumerate(metrics):
        pts = [(r["audio_s"], r["latency_s"]["asr"]) for r in m.get("datapoints", []) if "error" not in r and "audio_s" in r]
        if not pts:
            continue
        x, y = np.array(pts).T
        ax.scatter(x, y, s=36, color=SERIES[i % 8], marker=MARKERS[i % 8], edgecolors=SURFACE, linewidths=1.5, label=short[i], zorder=3)
        if len(x) > 1:
            slope, icept = np.polyfit(x, y, 1)
            xs = np.array([0, x.max()])
            ax.plot(xs, icept + slope * xs, color=SERIES[i % 8], linewidth=2, alpha=0.8)
            ax.text(x.max(), icept + slope * x.max(), f"  {icept:.2f}s + {slope * 60:.2f}s/min", va="center", fontsize=8, color=INK_2)
    _style(ax, "ASR time vs clip length (every clip, with linear fit)", "clip length (s)")
    ax.set_ylabel("ASR seconds", color=INK_2, fontsize=9); ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    files.append(out / "4_asr_scaling.png"); fig.tight_layout(); fig.savefig(files[-1], dpi=150); plt.close(fig)
    return files


def cli() -> None:
    names = parse_args()
    report = compare(names)
    loaded = [load(n) for n in names]
    out = OUT_DIR / "__vs__".join(Path(n).name for n in names)
    if len(out.name) > 200:  # many runs: keep the folder name legal
        out = OUT_DIR / f"{len(names)}_runs_{'__'.join(Path(n).name[:15] for n in names)}"[:200]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report + "\n")
    print(report)
    print(f"\n[compare] rendering charts...")
    files = plot(names, [c for c, _ in loaded], [m for _, m in loaded], out)
    print(f"[compare] saved report.txt + {len(files)} charts -> {out}")
    for f in files:
        print(f"[compare]   {f.name}")


if __name__ == "__main__":
    cli()
