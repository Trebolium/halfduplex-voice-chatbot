"""Stratified sample (run from repo root: uv run python evaluate/sample_dataset.py [--copy]): 3 clips per duration bucket per dataset; the long-form bucket is built by joining clips. --copy writes them."""
import glob, os, random, sys
from collections import defaultdict
import numpy as np, soundfile as sf
from voicebot import config  # loads .env

OUT = config.TEST_AUDIO_PATH.rstrip("/")  # clips are written here (TEST_AUDIO_PATH in .env)
if not OUT:
    sys.exit("TEST_AUDIO_PATH is not set in .env - add e.g. TEST_AUDIO_PATH=./test/test_data/ so the sampled test clips have somewhere to go.")
BUCKETS = {"short": (1, 5), "medium": (5, 15), "long": (15, 30)}  # single clips
JOINED = (45, 120)  # long-form: beyond Whisper's 30s window, built by joining clips
N, MAXNAME = 3, 240
DATASETS = os.path.expanduser("~/Documents/datasets")  # holds LibriTTS/ and voxpopuli/
random.seed(0)
existing = {os.path.basename(p)[:-4] for p in glob.glob(f"{OUT}/**/*.wav", recursive=True)}

def clips(name):
    """(wav, txt, dur, group) for every usable clip; group = chapter for LibriTTS (same speaker + book)."""
    out = []
    root = f"{DATASETS}/{name}"
    for w in sorted(glob.glob(f"{root}/**/*.wav", recursive=True)):
        t = w[:-4] + (".normalized.txt" if name == "LibriTTS" else ".txt")
        s = os.path.basename(w)[:-4]
        if not os.path.exists(t) or s in existing:
            continue
        if not open(t).read().strip():
            continue
        out.append((w, t, sf.info(w).duration, os.path.dirname(w)))
    return out

def stem(c): return os.path.basename(c[0])[:-4]
def speaker(c): return stem(c).split("_")[0] if "_" in stem(c) else stem(c)

def pick_singles(pool, lo, hi, used):
    """N random clips in [lo, hi), different speakers where the dataset has speaker ids."""
    cands = [c for c in pool if lo <= c[2] < hi and c[0] not in used]
    random.shuffle(cands)
    out, spk = [], set()
    for c in cands:
        if speaker(c) not in spk:
            out.append([c]); spk.add(speaker(c)); used.add(c[0])
        if len(out) == N: break
    return out

def pick_joined(pool, name, used):
    """LibriTTS: consecutive clips within one chapter. voxpopuli: unrelated segments, so random unused clips."""
    lo, hi = JOINED
    out = []
    if name == "LibriTTS":
        chapters = defaultdict(list)
        for c in pool: chapters[c[3]].append(c)
        keys = list(chapters); random.shuffle(keys)
        for k in keys:
            g = [c for c in chapters[k] if c[0] not in used]
            starts = list(range(len(g))); random.shuffle(starts)
            for i in starts:
                run, tot = [], 0
                for c in g[i:]:
                    run.append(c); tot += c[2]
                    if tot >= lo: break
                if lo <= tot <= hi and len("+".join(map(stem, run))) <= MAXNAME:
                    out.append(run); used.update(c[0] for c in run); break
            if len(out) == N: break
    else:
        free = [c for c in pool if c[0] not in used]; random.shuffle(free)
        for _ in range(N):
            run, tot = [], 0
            while free and tot < lo:
                c = free.pop(); run.append(c); tot += c[2]
            if lo <= tot <= hi:
                out.append(sorted(run, key=stem))
    return out

copy = "--copy" in sys.argv
for name in ["LibriTTS", "voxpopuli"]:
    pool, used = clips(name), set()
    picks = {b: pick_singles(pool, lo, hi, used) for b, (lo, hi) in BUCKETS.items()}
    picks["longform"] = pick_joined(pool, name, used)
    dest = f"{OUT}/{name.lower()}"
    for b, runs in picks.items():
        print(f"{name} {b}: {len(runs)}/{N}")
        for r in runs:
            new = "+".join(map(stem, r))
            print(f"  {sum(c[2] for c in r):6.2f}s {len(r):2d} clip(s)  {new[:100]}{'...' if len(new) > 100 else ''}")
            if copy:
                os.makedirs(dest, exist_ok=True)
                audio = [sf.read(c[0], dtype="float32") for c in r]
                assert len({sr for _, sr in audio}) == 1, f"mixed sample rates in {new}"
                sf.write(f"{dest}/{new}.wav", np.concatenate([a for a, _ in audio]), audio[0][1], subtype="PCM_16")
                open(f"{dest}/{new}.txt", "w").write(" ".join(open(c[1]).read().strip() for c in r) + "\n")
