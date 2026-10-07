"""INTERFACE: transcribe(wav_path) -> str. Local Whisper base on Apple Silicon via mlx-whisper."""
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("asr")

MODEL = "mlx-community/whisper-base-mlx"
INFO = {"backend": "mlx-whisper", "location": "local-apple-gpu", "model": MODEL}
# Anti-hallucination settings: don't feed earlier text back in, and drop low-confidence / repetitive / silent output
DECODE = dict(language="en", condition_on_previous_text=False, no_speech_threshold=0.6,
              logprob_threshold=-1.0, compression_ratio_threshold=2.4, temperature=0.0)


_path = ""


def load() -> None:
    """Resolve the model from the local HF cache (no network) and warm it with half a second of silence; mlx-whisper keeps it in memory."""
    global _path
    import mlx_whisper
    from voicebot.hf_cache import local_path
    print(f"[asr] loading {MODEL} from the local cache (first run downloads ~150MB)...")
    _path = local_path(MODEL)
    mlx_whisper.transcribe(np.zeros(8000, dtype=np.float32), path_or_hf_repo=_path, **DECODE)


def transcribe(wav_path: Path) -> str:
    """Transcribe a wav locally and return stripped English text."""
    import mlx_whisper
    print(f"[asr] transcribing {Path(wav_path).name} locally with {MODEL}")
    t0 = time.time()
    try:
        res = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=_path or MODEL, **DECODE)
    except Exception as e:
        raise RuntimeError(f"Local whisper failed ({type(e).__name__}: {e}). Check the wav exists and is 16kHz mono, "
                           "and that the model downloaded (re-run with --asr_model local-whisper on a connection).") from e
    text = (res.get("text") or "").strip()
    log.info("ASR done in %.2fs: %r", time.time() - t0, text)
    print(f"[asr] {time.time() - t0:.2f}s -> {text!r}")
    return text
