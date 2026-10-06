"""INTERFACE: trim_speech(wav_path) -> Path | None, plus is_speech/reset for live use. Silero VAD."""
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot.config import ARTEFACTS_DIR, MIN_SPEECH_MS, SAMPLE_RATE, VAD_THRESHOLD

log = logging.getLogger("vad")
INFO = {"backend": "silero-vad", "location": "local-cpu", "threshold": VAD_THRESHOLD, "min_speech_ms": MIN_SPEECH_MS}
FRAME_SAMPLES = 512  # Silero needs exactly 512 samples per call at 16kHz (32ms)
_model = None


def _load():
    """Load the Silero model once."""
    global _model
    if _model is None:
        print("[vad] loading Silero VAD...")
        from silero_vad import load_silero_vad
        _model = load_silero_vad()
    return _model


def reset() -> None:
    """Clear the model's streaming state; call before each new live recording."""
    _load().reset_states()


def is_speech(frame: np.ndarray) -> bool:
    """True if one 512-sample int16 frame contains speech."""
    import torch
    with torch.no_grad():
        return _load()(torch.from_numpy(frame.astype(np.float32) / 32768), SAMPLE_RATE).item() > VAD_THRESHOLD


def trim_speech(wav_path: Path) -> Path | None:
    """Keep only speech (with padding); write <name>_trimmed.wav. None if no speech >= MIN_SPEECH_MS."""
    import torch
    from silero_vad import get_speech_timestamps
    wav_path = Path(wav_path)
    print(f"[vad] trimming {wav_path.name}")
    t0 = time.perf_counter()
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != SAMPLE_RATE:  # linear resample (mic recordings are already 16kHz; this covers other test files)
        print(f"[vad] resampling {sr}Hz -> {SAMPLE_RATE}Hz")
        audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / sr)), np.arange(len(audio)), audio).astype(np.float32)
        sr = SAMPLE_RATE
    spans = get_speech_timestamps(torch.from_numpy(audio), _load(), sampling_rate=SAMPLE_RATE, threshold=VAD_THRESHOLD,
                                  min_speech_duration_ms=MIN_SPEECH_MS, speech_pad_ms=100)
    out = np.concatenate([audio[s["start"]:s["end"]] for s in spans]) if spans else np.zeros(0, np.float32)
    msg = f"[vad] {len(audio) / sr:.2f}s -> {len(out) / sr:.2f}s in {time.perf_counter() - t0:.2f}s"
    log.info(msg); print(msg)
    if not len(out):
        print("[vad] no speech found")
        return None
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / f"{wav_path.stem}_trimmed.wav"
    sf.write(str(out_path), out, SAMPLE_RATE, subtype="PCM_16")
    return out_path
