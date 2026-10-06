"""Shared helper for local TTS backends: write float/int audio to a wav in ARTEFACTS_DIR."""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot.config import ARTEFACTS_DIR

log = logging.getLogger("tts")


def save_wav(samples: np.ndarray, sample_rate: int, backend: str) -> Path:
    """Write mono samples to artefacts/<backend>_<timestamp>.wav and return the path."""
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"{backend}_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
    sf.write(str(path), np.asarray(samples).squeeze(), sample_rate)
    log.info("[TTS] %s wrote %s", backend, path.name)
    return path
