"""TTS via the edge-tts package (free Microsoft voices) + macOS afplay playback."""
import asyncio
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import edge_tts  # third-party package (absolute import, not this module)

from voicebot.config import ARTEFACTS_DIR, TTS_VOICE

log = logging.getLogger("tts")


def synthesize(text: str) -> Path:
    """Convert text to an mp3 in ARTEFACTS_DIR and return its path."""
    if not text or not text.strip():
        raise ValueError("synthesize() got empty text. Make sure the LLM returned a reply before calling TTS.")
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"reply_{datetime.now():%Y%m%d_%H%M%S_%f}.mp3"
    msg = f"[TTS] Synthesizing {len(text)} chars with voice {TTS_VOICE} -> {path.name}"
    log.info(msg); print(msg)
    t0 = time.time()
    try:
        asyncio.run(edge_tts.Communicate(text, TTS_VOICE).save(str(path)))
    except Exception as e:
        raise RuntimeError(
            f"edge-tts failed ({type(e).__name__}: {e}). Check your internet connection "
            f"and that TTS_VOICE '{TTS_VOICE}' is a valid voice (run `uv run edge-tts --list-voices`)."
        ) from e
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("edge-tts produced no audio. Check your internet connection and TTS_VOICE.")
    msg = f"[TTS] Done in {time.time() - t0:.2f}s ({path.stat().st_size} bytes)"
    log.info(msg); print(msg)
    return path


def play(path: Path) -> None:
    """Play an audio file with macOS afplay, blocking until finished."""
    msg = f"[TTS] Playing {Path(path).name}"
    log.info(msg); print(msg)
    try:
        subprocess.run(["afplay", str(path)], check=True)
    except FileNotFoundError as e:
        raise RuntimeError("afplay not found. It ships with macOS; on other OSes swap play() for ffplay/mpv.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"afplay failed (exit {e.returncode}). Check the file is valid audio: {path}") from e
