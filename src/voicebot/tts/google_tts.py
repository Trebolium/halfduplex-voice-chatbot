"""INTERFACE: synthesize(text) -> Path. Google Translate voice via gTTS (free, no key, unofficial endpoint)."""
import logging
from datetime import datetime
from pathlib import Path

from voicebot.config import ARTEFACTS_DIR

log = logging.getLogger("tts")
INFO = {"backend": "gtts", "location": "cloud", "model": "google-translate-tts / en"}


def synthesize(text: str) -> Path:
    """Convert text to an mp3 and return its path."""
    from gtts import gTTS
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"gtts_{datetime.now():%Y%m%d_%H%M%S_%f}.mp3"
    try:
        gTTS(text, lang="en").save(str(path))
    except Exception as e:
        raise RuntimeError(f"gTTS failed ({type(e).__name__}: {e}). Check your internet connection (it uses Google's free "
                           "translate endpoint, which can rate-limit) or use --tts_model edge-tts.") from e
    return path
