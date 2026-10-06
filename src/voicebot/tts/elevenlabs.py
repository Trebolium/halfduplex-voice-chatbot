"""INTERFACE: synthesize(text) -> Path. ElevenLabs Flash v2.5 over HTTPS; uses ELEVENLABS_API_KEY (free tier available)."""
import logging
from datetime import datetime
from pathlib import Path

import httpx

from voicebot import config
from voicebot.config import ARTEFACTS_DIR

log = logging.getLogger("tts")
MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # "George", a premade voice available on the free tier
INFO = {"backend": "elevenlabs", "location": "cloud", "model": f"{MODEL} / {DEFAULT_VOICE}"}


def load() -> None:
    """Fail early with a clear message if the key is missing; record the voice in INFO."""
    config.require("ELEVENLABS_API_KEY", config.ELEVENLABS_API_KEY)
    INFO["model"] = f"{MODEL} / {config.TTS_VOICE or DEFAULT_VOICE}"


def synthesize(text: str) -> Path:
    """Convert text to an mp3 and return its path."""
    key = config.require("ELEVENLABS_API_KEY", config.ELEVENLABS_API_KEY)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"elevenlabs_{datetime.now():%Y%m%d_%H%M%S_%f}.mp3"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.TTS_VOICE or DEFAULT_VOICE}?output_format=mp3_44100_128"
    for attempt in (1, 2):  # one retry: a dropped TLS connection happens occasionally
        try:
            r = httpx.post(url, timeout=30, headers={"xi-api-key": key}, json={"text": text, "model_id": MODEL})
            break
        except httpx.TransportError as e:
            if attempt == 2:
                raise RuntimeError(f"Could not reach ElevenLabs ({type(e).__name__}: {e}). Check your internet connection.") from e
            print(f"[TTS] ElevenLabs connection dropped ({type(e).__name__}), retrying once...")
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs returned {r.status_code}: {r.text[:200]}. Check ELEVENLABS_API_KEY, that the voice id "
                           "(--tts_voice) is a premade voice on your plan, and your free-tier credits.")
    path.write_bytes(r.content)
    return path
