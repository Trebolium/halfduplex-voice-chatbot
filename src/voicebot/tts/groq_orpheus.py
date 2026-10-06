"""INTERFACE: synthesize(text) -> Path. Orpheus (canopylabs/orpheus-v1-english) hosted on Groq; uses GROQ_API_KEY."""
import logging
from datetime import datetime
from pathlib import Path

import httpx

from voicebot import config
from voicebot.config import ARTEFACTS_DIR

log = logging.getLogger("tts")
MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "hannah"
INFO = {"backend": "groq-orpheus", "location": "cloud", "model": f"{MODEL} / {DEFAULT_VOICE}"}


def load() -> None:
    """Fail early with a clear message if the key is missing; record the voice in INFO."""
    config.require("GROQ_API_KEY", config.GROQ_API_KEY)
    INFO["model"] = f"{MODEL} / {config.TTS_VOICE or DEFAULT_VOICE}"


def synthesize(text: str) -> Path:
    """Convert text to a wav and return its path."""
    key = config.require("GROQ_API_KEY", config.GROQ_API_KEY)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"orpheus_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
    try:
        r = httpx.post("https://api.groq.com/openai/v1/audio/speech", timeout=30, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "input": text, "voice": config.TTS_VOICE or DEFAULT_VOICE, "response_format": "wav"})
    except httpx.HTTPError as e:
        raise RuntimeError(f"Could not reach Groq ({type(e).__name__}: {e}). Check your internet connection.") from e
    if r.status_code != 200:
        hint = ("Your Groq org admin must accept the model terms at https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english"
                if "terms" in r.text else "Groq's free tier allows ~10 Orpheus requests per minute: wait a few seconds and retry, or use another --tts_model."
                if r.status_code == 429 else "Check GROQ_API_KEY, the voice name (--tts_voice) and your Groq rate limits.")
        raise RuntimeError(f"Groq TTS returned {r.status_code}: {r.text[:200]}. {hint}")
    path.write_bytes(r.content)
    return path
