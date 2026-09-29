"""INTERFACE: transcribe(wav_path) -> str using Groq whisper turbo."""
import logging
import time
from pathlib import Path

from groq import Groq

from voicebot import config

log = logging.getLogger("asr")


def transcribe(wav_path: Path) -> str:
    """Send wav to Groq Whisper and return stripped English text."""
    key = config.require("GROQ_API_KEY", config.GROQ_API_KEY)
    print(f"[asr] transcribing {Path(wav_path).name} with {config.ASR_MODEL}")
    t0 = time.time()
    try:
        with open(wav_path, "rb") as f:
            res = Groq(api_key=key).audio.transcriptions.create(
                file=(Path(wav_path).name, f.read()), model=config.ASR_MODEL, language="en"
            )
    except Exception as e:
        raise RuntimeError(
            f"Groq transcription failed ({type(e).__name__}: {e}). Check GROQ_API_KEY is valid, "
            "your network connection, and that ASR_MODEL exists."
        ) from e
    text = (res.text or "").strip()
    log.info("ASR done in %.2fs: %r", time.time() - t0, text)
    print(f"[asr] {time.time() - t0:.2f}s -> {text!r}")
    return text
