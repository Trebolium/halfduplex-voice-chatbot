"""Central config: loads .env and holds constants shared by all modules."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
ARTEFACTS_DIR = ROOT / "artefacts"
USERS_DIR = ROOT / "data" / "users"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")  # optional OpenRouter provider tag to pin, e.g. deepinfra/turbo
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-large-v3-turbo")
TTS_VOICE = os.getenv("TTS_VOICE", "")  # optional voice override for the remote TTS backend (each backend has its own default)

SAMPLE_RATE = 16000          # Hz, mono int16 everywhere
MAX_RECORD_SECS = 120        # hard cap per recording
SILENCE_END_SECS = 1.0       # consecutive silence that ends a recording
MIN_SPEECH_MS = 300          # VAD drops speech segments shorter than this
VAD_THRESHOLD = 0.5          # Silero speech probability above this counts as speech
IDLE_TIMEOUT_SECS = 5        # no speech this long after a reply -> session ends and memory is saved

BASE_SYSTEM_PROMPT = (
    "You are a helpful voice AI assistant designed to assist the elderly. "
    "Be kind, engaging, and provide short responses, unless explicitly asked to elabourate."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")


def require(key_name: str, value: str) -> str:
    """Raise a clear error if an API key is missing."""
    if not value:
        raise RuntimeError(f"{key_name} is not set. Add it to the .env file in the project root (see .env.example).")
    return value
