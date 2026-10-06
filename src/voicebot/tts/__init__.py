"""TTS dispatcher: select_backend(name) picks the implementation; synthesize()/play() call it.

To add a backend: write tts/<name>.py with INFO + synthesize(text) -> Path (+ optional load()), then register it below.
"""
import importlib
import subprocess

from voicebot.tts.edge_tts import play  # noqa: F401  (afplay playback works for the mp3/wav of every backend)

# CLI name -> (module, uv dependency group needed or None, python package to check)
BACKENDS = {
    "piper": ("voicebot.tts.piper", "local-tts", "piper"),                     # fastest on CPU (~0.1-0.4s)
    "pocket-tts": ("voicebot.tts.pocket", "local-tts", "pocket_tts"),          # better quality, ~0.8-2s
    "kokoro": ("voicebot.tts.kokoro", "local-tts", "kokoro_onnx"),             # best-sounding small model, ~1-3.5s
    "kitten-tts": ("voicebot.tts.kitten", "local-tts", "espeakng_loader"),     # tiny, ~1.3-4s
    "gtts": ("voicebot.tts.google_tts", None, None),                           # remote, free, no key (fastest remote measured)
    "edge-tts": ("voicebot.tts.edge_tts", None, None),                         # remote Microsoft voices, free, no key
    "groq-orpheus": ("voicebot.tts.groq_orpheus", None, None),                 # remote, GROQ_API_KEY + accept model terms in the Groq console
    "elevenlabs": ("voicebot.tts.elevenlabs", None, None),                     # remote, ELEVENLABS_API_KEY (free tier)
}
LOCAL_BACKENDS = ["piper", "pocket-tts", "kokoro", "kitten-tts"]               # choices for --tts_model in local mode
REMOTE_BACKENDS = ["gtts", "edge-tts", "groq-orpheus", "elevenlabs"]           # choices for --tts_model in remote mode
DEFAULT = "piper"                                                              # local default
REMOTE_DEFAULT = "gtts"                                                        # lowest mean latency in evaluate/tts_remote_latency.py
_current = None


def _ensure_deps(name: str, group: str, package: str) -> None:
    """Install the uv dependency group for a local backend if its package is missing."""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"[tts] '{name}' needs the '{group}' dependencies - installing (uv sync --inexact --group {group})...")
        try:
            subprocess.run(["uv", "sync", "--inexact", "--group", group], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"Could not install the '{group}' dependencies ({e}). Run `uv sync --group {group}` manually "
                               "or use --tts_model edge-tts.") from e
        importlib.invalidate_caches()


def select_backend(name: str):
    """Activate a backend by CLI name (installing deps / downloading models first) and return its module."""
    global _current
    if name not in BACKENDS:
        raise ValueError(f"Unknown tts_model '{name}'. Choose one of: {', '.join(BACKENDS)}.")
    module, group, package = BACKENDS[name]
    print(f"[tts] backend: {name}")
    if group:
        _ensure_deps(name, group, package)
    _current = importlib.import_module(module)
    if hasattr(_current, "load"):
        _current.load()
    return _current


def current():
    """The active backend module (defaults to piper)."""
    return _current or select_backend(DEFAULT)


def synthesize(text):
    """Convert text to an audio file with the active backend and return its path."""
    return current().synthesize(text)
