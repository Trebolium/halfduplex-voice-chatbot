"""ASR dispatcher: select_backend(name) picks the implementation; transcribe(wav) calls it.

To add a backend: write asr/<name>.py with INFO + transcribe(wav_path) -> str (+ optional load()), then register it below.
"""
import importlib
import subprocess

# CLI name -> (module, uv dependency group needed or None)
BACKENDS = {
    "remote-whisper": ("voicebot.asr.groq_asr", None),         # Groq whisper-large-v3-turbo (cloud)
    "local-whisper": ("voicebot.asr.mlx_whisper", "local-asr"),  # mlx-whisper (Apple Silicon, local)
}
DEFAULT = "remote-whisper"
_current = None


def _ensure_deps(name: str, group: str) -> None:
    """Install the uv dependency group for a local backend if its package is missing."""
    try:
        importlib.import_module("mlx_whisper")
    except ImportError:
        print(f"[asr] '{name}' needs the '{group}' dependencies - installing (uv sync --inexact --group {group})...")
        try:
            subprocess.run(["uv", "sync", "--inexact", "--group", group], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"Could not install the '{group}' dependencies ({e}). Run `uv sync --group {group}` manually "
                               "or use --asr_model remote-whisper.") from e
        importlib.invalidate_caches()


def select_backend(name: str):
    """Activate a backend by CLI name (installing/downloading local deps first) and return its module."""
    global _current
    if name not in BACKENDS:
        raise ValueError(f"Unknown asr_model '{name}'. Choose one of: {', '.join(BACKENDS)}.")
    module, group = BACKENDS[name]
    print(f"[asr] backend: {name}")
    if group:
        _ensure_deps(name, group)
    _current = importlib.import_module(module)
    if hasattr(_current, "load"):
        _current.load()
    return _current


def current():
    """The active backend module (defaults to remote-whisper)."""
    return _current or select_backend(DEFAULT)


def transcribe(wav_path):
    """Transcribe with the active backend."""
    return current().transcribe(wav_path)
