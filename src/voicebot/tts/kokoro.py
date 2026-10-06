"""INTERFACE: synthesize(text) -> Path. Kokoro-82M via kokoro-onnx (CPU)."""
from pathlib import Path

from voicebot.config import ROOT
from voicebot.tts._common import save_wav

MODEL = ROOT / "models" / "kokoro-v1.0.onnx"
VOICES = ROOT / "models" / "voices-v1.0.bin"
VOICE = "af_heart"
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
INFO = {"backend": "kokoro-onnx", "location": "local-cpu", "model": f"kokoro-v1.0 / {VOICE}"}
_model = None


def load():
    """Load the model once (kept separate so benchmarks can exclude it)."""
    global _model
    if _model is None:
        if not MODEL.exists() or not VOICES.exists():
            import urllib.request
            MODEL.parent.mkdir(exist_ok=True)
            for f in (MODEL, VOICES):
                print(f"[tts] downloading {f.name} (first run, ~350MB total)...")
                try:
                    urllib.request.urlretrieve(f"{RELEASE}/{f.name}", f)
                except Exception as e:
                    f.unlink(missing_ok=True)
                    raise RuntimeError(f"Could not download {f.name} ({e}). Check your connection, or download it from {RELEASE} into models/.") from e
        from kokoro_onnx import Kokoro
        _model = Kokoro(str(MODEL), str(VOICES))
    return _model


def synthesize(text: str) -> Path:
    """Convert text to a wav and return its path."""
    samples, sr = load().create(text, voice=VOICE, speed=1.0, lang="en-us")
    return save_wav(samples, sr, "kokoro")
