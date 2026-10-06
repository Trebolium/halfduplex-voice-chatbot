"""INTERFACE: synthesize(text) -> Path. Piper (en_US-lessac-medium) on CPU."""
import wave
from datetime import datetime
from pathlib import Path

from voicebot.config import ARTEFACTS_DIR, ROOT

MODEL = ROOT / "models" / "en_US-lessac-medium.onnx"
INFO = {"backend": "piper", "location": "local-cpu", "model": "en_US-lessac-medium"}
_voice = None
BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"


def _fetch(url: str, dest) -> None:
    """Download with Range-resume: the HF stream sometimes ends early, so keep going until the size matches."""
    import http.client
    import urllib.request
    expected = None
    dest.write_bytes(b"")
    for attempt in range(8):
        have = dest.stat().st_size
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req) as r, open(dest, "ab") as out:
                if expected is None:
                    expected = have + int(r.headers["Content-Length"])
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        except http.client.IncompleteRead as e:  # connection dropped mid-stream: bytes read so far are lost, so resume
            dest.write_bytes(dest.read_bytes() + e.partial)
        if dest.stat().st_size == expected:
            return
        print(f"[tts] download stopped at {dest.stat().st_size}/{expected} bytes, resuming ({attempt + 1}/8)...")
    raise IOError(f"got {dest.stat().st_size} of {expected} bytes")


def _download() -> None:
    """Fetch the voice + config, checking sizes (piper's own downloader returned a corrupt model here)."""
    MODEL.parent.mkdir(exist_ok=True)
    print("[tts] downloading Piper voice en_US-lessac-medium (first run, ~63MB)...")
    for f in (MODEL, MODEL.with_suffix(".onnx.json")):
        try:
            _fetch(f"{BASE}/{f.name}", f)
        except Exception as e:
            MODEL.unlink(missing_ok=True); f.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {f.name} ({e}). Check your connection and retry, or fetch it from {BASE} into models/.") from e


def load():
    """Load the voice once (kept separate so benchmarks can exclude it)."""
    global _voice
    if _voice is None:
        if not MODEL.exists():
            _download()
        from piper import PiperVoice
        _voice = PiperVoice.load(str(MODEL))
    return _voice


def synthesize(text: str) -> Path:
    """Convert text to a wav and return its path."""
    voice = load()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"piper_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
    with wave.open(str(path), "wb") as f:
        voice.synthesize_wav(text, f)
    return path
