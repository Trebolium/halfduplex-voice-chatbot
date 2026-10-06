"""INTERFACE: synthesize(text) -> Path. Kyutai Pocket TTS on CPU."""
from pathlib import Path

from voicebot.tts._common import save_wav

VOICE = "alba"
INFO = {"backend": "pocket-tts", "location": "local-cpu", "model": f"pocket-tts / {VOICE}"}
_model = _state = None


def load():
    """Load the model and voice state once (kept separate so benchmarks can exclude it)."""
    global _model, _state
    if _model is None:
        from pocket_tts import TTSModel
        _model = TTSModel.load_model()
        _state = _model.get_state_for_audio_prompt(VOICE)
    return _model


def synthesize(text: str) -> Path:
    """Convert text to a wav and return its path."""
    model = load()
    audio = model.generate_audio(_state, text, max_tokens=1000)
    return save_wav(audio.numpy(), model.sample_rate, "pocket")
