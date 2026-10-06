"""INTERFACE: synthesize(text) -> Path. KittenTTS mini-0.8 (80M, ONNX, CPU); runs the HF ONNX model directly (PyPI kittentts is outdated)."""
import re
from pathlib import Path

import numpy as np

from voicebot.tts._common import save_wav

REPO = "KittenML/kitten-tts-mini-0.8"
VOICE = "expr-voice-5-f"
SAMPLE_RATE = 24000
SYMBOLS = list('$;:,.!?¡¿—…"«»"" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ')
INDEX = {s: i for i, s in enumerate(SYMBOLS)}
INFO = {"backend": "kitten-tts", "location": "local-cpu", "model": f"{REPO} / {VOICE}"}
_state = None


def load():
    """Load the ONNX session, voices and phonemizer once (kept separate so benchmarks can exclude it)."""
    global _state
    if _state is None:
        try:
            import espeakng_loader
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as e:
            raise RuntimeError(f"{e}. Run: uv sync --group bench") from e
        EspeakWrapper.set_library(espeakng_loader.get_library_path())  # bundled espeak, no brew needed
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        sess = ort.InferenceSession(hf_hub_download(REPO, "kitten_tts_mini_v0_8.onnx"))
        voices = np.load(hf_hub_download(REPO, "voices.npz"))[VOICE]
        ph = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        _state = (sess, voices, ph)
    return _state


def synthesize(text: str) -> Path:
    """Convert text to a wav and return its path."""
    sess, voices, ph = load()
    phones = " ".join(re.findall(r"\w+|[^\w\s]", ph.phonemize([text])[0]))
    ids = np.array([[0] + [INDEX[c] for c in phones if c in INDEX] + [0]], dtype=np.int64)
    style = voices[min(len(text), len(voices) - 1)][None]  # style row is picked by text length
    audio = sess.run(None, {"input_ids": ids, "style": style, "speed": np.array([1.0], dtype=np.float32)})[0]
    return save_wav(audio[:-5000], SAMPLE_RATE, "kitten")  # model emits trailing silence/noise
