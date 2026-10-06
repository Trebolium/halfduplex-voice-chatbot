"""Real end-to-end test (needs keys + network + macOS afconvert): spoken audio -> VAD -> Groq -> LLM -> TTS -> memory."""
import json
import subprocess
from pathlib import Path

import pytest

from voicebot import config
from voicebot.asr.groq_asr import transcribe
from voicebot.asr.vad import trim_speech
from voicebot.llm.openrouter import reply
from voicebot.memory import store
from voicebot.tts.edge_tts import synthesize

pytestmark = pytest.mark.skipif(
    not (config.GROQ_API_KEY and config.OPENROUTER_API_KEY), reason="API keys not set in .env"
)


def _speech_wav(text: str) -> Path:
    """Make a 16kHz mono wav of spoken text (stands in for the mic)."""
    mp3 = synthesize(text)
    wav = mp3.with_suffix(".wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(mp3), str(wav)], check=True)
    return wav


def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USERS_DIR", tmp_path)
    user = store.load("Margaret")
    system = store.build_system_prompt(user)
    history: list[dict] = []

    for said in ["Hello, my name is Margaret and I love gardening. My roses are blooming.",
                 "Yes, I planted them with my late husband."]:
        print(f"\n[E2E] user says: {said}")
        trimmed = trim_speech(_speech_wav(said))
        assert trimmed is not None, "VAD removed all speech"
        text = transcribe(trimmed)
        print(f"[E2E] ASR heard: {text}")
        assert text
        answer = reply(system, history, text)
        print(f"[E2E] bot: {answer}")
        assert answer and synthesize(answer).stat().st_size > 1000
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]

    store.end_session(user, history)
    saved = json.loads((tmp_path / "margaret.json").read_text())
    print(f"[E2E] saved memory: {saved}")
    assert saved["sessions"] == 1 and saved["facts"], "memory extraction produced no facts"
