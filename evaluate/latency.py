"""Measures per-stage latency (ASR, LLM, TTS) on a wav file: uv run python evaluate/latency.py <file.wav>"""
import sys
import time
from pathlib import Path

from voicebot.asr.groq_asr import transcribe
from voicebot.llm.openrouter import reply
from voicebot.memory.store import build_system_prompt, load
from voicebot.tts.edge_tts import synthesize

wav = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit("Usage: latency.py <file.wav>")
t = time.time(); text = transcribe(wav); asr = time.time() - t
t = time.time(); ans = reply(build_system_prompt(load("eval_user")), [], text); llm = time.time() - t
t = time.time(); synthesize(ans); tts = time.time() - t
print(f"ASR {asr:.2f}s | LLM {llm:.2f}s | TTS {tts:.2f}s | total {asr + llm + tts:.2f}s")
