# HL-task: conversational voice bot

Press SPACE, talk, and the bot replies out loud, remembers you across sessions, and asks follow-up questions.

Pipeline: mic (spacebar, 3s silence ends) -> VAD trim -> Groq Whisper turbo -> OpenRouter LLM -> edge-tts -> playback -> repeat.

## Run
1. `cp .env.example .env` and fill in `GROQ_API_KEY`, `OPENROUTER_API_KEY` (set `LLM_MODEL` to the OpenRouter slug).
2. `uv sync`
3. `uv run voicebot` (macOS: grant your terminal Microphone + Accessibility permission).

## Layout
- `src/voicebot/input` mic + spacebar recorder
- `src/voicebot/asr` VAD + Groq transcription
- `src/voicebot/llm` OpenRouter client
- `src/voicebot/tts` edge-tts + playback
- `src/voicebot/memory` per-user memory (`data/users/<name>.json`)
- `test/` pytest, `evaluate/` latency script. Swap any module by replacing its file, keeping the function signature.

Docker: `docker build -t voicebot . && docker run voicebot` runs the tests (no audio devices in-container).
