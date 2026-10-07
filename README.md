# halfduplex-voice-chatbot

Press SPACE, talk, and the bot talks back. It remembers you across sessions (facts and a rolling summary are stored per user) and asks natural follow-up questions. It was built for older users, so replies are short, warm and plain-spoken.

```
mic ──► Silero VAD ──► ASR ──► LLM ──► TTS ──► speaker
        (ends on 1s     (speech   (reply +    (speech     (afplay)
         silence, trims  to text)  memory)     synthesis)
         non-speech)
```

Each stage is a swappable module with a one-function interface, and the per-turn latency of every stage is saved with the conversation.

## Local vs remote mode

The bot runs in one of two modes, chosen with `--mode` (default: **local**).

| | **local** (default) | **remote** |
|---|---|---|
| ASR | mlx-whisper, `whisper-base`, on the Mac GPU | Groq `whisper-large-v3-turbo` (cloud) |
| LLM | LFM2.5 1.2B (4-bit) via mlx-lm | any OpenRouter model (default Llama 3.3 70B on DeepInfra Turbo) |
| TTS | Piper `en_US-lessac-medium` on CPU | gTTS (Google Translate voice, free, no key; lowest latency measured) or edge-tts, ElevenLabs, Groq Orpheus |
| API keys | none | `GROQ_API_KEY`, `OPENROUTER_API_KEY` (plus `ELEVENLABS_API_KEY` only if you pick ElevenLabs) |
| Internet | only for the first-run downloads | every turn |
| Typical speed | fastest | slower |
| Typical quality | lower (a 1.2B-parameter LLM) | higher (a 70B-parameter LLM) |

> **The local setup is designed for Macs with Apple Silicon (M1 or later).** It uses MLX, which only runs on Apple Silicon, and `afplay` for playback. The local models were chosen to demonstrate a **low-latency pipeline** on modest hardware (developed and measured on an M2 MacBook Air with 8GB of RAM), not to give the best possible answers.
>
> **For better performance (answer quality), choose remote models.** The cost is computational time: remote calls add network and queueing delay, and larger models take longer to generate. Remote mode also needs API keys and costs money per request.

## Requirements

- **Local mode:** a Mac with Apple Silicon, a recent macOS (developed on Sequoia 15), **8GB RAM minimum** (see the memory note below), and about 3GB of free disk for the models.
- **Remote mode:** any machine that can run Python and has a microphone, plus the two API keys. Playback uses macOS `afplay`; on other systems swap `play()` in `src/voicebot/tts/edge_tts.py` for `ffplay` or `mpv`.
- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- A microphone and speakers.

## Installation (from scratch)

1. **Install uv** (the package manager) if you don't have it:
   ```bash
   brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone the repo and enter it:**
   ```bash
   git clone https://github.com/Trebolium/halfduplex-voice-chatbot
   cd halfduplex-voice-chatbot
   ```
3. **Install the dependencies.** For local mode (the default):
   ```bash
   uv sync --group local-asr --group local-llm --group local-tts
   ```
   For remote mode only, plain `uv sync` is enough. If you skip the groups, the bot installs any that are missing the first time you run a local mode, but doing it up front makes the first run clearer.
4. **Create your `.env`:**
   ```bash
   cp .env.example .env
   ```
   Local mode needs nothing in it. For remote mode, fill in `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)) and `OPENROUTER_API_KEY` ([openrouter.ai](https://openrouter.ai)).
5. **Grant macOS permissions** to the terminal (or IDE) you run it from, in *System Settings > Privacy & Security*: **Microphone**, plus **Accessibility** and **Input Monitoring** (needed to detect the SPACE key). Then restart that terminal.
6. **Run it:**
   ```bash
   uv run voicebot
   ```
   The **first local run downloads the models** and takes a few minutes. They are cached, so later runs start much faster:

   | Model | Size | Cached in |
   |---|---|---|
   | mlx-whisper base | ~150MB | `~/.cache/huggingface` |
   | LFM2.5 1.2B 4-bit | ~0.7GB | `~/.cache/huggingface` |
   | Piper voice | ~63MB | `models/` |

   On every local start the bot also **warms up** the models (about 20 seconds on an M2 Air). This moves the slow first-use cost out of your first sentence.

## Usage

```bash
uv run voicebot                                   # local mode: mlx-whisper + LFM2.5 1.2B + Piper
uv run voicebot --mode remote                     # Groq + OpenRouter + gTTS, models taken from .env
uv run voicebot --mode remote --llm_model google/gemini-2.5-flash --tts_model edge-tts --tts_voice en-GB-SoniaNeural
uv run voicebot --tts_model kokoro                # --tts_model works in local mode too
```

You will be asked for your name (it picks up your saved memory). Press **SPACE**, speak, and stop talking: one second of silence ends your turn. The bot replies out loud and then listens for your answer without needing SPACE again. After about 5 seconds of silence the session ends, and the bot updates its memory about you.

### Options

| Option | Applies to | Meaning |
|---|---|---|
| `--mode local\|remote` | both | Which stack to run. Default `local` (or `MODE` in `.env`). |
| `--asr_model NAME` | remote only | Groq transcription model. Default `ASR_MODEL` in `.env` (`whisper-large-v3-turbo`). |
| `--llm_model SLUG` | remote only | OpenRouter model slug, e.g. `google/gemini-2.5-flash`. Default `LLM_MODEL` in `.env`. |
| `--llm_provider TAG` | remote only | Pin an OpenRouter provider, e.g. `deepinfra/turbo`. Default `LLM_PROVIDER` in `.env`; empty means OpenRouter chooses. |
| `--tts_model NAME` | **both** | TTS backend. Local: `piper` (default), `pocket-tts`, `kokoro`, `kitten-tts`. Remote: `gtts` (default), `edge-tts`, `groq-orpheus`, `elevenlabs`. Defaults come from `LOCAL_TTS` / `REMOTE_TTS` in `.env`. Choosing a remote backend in local mode (or the reverse) gives a clear error. |
| `--tts_voice NAME` | remote only | Voice for the chosen remote TTS: an edge-tts voice (e.g. `en-GB-SoniaNeural`), an Orpheus voice (e.g. `hannah`), or an ElevenLabs voice id. Empty = the backend's default. Default `TTS_VOICE` in `.env`. |

- **In local mode `--asr_model`, `--llm_model`, `--llm_provider` and `--tts_voice` are ignored**, and the bot prints a warning if you pass any. `--tts_model` is honoured in both modes.
- If you pass `--llm_model` without `--llm_provider`, the provider pin from `.env` is dropped, because a pin only makes sense for the model it was chosen for.
- Remote mode checks both API keys at startup and tells you which one is missing.

### `.env` settings

| Variable | Used in | Default |
|---|---|---|
| `MODE` | both | `local` |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY` | remote | none |
| `ASR_MODEL` | remote | `whisper-large-v3-turbo` |
| `LLM_MODEL` | remote | `meta-llama/llama-3.3-70b-instruct` |
| `LLM_PROVIDER` | remote | `deepinfra/turbo` |
| `TTS_VOICE` | remote | empty (each backend's own default voice) |
| `LOCAL_TTS` | local | `piper` (also `pocket-tts`, `kokoro`, `kitten-tts`) |
| `REMOTE_TTS` | remote | `gtts` (also `edge-tts`, `groq-orpheus`, `elevenlabs`) |
| `ELEVENLABS_API_KEY` | remote, only for ElevenLabs | none |

## How the pieces work

- **Voice activity detection:** [Silero VAD](https://github.com/snakers4/silero-vad) ends your recording on silence and trims non-speech before ASR. If no speech is found, ASR is skipped entirely, which stops Whisper inventing text from silence or noise.
- **Whisper settings (local):** previous text is not fed back in, and low-confidence, repetitive or silent output is dropped, to reduce hallucinations.
- **Memory:** `data/users/<name>.json` holds your facts, a rolling summary and every turn. At the end of a session the LLM extracts new facts and updates the summary, and both go into the next session's system prompt. In local mode the 1.2B model does this extraction, which it does less reliably than a large model. If its output can't be parsed, the old memory is kept.
- **Text cleanup:** replies are stripped of markdown and emoji, since they are spoken aloud.

### What each turn records

Every turn in the user JSON stores, next to the text:

```json
"latency_s": {"vad": 0.22, "asr": 1.55, "llm": 1.48, "tts": 0.89},
"setup": {"asr": {"backend": "mlx-whisper", "model": "...", "location": "local-apple-gpu"},
          "llm": {...}, "tts": {...}, "vad": {...}, "config": {...}}
```

- `vad`: time to trim the recording. `asr`: the transcription call. `llm`: transcript in to full reply out. `tts`: reply returned to audio file written.
- `setup` records the backends, models, provider, local or cloud, and the non-secret config used for that turn. API keys are never saved.

## Performance (measured on an M2 MacBook Air, 8GB)

All figures are seconds, measured with the scripts in `evaluate/`. The remote numbers vary with the network and provider load.

| Stage | Local | Remote |
|---|---|---|
| ASR (12s clip) | ~1.6–2.7 (mlx-whisper) | ~0.5–0.7 (Groq) |
| LLM, full reply | ~0.6–0.9 (LFM2.5 1.2B) | ~2.6 (Llama 3.3 70B on DeepInfra Turbo); ~10 (Gemini 3.8 Flash, with large spread) |
| TTS, 200 chars | ~0.35 (Piper) | ~0.55–0.7 (gTTS, edge-tts, ElevenLabs) |

**Remote TTS comparison** (`evaluate/tts_remote_latency.py`, 10 runs each, mean seconds; latest complete run):

| Backend | Short (57 chars) | Long (201 chars) | Average | Notes |
|---|---|---|---|---|
| **gTTS** (default) | **0.22** | 0.69 | **0.45** | free, no key, unofficial Google endpoint |
| ElevenLabs Flash v2.5 | 0.37 | **0.61** | 0.49 | free tier, needs `ELEVENLABS_API_KEY` (default voice "George") |
| edge-tts | 0.43 | **0.56** | 0.49 | free, no key, Microsoft neural voices |
| Groq Orpheus | 0.99 | 2.16 | 1.58 | measured in an earlier run; see below |

- **gTTS is the default** because it has the lowest average latency, but the three leaders are within a few hundredths of a second of each other on average, and the order changed between runs (edge-tts won two earlier runs). gTTS is quickest on short replies, while edge-tts and ElevenLabs are quickest on long ones. Treat them as roughly equal on speed and pick on voice quality and reliability.
- gTTS uses Google's unofficial free endpoint, so it can rate-limit or change without notice, and its voice is generally regarded as more basic than the neural voices (a judgement from reputation; no listening test was done). Try `--tts_model edge-tts` or `elevenlabs` if the voice matters more than the last tenth of a second.
- **ElevenLabs** worked with its default voice. Its free tier has a monthly character allowance, so heavy use will run out.
- **Groq Orpheus is the slowest and not practical on the free tier:** it allows about 10 requests per minute and only **3,600 tokens per day**, which the benchmark itself used up. Its numbers come from one earlier run (`tts_remote_orpheus_earlier_run.json`), and it is absent from the latest chart and JSON because the daily quota was exhausted.
- A single connection drop to ElevenLabs happened once during benchmarking, so that backend retries once automatically.

Other local TTS models: Pocket TTS ~1.9s, Kokoro ~3.5s, Kitten TTS mini ~4.0s for 200 characters. Piper was 3–10x faster than all three. Charts and raw data are in `evaluate/results/`.

Things to know:
- **Speed vs quality:** the local LLM is much faster but noticeably weaker than the 70B remote model. In spot checks it sometimes invented names or details and didn't always ask a follow-up question.
- **Memory on 8GB:** all three local models fit together, but the machine can start swapping if it is already busy. Close other heavy apps, and expect the warm-up at startup to be slower in that case.
- **No streaming yet:** each stage finishes completely before the next starts, so the latencies above add up.

## Project layout

```
src/voicebot/
  main.py          CLI, mode selection, the turn loop
  config.py        .env loading and constants
  setup_info.py    builds the per-turn setup metadata
  input/           recorder.py: mic capture, SPACE to start, Silero endpointing
  asr/             __init__.py dispatcher; mlx_whisper.py (local), groq_asr.py (remote), vad.py (Silero)
  llm/             __init__.py dispatcher; mlx_lm.py (local), openrouter.py (remote)
  tts/             __init__.py dispatcher; piper.py, pocket.py, kokoro.py, kitten.py (local); google_tts.py, edge_tts.py, groq_orpheus.py, elevenlabs.py (remote)
  memory/          store.py: per-user JSON memory
test/              pytest suite; test_data/ has sample .wav files with matching .txt transcripts
evaluate/          latency scripts and results (charts and JSON)
models/            downloaded Piper and Kokoro model files (git-ignored)
data/users/        saved user memory (git-ignored)
artefacts/         recorded and generated audio (git-ignored)
```

### Swapping or adding a component

Each stage has a dispatcher (`__init__.py`) with a registry of backends. To add one, write a file with an `INFO` dict (backend name, model, `cloud` or `local`) and the stage's function, then register it:

- ASR: `transcribe(wav_path) -> str`
- LLM: `reply(system_prompt, history, user_text) -> str` and `complete(system_prompt, user_text) -> str`
- TTS: `synthesize(text) -> Path`

An optional `load()` is called once at startup (download and load the model there).

## Testing and evaluation

```bash
uv run pytest -q          # unit tests; live tests use your keys if present and skip if not
```

The suite includes a live end-to-end pipeline test (it needs the API keys and may hit transient provider rate limits; rerun if it does). The benchmark scripts in `evaluate/` (for example `uv run python evaluate/tts_top4_latency.py`) need the `bench` group: `uv sync --group bench --group dev --group local-asr --group local-llm --group local-tts`.

> **uv tip:** a plain `uv sync --group X` removes packages that belong to other groups. List every group you want in one command, as above.

Docker: `docker build -t voicebot . && docker run voicebot` runs the tests only (containers have no microphone or speakers, and MLX needs a Mac).

## Troubleshooting

- **"Local mode uses MLX, which needs an Apple Silicon Mac":** use `--mode remote`.
- **SPACE does nothing or the keyboard listener dies:** grant Accessibility and Input Monitoring to your terminal, then restart it.
- **"Could not open microphone":** grant Microphone permission and check the input device.
- **First local run seems stuck:** it is downloading about 2.4GB; run again if the connection dropped (downloads resume or restart cleanly).
- **First turn is slow in local mode:** the machine is probably low on memory; close other apps.
- **Remote 404 / "model not found":** check the model slug, and clear `LLM_PROVIDER` if you pinned a provider that doesn't host that model.
- **Remote 429 "rate-limited":** the pinned provider is overloaded; retry, or clear `LLM_PROVIDER` so OpenRouter can use another host.
- **Odd or missing text in a transcript:** check the `[vad]` lines in the terminal; if VAD found no speech, ASR is skipped.

## To do

- [x] Make ASR more robust: local/remote ASR option and Silero VAD (done). Still to try: Parakeet and Moonshine.
- [ ] Listen to the remote voices (gTTS, edge-tts, ElevenLabs) before settling the remote default: the three are within noise on latency, so voice quality should decide.
- [ ] Ensure any failings or bad connections are explicitly reported in the terminal logs.
- [ ] Try TTS models that use their own integrated LLM to tackle latency.
- [ ] Listening test of the local TTS voices (Piper vs Pocket vs Kokoro): only latency has been measured so far.
- [ ] Make use of ASR streaming suggestions as described in a separate Claude chat.
- [ ] Build a set of realistic conversation prompts to compare local LLMs' quality (use of the user's name, plain spoken text, follow-up questions) before choosing a local default.
