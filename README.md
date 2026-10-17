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
| LLM | Llama 3.2 3B Instruct (4-bit) via mlx-lm | any OpenRouter model (default Llama 3.3 70B on DeepInfra Turbo) |
| TTS | Piper `en_US-lessac-medium` on CPU | gTTS (Google Translate voice, free, no key; lowest latency measured) or edge-tts, ElevenLabs, Groq Orpheus |
| API keys | none | `GROQ_API_KEY`, `OPENROUTER_API_KEY` (plus `ELEVENLABS_API_KEY` only if you pick ElevenLabs) |
| Internet | only for the first-run downloads | every turn |
| Typical speed | fastest | slower |
| Typical quality | lower (a 3B-parameter LLM) | higher (a 70B-parameter LLM) |

> **The local setup is designed for Macs with Apple Silicon (M1 or later).** It uses MLX, which only runs on Apple Silicon, and `afplay` for playback. The local models were chosen to demonstrate a **low-latency pipeline** on modest hardware (developed and measured on an M2 MacBook Air with 8GB of RAM), not to give the best possible answers.
>
> **For better performance (answer quality), choose remote models.** The cost is computational time: remote calls add network and queueing delay, and larger models take longer to generate. Remote mode also needs API keys and costs money per request.

## Requirements

- **Local mode:** a Mac with Apple Silicon, a recent macOS (developed on Sequoia 15), **8GB RAM minimum** (see the memory note below), and about 4GB of free disk for the models.
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
   | Llama 3.2 3B 4-bit | ~1.7GB | `~/.cache/huggingface` |
   | Piper voice | ~63MB | `models/` |

   On every local start the bot also **warms up** the models (about 20 seconds on an M2 Air). This moves the slow first-use cost out of your first sentence.

## Usage

```bash
uv run voicebot                                   # local mode: mlx-whisper + Llama 3.2 3B + Piper
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
| `--test_start STAGE` | both | Test run: start at `VAD`, `ASR`, `LLM` or `TTS` and go to the end. See [Testing a slice of the pipeline](#testing-a-slice-of-the-pipeline). |
| `--test_end STAGE` | both | Test run: stop after this stage (inclusive); starts at `VAD` unless `--test_start` is also given. |
| `--test_audio FILE.wav` | test runs | Input for a `VAD` or `ASR` start (replaces the microphone). |
| `--test_text FILE.txt` | test runs | Input for an `LLM` or `TTS` start. |

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
- **Memory:** `data/users/<name>/<name>.json` holds your facts, a rolling summary and every turn. At the end of a session the LLM extracts new facts and updates the summary, and both go into the next session's system prompt. In local mode the small local model does this extraction, which it does less reliably than a large model. If its output can't be parsed, the old memory is kept.
- **Per-user files:** each user has a folder, and every turn's files are stored there once the reply has finished playing:
  ```
  data/users/<name>/
    <name>.json     memory and turn log
    audio/          input_<time>.wav (your recording), input_<time>_trimmed.wav (after VAD), and the bot's reply audio (e.g. piper_<time>.wav)
    text/           input_<time>.txt (what ASR heard) and <reply audio name>.txt (the bot's reply), named after the audio they belong to
  ```
  Recordings with no speech, and anything produced outside a user session (benchmarks, tests), stay in `artefacts/`. If a turn fails before it finishes, its files also stay in `artefacts/`.
- **Text cleanup:** replies are stripped of markdown and emoji, since they are spoken aloud.

### What each turn records

Every turn in the user JSON stores, next to the text:

```json
"latency_s": {"vad": 0.22, "asr": 1.55, "llm": 1.48, "tts": 0.89},
"setup": {"asr": {"backend": "mlx-whisper", "model": "...", "location": "local-apple-gpu"},
          "llm": {...}, "tts": {...}, "vad": {...}, "config": {...}}
```

- `vad`, `asr`, `llm` and `tts` are the per-stage latencies defined in [Latency: what is measured](#latency-what-is-measured) below.
- `setup` records the backends, models, provider, local or cloud, and the non-secret config used for that turn. API keys are never saved.

## Latency: what is measured

"Latency" here means the **wall-clock time one pipeline stage takes** to turn its input into its output, in seconds. Each stage is timed separately, from the moment the previous stage hands over its result to the moment this stage returns its own. Nothing is timed across stages, so the figures add up but overlap nothing (there is no streaming).

```
 you speak ─► 1s of silence ─► [ VAD ] ─► [ ASR ] ─► [ LLM ] ─► [ TTS ] ─► playback
 (not timed)   (not timed)     vad       asr         llm         tts       (not timed)
```

| Stage | Timer starts | Timer stops | Includes |
|---|---|---|---|
| **vad** | The recording has finished (the 1 second of silence that ends your turn has already passed) and the saved audio file is about to be read. | The speech-only audio has been written to a new file, or VAD has decided there is no speech. | Reading the file, running Silero VAD, cutting out the non-speech parts, and writing the trimmed file. |
| **asr** | The trimmed audio is ready to be transcribed. | The transcript text has been returned. | Local: running Whisper on the Mac GPU. Remote: uploading the audio to Groq, Groq's processing, and the response coming back. Skipped (not recorded) if VAD found no speech. |
| **llm** | The transcript text has been returned by ASR. | The complete reply text has been returned. | Assembling the message list from the (already built) system prompt, history and transcript, the model generating the whole reply, the markdown/emoji cleanup of the reply, and, in remote mode, the network round trip to OpenRouter and the provider's queueing time. It is the time to the **full** reply, not to the first word. |
| **tts** | The complete reply text has been returned by the LLM. | The audio file for the reply has been written to disk and is ready to play. | Speech synthesis only (local: on the CPU; remote: the network round trip to the TTS service). Writing the audio file is included. |

**Not included in any of these figures:**
- Waiting for you to press SPACE or start talking, and the time you spend speaking.
- The 1 second of silence the recorder waits for before deciding you have finished.
- **Playback** of the reply. The timer stops when the audio file exists, not when you hear it.
- Model loading and warm-up at startup.
- The end-of-session memory update (the LLM call that extracts facts and updates the summary).

**What you actually wait for.** The delay from the moment you stop talking to the moment the bot starts speaking is about `1s silence + vad + asr + llm + tts` (plus a little for starting playback). The saved per-turn numbers cover everything except the 1s silence and the playback start.

**How the benchmarks in `evaluate/` measure.** They call the same stage functions as the bot, with the same start and stop points as the table above, and use a stopwatch around each call:
- **ASR:** a recorded `.wav` file in, transcript out. The clips are sent as they are, without the VAD trim step, and the clip length is given with each result (for example "12s clip").
- **LLM:** starts when the transcript has just been produced and stops when the full reply is returned. The transcript is generated once per clip, so every model gets the same input. One untimed warm-up call is made first to set up the connection.
- **TTS:** starts when the LLM reply has just been returned and stops when the audio file is written. Texts are 57 and 201 characters long. One untimed warm-up call is made first (it covers connection setup and, for local models, first-use costs), and the results are the mean of 10 runs. Model load time is reported separately where it is measured, not counted in these figures.
- Remote results depend on your network and the provider's load at that moment, so they vary from run to run.

## Performance (measured on an M2 MacBook Air, 8GB)

All figures are seconds, measured with the scripts in `evaluate/`. The remote numbers vary with the network and provider load.

| Stage | Local | Remote |
|---|---|---|
| ASR (12s clip) | ~1.6–2.7 (mlx-whisper) | ~0.5–0.7 (Groq) |
| LLM, full reply | ~1.8–1.9 (Llama 3.2 3B); ~0.6–0.9 for the 1B-class models | ~2.6 (Llama 3.3 70B on DeepInfra Turbo); ~10 (Gemini 3.8 Flash, with large spread) |
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

**Local LLM comparison** (`evaluate/llm_follow.py`; 6 prompts x 2 runs per model through the real system prompt; checks are pass/fail against the style rules, so treat them as a rough ranking from a small sample):

| Model | All checks passed | Mean reply (s) | Peak memory |
|---|---|---|---|
| LFM2.5 1.2B | 42% | 0.93 | 0.87 GB |
| Gemma 3 1B | 42% | 0.84 | 0.91 GB |
| Llama 3.2 1B | 33% | 1.31 | 0.94 GB |
| Qwen3 1.7B | 58% | 1.09 | 1.25 GB |
| **Llama 3.2 3B** (current default) | 58% | 2.72 | 2.19 GB |

Qwen3 1.7B scored the same as Llama 3B at about 40% of the latency and half the memory, so it is the next model worth trying live. Llama 3B at 2.2GB peak leaves little headroom on an 8GB machine (see [Memory pressure](#memory-pressure-on-8gb)).

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
  pipeline_test.py once-off test runs of any slice of VAD/ASR/LLM/TTS (--test_start/--test_end)
  input/           recorder.py: mic capture, SPACE to start, Silero endpointing
  asr/             __init__.py dispatcher; mlx_whisper.py (local), groq_asr.py (remote), vad.py (Silero)
  llm/             __init__.py dispatcher; mlx_lm.py (local), openrouter.py (remote)
  tts/             __init__.py dispatcher; piper.py, pocket.py, kokoro.py, kitten.py (local); google_tts.py, edge_tts.py, groq_orpheus.py, elevenlabs.py (remote)
  memory/          store.py: per-user JSON memory
test/              pytest suite; test_data/ has sample .wav files with matching .txt transcripts
evaluate/          latency and LLM instruction-following scripts and results (charts and JSON)
models/            downloaded Piper and Kokoro model files (git-ignored)
data/users/        one folder per user: memory json, audio/ and text/ (git-ignored)
artefacts/         audio from benchmarks, tests and no-speech recordings (git-ignored)
```

### Swapping or adding a component

Each stage has a dispatcher (`__init__.py`) with a registry of backends. To add one, write a file with an `INFO` dict (backend name, model, `cloud` or `local`) and the stage's function, then register it:

- ASR: `transcribe(wav_path) -> str`
- LLM: `reply(system_prompt, history, user_text) -> str` and `complete(system_prompt, user_text) -> str`
- TTS: `synthesize(text) -> Path`

An optional `load()` is called once at startup (download and load the model there).

## Testing and evaluation

```bash
uv run python -m pytest -q   # unit tests; live tests use your keys if present and skip if not
```

> Use `python -m pytest` rather than bare `uv run pytest`: it guarantees the project's own environment and `src` path are used.

### Testing a slice of the pipeline

Run any contiguous part of `VAD -> ASR -> LLM -> TTS` once, from a file instead of the microphone:

```bash
uv run voicebot --test_start ASR --test_end LLM --test_audio test/test_data/57s_anecdote.wav   # ASR then LLM, prints text
uv run voicebot --test_start VAD --test_audio test/test_data/57s_anecdote.wav                  # everything, VAD to TTS
uv run voicebot --test_end ASR --test_audio test/test_data/002.wav                              # VAD then ASR
uv run voicebot --test_start LLM --test_end TTS --test_text test/test_data/57s_anecdote.txt    # text in, audio file out
```

Test runs are single-shot: no mic, no loop, no name prompt, no memory saved, **no audio playback**. Each stage prints its output and its time. Bad flag combinations stop with a message saying what to change. Full rules are in the [detailed guide](#detailed-guide-how-each-part-works-and-how-it-is-tested) below.

The suite includes a live end-to-end pipeline test (it needs the API keys and may hit transient provider rate limits; rerun if it does). The benchmark scripts in `evaluate/` (for example `uv run python evaluate/tts_top4_latency.py`) need the `bench` group: `uv sync --group bench --group dev --group local-asr --group local-llm --group local-tts`.

> **uv tip:** a plain `uv sync --group X` removes packages that belong to other groups. List every group you want in one command, as above.

Docker: `docker build -t voicebot . && docker run voicebot` runs the tests only (containers have no microphone or speakers, and MLX needs a Mac).

## Troubleshooting

- **"Local mode uses MLX, which needs an Apple Silicon Mac":** use `--mode remote`.
- **SPACE does nothing or the keyboard listener dies:** grant Accessibility and Input Monitoring to your terminal, then restart it.
- **"Could not open microphone":** grant Microphone permission and check the input device.
- **First local run seems stuck:** it is downloading about 3.5GB; run again if the connection dropped (downloads resume or restart cleanly).
- **First turn is slow in local mode:** the machine is probably low on memory (see [Memory pressure](#memory-pressure-on-8gb)); close other apps.
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
- [ ] Build a set of realistic conversation prompts to compare local LLMs' quality (use of the user's name, plain spoken text, follow-up questions) before choosing a local default. A first version exists in `evaluate/llm_follow.py`; its checks are crude and the sample is small.
- [ ] Ensure that summary is confined to a set number of tokens/chars

## Detailed guide: how each part works and how it is tested

### 1. A conversation turn, step by step

1. **Input** (`input/recorder.py`): the first turn waits for SPACE; later turns start listening straight away. Audio is captured at 16kHz. Silero VAD watches the stream and ends the recording after 1 second of silence. If nothing is said for about 8 seconds, the session ends.
2. **VAD trim** (`asr/vad.py`): Silero finds the speech segments in the saved recording and writes a new file with only the speech. If no speech (at least 300ms) is found, ASR is skipped and the bot asks you to try again. This stops Whisper inventing text from silence.
3. **ASR** (`asr/`): turns the trimmed wav into text. `local-whisper` runs `whisper-base` on the Mac GPU through mlx-whisper; `remote-whisper` uploads to Groq. Output that looks like a hallucination (very low confidence, repeated, or empty) is dropped.
4. **LLM** (`llm/`): builds a message list of the system prompt, the history and the transcript, and generates the complete reply. The system prompt is the base prompt plus style rules (short, warm, plain text, one follow-up question) plus what the bot remembers about you. The reply is cleaned of markdown and emoji because it is spoken.
5. **TTS** (`tts/`): turns the reply into an audio file. Only the time to produce the file is measured.
6. **Playback**: `afplay` plays the file. Playback is not timed.
7. **Save**: after each turn the text, per-stage latencies and the setup used are written to the user's JSON, and the audio and text files are moved into the user's folder.
8. **End of session**: the LLM is called once more to extract memory (below).

### 2. Memory

- **Where:** `data/users/<name>/<name>.json`, with `facts` (list of short strings), `summary` (one rolling paragraph), `sessions`, `last_seen`, and a `"cascading conversation"` list with every turn.
- **When it is written:** every turn is appended and saved immediately, so a crash loses nothing from the log. Facts and the summary are only updated at the end of a session (also on Ctrl+C or a crash).
- **How facts are extracted:** the transcript of the session is sent to the active LLM with the existing facts and summary, and the model must answer with JSON only: `{"new_facts": [...], "summary": "..."}`. New facts are added unless an identical one exists (case-insensitive); the summary is replaced by the model's updated version. If the JSON can't be parsed, the old memory is kept.
- **How it is used:** at the next session, the facts and summary are appended to the system prompt. A first-time user instead gets an instruction to introduce itself and ask a getting-to-know-you question.
- **Known limits:** a small model sometimes records facts about itself rather than the user. Facts only accumulate and the summary has no size limit yet (see the to-do list).

### 3. Swappable modules

Each stage dispatcher (`asr/__init__.py`, `llm/__init__.py`, `tts/__init__.py`) holds a registry of backend names. `select_backend(name)` imports the chosen module, installs its `uv` dependency group if needed and calls its optional `load()`. The rest of the app only calls the dispatcher (`transcribe`, `reply`/`complete`, `synthesize`), so changing a model means changing one file or one registry line. Changing the local LLM is one constant: `DEFAULT_MODEL` in `src/voicebot/llm/mlx_lm.py` (and the test that asserts it).

### 4. Test runs: `--test_start` / `--test_end`

Stages in order: `VAD`, `ASR`, `LLM`, `TTS` (case-insensitive).

| You give | Runs |
|---|---|
| `--test_start X` only | X to the end (TTS) |
| `--test_end X` only | the beginning (VAD) to X, inclusive |
| both | X to Y, inclusive (start must not come after end) |

Inputs are always files, so runs are repeatable:

| Run starts at | Needs | Errors if |
|---|---|---|
| `VAD` or `ASR` | `--test_audio file.wav` | `--test_text` is given, or `--test_audio` is missing |
| `LLM` or `TTS` | `--test_text file.txt` (non-empty) | `--test_audio` is given, or `--test_text` is missing |

Other errors: input file not found, `--test_audio`/`--test_text` without `--test_start`/`--test_end`, and VAD or ASR finding no speech in the file. Only the backends in the slice are loaded (no warm-up). In local mode, a slice that includes the LLM is the slow part to start, because the model loads first. The LLM stage uses a fresh "friend" system prompt, so nothing from `data/users/` influences the result. The code is in `src/voicebot/pipeline_test.py`.

### 5. Automated tests (`test/`)

Run `uv run python -m pytest -q`. Most tests mock the other stages so they are fast and need no keys:

| File | Covers |
|---|---|
| `test_main.py` | the turn loop with every stage mocked, and the no-speech path |
| `test_pipeline_test.py` | `--test_*` flag combinations (valid and invalid), only the slice's stages being called, no playback in test runs |
| `test_memory.py`, `test_setup_info.py` | memory load/save/extraction (incl. bad JSON), per-turn setup metadata |
| `test_asr*.py`, `test_llm*.py`, `test_tts*.py`, `test_recorder.py` | each backend and its dispatcher |
| `test_pipeline.py` | live end-to-end run through real services (needs keys; skips without them) |

Rule used in this repo: when a test is added, it is re-run after every later change, and removed or amended when it stops applying.

### 6. Benchmarks (`evaluate/`)

Scripts use the same stage functions and start/stop points as the app (see [Latency](#latency-what-is-measured)). `llm_follow.py` also scores replies against the style rules and records peak GPU memory; its raw replies are in `evaluate/results/llm_follow.json`. Results and charts are written to `evaluate/results/`.

### Memory pressure on 8GB

All memory on an Apple Silicon Mac is shared by the CPU and GPU. If RAM runs out, macOS swaps to disk, and replies become erratic (for example 2.7s suddenly taking 8s). Llama 3.2 3B peaks at about 2.2GB, plus the other models and the OS. To check: open Activity Monitor > Memory and watch the **Memory Pressure** graph (green is fine, yellow is tight, red is swapping), or run `sysctl vm.swapusage`. If it goes yellow, close other apps or switch `DEFAULT_MODEL` to a 1B-class model.
