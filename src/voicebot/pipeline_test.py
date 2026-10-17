"""Once-off test runs of any contiguous slice of the pipeline (VAD -> ASR -> LLM -> TTS) from a wav or text file, no mic and no loop."""
import time
from pathlib import Path

from voicebot import asr, llm, tts
from voicebot.asr import transcribe, vad
from voicebot.llm import reply
from voicebot.memory import store
from voicebot.tts import synthesize

STAGES = ["VAD", "ASR", "LLM", "TTS"]
NEEDS_AUDIO = {"VAD", "ASR"}  # stages that start from a wav; LLM/TTS start from text


def validate(args) -> tuple[str, str] | None:
    """Return (start, end) if a test run was requested (None for the normal live app); raise a clear error on bad flag combos."""
    if not (args.test_start or args.test_end or args.test_audio or args.test_text):
        return None
    start, end = args.test_start or STAGES[0], args.test_end or STAGES[-1]
    if not (args.test_start or args.test_end):
        raise ValueError("--test_audio/--test_text only apply to test runs. Add --test_start and/or --test_end (VAD|ASR|LLM|TTS).")
    if STAGES.index(start) > STAGES.index(end):
        raise ValueError(f"--test_start {start} comes after --test_end {end}. Pipeline order is {' -> '.join(STAGES)}.")
    if start in NEEDS_AUDIO:
        if args.test_text:
            raise ValueError(f"A test starting at {start} needs audio, not text. Use --test_audio <file.wav> instead of --test_text.")
        if not args.test_audio:
            raise ValueError(f"A test starting at {start} needs --test_audio <file.wav> (live recording is disabled in test runs).")
    else:
        if args.test_audio:
            raise ValueError(f"A test starting at {start} takes text, not audio. Use --test_text <file.txt> instead of --test_audio.")
        if not args.test_text:
            raise ValueError(f"A test starting at {start} needs --test_text <file.txt>.")
    path = Path(args.test_audio or args.test_text)
    if not path.is_file():
        raise FileNotFoundError(f"Test input '{path}' not found. Check the path (relative to where you run `uv run voicebot`).")
    return start, end


def run(start: str, end: str, args, names: dict) -> None:
    """Select only the backends the slice uses, run it once, print each stage's output + timing."""
    stages = STAGES[STAGES.index(start):STAGES.index(end) + 1]
    print(f"[test] running {' -> '.join(stages)} once (no mic, no loop)")
    if "VAD" in stages: vad.reset()
    if "ASR" in stages: asr.select_backend(names["asr"])
    if "LLM" in stages: llm.select_backend(names["llm"])
    if "TTS" in stages: tts.select_backend(names["tts"])
    audio = Path(args.test_audio) if args.test_audio else None
    text = Path(args.test_text).read_text().strip() if args.test_text else None
    if args.test_text and not text:
        raise ValueError(f"{args.test_text} is empty. Put the text to send to {start} in it.")
    t_all = time.time()
    for stage in stages:
        print(f"\n[test] >>> {stage}")
        t0 = time.time()
        if stage == "VAD":
            audio = trimmed = vad.trim_speech(audio)
            if trimmed is None:
                raise RuntimeError("VAD found no speech in the test audio. Try a clip with audible speech.")
            print(f"[test] VAD kept speech -> {trimmed}")
        elif stage == "ASR":
            text = transcribe(audio)
            print(f"[test] ASR text: {text!r}")
            if not text:
                raise RuntimeError("ASR returned empty text for the test audio. Try a clearer clip.")
        elif stage == "LLM":
            system = store.build_system_prompt({"name": "friend", "sessions": 0, "facts": []})  # fresh user, nothing saved
            text = reply(system, [], text)
            print(f"[test] LLM reply: {text}")
        else:
            audio = synthesize(text)
            print(f"[test] TTS audio -> {audio}")
        print(f"[test] {stage} took {time.time() - t0:.2f}s")
    print(f"\n[test] done: {' -> '.join(stages)} in {time.time() - t_all:.2f}s")
