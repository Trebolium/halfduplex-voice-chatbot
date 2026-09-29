"""Main loop: input -> ASR -> LLM -> TTS -> repeat until the user goes quiet."""
import logging
import time

from voicebot import config  # noqa: F401  (loads .env, sets up logging)
from voicebot.asr.groq_asr import transcribe
from voicebot.asr.vad import trim_speech
from voicebot.input.recorder import record
from voicebot.llm.openrouter import reply
from voicebot.memory import store
from voicebot.tts.edge_tts import play, synthesize

log = logging.getLogger("main")


def run_turn(system_prompt: str, history: list[dict], first: bool, timings: dict | None = None) -> bool:
    """One loop iteration. Returns False when the session should end."""
    t = timings if timings is not None else {}
    print("\n[1/5] INPUT: " + ("press SPACE to speak..." if first else "listening for your reply..."))
    wav = record(wait_for_space=first)
    if wav is None:
        print("No speech heard - ending session.")
        return False

    print("[2/5] ASR: trimming silence with VAD...")
    trimmed = trim_speech(wav)
    if trimmed is None:
        print("No speech >=300ms detected - ending session.")
        return False
    t0 = time.time()
    text = transcribe(trimmed)
    t["asr"] = time.time() - t0
    print(f"      You said: {text!r}")
    if not text:
        print("Empty transcription - ending session.")
        return False

    print("[3/5] LLM: generating reply...")
    t0 = time.time()
    answer = reply(system_prompt, history, text)
    t["llm"] = time.time() - t0
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
    print(f"      Bot: {answer}")

    print("[4/5] TTS: synthesizing speech...")
    t0 = time.time()
    audio = synthesize(answer)
    t["tts"] = time.time() - t0

    print("[5/5] PLAYBACK...")
    play(audio)
    return True


def main() -> None:
    name = input("Who is speaking? Enter your name: ").strip() or "friend"
    user = store.load(name)
    system_prompt = store.build_system_prompt(user)
    history: list[dict] = []
    first = True
    try:
        while run_turn(system_prompt, history, first):
            first = False
    except KeyboardInterrupt:
        print("\nInterrupted - saving memory.")
    print("\n[MEMORY] Updating what I know about you...")
    store.end_session(user, history)
    print("Goodbye!")


if __name__ == "__main__":
    main()
