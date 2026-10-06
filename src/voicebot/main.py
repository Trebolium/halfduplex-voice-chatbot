"""Main loop: input -> ASR -> LLM -> TTS -> repeat until the user goes quiet."""
import argparse
import logging
import os
import platform
import time

from voicebot import config, setup_info  # noqa: F401  (config loads .env, sets up logging)
from voicebot import asr, llm
from voicebot.asr import transcribe, vad
from voicebot.asr.vad import trim_speech
from voicebot.input.recorder import record
from voicebot.llm import reply
from voicebot.memory import store
from voicebot import tts
from voicebot.tts import play, synthesize

log = logging.getLogger("main")


def run_turn(system_prompt: str, history: list[dict], first: bool, timings: dict | None = None, user: dict | None = None) -> bool:
    """One loop iteration. Returns False when the session should end."""
    t = timings if timings is not None else {}
    print("\n[1/5] INPUT: " + ("press SPACE to speak..." if first else "listening for your reply..."))
    wav = record(wait_for_space=first)
    if wav is None:
        print("No speech heard - ending session.")
        return False

    print("[2/5] ASR: trimming silence (Silero VAD), then transcribing...")
    t0 = time.time()
    trimmed = trim_speech(wav)
    t["vad"] = time.time() - t0
    if trimmed is None:
        print("VAD found no speech in the recording - skipping ASR to avoid hallucinated text.")
        return True
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
    if user is not None:
        # saved now so nothing is lost if the session dies; latencies: asr = transcribe call, llm = transcript -> full reply
        store.log_turn(user, text, answer, {"vad": t["vad"], "asr": t["asr"], "llm": t["llm"]}, setup_info.build(asr.current(), llm.current(), tts.current(), vad))

    print("[4/5] TTS: synthesizing speech...")
    t0 = time.time()
    audio = synthesize(answer)
    t["tts"] = time.time() - t0  # LLM reply returned -> audio file written
    if user is not None:
        store.add_latency(user, "tts", t["tts"])

    print("[5/5] PLAYBACK...")
    play(audio)
    return True




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: --mode picks the whole stack; the model flags only apply in remote mode."""
    p = argparse.ArgumentParser(description="Voice bot: local mode (default) runs everything on this Mac; remote mode calls cloud APIs.")
    p.add_argument("--mode", choices=["local", "remote"], default=os.getenv("MODE", "local"),
                   help="local = mlx-whisper + LFM2.5 1.2B + Piper (Apple Silicon, no API keys); remote = Groq + OpenRouter + edge-tts")
    p.add_argument("--asr_model", help="remote only: Groq transcription model (default ASR_MODEL in .env, whisper-large-v3-turbo)")
    p.add_argument("--llm_model", help="remote only: OpenRouter model slug (default LLM_MODEL in .env)")
    p.add_argument("--llm_provider", help="remote only: OpenRouter provider tag to pin, e.g. deepinfra/turbo (default LLM_PROVIDER in .env)")
    p.add_argument("--tts_model", help=f"TTS backend, works in both modes. local: {'|'.join(tts.LOCAL_BACKENDS)} (default {tts.DEFAULT}); "
                                       f"remote: {'|'.join(tts.REMOTE_BACKENDS)} (default {tts.REMOTE_DEFAULT})")
    p.add_argument("--tts_voice", help="remote only: voice for the chosen remote TTS (edge-tts voice name, Orpheus voice, or ElevenLabs voice id)")
    return p.parse_args(argv)


def configure(args: argparse.Namespace) -> dict:
    """Validate the environment for the chosen mode, apply remote overrides, and return the backend names to use."""
    flags = {k: v for k, v in {"--asr_model": args.asr_model, "--llm_model": args.llm_model,
                               "--llm_provider": args.llm_provider, "--tts_voice": args.tts_voice}.items() if v}
    if args.mode == "local":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Local mode uses MLX, which needs an Apple Silicon Mac. Run on a Mac with an M-series chip, or use --mode remote.")
        if flags:
            print(f"[main] WARNING: {', '.join(flags)} ignored in local mode (they only apply with --mode remote).")
        choice = args.tts_model or os.getenv("LOCAL_TTS", tts.DEFAULT)
        if choice not in tts.LOCAL_BACKENDS:
            raise ValueError(f"--tts_model '{choice}' is not a local TTS. Local options: {', '.join(tts.LOCAL_BACKENDS)} "
                             f"(remote ones like '{tts.REMOTE_DEFAULT}' need --mode remote).")
        return {"asr": "local-whisper", "llm": "local-llm", "tts": choice}
    choice = args.tts_model or os.getenv("REMOTE_TTS", tts.REMOTE_DEFAULT)
    if choice not in tts.REMOTE_BACKENDS:
        raise ValueError(f"--tts_model '{choice}' is not a remote TTS. Remote options: {', '.join(tts.REMOTE_BACKENDS)} "
                         f"(local ones like '{tts.DEFAULT}' need --mode local).")
    config.require("GROQ_API_KEY", config.GROQ_API_KEY)
    config.require("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
    if choice == "elevenlabs":
        config.require("ELEVENLABS_API_KEY", config.ELEVENLABS_API_KEY)
    if args.asr_model: config.ASR_MODEL = args.asr_model
    if args.llm_model:
        config.LLM_MODEL = args.llm_model
        if not args.llm_provider and config.LLM_PROVIDER:  # a provider pin is specific to one model, so don't carry it over
            print(f"[main] --llm_model given without --llm_provider: dropping the LLM_PROVIDER pin '{config.LLM_PROVIDER}' from .env.")
            config.LLM_PROVIDER = ""
    if args.llm_provider: config.LLM_PROVIDER = args.llm_provider
    if args.tts_voice: config.TTS_VOICE = args.tts_voice
    return {"asr": "remote-whisper", "llm": "remote-llm", "tts": choice}


def warm_up() -> None:
    """Run each local model once so the first real turn is not slow (weights get paged in, kernels compile)."""
    print("[main] warming up local models (avoids a slow first turn)...")
    t0 = time.time()
    asr.current().load()
    llm.reply("Be brief.", [], "Hello")
    tts.synthesize("Hello.")
    print(f"[main] warm-up done in {time.time() - t0:.1f}s")


def main() -> None:
    args = parse_args()
    names = configure(args)
    print(f"[main] mode={args.mode}: asr={names['asr']} llm={names['llm']} tts={names['tts']}")
    asr.select_backend(names["asr"])
    llm.select_backend(names["llm"])
    tts.select_backend(names["tts"])
    if args.mode == "local":
        warm_up()
    vad.reset()  # loads Silero now so the first turn is not slowed down
    name = input("Who is speaking? Enter your name: ").strip() or "friend"
    user = store.load(name)
    system_prompt = store.build_system_prompt(user)
    history: list[dict] = []
    first = True
    try:
        while run_turn(system_prompt, history, first, user=user):
            first = False
    except KeyboardInterrupt:
        print("\nInterrupted - saving memory.")
    finally:  # always save memory, even if a stage crashed
        print("\n[MEMORY] Updating what I know about you...")
        store.end_session(user, history)
    print("Goodbye!")


if __name__ == "__main__":
    main()
