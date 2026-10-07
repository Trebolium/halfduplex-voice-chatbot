"""INTERFACE: record(wait_for_space) -> Path | None. Returns a 16kHz mono wav, or None if no speech began within IDLE_TIMEOUT_SECS."""
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot.config import (ARTEFACTS_DIR, IDLE_TIMEOUT_SECS, MAX_RECORD_SECS,
                             SAMPLE_RATE, SILENCE_END_SECS)

FRAME_SAMPLES = 512  # Silero VAD frame size
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32


def endpoint_state(is_speech: bool, silence_frames: int, rec_frames: int) -> tuple[int, bool]:
    """Pure endpoint logic: returns (new_silence_frames, done) after one recorded frame."""
    silence_frames = 0 if is_speech else silence_frames + 1
    done = (silence_frames * FRAME_MS / 1000 >= SILENCE_END_SECS
            or rec_frames * FRAME_MS / 1000 >= MAX_RECORD_SECS)
    return silence_frames, done


def _keyboard_trusted() -> bool:
    """True if macOS lets this process (your terminal/IDE) monitor key presses; non-macOS assumes yes."""
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return True
    return bool(AXIsProcessTrusted())


def _wait_for_space() -> None:
    """Block until SPACE is pressed (pynput), or ENTER if keyboard monitoring is not permitted."""
    if not _keyboard_trusted():
        print("[recorder] Keyboard monitoring is not permitted for this terminal/IDE, so SPACE cannot be detected. "
              "To enable it: System Settings > Privacy & Security > Accessibility and Input Monitoring > add your terminal/IDE, then restart it.")
        try:
            import sys, termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)  # drop stray keystrokes typed earlier
        except Exception:
            pass
        input("[recorder] Meanwhile, press ENTER to start recording: ")
        return
    try:
        from pynput import keyboard
        pressed = threading.Event()
        listener = keyboard.Listener(on_press=lambda k: pressed.set() if k == keyboard.Key.space else None)
        listener.start()
    except Exception as e:
        raise RuntimeError(f"Keyboard listener failed ({e}). On macOS grant your terminal/IDE Accessibility "
                           "and Input Monitoring permission (System Settings > Privacy & Security), then restart it.") from e
    try:
        while not pressed.wait(0.1):
            if not listener.is_alive():
                raise RuntimeError("Keyboard listener died. Grant Accessibility permission to your terminal/IDE in "
                                   "System Settings > Privacy & Security, then restart it.")
    finally:
        listener.stop()


def record(wait_for_space: bool) -> Path | None:
    """Record one utterance from the mic; save wav to ARTEFACTS_DIR and return its path (None on idle timeout)."""
    import sounddevice as sd
    from voicebot.asr import vad
    vad.reset()
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=FRAME_SAMPLES, callback=cb)
        stream.start()
    except Exception as e:
        raise RuntimeError(f"Could not open microphone ({e}). Check a mic is connected and that your terminal/IDE "
                           "has Microphone permission (macOS System Settings > Privacy & Security).") from e

    frames: list[np.ndarray] = []
    try:
        started = False
        if wait_for_space:
            _wait_for_space()
            while not q.empty():  # drop audio buffered before keypress
                q.get_nowait()
        print("[recorder] Listening - speak now..." if wait_for_space else "[recorder] Listening...")
        t0 = time.time()  # after a keypress the user still gets IDLE_TIMEOUT_SECS to start speaking
        silence, rec = 0, 0
        while True:
            try:
                f = q.get(timeout=1)
            except queue.Empty:
                raise RuntimeError("No audio from microphone for 1s. Check the input device and Microphone permission.")
            is_speech = vad.is_speech(f)
            if not started:
                if is_speech:
                    started = True
                    print("[recorder] Speech detected, recording started.")
                elif time.time() - t0 > IDLE_TIMEOUT_SECS:
                    print("[recorder] No speech before idle timeout.")
                    return None
                else:
                    continue
            frames.append(f)
            rec += 1
            silence, done = endpoint_state(is_speech, silence, rec)
            if done:
                print(f"[recorder] Recording ended ({silence * FRAME_MS / 1000:.1f}s silence, {rec * FRAME_MS / 1000:.1f}s total).")
                break
    finally:
        stream.stop()
        stream.close()

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"input_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(path, np.concatenate(frames), SAMPLE_RATE, subtype="PCM_16")
    print(f"[recorder] Saved {path}")
    return path
