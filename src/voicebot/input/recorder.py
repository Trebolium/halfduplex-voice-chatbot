"""INTERFACE: record(wait_for_space) -> Path | None. Returns a 16kHz mono wav, or None if no speech began within IDLE_TIMEOUT_SECS."""
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot.config import (ARTEFACTS_DIR, IDLE_TIMEOUT_SECS, MAX_RECORD_SECS,
                             SAMPLE_RATE, SILENCE_END_SECS)

log = logging.getLogger("recorder")
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def endpoint_state(is_speech: bool, silence_frames: int, rec_frames: int) -> tuple[int, bool]:
    """Pure endpoint logic: returns (new_silence_frames, done) after one recorded frame."""
    silence_frames = 0 if is_speech else silence_frames + 1
    done = (silence_frames * FRAME_MS / 1000 >= SILENCE_END_SECS
            or rec_frames * FRAME_MS / 1000 >= MAX_RECORD_SECS)
    return silence_frames, done


def _wait_for_space(stop: threading.Event) -> None:
    """Block until spacebar is pressed (pynput)."""
    try:
        from pynput import keyboard
        pressed = threading.Event()
        listener = keyboard.Listener(on_press=lambda k: pressed.set() if k == keyboard.Key.space else None)
        listener.start()
    except Exception as e:
        raise RuntimeError(f"Keyboard listener failed ({e}). On macOS grant your terminal/IDE Accessibility "
                           "and Input Monitoring permission (System Settings > Privacy & Security), and Microphone too, then restart it.") from e
    print("[recorder] Press SPACE to start recording...")
    try:
        while not pressed.wait(0.1):
            if not listener.is_alive():
                raise RuntimeError("Keyboard listener died. Grant Accessibility permission to your terminal/IDE in "
                                   "System Settings > Privacy & Security, then restart it.")
    finally:
        listener.stop()
    log.info("Space pressed")
    print("[recorder] Space pressed.")


def record(wait_for_space: bool) -> Path | None:
    """Record one utterance from the mic; save wav to ARTEFACTS_DIR and return its path (None on idle timeout)."""
    import sounddevice as sd
    import webrtcvad
    vad = webrtcvad.Vad(2)
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
        print("[recorder] Listening...")
        log.info("Listening (wait_for_space=%s)", wait_for_space)
        started = False
        if wait_for_space:
            _wait_for_space(threading.Event())
            while not q.empty():  # drop audio buffered before keypress
                q.get_nowait()
            started = True
            print("[recorder] Recording started.")
        t0 = time.time()
        silence, rec = 0, 0
        while True:
            try:
                f = q.get(timeout=1)
            except queue.Empty:
                raise RuntimeError("No audio from microphone for 1s. Check the input device and Microphone permission.")
            is_speech = vad.is_speech(f.tobytes(), SAMPLE_RATE)
            if not started:
                if is_speech:
                    started = True
                    print("[recorder] Speech detected, recording started.")
                    log.info("Recording started")
                elif time.time() - t0 > IDLE_TIMEOUT_SECS:
                    print("[recorder] No speech before idle timeout.")
                    log.info("Idle timeout, no speech")
                    return None
                else:
                    continue
            frames.append(f)
            rec += 1
            silence, done = endpoint_state(is_speech, silence, rec)
            if done:
                print(f"[recorder] Recording ended ({silence * FRAME_MS / 1000:.1f}s silence, {rec * FRAME_MS / 1000:.1f}s total).")
                log.info("Silence/max reached, stopping")
                break
    finally:
        stream.stop()
        stream.close()

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f"input_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(path, np.concatenate(frames), SAMPLE_RATE, subtype="PCM_16")
    print(f"[recorder] Saved {path}")
    log.info("Saved %s", path)
    return path
