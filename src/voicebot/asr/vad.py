"""INTERFACE: trim_speech(wav_path) -> Path | None. Writes trimmed wav; None if no speech >=300ms."""
import logging
import wave
from pathlib import Path

import webrtcvad

from voicebot.config import ARTEFACTS_DIR, MIN_SPEECH_MS, SAMPLE_RATE

log = logging.getLogger("vad")
FRAME_MS = 30
PAD_FRAMES = 3  # frames of padding kept around each speech run


def trim_speech(wav_path: Path) -> Path | None:
    """Keep only speech runs >= MIN_SPEECH_MS (with padding); write <name>_trimmed.wav."""
    wav_path = Path(wav_path)
    print(f"[vad] trimming {wav_path.name}")
    with wave.open(str(wav_path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{wav_path} must be {SAMPLE_RATE}Hz mono int16. Re-record or resample with ffmpeg.")
        pcm = w.readframes(w.getnframes())

    size = SAMPLE_RATE * FRAME_MS // 1000 * 2  # bytes per frame
    frames = [pcm[i:i + size] for i in range(0, len(pcm) - size + 1, size)]
    vad = webrtcvad.Vad(2)
    flags = [vad.is_speech(f, SAMPLE_RATE) for f in frames]

    # find speech runs long enough
    min_frames = -(-MIN_SPEECH_MS // FRAME_MS)
    keep = [False] * len(frames)
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and flags[j]:
            j += 1
        if j - i >= min_frames:
            for k in range(max(0, i - PAD_FRAMES), min(len(frames), j + PAD_FRAMES)):
                keep[k] = True
        i = j

    before = len(frames) * FRAME_MS / 1000
    out = b"".join(f for f, k in zip(frames, keep) if k)
    after = len(out) / 2 / SAMPLE_RATE
    log.info("VAD: %.2fs -> %.2fs", before, after)
    print(f"[vad] {before:.2f}s -> {after:.2f}s")
    if not out:
        log.info("VAD: no speech found")
        print("[vad] no speech found")
        return None

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / f"{wav_path.stem}_trimmed.wav"
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(out)
    return out_path
