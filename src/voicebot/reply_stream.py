"""Sentence-level LLM -> TTS handoff: tokens are cut into sentences as they arrive and each sentence is synthesized on a
TTS thread while the LLM keeps generating, so the first audio is ready long before the full reply is. To swap the cut rule, edit next_sentence()."""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot import config
from voicebot.llm.openrouter import _clean

MIN_CHARS = 20  # don't cut shorter sentences ("Sure!") - too choppy to speak and abbreviations like "Dr." stay attached
_END = re.compile(r'[.!?]["\')\]]*\s')  # sentence end followed by whitespace = the sentence is complete
_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")  # one TTS thread, in order; separate from the MLX stream worker


def next_sentence(text: str, start: int) -> int | None:
    """End index of the first complete sentence (>= MIN_CHARS) in text[start:], or None while it is still being written."""
    for m in _END.finditer(text, start):
        if m.end() - start >= MIN_CHARS:
            return m.end()
    return None


class ReplyStream:
    """A reply being produced: text chunks (one per sentence) with their audio futures, filled by the LLM worker, read by the player."""

    def __init__(self, synthesize=None):
        self.synthesize = synthesize  # fn(text) -> Path; None = no TTS (tests)
        self.chunks: list[dict] = []  # {text, text_t, future}
        self.cv = threading.Condition()
        self.done, self.error, self.reply = False, None, ""
        self.text, self.text_t = None, None  # the user's transcript this reply answers
        self.text_ready = threading.Event()

    def set_text(self, text: str, text_t: float) -> None:
        self.text, self.text_t = text, text_t
        self.text_ready.set()

    def add(self, sentence: str) -> None:
        """Queue one sentence for TTS right away (called from the LLM thread)."""
        sentence = _clean(sentence)
        if not sentence:
            return
        future = _pool.submit(self._synth, sentence) if self.synthesize else None
        print(f"[reply] sentence {len(self.chunks) + 1} ready -> TTS: {sentence[:60]!r}")
        with self.cv:
            self.chunks.append({"text": sentence, "text_t": time.time(), "future": future})
            self.cv.notify_all()

    def _synth(self, text: str) -> dict:
        t0 = time.time()
        path = self.synthesize(text)
        return {"path": path, "t": time.time(), "secs": time.time() - t0}

    def finish(self, reply: str) -> None:
        with self.cv:
            self.reply, self.done = reply, True
            self.cv.notify_all()

    def fail(self, e: Exception) -> None:
        with self.cv:
            self.error, self.done = e, True
            self.cv.notify_all()
        self.text_ready.set()

    def cancel(self) -> None:
        """Drop sentences whose TTS has not started (used when a speculative reply goes stale)."""
        with self.cv:
            for c in self.chunks:
                if c["future"]:
                    c["future"].cancel()
            self.done = True
            self.cv.notify_all()

    def wait_first(self) -> dict | None:
        """Block until the first sentence exists; None if the reply ended empty. Raises the LLM error, if any."""
        with self.cv:
            self.cv.wait_for(lambda: self.chunks or self.done)
        if self.error:
            raise self.error
        return self.chunks[0] if self.chunks else None

    def audio(self):
        """Yield (chunk, tts result) in order as each sentence's audio is ready, until the reply is complete."""
        i = 0
        while True:
            with self.cv:
                self.cv.wait_for(lambda: i < len(self.chunks) or self.done)
                if i >= len(self.chunks):
                    break
                chunk = self.chunks[i]
            yield chunk, (chunk["future"].result() if chunk["future"] else None)
            i += 1
        if self.error:
            raise self.error


def join_audio(paths: list[Path]) -> Path:
    """Concatenate sentence audio into one wav (for archiving / eval); a single file is returned as is."""
    if len(paths) == 1:
        return paths[0]
    parts = [sf.read(str(p), dtype="float32") for p in paths]
    sr = parts[0][1]
    out = config.ARTEFACTS_DIR / f"reply_{Path(paths[0]).stem}_joined.wav"
    sf.write(str(out), np.concatenate([a if a.ndim == 1 else a.mean(axis=1) for a, _ in parts]), sr)
    return out
