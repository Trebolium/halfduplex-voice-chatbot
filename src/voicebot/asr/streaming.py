"""Streaming turn: partial ASR while the user speaks + a speculative LLM start at the first pause.

Goal: cut the wait between the end of speech and the LLM reply, whatever the LLM is. The recorder (or the eval's
real-time file player) calls feed() for every 32ms frame. A background worker:
  1. re-transcribes the audio so far every STREAM_PARTIAL_EVERY_S while speech continues (partials -> StabilityTracker),
  2. at a pause of STREAM_PAUSE_SECS transcribes all speech so far; if that partial retracts nothing already stable,
     it starts the LLM immediately - i.e. during the end-of-turn silence the recorder is waiting out anyway.
If speech resumes, the speculative reply is marked stale: a streaming backend (local mlx-lm) stops generating at the next
token, others finish and are discarded. Replies are cut into sentences and each is sent to TTS while the LLM is still
generating (reply_stream.ReplyStream). finish() (called at the endpoint) returns as soon as the FIRST sentence's audio is
ready, with the live ReplyStream and how long the caller waited for each stage after the endpoint.
All model calls run on ONE process-wide worker thread, in order: local MLX models are never used concurrently or from a
fresh thread each turn.
"""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicebot import config
from voicebot.asr.stability import StabilityTracker, norm_words
from voicebot.reply_stream import ReplyStream, next_sentence

FRAME_S = 512 / config.SAMPLE_RATE
_jobs: queue.Queue = queue.Queue()
_worker: threading.Thread | None = None


def _reload_llm_here() -> None:
    """MLX streams are thread-local: a model loaded on the main thread fails on this worker ('no Stream(gpu, 1)'), so reload it here."""
    from voicebot import llm
    backend = llm.current()
    if hasattr(backend, "load") and hasattr(backend, "model_id") and backend.model_id():
        print("[stream] reloading the local LLM on the stream worker thread (MLX streams are per-thread)...")
        backend.load(backend.model_id())


def _run_worker() -> None:
    """Process-wide worker: runs submitted callables one at a time, forever."""
    _reload_llm_here()
    while True:
        fn = _jobs.get()
        try:
            fn()
        finally:
            _jobs.task_done()


def _submit(fn) -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_run_worker, daemon=True, name="stream-worker")
        _worker.start()
    _jobs.put(fn)


class StreamingSession:
    """One user turn. feed() per frame, then finish() at the endpoint. transcribe/reply/stream_reply are injectable for tests."""

    def __init__(self, system_prompt: str, history: list[dict], transcribe=None, reply=None, stream_reply="auto", synthesize=None,
                 every_s: float | None = None, pause_s: float | None = None, window: int | None = None):
        from voicebot import asr, llm
        self.transcribe, self.reply = transcribe or asr.transcribe, reply or llm.reply
        if stream_reply == "auto":  # cancellable token stream if the active LLM backend has one (local mlx-lm does)
            stream_reply = getattr(llm.current(), "stream_reply", None) if reply is None else None
        self.stream_reply, self.synthesize = stream_reply, synthesize  # synthesize: fn(text)->Path run per sentence; None = no TTS
        self.system_prompt, self.history = system_prompt, list(history)  # copy: a speculative reply must not touch history
        self.every = int((every_s or config.STREAM_PARTIAL_EVERY_S) * config.SAMPLE_RATE)
        self.pause_frames = int(round((pause_s or config.STREAM_PAUSE_SECS) / FRAME_S))
        self.tracker = StabilityTracker(window or config.STREAM_WINDOW)
        self.frames: list[np.ndarray] = []
        self.n = 0                  # samples received
        self.speech_end = 0         # sample index just after the last speech frame
        self.silence = 0            # consecutive non-speech frames
        self.last_partial_n = 0     # samples covered by the last queued periodic partial
        self.pause_queued_for = -1  # speech_end value a pause partial was already queued for
        self.pending = 0            # this session's jobs not yet finished
        self.lock = threading.Condition()  # also signals finish() when a speculation starts or a job ends
        self.latest = None          # (speech_end covered, text, time ready) from the newest partial
        self.spec = None            # speculative reply: dict(speech_end, text, stream, stale)
        self.stats = {"partials": 0, "partial_failures": 0, "retractions": 0, "speculations": 0,
                      "stale_speculations": 0, "cancelled_generations": 0, "skipped_partials": 0}

    # ---- called from the audio loop (must stay cheap) ----
    def feed(self, frame: np.ndarray, is_speech: bool) -> None:
        """Add one frame; queue partials when due."""
        self.frames.append(frame)
        self.n += len(frame)
        if is_speech:
            self.silence = 0
            self.speech_end = self.n
            if self.spec and not self.spec["stale"]:
                self.spec["stale"] = True
                self.stats["stale_speculations"] += 1
                print("[stream] speech resumed - discarding the speculative reply")
            if self.n - self.last_partial_n >= self.every and self.pending == 0:  # skip while the worker is busy
                self.last_partial_n = self.n
                self._queue(at_pause=False)
        else:
            self.silence += 1
            if self.silence >= self.pause_frames and self.pause_queued_for != self.speech_end and self.speech_end > 0:
                self.pause_queued_for = self.speech_end
                self._queue(at_pause=True)

    def _queue(self, at_pause: bool) -> None:
        audio, speech_end = np.concatenate(self.frames), self.speech_end
        with self.lock:
            self.pending += 1
        _submit(lambda: self._job(audio, speech_end, at_pause))

    # ---- worker side ----
    def _job(self, audio: np.ndarray, speech_end: int, at_pause: bool) -> None:
        try:
            if at_pause and speech_end != self.speech_end:  # speech resumed before we got to it: superseded
                self.stats["skipped_partials"] += 1
                return
            self._partial(audio, speech_end, at_pause)
        except Exception as e:  # a failed partial is never fatal: finish() falls back to a normal transcription
            self.stats["partial_failures"] += 1
            print(f"[stream] WARNING: partial step failed ({type(e).__name__}: {e}); will fall back at the endpoint if needed.")
        finally:
            with self.lock:
                self.pending -= 1
                self.lock.notify_all()

    def _partial(self, audio: np.ndarray, speech_end: int, at_pause: bool) -> None:
        """Transcribe a snapshot, update stability, and start the speculative LLM at a clean pause."""
        text = self._transcribe(audio, "partial")
        upd = self.tracker.update(norm_words(text))
        self.stats["partials"] += 1
        if upd.retracted_from is not None:
            self.stats["retractions"] += 1
        self.latest = (speech_end, text, time.time())
        print(f"[stream] partial {self.stats['partials']} ({len(audio) / config.SAMPLE_RATE:.1f}s{', pause' if at_pause else ''}): "
              f"{upd.stable_len}/{len(upd.words)} words stable" + (f", retracted {upd.retracted}" if upd.retracted else ""))
        if at_pause and text and upd.retracted_from is None and speech_end == self.speech_end:
            stream = ReplyStream(self.synthesize)
            stream.set_text(text, time.time())
            spec = {"speech_end": speech_end, "text": text, "stream": stream, "stale": False}
            self.stats["speculations"] += 1
            print("[stream] pause with no retractions - starting the LLM speculatively")
            with self.lock:
                self.spec = spec
                self.lock.notify_all()  # finish() may already be waiting for this
            self._generate(text, stream, spec)

    def _generate(self, text: str, stream: ReplyStream, spec: dict | None = None) -> None:
        """Fill `stream` with the reply, one sentence at a time (each goes to TTS at once). A speculation that goes stale is cancelled at the next token."""
        try:
            if self.stream_reply is None:  # no token stream (remote LLM): the whole reply is one sentence
                reply = self.reply(self.system_prompt, self.history, text)
                stream.add(reply)
                stream.finish(reply)
                return
            from voicebot.llm.openrouter import _clean
            out, pos = "", 0
            for out, _ in self.stream_reply(self.system_prompt, self.history, text):
                if spec and spec["stale"]:
                    self.stats["cancelled_generations"] += 1
                    print("[stream] speculative generation stopped (speech resumed)")
                    stream.cancel()
                    return
                while (end := next_sentence(out, pos)):
                    stream.add(out[pos:end])
                    pos = end
            stream.add(out[pos:])
            stream.finish(_clean(out))
        except Exception as e:
            stream.fail(e)
            raise

    def _transcribe(self, audio: np.ndarray, name: str) -> str:
        config.ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.ARTEFACTS_DIR / f"stream_{name}.wav"
        sf.write(str(path), audio, config.SAMPLE_RATE, subtype="PCM_16")
        return self.transcribe(path)

    # ---- called at the endpoint ----
    def _spec_ok(self) -> bool:
        sp = self.spec
        return bool(sp and not sp["stale"] and sp["speech_end"] == self.speech_end and sp["stream"].error is None)

    def finish(self) -> dict:
        """Return once the first sentence's audio is ready: the live ReplyStream (play stream.audio(), then read stream.reply)
        plus the waits after the endpoint (asr_wait_s: transcript, llm_wait_s: first sentence, tts_wait_s: its audio)."""
        t_end = time.time()
        with self.lock:
            self.lock.wait_for(lambda: self._spec_ok() or self.pending == 0)
            use_spec = self._spec_ok()
        if use_spec:
            stream, path = self.spec["stream"], "speculative"
        else:
            stream = ReplyStream(self.synthesize)
            path = "partial" if self.latest and self.latest[0] == self.speech_end and self.latest[1] else "fallback"
            _submit(lambda: self._fallback_job(stream))  # queued after this turn's jobs
        stream.text_ready.wait()
        first = stream.wait_first()  # raises the LLM error, if any
        if first is None:
            return {"text": stream.text, "stream": stream, "asr_wait_s": 0.0, "llm_wait_s": 0.0, "tts_wait_s": 0.0, "path": path, **self.stats}
        audio_t = first["future"].result()["t"] if first["future"] else first["text_t"]
        asr_wait = max(0.0, stream.text_t - t_end)
        llm_wait = max(0.0, first["text_t"] - max(t_end, stream.text_t))
        tts_wait = max(0.0, audio_t - max(t_end, first["text_t"]))
        print(f"[stream] endpoint -> first audio in {asr_wait + llm_wait + tts_wait:.2f}s via {path} (asr wait {asr_wait:.2f}s, "
              f"llm wait {llm_wait:.2f}s, tts wait {tts_wait:.2f}s; {self.stats})")
        return {"text": stream.text, "stream": stream, "asr_wait_s": asr_wait, "llm_wait_s": llm_wait, "tts_wait_s": tts_wait,
                "path": path, **self.stats}

    def _fallback_job(self, stream: ReplyStream) -> None:
        """No usable speculation: use the newest partial if it covers all the speech, else transcribe the whole turn, then generate."""
        try:
            if self.latest and self.latest[0] == self.speech_end and self.latest[1]:
                text, text_t = self.latest[1], self.latest[2]
            else:
                print("[stream] no usable partial covers all the speech - transcribing the whole turn now")
                text = self._transcribe(np.concatenate(self.frames) if self.frames else np.zeros(1, np.int16), "final")
                text_t = time.time()
            stream.set_text(text, text_t)
            if text:
                self._generate(text, stream)
            else:
                stream.finish("")
        except Exception as e:
            stream.fail(e)


def load_16k(path: Path) -> np.ndarray:
    """Read a clip as 16kHz mono int16 for the eval's real-time player (linear resample if needed)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        n = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)
