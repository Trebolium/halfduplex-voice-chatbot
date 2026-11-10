"""Stable-prefix tracking over ASR partials (adapted from github.com/Orenil/streaming-asr-normalizer, MIT licence).

Each partial is a full re-transcription of the audio so far. A word position becomes "stable" once the same word has been
seen there in `window` consecutive partials; promotion is contiguous from the front. Every partial is also diffed against
the already-stable prefix, so a late change to a stable word (or a shorter hypothesis) is reported as a retraction instead
of being silently rewritten. Words are compared after lowercasing and stripping punctuation, since Whisper re-punctuates.
"""
import re
from dataclasses import dataclass, field


def norm_words(text: str) -> list[str]:
    """Comparable words: lowercase, punctuation stripped (apostrophes kept)."""
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


@dataclass
class StabilityUpdate:
    """Result of one partial: how much is stable now, what got newly promoted, and any retraction."""
    words: list[str]
    stable_len: int
    newly_stable: list[str] = field(default_factory=list)
    retracted_from: int | None = None  # first stable index that was invalidated, or None
    retracted: list[str] = field(default_factory=list)


class StabilityTracker:
    """Pure windowing logic over word lists - no ASR, no I/O."""

    def __init__(self, window: int = 2):
        if window < 1:
            raise ValueError("window must be >= 1 (number of consecutive agreeing partials before a word is stable)")
        self.window = window
        self.stable: list[str] = []
        self._pending: list[tuple[str, int]] = []  # (word, consecutive count) for the unconfirmed suffix

    def update(self, words: list[str]) -> StabilityUpdate:
        """Feed one partial (already normalised words); returns the new stable state."""
        # 1. Late correction: does this partial disagree with (or fall short of) the stable prefix?
        cut = next((i for i, (a, b) in enumerate(zip(words, self.stable)) if a != b), None)
        if cut is None and len(words) < len(self.stable):
            cut = len(words)
        retracted = []
        if cut is not None:
            retracted, self.stable, self._pending = self.stable[cut:], self.stable[:cut], []
        # 2. Streak counts for the live suffix (same word at the same position -> count + 1).
        suffix = words[len(self.stable):]
        self._pending = [(w, self._pending[i][1] + 1 if i < len(self._pending) and self._pending[i][0] == w else 1)
                         for i, w in enumerate(suffix)]
        # 3. Promote the longest front run that has met the window.
        k = 0
        while k < len(self._pending) and self._pending[k][1] >= self.window:
            k += 1
        newly = [w for w, _ in self._pending[:k]]
        self.stable += newly
        self._pending = self._pending[k:]
        return StabilityUpdate(words, len(self.stable), newly, cut, retracted)
