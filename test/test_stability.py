"""StabilityTracker (stable prefix over ASR partials) and the word normaliser it compares with."""
import pytest

from voicebot.asr.stability import StabilityTracker, norm_words


def test_norm_words():
    assert norm_words("Hello, World! It's me.") == ["hello", "world", "it's", "me"]


def test_words_stabilise_after_window():
    t = StabilityTracker(window=2)
    assert t.update(["the", "cat"]).stable_len == 0
    u = t.update(["the", "cat", "sat"])  # 'the','cat' seen twice -> stable; 'sat' once
    assert u.newly_stable == ["the", "cat"] and u.stable_len == 2 and u.retracted_from is None
    assert t.update(["the", "cat", "sat"]).newly_stable == ["sat"]


def test_promotion_is_contiguous():
    t = StabilityTracker(window=2)
    t.update(["a", "b", "c"])
    t.update(["a", "x", "c"])  # 'c' has a streak but 'x' does not -> only 'a' promotes
    assert t.stable == ["a"]


def test_late_correction_is_retracted():
    t = StabilityTracker(window=2)
    t.update(["i", "saw", "their"]); t.update(["i", "saw", "their"])
    assert t.stable == ["i", "saw", "their"]
    u = t.update(["i", "saw", "there", "car"])
    assert u.retracted_from == 2 and u.retracted == ["their"] and t.stable == ["i", "saw"]


def test_shorter_hypothesis_retracts_tail():
    t = StabilityTracker(window=1)
    t.update(["one", "two", "three"])
    u = t.update(["one", "two"])
    assert u.retracted_from == 2 and u.retracted == ["three"]


def test_bad_window():
    with pytest.raises(ValueError, match="window"):
        StabilityTracker(window=0)
