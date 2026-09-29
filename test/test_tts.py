import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from voicebot.tts import edge_tts as tts


class FakeComm:
    def __init__(self, text, voice): pass
    async def save(self, p): open(p, "wb").write(b"fake-mp3")


def test_synthesize_mocked(tmp_path):
    with patch.object(tts, "ARTEFACTS_DIR", tmp_path), patch.object(tts.edge_tts, "Communicate", FakeComm):
        p = tts.synthesize("hello")
    assert p.parent == tmp_path and p.suffix == ".mp3" and p.read_bytes() == b"fake-mp3"


def test_synthesize_empty():
    with pytest.raises(ValueError):
        tts.synthesize("  ")


def test_synthesize_network_error(tmp_path):
    class Bad(FakeComm):
        async def save(self, p): raise OSError("no network")
    with patch.object(tts, "ARTEFACTS_DIR", tmp_path), patch.object(tts.edge_tts, "Communicate", Bad):
        with pytest.raises(RuntimeError, match="internet"):
            tts.synthesize("hi")


def test_play_mocked(tmp_path):
    with patch.object(tts.subprocess, "run") as run:
        tts.play(tmp_path / "a.mp3")
    assert run.call_args[0][0][0] == "afplay"


def test_play_missing_afplay(tmp_path):
    with patch.object(tts.subprocess, "run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="afplay"):
            tts.play(tmp_path / "a.mp3")


@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1")
def test_synthesize_real_network(tmp_path):
    with patch.object(tts, "ARTEFACTS_DIR", tmp_path):
        p = tts.synthesize("Hello, this is a test.")
    assert p.stat().st_size > 1000
