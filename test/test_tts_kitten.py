from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from voicebot.tts import kitten


def test_synthesize_mocked(tmp_path):
    sess = MagicMock(); sess.run.return_value = [np.zeros(8000, dtype=np.float32)]
    ph = MagicMock(); ph.phonemize.return_value = ["həloʊ"]
    with patch.object(kitten, "_state", (sess, np.zeros((400, 256), dtype=np.float32), ph)), \
            patch("voicebot.tts._common.ARTEFACTS_DIR", tmp_path):
        p = kitten.synthesize("hello")
    assert p.parent == tmp_path and sf.info(str(p)).samplerate == 24000
