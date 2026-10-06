from voicebot import setup_info
from voicebot.asr import groq_asr, mlx_whisper, vad
from voicebot.llm import mlx_lm, openrouter
from voicebot.tts import edge_tts, piper


def test_setup_has_models_locations_and_no_keys():
    s = setup_info.build(groq_asr, openrouter, piper, vad)
    assert s["asr"]["location"] == "cloud" and s["llm"]["model"] and s["tts"]["location"] == "local-cpu"
    assert "KEY" not in str(s).upper().replace("MAX_TOKENS", "")
    assert setup_info.build(groq_asr, openrouter, edge_tts, vad)["tts"]["backend"] == "edge-tts"
    local = setup_info.build(mlx_whisper, openrouter, piper, vad)
    assert local["asr"]["location"] == "local-apple-gpu" and local["vad"]["backend"] == "silero-vad"


def test_setup_for_local_llm():
    llm = setup_info.build(groq_asr, mlx_lm, piper, vad)["llm"]
    assert llm["location"] == "local-apple-gpu" and "LFM2.5" in llm["model"] and llm["max_tokens"] and "provider" in llm
    remote = setup_info.build(groq_asr, openrouter, piper, vad)["llm"]
    assert remote["location"] == "cloud" and remote["max_tokens"] == openrouter.REPLY_MAX_TOKENS and remote["provider"]
