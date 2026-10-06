"""Builds the setup metadata (models, cloud/local, non-secret configs) saved alongside each turn."""
from voicebot import config


def build(asr, llm, tts, vad) -> dict:
    """Describe the pipeline from the INFO dict each component module carries; API keys are never included."""
    cloud = llm.INFO.get("location") == "cloud"
    remote_llm = {"model": config.LLM_MODEL, "provider": config.LLM_PROVIDER or "openrouter-default"} if cloud else {}
    return {
        "vad": dict(vad.INFO),
        "asr": {"model": config.ASR_MODEL, **asr.INFO},
        "llm": {**remote_llm, **llm.INFO},  # local backends carry their own model/provider in INFO
        "tts": dict(tts.INFO),
        "config": {"sample_rate": config.SAMPLE_RATE, "max_record_secs": config.MAX_RECORD_SECS,
                   "silence_end_secs": config.SILENCE_END_SECS, "min_speech_ms": config.MIN_SPEECH_MS,
                   "idle_timeout_secs": config.IDLE_TIMEOUT_SECS, "system_prompt_base": config.BASE_SYSTEM_PROMPT},
    }
