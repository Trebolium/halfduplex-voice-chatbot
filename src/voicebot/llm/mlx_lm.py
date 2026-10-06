"""INTERFACE: reply(system_prompt, history, user_text) -> str, complete(system_prompt, user_text) -> str. Local LFM2.5 1.2B via mlx-lm."""
import logging
import time

from voicebot.llm.openrouter import _clean

log = logging.getLogger("llm")

DEFAULT_MODEL = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
MAX_TOKENS = 200
INFO = {"backend": "mlx-lm", "location": "local-apple-gpu", "model": DEFAULT_MODEL, "provider": "mlx-lm", "max_tokens": MAX_TOKENS}  # model updated by load()
_model = _tok = None
_model_id = ""


def model_id() -> str:
    """The currently loaded model repo id."""
    return _model_id


def load(model: str = DEFAULT_MODEL) -> None:
    """Download (first run) and load a model, replacing any previously loaded one."""
    global _model, _tok, _model_id
    from mlx_lm import load as mlx_load
    print(f"[llm] loading local model {model} (first run downloads ~0.7GB from Hugging Face, then it is cached)...")
    _model, _tok, _model_id = None, None, model
    import gc
    import mlx.core as mx
    gc.collect(); mx.clear_cache()  # free the previous model before loading the next (8GB RAM)
    _model, _tok = mlx_load(model)
    INFO["model"] = model


def stream_reply(system_prompt: str, history: list[dict], user_text: str):
    """Yield (text_so_far, generation_response) as tokens are produced."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    if _model is None:
        load()
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_text}]
    prompt = _tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text = ""
    for r in stream_generate(_model, _tok, prompt, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.7)):
        text += r.text
        yield text, r


def reply(system_prompt: str, history: list[dict], user_text: str) -> str:
    """Return the full spoken reply (markdown stripped), same contract as llm.openrouter.reply."""
    t0 = time.time()
    text = ""
    for text, _ in stream_reply(system_prompt, history, user_text):
        pass
    text = _clean(text)
    log.info("Local LLM reply in %.2fs (%s): %r", time.time() - t0, _model_id, text[:80])
    if not text:
        raise RuntimeError(f"Local LLM '{_model_id}' returned an empty response. Try another model or re-run.")
    return text


def complete(system_prompt: str, user_text: str) -> str:
    """One-shot raw completion (no markdown cleaning; JSON keys contain underscores), used for memory extraction."""
    text = ""
    for text, _ in stream_reply(system_prompt, [], user_text):
        pass
    text = text.strip()
    if not text:
        raise RuntimeError(f"Local LLM '{_model_id}' returned an empty response. Try another model or re-run.")
    return text
