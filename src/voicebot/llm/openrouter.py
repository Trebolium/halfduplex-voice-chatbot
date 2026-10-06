"""INTERFACE: reply(system_prompt, history, user_text) -> str. history = [{"role","content"}, ...].

OpenRouter implementation via the openai SDK.
"""
import logging
import re
import time

import openai
from openai import OpenAI

from voicebot import config

log = logging.getLogger("llm")


BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT_SECS = 30
REPLY_MAX_TOKENS = 1000
COMPLETE_MAX_TOKENS = 1000
INFO = {"backend": "openrouter", "location": "cloud", "max_tokens": REPLY_MAX_TOKENS, "timeout_s": TIMEOUT_SECS}  # model/provider: config.LLM_MODEL / LLM_PROVIDER


_clients: dict[str, OpenAI] = {}


def _client() -> OpenAI:
    """Return a cached OpenRouter client so the connection is reused (fails clearly if the key is missing)."""
    key = config.require("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
    if key not in _clients:
        _clients[key] = OpenAI(base_url=BASE_URL, api_key=key, timeout=TIMEOUT_SECS)
    return _clients[key]


# emoji, pictographs, dingbats, flags, variation selectors, zero-width chars/joiners, keycap combiner
_NON_SPEECH = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U00002300-\U000023FF"
                         "\U0001F1E6-\U0001F1FF\uFE00-\uFE0F\u200B-\u200F\u2060\u20E3\U000E0020-\U000E007F]")


def _clean(text: str) -> str:
    """Strip markdown symbols and emoji/pictographs (output is spoken aloud), then tidy whitespace."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = _NON_SPEECH.sub("", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" +([.!?,])", r"\1", text).strip()


def _chat(messages: list[dict], max_tokens: int, clean: bool = True) -> str:
    """Call the model, log latency/preview, and translate errors into fixes."""
    model = config.LLM_MODEL
    log.info("LLM request -> %s (%d messages)", model, len(messages))
    t0 = time.time()
    try:
        extra = {"provider": {"order": [config.LLM_PROVIDER], "allow_fallbacks": False}} if config.LLM_PROVIDER else None
        resp = _client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, extra_body=extra)
    except openai.AuthenticationError as e:
        raise RuntimeError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.") from e
    except openai.NotFoundError as e:
        raise RuntimeError(f"Model '{model}' not found on OpenRouter. Fix LLM_MODEL in .env (e.g. google/gemini-2.5-flash)."
                           + (f" The request is pinned to provider '{config.LLM_PROVIDER}' (LLM_PROVIDER / --llm_provider); that provider may not host this model - clear the pin or change it." if config.LLM_PROVIDER else "")) from e
    except openai.BadRequestError as e:
        raise RuntimeError(f"OpenRouter bad request (is LLM_MODEL='{model}' a valid slug?): {e}") from e
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise RuntimeError("Could not reach OpenRouter. Check your internet connection and retry.") from e
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    text = _clean(text) if clean else text.strip()
    finish = getattr(resp.choices[0], "finish_reason", None) if resp.choices else None
    log.info("LLM reply in %.2fs (finish_reason=%s): %r", time.time() - t0, finish, text[:80])
    if finish == "length":
        log.warning("LLM reply was cut off by max_tokens=%d. Raise REPLY_MAX_TOKENS in llm/openrouter.py.", max_tokens)
    if not text:
        raise RuntimeError(f"LLM returned an empty response. Try again or change LLM_MODEL ('{model}') in .env.")
    return text


def reply(system_prompt: str, history: list[dict], user_text: str) -> str:
    """Return the assistant's spoken reply; does not mutate history."""
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_text}]
    return _chat(messages, REPLY_MAX_TOKENS)


def complete(system_prompt: str, user_text: str) -> str:
    """One-shot completion (used by memory for fact extraction)."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    return _chat(messages, COMPLETE_MAX_TOKENS, clean=False)  # raw: JSON keys contain underscores
