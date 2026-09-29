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
REPLY_MAX_TOKENS = 300
COMPLETE_MAX_TOKENS = 500


def _client() -> OpenAI:
    """Build an OpenRouter client (fails clearly if the key is missing)."""
    key = config.require("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=TIMEOUT_SECS)


def _clean(text: str) -> str:
    """Strip stray markdown symbols since the output is spoken aloud."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"[ \t]+", " ", text).strip()


def _chat(messages: list[dict], max_tokens: int, clean: bool = True) -> str:
    """Call the model, log latency/preview, and translate errors into fixes."""
    model = config.LLM_MODEL
    log.info("LLM request -> %s (%d messages)", model, len(messages))
    t0 = time.time()
    try:
        resp = _client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    except openai.AuthenticationError as e:
        raise RuntimeError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.") from e
    except openai.NotFoundError as e:
        raise RuntimeError(f"Model '{model}' not found on OpenRouter. Fix LLM_MODEL in .env (e.g. google/gemini-2.5-flash).") from e
    except openai.BadRequestError as e:
        raise RuntimeError(f"OpenRouter bad request (is LLM_MODEL='{model}' a valid slug?): {e}") from e
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        raise RuntimeError("Could not reach OpenRouter. Check your internet connection and retry.") from e
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    text = _clean(text) if clean else text.strip()
    log.info("LLM reply in %.2fs: %r", time.time() - t0, text[:80])
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
