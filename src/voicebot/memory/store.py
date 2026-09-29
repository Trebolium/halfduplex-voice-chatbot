"""Per-user memory: JSON file per user, facts + rolling summary, injected into the system prompt."""
import json
import logging
import re
from datetime import datetime, timezone

from voicebot import config

log = logging.getLogger("memory")

STYLE_RULES = (
    " Speak like a warm, human friend: natural, with contractions, and occasionally react to what they said."
    " End most replies with ONE natural follow-up question that builds on what they just said or on things"
    " you remember, so you get to know them better over time. Never interrogate and never list questions."
    " Use plain spoken text only: no markdown, no emojis, no bullet points."
    " Keep replies short unless asked to elaborate."
)


def _path(name: str):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "user"
    return config.USERS_DIR / f"{slug}.json"


def _save(user: dict) -> None:
    p = _path(user["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(user, indent=2))
    log.info("Saved memory for %s -> %s", user["name"], p)


def load(name: str) -> dict:
    """Load a user's record, creating a fresh one if missing."""
    p = _path(name)
    if p.exists():
        user = json.loads(p.read_text())
        log.info("Welcome back %s (%d past sessions, %d facts)", name, user["sessions"], len(user["facts"]))
        return user
    log.info("New user: %s", name)
    user = {"name": name, "facts": [], "summary": "", "sessions": 0, "last_seen": ""}
    _save(user)
    return user


def build_system_prompt(user: dict) -> str:
    """Base prompt + friendly style rules + what we remember about the user."""
    prompt = config.BASE_SYSTEM_PROMPT + STYLE_RULES
    if user.get("sessions", 0) == 0 and not user.get("facts"):
        prompt += (
            f" This is your first meeting with {user['name']}. Introduce yourself briefly, greet them by name,"
            " and ask one gentle getting-to-know-you question."
        )
    else:
        prompt += f" You are talking with {user['name']}, whom you've spoken with before."
        if user.get("facts"):
            prompt += " Things you know about them: " + "; ".join(user["facts"]) + "."
        if user.get("summary"):
            prompt += " Summary of past conversations: " + user["summary"]
    log.info("Built system prompt (%d chars) for %s", len(prompt), user["name"])
    return prompt


def _parse(raw: str) -> dict:
    """Parse LLM JSON, stripping code fences; raises on failure."""
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def end_session(user: dict, history: list[dict]) -> None:
    """Extract new facts + updated summary from the transcript via the LLM, then save."""
    if not history:
        log.info("Empty history for %s, nothing to remember", user["name"])
        return
    from voicebot.llm.openrouter import complete  # lazy so tests can mock

    log.info("Ending session for %s: extracting memory from %d messages", user["name"], len(history))
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    system = (
        "You maintain long-term memory about a user of a voice assistant. Reply with ONLY JSON of the form "
        '{"new_facts": ["..."], "summary": "..."}. new_facts: short durable facts about the user learned in this '
        "transcript that are not already known. summary: an updated brief rolling summary of all conversations "
        "so far, merging the existing summary with this transcript."
    )
    prompt = (
        f"Existing facts: {json.dumps(user['facts'])}\nExisting summary: {user['summary'] or '(none)'}\n\n"
        f"Transcript:\n{transcript}"
    )
    try:
        data = _parse(complete(system, prompt))
        new_facts = [str(f).strip() for f in data.get("new_facts", []) if str(f).strip()]
        summary = str(data.get("summary", user["summary"]))
    except Exception as e:
        log.warning("Could not parse memory update (%s); keeping old memory. Check LLM_MODEL output.", e)
        new_facts, summary = [], user["summary"]
    known = {f.lower() for f in user["facts"]}
    for f in new_facts:
        if f.lower() not in known:
            user["facts"].append(f)
            known.add(f.lower())
    user["summary"] = summary
    user["sessions"] += 1
    user["last_seen"] = datetime.now(timezone.utc).isoformat()
    log.info("Memory updated: %d facts total, session #%d", len(user["facts"]), user["sessions"])
    _save(user)
