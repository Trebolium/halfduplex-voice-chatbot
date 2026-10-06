"""LLM dispatcher: select_backend(name) picks the implementation; reply()/complete() call it.

To add a backend: write llm/<name>.py with INFO + reply(system_prompt, history, user_text) + complete(system_prompt, user_text)
(+ optional load()), then register it below.
"""
import importlib
import subprocess

# CLI name -> (module, uv dependency group needed or None, python package to check)
BACKENDS = {
    "remote-llm": ("voicebot.llm.openrouter", None, None),         # OpenRouter (LLM_MODEL / LLM_PROVIDER in .env)
    "local-llm": ("voicebot.llm.mlx_lm", "local-llm", "mlx_lm"),  # LFM2.5 1.2B via mlx-lm (Apple Silicon, local)
}
DEFAULT = "remote-llm"
_current = None


def _ensure_deps(name: str, group: str, package: str) -> None:
    """Install the uv dependency group for a local backend if its package is missing."""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"[llm] '{name}' needs the '{group}' dependencies - installing (uv sync --inexact --group {group})...")
        try:
            subprocess.run(["uv", "sync", "--inexact", "--group", group], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"Could not install the '{group}' dependencies ({e}). Run `uv sync --group {group}` manually "
                               "or use --llm_model remote-llm.") from e
        importlib.invalidate_caches()


def select_backend(name: str):
    """Activate a backend by CLI name (installing/downloading local deps first) and return its module."""
    global _current
    if name not in BACKENDS:
        raise ValueError(f"Unknown llm_model '{name}'. Choose one of: {', '.join(BACKENDS)}.")
    module, group, package = BACKENDS[name]
    print(f"[llm] backend: {name}")
    if group:
        _ensure_deps(name, group, package)
    _current = importlib.import_module(module)
    if hasattr(_current, "load"):
        _current.load()
    return _current


def current():
    """The active backend module (defaults to remote-llm)."""
    return _current or select_backend(DEFAULT)


def reply(system_prompt, history, user_text):
    """Spoken reply from the active backend."""
    return current().reply(system_prompt, history, user_text)


def complete(system_prompt, user_text):
    """One-shot raw completion from the active backend (used for memory extraction)."""
    return current().complete(system_prompt, user_text)
