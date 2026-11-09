"""Compare eval runs - same as `uv run voicebot-compare`. Run: uv run python evaluate/compare_results.py --compare_results <run> <run> [...]"""
from voicebot.compare import cli

cli()
