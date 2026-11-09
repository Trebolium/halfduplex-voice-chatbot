"""Master eval (latency + WER) - same as `uv run voicebot-eval`. Run: uv run python evaluate/run_eval.py [--data_dir DIR] [--mode ...]"""
from voicebot.evaluation import cli

cli()
