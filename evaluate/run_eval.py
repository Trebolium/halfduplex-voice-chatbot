"""Master eval (latency + WER) - same as `uv run voicebot-eval`. Run: uv run python evaluate/run_eval.py [--mode ...] (clips from TEST_AUDIO_PATH in .env)"""
from voicebot.evaluation import cli

cli()
