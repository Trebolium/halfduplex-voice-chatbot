# Builds deps and runs tests. Mic/speaker access is not available in Docker on macOS; run the bot natively with `uv run voicebot`.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 libsndfile1 gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
COPY test ./test
RUN uv sync
CMD ["uv", "run", "pytest"]
