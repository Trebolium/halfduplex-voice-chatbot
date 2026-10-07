"""Resolve a Hugging Face repo id to a local snapshot path, using the cache only (no network) when possible."""


def local_path(repo: str) -> str:
    """Return the cached snapshot dir for repo; download it once if it is not cached yet."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        print(f"[hf] {repo} is not fully cached - downloading it (one-off, needs internet)...")
        try:
            return snapshot_download(repo)
        except Exception as e:
            raise RuntimeError(f"Could not download {repo} ({e}). Connect to the internet once so it can be cached, then re-run.") from e
