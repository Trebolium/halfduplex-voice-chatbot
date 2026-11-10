"""Memory diagnostics for 8GB Macs: cap MLX memory (error instead of a hard freeze) and log usage after each stage."""
import logging
import os
import subprocess

log = logging.getLogger("mem")
LIMIT_GB = float(os.getenv("MLX_MEMORY_LIMIT_GB", "5"))  # override in .env; 0 disables the cap
CACHE_GB = float(os.getenv("MLX_CACHE_LIMIT_GB", "1"))
_applied = False


def apply_limit() -> None:
    """Set MLX memory + cache limits once, so overuse raises an error instead of freezing/rebooting the Mac."""
    global _applied
    if _applied or LIMIT_GB <= 0:
        return
    try:
        import mlx.core as mx
        mx.set_memory_limit(int(LIMIT_GB * 1024**3))
        mx.set_cache_limit(int(CACHE_GB * 1024**3))
        _applied = True
        print(f"[mem] MLX memory limit {LIMIT_GB}GB, cache limit {CACHE_GB}GB (set MLX_MEMORY_LIMIT_GB in .env to change, 0 to disable)")
    except Exception as e:
        print(f"[mem] could not set MLX limits ({e}); update mlx (uv add -U mlx) if this is an old version")


def _swap() -> str:
    """macOS swap usage string, e.g. 'used = 2048.00M'."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
        return out.split("used =")[1].split()[0]
    except Exception:
        return "?"


def log_mem(stage: str) -> None:
    """Print + log MLX active/peak/cache memory and system swap after a pipeline stage."""
    try:
        import mlx.core as mx
        gb = lambda n: n / 1024**3  # noqa: E731
        msg = f"after {stage}: mlx active={gb(mx.get_active_memory()):.2f}GB peak={gb(mx.get_peak_memory()):.2f}GB cache={gb(mx.get_cache_memory()):.2f}GB swap={_swap()}"
    except Exception as e:
        msg = f"after {stage}: mlx stats unavailable ({e}) swap={_swap()}"
    print(f"[mem] {msg}")
    log.info(msg)
