"""Runtime helpers — process rank, GPU detection, precision and attention."""

from __future__ import annotations

import os
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------

def is_rank_zero() -> bool:
    """True when this process is global rank 0 (or not running distributed).

    Uses ``RANK`` rather than ``LOCAL_RANK`` so that in multi-node runs exactly
    one process across the whole job — not one per node — writes the MLflow
    run, checkpoints and reports.
    """
    return int(os.environ.get("RANK", "0")) == 0


def local_rank() -> int:
    """This process's GPU index on its own node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed_launch() -> bool:
    """True inside a worker that a launcher (``@distributed``, torchrun) started."""
    return os.environ.get("WORLD_SIZE") is not None and os.environ.get("RANK") is not None


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_gpu_count() -> int:
    """Return the number of NVIDIA GPUs.

    Prefers ``nvidia-smi`` so counting GPUs does not initialise CUDA in the
    parent process — which matters before forking training workers.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def resolve_precision(requested: str = "auto") -> str:
    """Resolve a precision setting against the hardware actually present.

    ``bf16`` needs Ampere or newer. Every AI Runtime accelerator has it, but
    the same code runs on CPU in CI, where neither half precision is usable.

    Returns one of ``"bf16"``, ``"fp16"`` or ``"fp32"``.
    """
    import torch

    if not torch.cuda.is_available():
        return "fp32"

    def _bf16_ok() -> bool:
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False

    if requested == "auto":
        return "bf16" if _bf16_ok() else "fp16"

    if requested == "bf16" and not _bf16_ok():
        print(
            "Warning: precision 'bf16' requested but this GPU does not support "
            "it; falling back to fp16."
        )
        return "fp16"

    return requested


def resolve_attn_implementation(requested: str = "auto") -> Optional[str]:
    """Pick an attention implementation that is actually installed and usable.

    FlashAttention-2 is not part of the AI Runtime environment, and it needs
    Ampere or newer.  ``auto`` tries FA2, then SDPA, then eager.
    """
    import torch

    if requested and requested != "auto":
        return requested

    if not torch.cuda.is_available():
        return "sdpa"

    try:
        import flash_attn  # noqa: F401

        if torch.cuda.is_bf16_supported():
            return "flash_attention_2"
    except ImportError:
        pass

    return "sdpa"
