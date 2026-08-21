"""Databricks environment helpers — GPU detection, NCCL setup, data staging."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_databricks() -> bool:
    return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None


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
    """True when a launcher (TorchDistributor, torchrun) set up the process group."""
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


def setup_nccl_env() -> None:
    """Configure NCCL environment variables for Databricks multi-GPU networking."""
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def resolve_precision(requested: str = "auto") -> str:
    """Resolve a precision setting against the hardware actually present.

    ``bf16`` needs Ampere or newer; V100 and T4 are still common on Databricks
    GPU pools, and ``bf16: true`` is the shipped default in every config.

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

    FlashAttention-2 used to be hardcoded on whenever CUDA was present, with
    no capability check and no fallback: it is not part of DBR ML, and it
    needs Ampere or newer.  ``auto`` tries FA2, then SDPA, then eager.
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


# ---------------------------------------------------------------------------
# Data staging for /Volumes/ → local disk
# ---------------------------------------------------------------------------

_VOLUMES_ROOT = "/Volumes"


def volumes_staging_path(volumes_path: str, local_root: str) -> str:
    """Map a ``/Volumes/...`` path to its deterministic local staging path.

    Split out so the path arithmetic can be tested without touching the
    filesystem.  Uses ``removeprefix`` rather than ``lstrip``: ``lstrip``
    strips *characters*, so ``lstrip("/Volumes/")`` would eat into the first
    real path segment and could collide distinct sources onto one directory.
    """
    relative = volumes_path.removeprefix("/Volumes/").rstrip("/")
    return os.path.join(local_root, relative)


def stage_data_to_local(path: str, local_root: str = "/tmp/staged_data") -> str:
    """Copy a /Volumes/ path to local disk so worker processes can read it.

    Returns the original path unchanged if it is not a /Volumes/ path.
    """
    if not path.startswith("/Volumes/"):
        return path

    local_path = volumes_staging_path(path, local_root)

    if os.path.exists(local_path):
        return local_path

    # ``_VOLUMES_ROOT`` is indirected so tests can rebase the source tree.
    src = Path(_VOLUMES_ROOT) / path.removeprefix("/Volumes/").rstrip("/")
    dst = Path(local_path)

    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    elif src.is_dir():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dst))
    else:
        return path

    print(f"Staged {path} → {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Runtime dependency installation
# ---------------------------------------------------------------------------

def ensure_runtime_requirements(requirements_file: str | Path) -> None:
    """Install ``requirements_file`` if it exists, once, on rank 0 only.

    Databricks job clusters should normally declare these as cluster libraries;
    this is the fallback for running the entry points from a checkout.  Set
    ``SLM_SKIP_RUNTIME_INSTALL=1`` to opt out.

    Called explicitly from ``main()`` — never at import time, so importing a
    job module has no side effects.
    """
    if os.environ.get("SLM_SKIP_RUNTIME_INSTALL"):
        return
    if not is_rank_zero():
        return

    path = Path(requirements_file)
    if not path.exists():
        return

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(path)],
        stdout=subprocess.DEVNULL,
    )
