"""Databricks environment helpers — GPU detection, NCCL setup, data staging."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def get_gpu_count() -> int:
    """Return the number of CUDA GPUs available, falling back to 0."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def setup_nccl_env() -> None:
    """Configure NCCL environment variables for Databricks multi-GPU networking."""
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")


def stage_data_to_local(path: str, local_root: str = "/tmp/staged_data") -> str:
    """Copy a /Volumes/ path to local disk so DDP workers can access it.

    Returns the original path unchanged if it is not a /Volumes/ path.
    """
    if not path.startswith("/Volumes/"):
        return path

    relative = path.lstrip("/")
    local_path = os.path.join(local_root, relative)

    if os.path.exists(local_path):
        return local_path

    src = Path(path)
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
