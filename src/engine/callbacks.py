"""HF Trainer callbacks for checkpoint management and early stopping."""

from __future__ import annotations

import os
import shutil

from transformers import (
    EarlyStoppingCallback as _HFEarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class VolumeCheckpointCallback(TrainerCallback):
    """Copy Trainer checkpoints to a persistent /Volumes/ directory.

    Ensures checkpoints survive ephemeral cluster storage.
    """

    def __init__(self, volume_dir: str):
        self.volume_dir = volume_dir
        os.makedirs(self.volume_dir, exist_ok=True)
        print(f"Checkpoints will be copied to volume: {self.volume_dir}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        dest = os.path.join(self.volume_dir, f"checkpoint-{state.global_step}")
        try:
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(ckpt_dir, dest)
            print(f"Copied checkpoint to volume: {dest}")
        except Exception as e:
            print(f"Warning: failed to copy checkpoint to volume: {e}")


class EarlyStoppingCallback(_HFEarlyStoppingCallback):
    """Thin wrapper around HF's EarlyStoppingCallback with sensible defaults."""

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
