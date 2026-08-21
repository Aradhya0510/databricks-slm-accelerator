"""TrainingEngine — unified orchestrator that wires task, model, data, and trainer.

Each Task creates and returns the appropriate TRL/HF Trainer directly.
The engine orchestrates the flow: config -> task -> model -> data -> trainer -> train.

Single-node multi-GPU goes through ``TorchDistributor(local_mode=True)``, which
launches one process per GPU so HF Trainer runs real DDP.  Running this as a
plain process with several GPUs visible would give DataParallel instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..serving.artifacts import LOGGED_MODEL_PARAM, log_model_artifacts
from ..utils.environment import (
    get_gpu_count,
    is_distributed_launch,
    is_rank_zero,
    setup_nccl_env,
    stage_data_to_local,
    world_size,
)
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback

_TASK_MODULES = {
    "instruction_tuning": "src.tasks.instruction_tuning",
    "dpo": "src.tasks.dpo",
    "text_classification": "src.tasks.text_classification",
}


class TrainingEngine:
    """High-level orchestrator: config in -> metrics out."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        num_gpus: Optional[int] = None,
        distributed_mode: Literal["auto", "single", "local", "multinode"] = "auto",
    ) -> Dict[str, Any]:
        """Run fine-tuning.

        Args:
            num_gpus: GPUs to use.  Auto-detected when ``None``.
            distributed_mode:
                ``"auto"`` (default) — one process per visible GPU on this node.
                ``"single"`` — force a single process.
                ``"local"`` — single-node multi-process DDP.
                ``"multinode"`` — distribute across Spark workers.
        """
        if num_gpus is None:
            num_gpus = get_gpu_count()
        num_gpus = max(num_gpus, 1)

        # Already inside a launched worker: just train, one process one GPU.
        if is_distributed_launch():
            return self._train_fn(num_gpus=1)

        if distributed_mode == "auto":
            distributed_mode = "local" if num_gpus > 1 else "single"

        if distributed_mode == "single":
            if num_gpus > 1:
                # Pin to one GPU so HF Trainer cannot silently pick DataParallel.
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            return self._train_fn(num_gpus=1)

        self._stage_volumes_data()
        return self._train_distributed(num_gpus, local_mode=(distributed_mode == "local"))

    # ------------------------------------------------------------------
    # Core training flow (one process, one device)
    # ------------------------------------------------------------------
    def _train_fn(self, num_gpus: int = 1) -> Dict[str, Any]:
        import importlib

        import mlflow
        from transformers import set_seed

        config = self.config
        is_writer = is_rank_zero()

        set_seed(config.training.seed)

        # --- lazily import only the required task to avoid pulling heavy
        # dependencies (e.g. trl for DPO) when they are not needed ---
        task_type = config.model.task_type
        if task_type in _TASK_MODULES:
            importlib.import_module(_TASK_MODULES[task_type])
        else:
            for mod in _TASK_MODULES.values():
                importlib.import_module(mod)

        task = TaskRegistry.get(task_type)

        # --- model + tokenizer (quantization + PEFT applied inside task) ---
        model, tokenizer = task.load_model_and_tokenizer(config)

        # --- datasets ---
        train_ds, val_ds = task.prepare_datasets(config, tokenizer)
        print(f"Training samples: {len(train_ds)}")
        if val_ds:
            print(f"Validation samples: {len(val_ds)}")

        # --- NCCL env for Databricks networking ---
        if world_size() > 1:
            setup_nccl_env()

        # --- callbacks ---
        callbacks = []
        if config.training.volume_checkpoint_dir:
            callbacks.append(
                VolumeCheckpointCallback(
                    config.training.volume_checkpoint_dir,
                    save_total_limit=config.training.save_top_k + 1,
                )
            )
        if config.training.early_stopping_patience > 0 and val_ds is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience,
                )
            )

        # --- MLflow: one run for the whole lifecycle ---
        # The run must be opened *before* the trainer is built.  HF's
        # MLflowCallback checks mlflow.active_run(): with no run it creates one
        # and sets _auto_end_run=True, then closes it in on_train_end — so
        # everything logged after trainer.train() (the final evaluate, and the
        # model artifacts) landed in fresh, orphaned runs, separate from the
        # metrics.
        mlflow.set_experiment(config.mlflow.experiment_name)
        run_ctx = mlflow.start_run(run_name=config.mlflow.run_name) if is_writer else None

        try:
            if is_writer:
                if config.mlflow.tags:
                    mlflow.set_tags(config.mlflow.tags)
                mlflow.log_params(self._run_params(num_gpus))

            trainer = task.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=config,
                callbacks=callbacks,
            )

            trainer.train()

            metrics = {}
            if val_ds is not None:
                metrics = trainer.evaluate()

            # --- log the final model, inside the same run ---
            # Deliberately not wrapped in try/except: a run that trained but
            # persisted no model has failed, and used to exit cleanly with only
            # a printed warning.
            if is_writer and config.mlflow.log_model:
                model_uri = log_model_artifacts(
                    trainer.model,
                    tokenizer,
                    base_model_name=config.model.model_name,
                    artifact_format=config.mlflow.artifact_format,
                )
                mlflow.log_param(LOGGED_MODEL_PARAM, model_uri)

            if is_writer and config.training.volume_checkpoint_dir:
                mlflow.log_param("checkpoint_dir", config.training.volume_checkpoint_dir)

        finally:
            if run_ctx is not None:
                mlflow.end_run()

        return metrics

    def _run_params(self, num_gpus: int) -> Dict[str, Any]:
        """Parameters worth being able to find a run by, later."""
        config = self.config
        return {
            "task_type": config.model.task_type,
            "model_name": config.model.model_name,
            "quantization": config.model.quantization,
            "use_peft": config.model.use_peft,
            "lora_r": config.model.lora_r,
            "lora_alpha": config.model.lora_alpha,
            "max_epochs": config.training.max_epochs,
            "batch_size": config.data.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "max_seq_length": config.data.max_seq_length,
            "completion_only_loss": config.training.completion_only_loss,
            "num_gpus": num_gpus,
            "world_size": world_size(),
            "seed": config.training.seed,
        }

    # ------------------------------------------------------------------
    # TorchDistributor paths
    # ------------------------------------------------------------------
    def _train_distributed(self, num_gpus: int, local_mode: bool) -> Dict[str, Any]:
        """Launch one process per GPU via TorchDistributor."""
        config_dict = self.config.model_dump()

        def train_fn():
            import os

            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")

            from src.config.schema import PipelineConfig
            from src.engine.engine import TrainingEngine

            config = PipelineConfig(**config_dict)
            return TrainingEngine(config)._train_fn(num_gpus=1)

        from pyspark.ml.torch.distributor import TorchDistributor

        return TorchDistributor(
            num_processes=num_gpus,
            local_mode=local_mode,
            use_gpu=True,
        ).run(train_fn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stage_volumes_data(self) -> None:
        cfg = self.config
        if cfg.data.train_data_path:
            cfg.data.train_data_path = stage_data_to_local(cfg.data.train_data_path)
        if cfg.data.val_data_path:
            cfg.data.val_data_path = stage_data_to_local(cfg.data.val_data_path)
        if cfg.data.test_data_path:
            cfg.data.test_data_path = stage_data_to_local(cfg.data.test_data_path)
