"""TrainingEngine — unified orchestrator that wires task, model, data, and trainer.

Each Task creates and returns the appropriate TRL/HF Trainer directly.
The engine orchestrates the flow: config -> task -> model -> data -> trainer -> train.

Training runs on whatever AI Runtime accelerator the notebook is attached to.
A single GPU trains in-process; a multi-GPU node (8xH100, 8xB300) goes through
``@distributed``, which launches one process per GPU so HF Trainer runs real
DDP.  Running one process with several GPUs visible would give DataParallel.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..serving.artifacts import LOGGED_MODEL_PARAM, log_model_artifacts
from ..utils.environment import (
    get_gpu_count,
    is_distributed_launch,
    is_rank_zero,
    world_size,
)
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback

_TASK_MODULES = {
    "instruction_tuning": "src.tasks.instruction_tuning",
    "dpo": "src.tasks.dpo",
    "text_classification": "src.tasks.text_classification",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingResult:
    """What a training run produced: the MLflow run that holds it, and metrics."""

    run_id: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


class TrainingEngine:
    """High-level orchestrator: config in -> trained, logged model out."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, num_gpus: Optional[int] = None) -> TrainingResult:
        """Run fine-tuning on the GPUs attached to this process.

        Args:
            num_gpus: GPUs to train on. Defaults to every GPU on the node.
        """
        # Already inside a launched worker: one process, one GPU.
        if is_distributed_launch():
            return self._train_fn()

        if num_gpus is None:
            num_gpus = get_gpu_count()

        if num_gpus > 1:
            return self._train_distributed(num_gpus)
        return self._train_fn()

    # ------------------------------------------------------------------
    # Core training flow (one process, one device)
    # ------------------------------------------------------------------
    def _train_fn(self) -> TrainingResult:
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
        #
        # Under ``@distributed`` AI Runtime has already opened a run for the
        # whole fan-out and hands it to every rank as MLFLOW_RUN_ID; starting
        # a second one would split the model from its training metrics.
        run_ctx = None
        run_id = None
        if is_writer:
            managed_run_id = os.environ.get("MLFLOW_RUN_ID")
            if managed_run_id:
                run_ctx = mlflow.start_run(run_id=managed_run_id)
                mlflow.set_tag("mlflow.runName", config.mlflow.run_name)
            else:
                mlflow.set_experiment(config.mlflow.experiment_name)
                run_ctx = mlflow.start_run(run_name=config.mlflow.run_name)
            run_id = run_ctx.info.run_id

        try:
            if is_writer:
                if config.mlflow.tags:
                    mlflow.set_tags(config.mlflow.tags)
                mlflow.log_params(self._run_params())

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

        return TrainingResult(run_id=run_id, metrics=metrics)

    def _run_params(self) -> Dict[str, Any]:
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
            "num_gpus": world_size(),
            "seed": config.training.seed,
        }

    # ------------------------------------------------------------------
    # Multi-GPU: one process per GPU via AI Runtime's @distributed
    # ------------------------------------------------------------------
    def _train_distributed(self, num_gpus: int) -> TrainingResult:
        import mlflow
        from databricks.air import distributed

        config_dict = self.config.model_dump()
        project_root = str(_PROJECT_ROOT)

        # Nested so it is pickled by value: each rank is a fresh process that
        # has not imported this package, and may not have it on sys.path.
        def train_worker(config_dict: Dict[str, Any], project_root: str) -> Dict[str, Any]:
            import sys

            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.config.schema import PipelineConfig
            from src.engine.engine import TrainingEngine

            return asdict(TrainingEngine(PipelineConfig(**config_dict))._train_fn())

        # .distributed() opens its MLflow run in the active experiment.
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        # The decorator's default timeout is three hours, which a real
        # fine-tune can exceed; the job's own timeout bounds the run instead.
        launcher = distributed(num_accelerators=num_gpus, timeout=None)(train_worker)
        results = launcher.distributed(config_dict, project_root)

        # Every rank returns; only rank 0 opened the run and holds its id.
        for result in results or []:
            if result and result.get("run_id"):
                return TrainingResult(**result)
        raise RuntimeError(
            f"Distributed training on {num_gpus} GPUs returned no MLflow run id."
        )
