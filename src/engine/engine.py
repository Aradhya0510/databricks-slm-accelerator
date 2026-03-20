"""TrainingEngine — unified orchestrator that wires task, model, data, and trainer.

Each Task creates and returns the appropriate TRL/HF Trainer directly.
The engine orchestrates the flow: config -> task -> model -> data -> trainer -> train.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..utils.environment import get_gpu_count, setup_nccl_env, stage_data_to_local
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback


class TrainingEngine:
    """High-level orchestrator: config in -> metrics out.

    By default uses native HF Trainer DDP for multi-GPU on a single node.
    Pass ``distributed_mode="torchd"`` for multi-node via TorchDistributor.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        num_gpus: Optional[int] = None,
        distributed_mode: Literal["native", "torchd"] = "native",
    ) -> Dict[str, Any]:
        if num_gpus is None:
            num_gpus = get_gpu_count()
        num_gpus = max(num_gpus, 1)

        if num_gpus > 1:
            self._stage_volumes_data()

        if distributed_mode == "torchd" and num_gpus > 1:
            return self._train_torchd(num_gpus)

        return self._train_fn(num_gpus=num_gpus)

    # ------------------------------------------------------------------
    # Core training flow
    # ------------------------------------------------------------------
    def _train_fn(self, num_gpus: int = 1) -> Dict[str, Any]:
        config = self.config

        # --- lazily import only the required task to avoid pulling heavy
        # dependencies (e.g. trl for DPO) when they are not needed ---
        _task_modules = {
            "instruction_tuning": "src.tasks.instruction_tuning",
            "dpo": "src.tasks.dpo",
            "text_classification": "src.tasks.text_classification",
        }
        import importlib
        task_type = config.model.task_type
        if task_type in _task_modules:
            importlib.import_module(_task_modules[task_type])
        else:
            for mod in _task_modules.values():
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
        if num_gpus > 1:
            setup_nccl_env()

        # --- MLflow setup ---
        try:
            import mlflow
            mlflow.set_experiment(config.mlflow.experiment_name)
        except Exception as e:
            print(f"Warning: MLflow experiment setup failed: {e}")

        # --- callbacks ---
        callbacks = []
        if config.training.volume_checkpoint_dir:
            callbacks.append(
                VolumeCheckpointCallback(config.training.volume_checkpoint_dir)
            )
        if config.training.early_stopping_patience > 0 and val_ds is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience,
                )
            )

        # --- create trainer (task-specific) ---
        trainer = task.create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            callbacks=callbacks,
        )

        # --- train ---
        trainer.train()

        # --- final eval ---
        metrics = {}
        if val_ds is not None:
            try:
                metrics = trainer.evaluate()
            except RuntimeError:
                # Databricks notebook env injects NotebookProgressCallback
                # which fails on standalone evaluate() after train().
                # Remove it and retry.
                try:
                    from transformers.utils.notebook import NotebookProgressCallback
                    trainer.remove_callback(NotebookProgressCallback)
                    metrics = trainer.evaluate()
                except Exception:
                    metrics = {
                        k: v for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items()
                        if k.startswith("eval_")
                    }

        # --- log final model (rank 0 only) ---
        try:
            import mlflow
            if int(os.environ.get("LOCAL_RANK", "0")) == 0 and config.mlflow.log_model:
                if config.model.use_peft:
                    # Save merged model or adapter
                    save_dir = os.path.join(config.training.checkpoint_dir, "final_model")
                    trainer.save_model(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    mlflow.log_artifacts(save_dir, artifact_path="model")
                else:
                    model_info = mlflow.transformers.log_model(
                        transformers_model={
                            "model": model,
                            "tokenizer": tokenizer,
                        },
                        name="model",
                    )
                    mlflow.log_param("logged_model_uri", model_info.model_uri)
        except Exception as e:
            print(f"Warning: MLflow model logging failed: {e}")

        return metrics

    # ------------------------------------------------------------------
    # TorchDistributor path (multi-node)
    # ------------------------------------------------------------------
    def _train_torchd(self, num_gpus: int) -> Dict[str, Any]:
        config_dict = self.config.model_dump()

        def train_fn():
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")

            from src.config.schema import PipelineConfig
            from src.engine.engine import TrainingEngine

            config = PipelineConfig(**config_dict)
            engine = TrainingEngine(config)
            return engine._train_fn(num_gpus=1)

        from pyspark.ml.torch.distributor import TorchDistributor

        return TorchDistributor(
            num_processes=num_gpus,
            local_mode=False,
            use_gpu=True,
        ).run(train_fn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stage_volumes_data(self) -> None:
        cfg = self.config
        cfg.data.train_data_path = stage_data_to_local(cfg.data.train_data_path)
        if cfg.data.val_data_path:
            cfg.data.val_data_path = stage_data_to_local(cfg.data.val_data_path)
        if cfg.data.test_data_path:
            cfg.data.test_data_path = stage_data_to_local(cfg.data.test_data_path)
