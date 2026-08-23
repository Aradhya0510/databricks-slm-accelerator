"""DPO (Direct Preference Optimization) task for alignment training.

Uses TRL's ``DPOTrainer`` with preference pair data (prompt/chosen/rejected).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from datasets import Dataset
from peft import TaskType
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from trl import DPOConfig, DPOTrainer

from ...config.schema import PipelineConfig
from ...model.loader import load_model_and_tokenizer
from ...model.peft_utils import apply_peft
from ...registry import TaskRegistry
from ...utils.environment import resolve_precision, warmup_kwargs
from ..base import BaseTask
from .formatting import load_preference_dataset


@TaskRegistry.register("dpo")
class DPOTask(BaseTask):
    """Alignment via Direct Preference Optimization."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def load_model_and_tokenizer(
        self, config: PipelineConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model, tokenizer = load_model_and_tokenizer(config.model)

        if config.model.use_peft:
            model = apply_peft(model, config.model, task_type=TaskType.CAUSAL_LM)

        return model, tokenizer

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def prepare_datasets(
        self,
        config: PipelineConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        train_ds = load_preference_dataset(config.data, split="train")
        val_ds = None
        if config.data.val_data_path or config.data.val_table:
            val_ds = load_preference_dataset(config.data, split="val")
        elif config.data.val_split_ratio > 0:
            split = train_ds.train_test_split(
                test_size=config.data.val_split_ratio,
                seed=config.training.seed,
            )
            train_ds, val_ds = split["train"], split["test"]

        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        config: PipelineConfig,
        callbacks: list | None = None,
    ) -> Trainer:
        precision = resolve_precision(config.training.precision)

        dpo_config = DPOConfig(
            output_dir=config.training.checkpoint_dir,
            num_train_epochs=config.training.max_epochs,
            per_device_train_batch_size=config.data.batch_size,
            per_device_eval_batch_size=config.data.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            **warmup_kwargs(config.training.warmup_ratio, DPOConfig),
            lr_scheduler_type=config.training.lr_scheduler_type,
            max_grad_norm=config.training.max_grad_norm,
            bf16=(precision == "bf16"),
            fp16=(precision == "fp16"),
            seed=config.training.seed,
            data_seed=config.training.seed,
            beta=config.training.dpo_beta,
            loss_type=config.training.dpo_loss_type,
            max_length=config.data.max_seq_length,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_steps=config.training.log_every_n_steps,
            load_best_model_at_end=val_dataset is not None,
            report_to="mlflow",
            save_total_limit=config.training.save_top_k + 1,
            dataloader_num_workers=config.data.num_workers,
            remove_unused_columns=False,
            deepspeed=config.training.deepspeed_config,
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=callbacks or [],
        )

        return trainer
