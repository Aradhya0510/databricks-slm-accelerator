"""Instruction-tuning (SFT) task — the primary SLM fine-tuning workflow.

Supports Alpaca and ShareGPT data formats.  Uses TRL's ``SFTTrainer``
directly rather than wrapping it in another layer.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from datasets import Dataset
from peft import TaskType
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from trl import SFTConfig, SFTTrainer

from ...config.schema import PipelineConfig
from ...model.loader import load_model_and_tokenizer
from ...model.peft_utils import apply_peft
from ...registry import TaskRegistry
from ..base import BaseTask
from .formatting import build_formatting_fn, load_dataset_from_config


@TaskRegistry.register("instruction_tuning")
class InstructionTuningTask(BaseTask):
    """Supervised fine-tuning via TRL's SFTTrainer."""

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
        train_ds = load_dataset_from_config(config.data, split="train")
        val_ds = None
        if config.data.val_data_path:
            val_ds = load_dataset_from_config(config.data, split="val")
        elif config.data.val_split_ratio > 0:
            split = train_ds.train_test_split(test_size=config.data.val_split_ratio, seed=42)
            train_ds, val_ds = split["train"], split["test"]

        formatting_fn = build_formatting_fn(config.data, tokenizer)
        if formatting_fn is not None:
            def _apply(examples):
                return {"text": formatting_fn(examples)}

            train_ds = train_ds.map(_apply, batched=True, remove_columns=train_ds.column_names)
            if val_ds is not None:
                val_ds = val_ds.map(_apply, batched=True, remove_columns=val_ds.column_names)

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
        sft_config = SFTConfig(
            output_dir=config.training.checkpoint_dir,
            num_train_epochs=config.training.max_epochs,
            per_device_train_batch_size=config.data.batch_size,
            per_device_eval_batch_size=config.data.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            lr_scheduler_type=config.training.lr_scheduler_type,
            max_grad_norm=config.training.max_grad_norm,
            bf16=config.training.bf16,
            fp16=config.training.fp16,
            max_length=config.data.max_seq_length,
            packing=config.training.packing,
            dataset_text_field="text",
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_steps=config.training.log_every_n_steps,
            load_best_model_at_end=val_dataset is not None,
            metric_for_best_model=config.training.monitor_metric if val_dataset else None,
            greater_is_better=(config.training.monitor_mode == "max") if val_dataset else None,
            report_to="mlflow",
            save_total_limit=config.training.save_top_k + 1,
            dataloader_num_workers=config.data.num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            deepspeed=config.training.deepspeed_config,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=callbacks or [],
        )

        return trainer
