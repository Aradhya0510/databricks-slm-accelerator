"""Text classification task — fine-tune a causal LM with a classification head.

Uses HuggingFace's ``AutoModelForSequenceClassification`` with optional
quantization + LoRA, and a standard HF ``Trainer``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from datasets import Dataset
from peft import TaskType
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from ...config.schema import PipelineConfig
from ...model.loader import load_model_and_tokenizer
from ...model.peft_utils import apply_peft
from ...registry import TaskRegistry
from ..base import BaseTask
from .formatting import load_classification_dataset, tokenize_classification_dataset


@TaskRegistry.register("text_classification")
class TextClassificationTask(BaseTask):
    """Sequence classification via a causal LM backbone + classification head."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def load_model_and_tokenizer(
        self, config: PipelineConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model, tokenizer = load_model_and_tokenizer(
            config.model, for_classification=True,
        )

        if config.model.use_peft:
            model = apply_peft(
                model, config.model, task_type=TaskType.SEQ_CLS,
            )

        return model, tokenizer

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def prepare_datasets(
        self,
        config: PipelineConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        train_ds = load_classification_dataset(config.data, split="train")
        val_ds = None
        if config.data.val_data_path:
            val_ds = load_classification_dataset(config.data, split="val")
        elif config.data.val_split_ratio > 0:
            split = train_ds.train_test_split(test_size=config.data.val_split_ratio, seed=42)
            train_ds, val_ds = split["train"], split["test"]

        train_ds = tokenize_classification_dataset(train_ds, tokenizer, config.data)
        if val_ds:
            val_ds = tokenize_classification_dataset(val_ds, tokenizer, config.data)

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
        training_args = TrainingArguments(
            output_dir=config.training.checkpoint_dir,
            num_train_epochs=config.training.max_epochs,
            per_device_train_batch_size=config.data.batch_size,
            per_device_eval_batch_size=config.data.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            lr_scheduler_type=config.training.lr_scheduler_type,
            max_grad_norm=config.training.max_grad_norm,
            bf16=config.training.bf16,
            fp16=config.training.fp16,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_steps=config.training.log_every_n_steps,
            load_best_model_at_end=val_dataset is not None,
            metric_for_best_model="eval_accuracy" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            report_to="mlflow",
            save_total_limit=config.training.save_top_k + 1,
            dataloader_num_workers=config.data.num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            deepspeed=config.training.deepspeed_config,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks or [],
        )

        return trainer

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        total = len(labels)
        correct = (preds == labels).sum()
        accuracy = correct / max(total, 1)

        num_classes = logits.shape[-1]
        metrics = {"accuracy": float(accuracy)}

        precisions, recalls, f1s = [], [], []
        for c in range(num_classes):
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        metrics["precision_macro"] = float(np.mean(precisions))
        metrics["recall_macro"] = float(np.mean(recalls))
        metrics["f1_macro"] = float(np.mean(f1s))

        return metrics
