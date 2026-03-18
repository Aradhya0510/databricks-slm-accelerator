"""Base task interface for the SLM pipeline.

This ABC defines the contract explicitly so new tasks are self-documenting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer

from ..config.schema import PipelineConfig


class BaseTask(ABC):
    """Every SLM task must implement these methods.

    The ``TrainingEngine`` calls them in order:

    1. ``load_model_and_tokenizer`` — load (+ optionally quantize & apply PEFT)
    2. ``prepare_datasets`` — return formatted HF Datasets ready for training
    3. ``create_trainer`` — return a fully configured Trainer (TRL or HF)
    4. ``compute_metrics`` — optional hook used by the Trainer during eval
    """

    @abstractmethod
    def load_model_and_tokenizer(
        self, config: PipelineConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer, applying quantization + PEFT as needed."""

    @abstractmethod
    def prepare_datasets(
        self,
        config: PipelineConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Return ``(train_dataset, val_dataset)``; val may be ``None``."""

    @abstractmethod
    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        config: PipelineConfig,
        callbacks: list | None = None,
    ) -> Trainer:
        """Build and return the appropriate Trainer for this task."""

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Optional: task-specific metric computation.

        Override in subclasses that need custom eval metrics.
        Default returns empty dict (Trainer computes loss automatically).
        """
        return {}
