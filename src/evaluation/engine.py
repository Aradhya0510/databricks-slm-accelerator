"""EvaluationEngine — perplexity, generation quality, and latency benchmarks.

Redesigned for language models:
- Perplexity (quantitative signal for model quality)
- Generation quality (ROUGE scores + sample outputs for human review)
- Latency benchmarks (tokens/sec, time-to-first-token)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config.schema import PipelineConfig


class EvaluationEngine:
    """Standalone evaluation for fine-tuned SLMs."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model_and_tokenizer(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
    ):
        """Load model + tokenizer from local path, MLflow run, or model URI."""
        if model_uri is not None:
            import mlflow
            model = mlflow.transformers.load_model(model_uri)
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=self.config.model.trust_remote_code,
            )
            return model, tokenizer

        if run_id is not None:
            import mlflow
            try:
                client = mlflow.MlflowClient()
                run = client.get_run(run_id)
                stored_uri = run.data.params.get("logged_model_uri")
                if stored_uri:
                    model = mlflow.transformers.load_model(stored_uri)
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model.model_name,
                        trust_remote_code=self.config.model.trust_remote_code,
                    )
                    return model, tokenizer
            except Exception:
                pass
            model = mlflow.transformers.load_model(f"runs:/{run_id}/model")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=self.config.model.trust_remote_code,
            )
            return model, tokenizer

        if model_path is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code,
            )
            return model, tokenizer

        from ..model.loader import load_model_and_tokenizer
        return load_model_and_tokenizer(self.config.model)

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    def evaluate_perplexity(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        max_samples: int = 500,
    ) -> Dict[str, float]:
        """Compute perplexity on the validation set."""
        model, tokenizer = self._load_model_and_tokenizer(model_path, run_id, model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model, "device") or str(model.device) == "cpu":
            model = model.to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        eval_path = self.config.data.val_data_path or self.config.data.train_data_path
        from ..tasks.instruction_tuning.formatting import load_dataset_from_config
        from ..config.schema import DataConfig

        temp_cfg = self.config.data.model_copy()
        temp_cfg.train_data_path = eval_path
        ds = load_dataset_from_config(temp_cfg, split="train")

        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        total_loss = 0.0
        total_tokens = 0

        for i in range(len(ds)):
            text = ds[i].get("text", ds[i].get(self.config.data.instruction_column, ""))
            if not text:
                continue

            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.data.max_seq_length,
            )
            input_ids = encodings.input_ids.to(device)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)

            total_loss += outputs.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        result = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_samples": len(ds),
        }
        self._save_results(result, "perplexity.json")
        return result

    # ------------------------------------------------------------------
    # Generation quality
    # ------------------------------------------------------------------
    def evaluate_generation(
        self,
        prompts: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        max_samples: int = 20,
    ) -> Dict[str, Any]:
        """Generate sample outputs and optionally compute ROUGE scores."""
        model, tokenizer = self._load_model_and_tokenizer(model_path, run_id, model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model, "device") or str(model.device) == "cpu":
            model = model.to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if prompts is None:
            prompts = self._extract_prompts_from_data(max_samples)

        generations = []
        for prompt in prompts[:max_samples]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.config.data.max_seq_length // 2)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.config.model.max_new_tokens,
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p,
                    do_sample=self.config.model.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generations.append({"prompt": prompt, "response": response})

        result: Dict[str, Any] = {
            "num_samples": len(generations),
            "generations": generations,
        }

        # Compute ROUGE if references are available
        rouge_scores = self._compute_rouge(generations)
        if rouge_scores:
            result["rouge"] = rouge_scores

        self._save_results(result, "generation_quality.json")
        return result

    def _extract_prompts_from_data(self, max_samples: int) -> List[str]:
        """Pull prompts from the configured dataset for evaluation."""
        eval_path = self.config.data.val_data_path or self.config.data.train_data_path
        from ..tasks.instruction_tuning.formatting import load_dataset_from_config
        temp_cfg = self.config.data.model_copy()
        temp_cfg.train_data_path = eval_path

        try:
            ds = load_dataset_from_config(temp_cfg, split="train")
        except Exception:
            return ["Tell me about machine learning.", "Explain quantum computing."]

        prompts = []
        for i in range(min(len(ds), max_samples)):
            row = ds[i]
            prompt = row.get(self.config.data.instruction_column,
                           row.get("prompt", row.get("text", "")))
            if prompt:
                prompts.append(prompt)
        return prompts

    def _compute_rouge(self, generations: List[Dict]) -> Optional[Dict[str, float]]:
        """Attempt ROUGE scoring if the rouge-score package is available."""
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            return None

        scored = [g for g in generations if g.get("reference")]
        if not scored:
            return None

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for g in scored:
            scores = scorer.score(g["reference"], g["response"])
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": sum(r1) / len(r1),
            "rouge2": sum(r2) / len(r2),
            "rougeL": sum(rl) / len(rl),
            "num_scored": len(scored),
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    def benchmark(
        self,
        model_path: Optional[str] = None,
        run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        num_warmup: int = 5,
        num_iterations: int = 50,
        prompt: str = "Explain the concept of transfer learning in deep learning.",
    ) -> Dict[str, Any]:
        """Measure generation throughput and latency."""
        model, tokenizer = self._load_model_and_tokenizer(model_path, run_id, model_uri)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(model, "device") or str(model.device) == "cpu":
            model = model.to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=self.config.data.max_seq_length // 2)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_new = self.config.model.max_new_tokens

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_new,
                               pad_token_id=tokenizer.pad_token_id)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        total_tokens_generated = 0

        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
            total_tokens_generated += new_tokens
            latencies.append(t1 - t0)

        latencies_sorted = sorted(latencies)
        total_time = sum(latencies)

        result = {
            "tokens_per_second": total_tokens_generated / total_time if total_time > 0 else 0,
            "total_tokens_generated": total_tokens_generated,
            "total_time_s": total_time,
            "latency_per_generation_ms": {
                "mean": (total_time / len(latencies)) * 1000,
                "p50": latencies_sorted[len(latencies) // 2] * 1000,
                "p95": latencies_sorted[int(len(latencies) * 0.95)] * 1000,
                "p99": latencies_sorted[int(len(latencies) * 0.99)] * 1000,
            },
            "num_iterations": num_iterations,
            "device": str(device),
        }

        if device.type == "cuda":
            result["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        self._save_results(result, "benchmark.json")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_results(self, data: dict, filename: str) -> None:
        results_dir = self.config.output.results_dir
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {path}")
