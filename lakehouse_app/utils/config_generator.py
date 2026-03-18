"""Configuration Generator for SLM Fine-Tuning Pipeline."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigGenerator:
    """Generate YAML configurations from UI inputs."""

    TASK_MODELS = {
        "instruction_tuning": [
            {"name": "microsoft/Phi-3.5-mini-instruct", "display": "Phi-3.5 Mini Instruct", "family": "phi-3", "params": "3.8B"},
            {"name": "microsoft/Phi-4-mini-instruct", "display": "Phi-4 Mini Instruct", "family": "phi-4", "params": "3.8B"},
            {"name": "meta-llama/Llama-3.2-1B-Instruct", "display": "Llama 3.2 1B Instruct", "family": "llama", "params": "1B"},
            {"name": "meta-llama/Llama-3.2-3B-Instruct", "display": "Llama 3.2 3B Instruct", "family": "llama", "params": "3B"},
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "display": "Llama 3.1 8B Instruct", "family": "llama", "params": "8B"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "display": "Mistral 7B Instruct v0.3", "family": "mistral", "params": "7B"},
            {"name": "google/gemma-2-2b-it", "display": "Gemma 2 2B IT", "family": "gemma", "params": "2B"},
            {"name": "google/gemma-2-9b-it", "display": "Gemma 2 9B IT", "family": "gemma", "params": "9B"},
            {"name": "Qwen/Qwen2.5-3B-Instruct", "display": "Qwen 2.5 3B Instruct", "family": "qwen", "params": "3B"},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "display": "Qwen 2.5 7B Instruct", "family": "qwen", "params": "7B"},
        ],
        "dpo": [
            {"name": "microsoft/Phi-3.5-mini-instruct", "display": "Phi-3.5 Mini Instruct", "family": "phi-3", "params": "3.8B"},
            {"name": "meta-llama/Llama-3.2-3B-Instruct", "display": "Llama 3.2 3B Instruct", "family": "llama", "params": "3B"},
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "display": "Llama 3.1 8B Instruct", "family": "llama", "params": "8B"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "display": "Mistral 7B Instruct v0.3", "family": "mistral", "params": "7B"},
        ],
        "text_classification": [
            {"name": "microsoft/Phi-3.5-mini-instruct", "display": "Phi-3.5 Mini Instruct", "family": "phi-3", "params": "3.8B"},
            {"name": "meta-llama/Llama-3.2-1B-Instruct", "display": "Llama 3.2 1B Instruct", "family": "llama", "params": "1B"},
            {"name": "meta-llama/Llama-3.2-3B-Instruct", "display": "Llama 3.2 3B Instruct", "family": "llama", "params": "3B"},
            {"name": "google/gemma-2-2b-it", "display": "Gemma 2 2B IT", "family": "gemma", "params": "2B"},
        ],
    }

    DEFAULT_HYPERPARAMS = {
        "instruction_tuning": {"batch_size": 4, "learning_rate": 2e-4, "epochs": 3, "lora_r": 16},
        "dpo": {"batch_size": 2, "learning_rate": 5e-5, "epochs": 1, "lora_r": 16},
        "text_classification": {"batch_size": 8, "learning_rate": 2e-4, "epochs": 5, "lora_r": 8},
    }

    DATA_FORMAT_MAP = {
        "instruction_tuning": ["alpaca", "sharegpt"],
        "dpo": ["preference"],
        "text_classification": ["csv"],
    }

    @classmethod
    def get_models_for_task(cls, task: str) -> List[Dict[str, str]]:
        return cls.TASK_MODELS.get(task, [])

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, str]]:
        for task_models in cls.TASK_MODELS.values():
            for model in task_models:
                if model["name"] == model_name:
                    return model
        return None

    @classmethod
    def get_data_formats(cls, task: str) -> List[str]:
        return cls.DATA_FORMAT_MAP.get(task, ["alpaca"])

    @classmethod
    def get_default_hyperparams(cls, task: str) -> Dict[str, Any]:
        return cls.DEFAULT_HYPERPARAMS.get(task, cls.DEFAULT_HYPERPARAMS["instruction_tuning"])

    @classmethod
    def generate_config(
        cls,
        task: str,
        model_name: str,
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        mlflow_config: Dict[str, Any],
        output_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_info = cls.get_model_info(model_name) or {}

        config = {
            "model": {
                "model_name": model_name,
                "task_type": task,
                "trust_remote_code": True,
                "quantization": lora_config.get("quantization", "4bit"),
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "use_double_quant": True,
                "use_peft": lora_config.get("use_peft", True),
                "lora_r": lora_config.get("lora_r", 16),
                "lora_alpha": lora_config.get("lora_alpha", 32),
                "lora_dropout": lora_config.get("lora_dropout", 0.05),
                "max_new_tokens": training_config.get("max_new_tokens", 512),
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "data": {
                "train_data_path": data_config.get("train_data_path", ""),
                "val_data_path": data_config.get("val_data_path"),
                "data_format": data_config.get("data_format", "alpaca"),
                "max_seq_length": data_config.get("max_seq_length", 2048),
                "batch_size": data_config.get("batch_size", 4),
                "num_workers": data_config.get("num_workers", 4),
            },
            "training": {
                "max_epochs": training_config.get("epochs", 3),
                "learning_rate": training_config.get("learning_rate", 2e-4),
                "weight_decay": training_config.get("weight_decay", 0.01),
                "warmup_ratio": training_config.get("warmup_ratio", 0.1),
                "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
                "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 4),
                "gradient_checkpointing": True,
                "max_grad_norm": 1.0,
                "bf16": True,
                "packing": training_config.get("packing", False),
                "early_stopping_patience": training_config.get("early_stopping_patience", 3),
                "monitor_metric": "eval_loss",
                "monitor_mode": "min",
                "checkpoint_dir": training_config.get("checkpoint_dir", "/tmp/checkpoints"),
                "volume_checkpoint_dir": training_config.get("volume_checkpoint_dir"),
                "save_top_k": 3,
                "log_every_n_steps": 10,
            },
            "mlflow": {
                "experiment_name": mlflow_config.get("experiment_name", f"/Users/default/slm_{task}"),
                "run_name": mlflow_config.get("run_name", f"{task}_{model_name.split('/')[-1]}"),
                "log_model": True,
                "tags": {
                    "framework": "trl",
                    "model_family": model_info.get("family", "unknown"),
                    "task": task,
                    "method": "qlora" if lora_config.get("quantization", "4bit") != "none" else "lora",
                },
            },
            "output": {
                "results_dir": output_config.get("results_dir", "/tmp/results"),
            },
            "serving": {
                "registered_model_name": output_config.get("registered_model_name"),
                "endpoint_name": output_config.get("endpoint_name"),
                "workload_size": "Small",
                "workload_type": "GPU_SMALL",
                "scale_to_zero": True,
            },
            "monitoring": {
                "drift_threshold": 0.1,
                "error_rate_threshold": 0.05,
                "latency_p95_threshold_ms": 2000,
            },
        }

        if task == "dpo":
            config["training"]["dpo_beta"] = training_config.get("dpo_beta", 0.1)
        if task == "text_classification":
            config["model"]["num_labels"] = data_config.get("num_labels", 2)

        return config

    @classmethod
    def save_config(cls, config: Dict[str, Any], file_path: str) -> str:
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        if file_path.startswith("/Volumes"):
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            w.files.upload(file_path, io.BytesIO(content.encode("utf-8")), overwrite=True)
        else:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        return file_path

    @classmethod
    def load_config(cls, file_path: str) -> Dict[str, Any]:
        if file_path.startswith("/Volumes"):
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            resp = w.files.download(file_path)
            return yaml.safe_load(resp.contents.read())
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        for section in ["model", "data", "training"]:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        if "model" in config:
            for field in ["model_name", "task_type"]:
                if field not in config["model"]:
                    errors.append(f"Missing required field: model.{field}")
        if "data" in config:
            if "train_data_path" not in config["data"]:
                errors.append("Missing required field: data.train_data_path")
        return len(errors) == 0, errors

    @classmethod
    def get_config_preview(cls, config: Dict[str, Any]) -> str:
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
