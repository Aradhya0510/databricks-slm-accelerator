"""Configuration Form Builder for SLM fine-tuning (Design System v1.0)."""

import streamlit as st
from typing import Any, Dict, Optional, Tuple

from components.theme import section_title
from utils.config_generator import ConfigGenerator


class ConfigFormBuilder:
    """Build dynamic configuration forms for SLM tasks."""

    @staticmethod
    def task_selector() -> str:
        task_options = {
            "Instruction Tuning (SFT)": "instruction_tuning",
            "DPO Alignment": "dpo",
            "Text Classification": "text_classification",
        }
        selected = st.selectbox(
            "Select Fine-Tuning Task",
            options=list(task_options.keys()),
            help="Choose the type of fine-tuning task",
        )
        return task_options[selected]

    @staticmethod
    def model_selector(task: str) -> Tuple[str, Dict[str, str]]:
        models = ConfigGenerator.get_models_for_task(task)
        if not models:
            st.error(f"No models available for task: {task}")
            return "", {}
        model_options = {f"{m['display']} ({m['params']})": m["name"] for m in models}
        selected = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            help="Choose a HuggingFace model to fine-tune",
        )
        model_name = model_options[selected]
        model_info = ConfigGenerator.get_model_info(model_name)
        with st.expander("Model Information"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Model ID:** `{model_name}`")
                st.markdown(f"**Parameters:** {model_info.get('params', 'Unknown')}")
            with c2:
                st.markdown(f"**Family:** {model_info.get('family', 'Unknown')}")
                st.markdown(f"**Task:** {task.replace('_', ' ').title()}")
        return model_name, model_info

    @staticmethod
    def lora_config_form(defaults: Optional[Dict] = None) -> Dict[str, Any]:
        defaults = defaults or {}
        config = {}
        c1, c2 = st.columns(2)
        with c1:
            config["use_peft"] = st.checkbox(
                "Enable LoRA", value=defaults.get("use_peft", True)
            )
            config["quantization"] = st.selectbox(
                "Quantization",
                options=["4bit", "8bit", "none"],
                index=0,
                help="QLoRA (4bit) is recommended for most use cases",
            )
        with c2:
            config["lora_r"] = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                max_value=256,
                value=defaults.get("lora_r", 16),
            )
            config["lora_alpha"] = st.number_input(
                "LoRA Alpha",
                min_value=1,
                max_value=512,
                value=defaults.get("lora_alpha", 32),
            )
        config["lora_dropout"] = st.slider(
            "LoRA Dropout",
            0.0,
            0.5,
            value=defaults.get("lora_dropout", 0.05),
            step=0.01,
        )
        return config

    @staticmethod
    def data_config_form(
        task: str, defaults: Optional[Dict] = None
    ) -> Dict[str, Any]:
        defaults = defaults or {}
        config = {}

        formats = ConfigGenerator.get_data_formats(task)
        config["data_format"] = st.selectbox(
            "Data Format", options=formats, help="Format of your training data"
        )

        config["train_data_path"] = st.text_input(
            "Training Data Path",
            value=defaults.get(
                "train_data_path",
                "/Volumes/<catalog>/<schema>/<volume>/data/train.jsonl",
            ),
        )
        config["val_data_path"] = (
            st.text_input(
                "Validation Data Path (optional)",
                value=defaults.get("val_data_path", ""),
            )
            or None
        )

        c1, c2 = st.columns(2)
        with c1:
            config["max_seq_length"] = st.number_input(
                "Max Sequence Length",
                min_value=128,
                max_value=8192,
                value=defaults.get("max_seq_length", 2048),
                step=128,
            )
            config["batch_size"] = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=64,
                value=defaults.get("batch_size", 4),
            )
        with c2:
            config["num_workers"] = st.number_input(
                "Num Workers",
                min_value=0,
                max_value=16,
                value=defaults.get("num_workers", 4),
            )
            if task == "text_classification":
                config["num_labels"] = st.number_input(
                    "Number of Labels",
                    min_value=2,
                    max_value=1000,
                    value=defaults.get("num_labels", 2),
                )

        if task == "instruction_tuning" and config["data_format"] == "alpaca":
            with st.expander("Column Mappings"):
                config["instruction_column"] = st.text_input(
                    "Instruction Column", value="instruction"
                )
                config["input_column"] = st.text_input("Input Column", value="input")
                config["output_column"] = st.text_input(
                    "Output Column", value="output"
                )
            config["system_prompt"] = (
                st.text_area(
                    "System Prompt (optional)",
                    value=defaults.get("system_prompt", ""),
                    help="System prompt prepended to every example",
                )
                or None
            )
        return config

    @staticmethod
    def training_config_form(
        task: str, defaults: Optional[Dict] = None
    ) -> Dict[str, Any]:
        defaults = defaults or ConfigGenerator.get_default_hyperparams(task)
        config = {}

        c1, c2, c3 = st.columns(3)
        with c1:
            config["epochs"] = st.number_input(
                "Epochs",
                min_value=1,
                max_value=100,
                value=defaults.get("epochs", 3),
            )
        with c2:
            config["learning_rate"] = st.number_input(
                "Learning Rate",
                min_value=1e-7,
                max_value=1.0,
                value=defaults.get("learning_rate", 2e-4),
                format="%.2e",
            )
        with c3:
            config["weight_decay"] = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=1.0,
                value=defaults.get("weight_decay", 0.01),
                format="%.3f",
            )

        c1, c2 = st.columns(2)
        with c1:
            config["warmup_ratio"] = st.slider(
                "Warmup Ratio", 0.0, 0.5, value=0.1, step=0.05
            )
            config["lr_scheduler_type"] = st.selectbox(
                "LR Scheduler",
                options=["cosine", "linear", "constant", "constant_with_warmup"],
            )
        with c2:
            config["gradient_accumulation_steps"] = st.number_input(
                "Gradient Accumulation Steps", min_value=1, max_value=64, value=4
            )
            config["packing"] = st.checkbox(
                "Enable Packing",
                value=False,
                help="Pack multiple examples into one sequence",
            )

        if task == "dpo":
            config["dpo_beta"] = st.slider(
                "DPO Beta",
                0.01,
                1.0,
                value=0.1,
                step=0.01,
                help="KL penalty coefficient for DPO",
            )

        section_title("Checkpointing")
        config["checkpoint_dir"] = st.text_input(
            "Checkpoint Directory", value="/tmp/checkpoints"
        )
        config["volume_checkpoint_dir"] = (
            st.text_input("Volume Checkpoint Directory (persistent)", value="")
            or None
        )
        config["early_stopping_patience"] = st.number_input(
            "Early Stopping Patience", min_value=0, max_value=50, value=3
        )
        return config

    @staticmethod
    def mlflow_config_form(defaults: Optional[Dict] = None) -> Dict[str, Any]:
        defaults = defaults or {}
        config = {}
        config["experiment_name"] = st.text_input(
            "Experiment Name (Workspace Path)",
            value=defaults.get(
                "experiment_name",
                "/Users/<email@databricks.com>/slm_experiments",
            ),
        )
        st.info(
            "Experiment name must be an absolute workspace path, e.g., "
            "`/Users/your.email@databricks.com/slm_sft`"
        )
        config["run_name"] = st.text_input(
            "Run Name", value=defaults.get("run_name", "")
        )
        return config

    @staticmethod
    def output_config_form(defaults: Optional[Dict] = None) -> Dict[str, Any]:
        defaults = defaults or {}
        config = {}
        config["results_dir"] = st.text_input(
            "Results Directory",
            value=defaults.get(
                "results_dir", "/Volumes/<catalog>/<schema>/<volume>/results"
            ),
        )
        return config
