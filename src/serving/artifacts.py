"""The contract between training and registration for model artifacts.

The default configuration is QLoRA — ``use_peft: true``, ``quantization:
4bit`` — and it did not survive its own registration step.  Training called
``trainer.save_model()``, which for a PEFT model writes an *adapter*
directory (``adapter_config.json`` plus adapter weights and no base
``config.json``), and registration then called
``AutoModelForCausalLM.from_pretrained`` on it, which raises.

There are two defensible answers, and the important thing is to pick one and
record which was used rather than leaving registration to guess:

``merged``
    Merge the adapter into the base weights and save a standalone model.
    Larger on disk, but serving is an ordinary ``from_pretrained`` and needs
    no PEFT at inference time.  This is the default because it makes the
    serving path simple and independent of the base model still being
    reachable.

``adapter``
    Save the adapter alone and record the base model id alongside it.  Small,
    but serving must fetch the base model and apply the adapter, so the base
    has to stay available.

Either way :func:`resolve_model_dir` hands back one local directory that
``AutoModelForCausalLM.from_pretrained`` and ``AutoTokenizer.from_pretrained``
both accept.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional, Tuple

import mlflow

# Run param carrying the logged model URI.
LOGGED_MODEL_PARAM = "logged_model_uri"

# Run tag recording how the weights were saved.
ARTIFACT_FORMAT_TAG = "model_artifact_format"

MERGED_FORMAT = "merged"
ADAPTER_FORMAT = "adapter"

# Written next to an adapter so serving can find the base model it needs.
ADAPTER_METADATA_FILE = "adapter_base_model.json"


def is_peft_model(model: Any) -> bool:
    """True when *model* is a PEFT wrapper.

    An isinstance check, not an attribute probe: ``get_peft_model`` attaches
    ``peft_config`` to the *wrapped base model* as well, so ``hasattr`` says
    yes for the unwrapped model too and merging would be attempted on
    something that cannot merge.
    """
    try:
        from peft import PeftModel
    except ImportError:
        return False

    return isinstance(model, PeftModel)


def log_model_artifacts(
    model: Any,
    tokenizer: Any,
    *,
    base_model_name: str,
    artifact_format: str = MERGED_FORMAT,
    artifact_name: str = "model",
) -> str:
    """Persist *model* and *tokenizer* to the active run; return the artifact URI.

    Must be called inside an active MLflow run.  Raises rather than warning:
    a training run that persisted no model has failed, and used to exit
    cleanly having saved nothing.
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError(
            "log_model_artifacts must be called inside an active MLflow run. "
            "Without one the artifacts land in a fresh, orphaned run, separate "
            "from the metrics."
        )

    if tokenizer is None:
        raise ValueError(
            "tokenizer is required: a language model logged without its "
            "tokenizer cannot be reloaded for registration or serving."
        )

    tmpdir = tempfile.mkdtemp(prefix="slm_model_")

    if is_peft_model(model) and artifact_format == MERGED_FORMAT:
        saved_format = _save_merged(model, tokenizer, tmpdir)
    elif is_peft_model(model):
        saved_format = _save_adapter(model, tokenizer, tmpdir, base_model_name)
    else:
        model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        saved_format = MERGED_FORMAT

    mlflow.log_artifacts(tmpdir, artifact_path=artifact_name)
    mlflow.set_tag(ARTIFACT_FORMAT_TAG, saved_format)

    return f"runs:/{run.info.run_id}/{artifact_name}"


def _save_merged(model: Any, tokenizer: Any, out_dir: str) -> str:
    """Merge LoRA weights into the base model and save a standalone checkpoint."""
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return MERGED_FORMAT


def _save_adapter(model: Any, tokenizer: Any, out_dir: str, base_model_name: str) -> str:
    """Save the adapter alone, recording which base model it belongs to."""
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # tokenizer.name_or_path points at the local directory once reloaded, so
    # the base model id has to be written down explicitly here.
    with open(os.path.join(out_dir, ADAPTER_METADATA_FILE), "w") as f:
        json.dump({"base_model_name_or_path": base_model_name}, f, indent=2)

    return ADAPTER_FORMAT


def read_adapter_base_model(model_dir: str) -> Optional[str]:
    """Return the base model id recorded beside an adapter, if any."""
    path = os.path.join(model_dir, ADAPTER_METADATA_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f).get("base_model_name_or_path")

    # Fall back to what PEFT itself records.
    adapter_config = os.path.join(model_dir, "adapter_config.json")
    if os.path.isfile(adapter_config):
        with open(adapter_config) as f:
            return json.load(f).get("base_model_name_or_path")

    return None


def is_adapter_dir(model_dir: str) -> bool:
    """True when *model_dir* holds a PEFT adapter rather than a full model."""
    return os.path.isfile(os.path.join(model_dir, "adapter_config.json")) and not (
        os.path.isfile(os.path.join(model_dir, "config.json"))
    )


def resolve_model_dir(
    run_id: Optional[str] = None,
    model_uri: Optional[str] = None,
    artifact_name: str = "model",
) -> Tuple[str, str]:
    """Download the artifacts and return ``(local_dir, artifact_format)``.

    Resolution order is *model_uri*, then the URI recorded on *run_id*, then
    the ``runs:/`` fallback.
    """
    uri, recorded_format = _resolve_uri_and_format(run_id, model_uri, artifact_name)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)

    # Trust what is on disk over what was tagged; the tag can be missing on
    # runs produced before it existed.
    if is_adapter_dir(local_path):
        return local_path, ADAPTER_FORMAT

    if not os.path.isfile(os.path.join(local_path, "config.json")):
        raise RuntimeError(
            f"Artifact at '{uri}' is neither a full model (no config.json) nor "
            f"a PEFT adapter (no adapter_config.json). Contents: "
            f"{sorted(os.listdir(local_path))}"
        )

    return local_path, recorded_format or MERGED_FORMAT


def _resolve_uri_and_format(
    run_id: Optional[str],
    model_uri: Optional[str],
    artifact_name: str,
) -> Tuple[str, Optional[str]]:
    if model_uri:
        return model_uri, None

    if not run_id:
        raise ValueError("resolve_model_dir needs either run_id or model_uri")

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    recorded_format = run.data.tags.get(ARTIFACT_FORMAT_TAG)
    stored_uri = run.data.params.get(LOGGED_MODEL_PARAM)
    if stored_uri:
        return stored_uri, recorded_format

    return f"runs:/{run_id}/{artifact_name}", recorded_format


def load_model_from_dir(model_dir: str, task_type: str = "instruction_tuning"):
    """Load a model from a resolved directory, handling both artifact formats."""
    import torch

    if task_type == "text_classification":
        from transformers import AutoModelForSequenceClassification as AutoModel
    else:
        from transformers import AutoModelForCausalLM as AutoModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if not is_adapter_dir(model_dir):
        return AutoModel.from_pretrained(model_dir, torch_dtype=dtype)

    base_model_name = read_adapter_base_model(model_dir)
    if not base_model_name:
        raise RuntimeError(
            f"{model_dir} holds a PEFT adapter but records no base model. "
            f"Re-run training, or register with artifact_format='merged'."
        )

    from peft import PeftModel

    base = AutoModel.from_pretrained(base_model_name, torch_dtype=dtype)
    return PeftModel.from_pretrained(base, model_dir).merge_and_unload()
