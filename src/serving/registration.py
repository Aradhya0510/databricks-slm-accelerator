"""Model registration: save artifacts, log PyFunc, register to Unity Catalog."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models import infer_signature

from .artifacts import load_model_from_dir, resolve_model_dir, save_standalone_model


def _set_uc_registry() -> None:
    """Point the MLflow registry at Unity Catalog.

    Three-level ``catalog.schema.model`` names are only valid against the UC
    registry; relying on the workspace default failed with an opaque
    name-format error anywhere that default was not already UC.
    """
    try:
        if mlflow.get_registry_uri() != "databricks-uc":
            mlflow.set_registry_uri("databricks-uc")
    except Exception as exc:  # noqa: BLE001 - non-Databricks tracking backends
        print(f"Note: could not set the Unity Catalog registry URI ({exc}).")


def _use_run_experiment(run_id: str) -> None:
    """Log the PyFunc into the same experiment as the training run.

    ``mlflow.pyfunc.log_model`` creates a logged model, which needs an
    experiment id.  A notebook has a default one; a job's Python entry point
    does not, and the API rejects the call with a missing-field error.
    """
    try:
        experiment_id = mlflow.MlflowClient().get_run(run_id).info.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)
    except Exception as exc:  # noqa: BLE001 - keep an explicit experiment optional
        print(f"Note: could not adopt the run's experiment ({exc}).")


def register_model(
    run_id: str,
    registered_model_name: str,
    *,
    task_type: str = "instruction_tuning",
    model_uri: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
    validate: bool = True,
    test_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a PyFunc model to MLflow and register it to Unity Catalog.

    Args:
        run_id: MLflow run ID whose model artifacts to wrap.
        registered_model_name: Three-level UC name (catalog.schema.model).
        task_type: Selects the PyFunc wrapper class.
        model_uri: Direct model URI from log_model.
        aliases: Aliases for the new version (e.g. ["champion"]). "latest" is
            reserved by the registry and cannot be used.
        tags: Tags to attach to the model version.
        validate: If True, run a local prediction test before registering.
        test_prompt: Optional prompt for validation.
    """
    aliases = aliases or ["champion"]
    tags = tags or {}

    _set_uc_registry()
    _use_run_experiment(run_id)

    # Resolve the artifacts. This handles both formats the training run may
    # have written: a merged model, or a PEFT adapter plus its recorded base
    # model. Calling AutoModelForCausalLM.from_pretrained on an adapter
    # directory — which is what the default QLoRA config produces — used to
    # raise here, so the headline workflow never reached registration.
    artifact_path, artifact_format = resolve_model_dir(
        run_id=run_id, model_uri=model_uri,
    )
    print(f"Resolved {artifact_format} artifacts from {artifact_path}")

    from transformers import AutoTokenizer

    tmpdir = tempfile.mkdtemp(prefix="slm_pyfunc_")
    model_dir = os.path.join(tmpdir, "model_artifacts")

    model = load_model_from_dir(artifact_path, task_type=task_type)
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)

    # Always hand serving a standalone model directory, whichever format the
    # training run produced.
    save_standalone_model(model, tokenizer, model_dir)

    # Build signature
    import pandas as pd

    if task_type == "text_classification":
        input_example = pd.DataFrame([{"text": "Sample text for classification"}])
        output_example = [{"label": 0, "label_name": "class_0",
                           "confidence": 0.95, "status": "success"}]
    else:
        input_example = pd.DataFrame([{"prompt": "Explain machine learning."}])
        output_example = [{"response": "Machine learning is...", "status": "success"}]

    signature = infer_signature(input_example, output_example)

    # Select PyFunc wrapper
    if task_type == "text_classification":
        from .pyfunc import TextClassificationPyFuncModel
        pyfunc_model = TextClassificationPyFuncModel()
        artifact_name = "classification_pyfunc"
    else:
        from .pyfunc import TextGenerationPyFuncModel
        pyfunc_model = TextGenerationPyFuncModel()
        artifact_name = "text_generation_pyfunc"

    pip_requirements = [
        "mlflow>=3.1",
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "accelerate>=0.28",
    ]

    model_info = mlflow.pyfunc.log_model(
        name=artifact_name,
        python_model=pyfunc_model,
        artifacts={"model_dir": model_dir},
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
    )

    pyfunc_model_uri = model_info.model_uri

    # Validate
    if validate and test_prompt:
        print("Validating model with test prompt...")
        test_input = pd.DataFrame([{"prompt": test_prompt}])
        loaded = mlflow.pyfunc.load_model(pyfunc_model_uri)
        result = loaded.predict(test_input)
        print(f"Validation result: {result[0].get('status', 'unknown')}")

    # Register to Unity Catalog
    mv = mlflow.register_model(pyfunc_model_uri, registered_model_name)
    version = mv.version

    client = mlflow.MlflowClient()
    for alias in aliases:
        client.set_registered_model_alias(registered_model_name, alias, version)
    for k, v in tags.items():
        client.set_model_version_tag(registered_model_name, version, k, v)

    print(f"Registered {registered_model_name} version {version}")
    return {
        "model_uri": pyfunc_model_uri,
        "model_version": version,
        "registered_model_name": registered_model_name,
    }
