"""Model registration: save artifacts, log PyFunc, register to Unity Catalog."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models import infer_signature


def _resolve_model_artifacts(
    run_id: str,
    model_uri: Optional[str] = None,
) -> str:
    """Download the model artifacts logged during training."""
    if model_uri:
        return mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    stored_uri = run.data.params.get("logged_model_uri")
    if stored_uri:
        return mlflow.artifacts.download_artifacts(artifact_uri=stored_uri)

    return mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model",
    )


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
        aliases: Aliases for the new version (e.g. ["champion"]).
        tags: Tags to attach to the model version.
        validate: If True, run a local prediction test before registering.
        test_prompt: Optional prompt for validation.
    """
    aliases = aliases or ["champion", "latest"]
    tags = tags or {}

    artifact_path = _resolve_model_artifacts(run_id, model_uri)

    # Save model + tokenizer to a clean temp directory
    tmpdir = tempfile.mkdtemp(prefix="slm_pyfunc_")
    model_dir = os.path.join(tmpdir, "model_artifacts")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if task_type == "text_classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(artifact_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(artifact_path)

    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

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
