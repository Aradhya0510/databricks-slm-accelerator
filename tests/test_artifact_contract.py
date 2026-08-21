"""The training → registration → serving artifact contract.

The default config is QLoRA (use_peft: true, quantization: 4bit), and it did
not survive its own registration step: training saved a PEFT *adapter*
directory and registration called AutoModelForCausalLM.from_pretrained on it,
which raises because there is no base config.json.

These round-trip a real — if tiny — model through the whole chain on CPU,
with no network.
"""

from __future__ import annotations

import tempfile

import pytest

mlflow = pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from src.serving.artifacts import (  # noqa: E402
    ADAPTER_FORMAT,
    ARTIFACT_FORMAT_TAG,
    LOGGED_MODEL_PARAM,
    MERGED_FORMAT,
    is_adapter_dir,
    is_peft_model,
    load_model_from_dir,
    log_model_artifacts,
    read_adapter_base_model,
    resolve_model_dir,
)


@pytest.fixture(scope="module", autouse=True)
def tracking_uri():
    d = tempfile.mkdtemp(prefix="mlflow_slm_test_")
    mlflow.set_tracking_uri(f"sqlite:///{d}/mlflow.db")
    mlflow.set_experiment("slm-artifact-contract")
    yield


@pytest.fixture
def tiny_lm():
    """A minimal Llama-architecture causal LM, built in code."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


@pytest.fixture
def tiny_tokenizer(chat_tokenizer):
    return chat_tokenizer


@pytest.fixture
def tiny_peft_model(tiny_lm):
    """The tiny LM wrapped in LoRA — what training actually hands to the logger."""
    from peft import LoraConfig, TaskType, get_peft_model

    return get_peft_model(
        tiny_lm,
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        ),
    )


def test_peft_models_are_recognised(tiny_peft_model):
    """An isinstance check, not an attribute probe.

    get_peft_model attaches peft_config to the wrapped base model too, so a
    hasattr-based check reports True for the *unwrapped* model as well — and
    merging would then be attempted on something that cannot merge.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    plain = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))

    assert is_peft_model(tiny_peft_model) is True
    assert is_peft_model(plain) is False


def test_logging_outside_a_run_is_rejected(tiny_lm, tiny_tokenizer):
    """Artifacts logged with no active run land in an orphaned run."""
    mlflow.end_run()
    with pytest.raises(RuntimeError, match="active MLflow run"):
        log_model_artifacts(tiny_lm, tiny_tokenizer, base_model_name="tiny")


def test_logging_without_a_tokenizer_is_rejected(tiny_lm):
    with mlflow.start_run():
        with pytest.raises(ValueError, match="tokenizer is required"):
            log_model_artifacts(tiny_lm, None, base_model_name="tiny")


def test_merged_format_produces_a_standalone_model(tiny_peft_model, tiny_tokenizer):
    """The default: registration must be an ordinary from_pretrained."""
    from transformers import AutoModelForCausalLM

    with mlflow.start_run() as run:
        uri = log_model_artifacts(
            tiny_peft_model, tiny_tokenizer,
            base_model_name="tiny", artifact_format=MERGED_FORMAT,
        )
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    model_dir, fmt = resolve_model_dir(run_id=run_id)

    assert fmt == MERGED_FORMAT
    assert not is_adapter_dir(model_dir)
    # This is the exact call that used to raise on the default config.
    reloaded = AutoModelForCausalLM.from_pretrained(model_dir)
    assert reloaded.config.hidden_size == 16


def test_adapter_format_records_its_base_model(tiny_peft_model, tiny_tokenizer, tmp_path):
    """Small artifacts are fine, but the base model must be recoverable.

    tokenizer.name_or_path points at the local directory once reloaded, so
    the base model id has to be written down explicitly.
    """
    with mlflow.start_run() as run:
        uri = log_model_artifacts(
            tiny_peft_model, tiny_tokenizer,
            base_model_name="meta-llama/Llama-3-8B", artifact_format=ADAPTER_FORMAT,
        )
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    model_dir, fmt = resolve_model_dir(run_id=run_id)

    assert fmt == ADAPTER_FORMAT
    assert is_adapter_dir(model_dir)
    assert read_adapter_base_model(model_dir) == "meta-llama/Llama-3-8B"


def test_the_format_is_recorded_on_the_run(tiny_peft_model, tiny_tokenizer):
    with mlflow.start_run() as run:
        log_model_artifacts(tiny_peft_model, tiny_tokenizer, base_model_name="tiny")
        run_id = run.info.run_id

    tags = mlflow.MlflowClient().get_run(run_id).data.tags
    assert tags[ARTIFACT_FORMAT_TAG] == MERGED_FORMAT


def test_a_plain_model_round_trips(tiny_lm, tiny_tokenizer):
    with mlflow.start_run() as run:
        uri = log_model_artifacts(tiny_lm, tiny_tokenizer, base_model_name="tiny")
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    model_dir, fmt = resolve_model_dir(run_id=run_id)
    assert fmt == MERGED_FORMAT
    assert load_model_from_dir(model_dir) is not None


def test_tokenizer_survives_the_round_trip(tiny_peft_model, tiny_tokenizer):
    from transformers import AutoTokenizer

    with mlflow.start_run() as run:
        uri = log_model_artifacts(tiny_peft_model, tiny_tokenizer, base_model_name="tiny")
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    model_dir, _ = resolve_model_dir(run_id=run_id)
    assert AutoTokenizer.from_pretrained(model_dir) is not None


def test_merged_weights_actually_differ_from_the_base(tiny_lm, tiny_tokenizer):
    """Merging must fold the adapter in, not just drop it."""
    import copy

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base = copy.deepcopy(tiny_lm)
    peft_model = get_peft_model(
        tiny_lm,
        LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                   target_modules=["q_proj"], bias="none",
                   task_type=TaskType.CAUSAL_LM),
    )
    # Give the adapter a non-zero contribution (lora_B starts at zero).
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.5)

    with mlflow.start_run() as run:
        uri = log_model_artifacts(peft_model, tiny_tokenizer, base_model_name="tiny")
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    model_dir, _ = resolve_model_dir(run_id=run_id)
    merged = AutoModelForCausalLM.from_pretrained(model_dir)

    base_q = base.model.layers[0].self_attn.q_proj.weight
    merged_q = merged.model.layers[0].self_attn.q_proj.weight
    assert not torch.allclose(base_q, merged_q), "adapter was not merged in"
