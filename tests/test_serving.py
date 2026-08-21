"""Serving wrapper input handling and artifact-format dispatch."""

from __future__ import annotations

import json

import pytest

from src.serving.pyfunc import _is_adapter_dir, _read_adapter_base_model


def test_adapter_directory_is_recognised(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "meta-llama/Llama-3-8B"})
    )
    assert _is_adapter_dir(str(tmp_path)) is True


def test_full_model_directory_is_not_an_adapter(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert _is_adapter_dir(str(tmp_path)) is False


def test_merged_directory_with_both_files_is_not_an_adapter(tmp_path):
    """A merged save can leave both files; config.json wins."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    assert _is_adapter_dir(str(tmp_path)) is False


def test_base_model_is_read_from_the_explicit_metadata_file(tmp_path):
    """tokenizer.name_or_path points at the local dir, so this must be explicit."""
    (tmp_path / "adapter_base_model.json").write_text(
        json.dumps({"base_model_name_or_path": "microsoft/Phi-3.5-mini-instruct"})
    )
    assert _read_adapter_base_model(str(tmp_path)) == "microsoft/Phi-3.5-mini-instruct"


def test_base_model_falls_back_to_peft_own_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "meta-llama/Llama-3-8B"})
    )
    assert _read_adapter_base_model(str(tmp_path)) == "meta-llama/Llama-3-8B"


def test_missing_base_model_returns_none(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert _read_adapter_base_model(str(tmp_path)) is None


def test_prompt_extraction_handles_the_documented_shapes():
    from src.serving.pyfunc import TextGenerationPyFuncModel

    wrapper = TextGenerationPyFuncModel()
    wrapper.tokenizer = type("T", (), {"chat_template": None})()

    assert wrapper._extract_prompt("plain string") == "plain string"
    assert wrapper._extract_prompt({"prompt": "p"}) == "p"
    assert wrapper._extract_prompt({"text": "t"}) == "t"
    assert wrapper._extract_prompt({"instruction": "i"}) == "i"


def test_unknown_prompt_shape_is_rejected():
    from src.serving.pyfunc import TextGenerationPyFuncModel

    wrapper = TextGenerationPyFuncModel()
    with pytest.raises(ValueError, match="Cannot extract prompt"):
        wrapper._extract_prompt(12345)


def test_input_normalisation_covers_serving_payload_shapes():
    from src.serving.pyfunc import TextGenerationPyFuncModel

    norm = TextGenerationPyFuncModel._normalize_input
    assert norm({"dataframe_records": [{"prompt": "a"}]}) == [{"prompt": "a"}]
    assert norm({"instances": [{"prompt": "b"}]}) == [{"prompt": "b"}]
    assert norm([{"prompt": "c"}]) == [{"prompt": "c"}]
