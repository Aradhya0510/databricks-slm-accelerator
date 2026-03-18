"""PyFunc wrappers for Databricks Model Serving — text generation and classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlflow


class TextGenerationPyFuncModel(mlflow.pyfunc.PythonModel):
    """Wraps a fine-tuned causal LM for Databricks Model Serving.

    Accepts chat-style input (messages array) or plain text prompts.
    """

    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model_dir = context.artifacts["model_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        except Exception:
            base_model_name = self.tokenizer.name_or_path
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(self.model, model_dir)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        import torch

        params = params or {}
        max_tokens = int(params.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        temperature = float(params.get("temperature", self.DEFAULT_TEMPERATURE))

        records = self._normalize_input(model_input)
        results = []

        for record in records:
            try:
                result = self._generate(record, max_tokens, temperature)
                results.append({"response": result, "status": "success"})
            except Exception as e:
                results.append({"response": "", "status": "error", "error": str(e)})

        return results

    def _generate(
        self,
        record: Any,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import torch

        prompt = self._extract_prompt(record)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _extract_prompt(self, record: Any) -> str:
        """Extract a text prompt from various input formats."""
        if isinstance(record, str):
            return record

        if isinstance(record, dict):
            # Chat-style messages
            if "messages" in record:
                messages = record["messages"]
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            if "prompt" in record:
                return record["prompt"]
            if "text" in record:
                return record["text"]
            if "instruction" in record:
                return record["instruction"]

        raise ValueError(f"Cannot extract prompt from input type: {type(record)}")

    @staticmethod
    def _normalize_input(model_input: Any) -> List[Any]:
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        if isinstance(model_input, dict):
            if "instances" in model_input:
                return model_input["instances"]
            if "inputs" in model_input:
                return model_input["inputs"]
            if "dataframe_records" in model_input:
                return model_input["dataframe_records"]
            return [model_input]
        if isinstance(model_input, list):
            return model_input
        return [model_input]


class TextClassificationPyFuncModel(mlflow.pyfunc.PythonModel):
    """Wraps a fine-tuned sequence classification model for Model Serving."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = context.artifacts["model_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.id2label = getattr(self.model.config, "id2label", None) or {}

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        import torch

        records = TextGenerationPyFuncModel._normalize_input(model_input)
        results = []

        for record in records:
            try:
                text = record.get("text", record) if isinstance(record, dict) else str(record)
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                predicted_class = probs.argmax().item()

                results.append({
                    "label": predicted_class,
                    "label_name": self.id2label.get(predicted_class, str(predicted_class)),
                    "confidence": probs[predicted_class].item(),
                    "status": "success",
                })
            except Exception as e:
                results.append({"status": "error", "error": str(e)})

        return results
