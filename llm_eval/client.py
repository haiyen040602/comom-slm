import os
import time
from typing import Dict, List

from openai import OpenAI


class BaseLLMClient:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)

        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_error}")


class HuggingFaceLocalClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        dtype: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace local mode requires transformers and torch. "
                "Please install requirements for hf-local provider."
            ) from exc

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_name = model

        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model_kwargs = {
            "device_map": "auto",
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        lines = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        import torch

        prompt = self._messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_output_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
