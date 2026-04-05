import os
import re
import time
from typing import Dict, List

from openai import OpenAI


class BaseLLMClient:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def generate_batch(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        # Fallback implementation for providers that do not support true batch requests.
        return [self.generate(messages) for messages in messages_batch]


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

        model_dtype = None
        if dtype == "auto" and torch.cuda.is_available():
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dtype == "float16":
            model_dtype = torch.float16
        elif dtype == "bfloat16":
            model_dtype = torch.bfloat16

        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "sdpa"

            # Small models are often faster on a single T4 than when sharded across 2 GPUs.
            match = re.search(r"(\d+(?:\.\d+)?)b", model.lower())
            if match and float(match.group(1)) <= 4.5 and torch.cuda.device_count() > 1 and not load_in_4bit:
                model_kwargs["device_map"] = {"": 0}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            compute_dtype = model_dtype if model_dtype is not None else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif model_dtype is not None:
            model_kwargs["dtype"] = model_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.model.eval()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = self.temperature > 0
            if self.temperature > 0:
                self.model.generation_config.temperature = self.temperature
            else:
                for attr in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                    if hasattr(self.model.generation_config, attr):
                        setattr(self.model.generation_config, attr, None)

    def _input_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _generation_kwargs(self) -> Dict[str, object]:
        do_sample = self.temperature > 0
        gen_kwargs: Dict[str, object] = {
            "max_new_tokens": self.max_output_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        return gen_kwargs

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
        inputs = {k: v.to(self._input_device()) for k, v in inputs.items()}
        gen_kwargs = self._generation_kwargs()

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate_batch(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        import torch

        if not messages_batch:
            return []

        prompts = [self._messages_to_prompt(messages) for messages in messages_batch]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
        )
        encoded = {k: v.to(self._input_device()) for k, v in encoded.items()}
        gen_kwargs = self._generation_kwargs()

        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **gen_kwargs)

        results: List[str] = []
        input_ids = encoded["input_ids"]
        # For decoder-only generation, generated tokens start after the padded input length
        # (same for all rows in a batch), not after each row's non-pad token count.
        prompt_len = input_ids.shape[1]

        for i in range(output_ids.shape[0]):
            gen_ids = output_ids[i][prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        return results
