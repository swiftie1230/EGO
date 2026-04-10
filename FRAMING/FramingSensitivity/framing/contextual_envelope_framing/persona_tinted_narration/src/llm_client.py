# llm_client.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional

import requests


# =====================================================
# OpenRouter client
# =====================================================

class OpenRouterClient:
    """
    Minimal OpenAI-compatible client for OpenRouter.

    Env:
      - OPENROUTER_API_KEY (required)
      - OR_SITE (optional)
      - OR_APP (optional)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        site: str = "",
        app: str = "",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY env var.")

        self.site = site or os.environ.get("OR_SITE", "")
        self.app = app or os.environ.get("OR_APP", "")
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 0.95,
        retries: int = 4,
        timeout: int = 120,
    ) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.site:
            headers["HTTP-Referer"] = self.site
        if self.app:
            headers["X-Title"] = self.app

        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Empty completion.")

                return content.strip()

            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"OpenRouter generate failed after retries. Last error: {last_err}"
        )


# =====================================================
# Hugging Face client
# =====================================================

class HuggingFaceClient:
    """
    Local HuggingFace inference client.

    Example:
        client = HuggingFaceClient(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            device="cuda:0",
            dtype="bfloat16"
        )
        text = client.generate(prompt)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "HuggingFace backend requires transformers + torch.\n"
                "pip install -U transformers accelerate torch"
            ) from e

        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if self.dtype not in dtype_map:
            raise ValueError(
                f"dtype must be one of {list(dtype_map.keys())}, got {self.dtype}"
            )

        torch_dtype = dtype_map[self.dtype]

        if self.device == "auto":
            device_map = "auto"
        else:
            device_map = {"": self.device}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,   # kept for interface compatibility
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 0.95,
        retries: int = 1,
        timeout: int = 0,
    ) -> str:
        """
        Same interface as OpenRouterClient.generate()
        """

        import torch

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Prefer chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
            #text = f"[USER]\n{prompt}\n\n[ASSISTANT]\n"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        for k in inputs:
            inputs[k] = inputs[k].to(self.model.device)
            
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.0

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(
            output[0][input_len:],
            skip_special_tokens=True,
        )

        return decoded.strip()

