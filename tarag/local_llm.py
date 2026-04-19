from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start_indexes = [idx for idx, char in enumerate(text) if char == start_char]
        for start in start_indexes:
            depth = 0
            for end in range(start, len(text)):
                char = text[end]
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : end + 1])
                        break
    return candidates


def parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response from model.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for candidate in _extract_json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Cannot parse JSON from model output: {text[:200]}")


class LocalLLM:
    def __init__(
        self,
        model_dir: str | Path,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = Path(model_dir)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local LLM path not found: {self.model_path}. "
                "Please put your model under ./models first."
            )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()

    def _build_inputs(self, system_prompt: str, user_prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = (
                f"System:\n{system_prompt}\n\n"
                f"User:\n{user_prompt}\n\n"
                "Assistant:\n"
            )
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"]
        return input_ids

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        input_ids = self._build_inputs(system_prompt, user_prompt).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        actual_temperature = self.temperature if temperature is None else temperature
        actual_max_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        do_sample = actual_temperature > 0
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id or eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=actual_max_tokens,
                do_sample=do_sample,
                temperature=max(actual_temperature, 1e-5),
                top_p=self.top_p,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: Any,
        max_new_tokens: int | None = None,
    ) -> Any:
        output = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        try:
            return parse_json_from_text(output)
        except ValueError:
            return fallback
