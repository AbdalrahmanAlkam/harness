"""Local single-forward-pass semantic decisions from next-token logits.

SemIf scores a small set of typed branches directly from a causal model's
logits. It never generates decision text or parses model output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Mapping


@dataclass(frozen=True)
class SemIfDecision:
    selected_option: str
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    margin: float
    latency_ms: float


class SemIfEngine:
    """Probe candidate option token logits with one lazy local model pass."""

    DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, device: str = "auto",
                 load_in_4bit: bool = False, temperature: float = 1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("SemIf temperature must be finite and greater than zero")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("SemIf device must be auto, cpu, cuda, or mps")
        self.model_name_or_path = str(model_name_or_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    @property
    def model_label(self) -> str:
        return Path(self.model_name_or_path).name or self.model_name_or_path

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("SemIf requires optional dependencies: install adaptive-harness[semif]") from exc

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            else:
                device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("SemIf device=cuda requested, but CUDA is unavailable")
            if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError("SemIf device=mps requested, but MPS is unavailable")
            if self.load_in_4bit and device != "cuda":
                raise RuntimeError("SemIf 4-bit quantization currently requires CUDA and bitsandbytes")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
            kwargs = {"local_files_only": True}
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise RuntimeError("Install bitsandbytes to use SemIf 4-bit quantization") from exc
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = torch.float16 if device in {"cuda", "mps"} else torch.float32
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
            if not self.load_in_4bit:
                model.to(device)
            model.eval()
            self._torch, self._tokenizer, self._model = torch, tokenizer, model

    @staticmethod
    def _branch_ids(tokenizer, count: int) -> list[int]:
        if count > 26:
            raise ValueError("SemIf supports at most 26 candidate options per decision")
        ids = []
        for index in range(count):
            token_ids = tokenizer.encode(f" {chr(65 + index)}", add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"Model tokenizer does not encode SemIf branch {chr(65 + index)} as one token")
            ids.append(token_ids[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError("Model tokenizer maps multiple SemIf branches to the same token")
        return ids

    def decide(self, context: str, options: Mapping[str, str]) -> SemIfDecision:
        if not isinstance(context, str) or not context.strip():
            raise ValueError("SemIf context cannot be empty")
        if not options or len(options) > 26:
            raise ValueError("SemIf requires between 1 and 26 candidate options")
        keys = list(options)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("SemIf option keys must be non-empty strings")
        self._load()
        torch, tokenizer, model = self._torch, self._tokenizer, self._model
        branches = self._branch_ids(tokenizer, len(keys))
        choices = "\n".join(f"{chr(65 + i)}: {key} — {description}" for i, (key, description) in enumerate(options.items()))
        prompt = ("Choose the option that best describes the context. Consider the complete meaning, "
                  "not just matching words. Reply with exactly one branch letter.\n"
                  f"Context: {context.strip()}\nOptions:\n{choices}\nDecision:")
        truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"  # retain the candidate branches and decision boundary on long prompts
        try:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        finally:
            tokenizer.truncation_side = truncation_side
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = "cpu"
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1, branches].float() / self.temperature
            probs_tensor = torch.softmax(logits, dim=0).cpu()
        probabilities = {key: float(probs_tensor[i]) for i, key in enumerate(keys)}
        if any(not math.isfinite(probability) or probability < 0 for probability in probabilities.values()):
            raise RuntimeError("SemIf produced invalid candidate probabilities")
        total_probability = sum(probabilities.values())
        if total_probability <= 0:
            raise RuntimeError("SemIf produced an empty probability distribution")
        probabilities = {key: value / total_probability for key, value in probabilities.items()}
        ranked = sorted(probabilities.values(), reverse=True)
        entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
        return SemIfDecision(max(probabilities, key=probabilities.get), ranked[0], probabilities,
                             entropy, ranked[0] - ranked[1] if len(ranked) > 1 else 1.0,
                             (time.perf_counter() - start) * 1000)
