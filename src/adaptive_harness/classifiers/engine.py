"""Selectable local and remote classification engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import os
import time
from typing import Mapping
from urllib.request import Request, urlopen

from adaptive_harness.classifiers.skill_classifier import SkillClassifier

DOMAIN_LABELS = ["coding", "research", "science", "audit"]
THINKING_LABELS = ["none", "low", "medium", "deep"]

TRAINING = {
    "coding": ["fix the Python bug", "refactor the API", "implement a new feature", "write unit tests", "review the git diff", "compile this project"],
    "research": ["survey academic literature", "find citations for this claim", "compare primary sources", "summarize recent papers", "write a literature review", "investigate historical evidence"],
    "science": ["solve a differential equation", "check numerical convergence", "simulate the physical model", "prove this theorem", "analyze the dataset statistically", "derive the mathematical formula"],
    "audit": ["audit this code for vulnerabilities", "check SQL injection risks", "review authentication security", "find OWASP weaknesses", "inspect for memory safety bugs", "threat model this service"],
    "none": ["git status", "list files", "read file", "show directory", "check version", "print current branch"],
    "low": ["write a simple function", "fix a typo", "add a unit test", "explain this code", "format a file", "update a comment"],
    "medium": ["refactor multiple modules", "design a data pipeline", "solve an algorithm problem", "debug a cross module issue", "implement a parser", "integrate two systems"],
    "deep": ["prove a formal theorem", "debug a concurrency deadlock", "design a distributed architecture", "resolve subtle race conditions", "verify a cryptographic protocol", "analyze a complex numerical method"],
}


@dataclass
class Classification:
    label: str
    probabilities: dict[str, float]
    latency_ms: float
    reasoning: str = ""

    @property
    def entropy(self) -> float:
        return -sum(p * math.log2(p) for p in self.probabilities.values() if p > 0)

    @property
    def margin(self) -> float:
        ranked = sorted(self.probabilities.values(), reverse=True)
        return ranked[0] - ranked[1] if len(ranked) > 1 else 1.0


def _normalize(scores: Mapping[str, float], labels: list[str]) -> dict[str, float]:
    if not labels:
        raise ValueError("Classifier labels cannot be empty")
    values = {}
    for key in labels:
        value = float(scores.get(key, 0))
        if not math.isfinite(value):
            raise ValueError(f"Invalid probability for {key}")
        values[key] = max(0.0, value)
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Classifier returned no valid probabilities")
    return {key: value / total for key, value in values.items()}


class BaseClassifierBackend(ABC):
    name = "base"
    model = ""

    @abstractmethod
    def classify(self, text: str, labels: list[str]) -> Classification:
        """Return one normalized probability distribution over labels."""


class SklearnBackend(BaseClassifierBackend):
    name = "sklearn"
    model = "TF-IDF + Logistic Regression"

    def __init__(self):
        self.skill = SkillClassifier()
        self._pipelines = {}

    def classify(self, text: str, labels: list[str]) -> Classification:
        start = time.perf_counter()
        if set(labels).issubset(set(self.skill.pipeline.classes_)):
            probabilities = _normalize(self.skill.classify(text).probabilities, labels)
        elif set(labels) in (set(DOMAIN_LABELS), set(THINKING_LABELS)):
            key = tuple(sorted(labels))
            if key not in self._pipelines:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                examples = [(sentence, label) for label in labels for sentence in TRAINING[label]]
                pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=500))
                pipeline.fit([sentence for sentence, _ in examples], [label for _, label in examples])
                self._pipelines[key] = pipeline
            pipeline = self._pipelines[key]
            raw = pipeline.predict_proba([text])[0]
            probabilities = _normalize(dict(zip(pipeline.classes_, raw)), labels)
        else:
            raise ValueError(f"Sklearn backend does not support labels: {labels}")
        return Classification(max(probabilities, key=probabilities.get), probabilities,
                              (time.perf_counter() - start) * 1000)


class OllamaBackend(BaseClassifierBackend):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:1.5b", endpoint: str = "http://localhost:11434/api/generate"):
        self.model, self.endpoint = model, endpoint

    def classify(self, text: str, labels: list[str]) -> Classification:
        start = time.perf_counter()
        prompt = f"Classify the task into exactly one of {labels}. Return JSON with label, confidence (0-1), reasoning. Task: {text}"
        data = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "format": "json"}).encode()
        with urlopen(Request(self.endpoint, data=data, headers={"Content-Type": "application/json"}), timeout=30) as response:
            payload = json.load(response)
        return _parse_response(payload.get("response", "{}"), labels, start)


class LocalHTTPBackend(BaseClassifierBackend):
    """OpenAI-compatible local endpoint, including llama.cpp server."""
    name = "local-slm"

    def __init__(self, model: str, endpoint: str):
        self.model, self.endpoint = model, endpoint

    def classify(self, text: str, labels: list[str]) -> Classification:
        start = time.perf_counter()
        data = json.dumps({"model": self.model, "messages": [{"role": "user", "content":
                           f"Classify into {labels}. Return JSON with label, confidence, reasoning. Task: {text}"}],
                           "temperature": 0}).encode()
        with urlopen(Request(self.endpoint, data=data, headers={"Content-Type": "application/json"}), timeout=30) as response:
            payload = json.load(response)
        return _parse_response(payload["choices"][0]["message"]["content"], labels, start)


class OpenRouterBackend(BaseClassifierBackend):
    name = "openrouter"

    def __init__(self, model: str = "google/gemini-2.5-flash-lite", api_key: str | None = None,
                 endpoint: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.model, self.api_key, self.endpoint = model, api_key or os.getenv("OPENROUTER_API_KEY"), endpoint

    def classify(self, text: str, labels: list[str]) -> Classification:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for openrouter classifier")
        start = time.perf_counter()
        data = json.dumps({"model": self.model, "response_format": {"type": "json_object"},
                           "messages": [{"role": "user", "content": f"Classify this task into {labels}. Return JSON with label, confidence (0-1), reasoning. Task: {text}"}]}).encode()
        request = Request(self.endpoint, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        return _parse_response(payload["choices"][0]["message"]["content"], labels, start)


def _parse_response(raw: str, labels: list[str], start: float) -> Classification:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Some local chat servers wrap otherwise valid JSON in markdown fences.
        first = raw.find("{")
        if first < 0:
            raise
        payload, _ = json.JSONDecoder().raw_decode(raw[first:])
    if not isinstance(payload, dict):
        raise ValueError("Classifier response must be a JSON object")
    label = payload["label"]
    if label not in labels:
        raise ValueError(f"Unknown classifier label: {label}")
    if isinstance(payload.get("probabilities"), dict):
        probs = _normalize(payload["probabilities"], labels)
    else:
        confidence = float(payload.get("confidence", 0.75))
        if not math.isfinite(confidence):
            raise ValueError("Classifier confidence must be finite")
        confidence = min(1.0, max(1.0 / len(labels), confidence))
        probs = {key: (confidence if key == label else (1-confidence)/(len(labels)-1)) for key in labels} if len(labels) > 1 else {label: 1.0}
    return Classification(max(probs, key=probs.get), probs, (time.perf_counter()-start)*1000,
                          str(payload.get("reasoning", "")))


class OnnxBackend(BaseClassifierBackend):
    """Local zero-shot semantic classification using ONNX embeddings and label descriptions."""
    name = "onnx"

    def __init__(self, model: str):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime and transformers for the onnx backend") from exc
        from pathlib import Path
        path = Path(model)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.session = ort.InferenceSession(str(path / "model.onnx"), providers=["CPUExecutionProvider"])

    def _embed(self, text: str):
        import numpy as np
        inputs = self.tokenizer(text, return_tensors="np", truncation=True, max_length=256)
        names = {item.name for item in self.session.get_inputs()}
        outputs = self.session.run(None, {k: v.astype("int64") for k, v in inputs.items() if k in names})[0]
        if outputs.ndim == 3:
            mask = inputs.get("attention_mask")
            if mask is None:
                embedding = outputs.mean(axis=1)[0]
            else:
                weights = mask[..., None]
                embedding = ((outputs * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1))[0]
        elif outputs.ndim == 2:
            embedding = outputs[0]
        else:
            raise ValueError(f"Unsupported ONNX embedding output shape: {outputs.shape}")
        return embedding / (np.linalg.norm(embedding) + 1e-12)

    def classify(self, text: str, labels: list[str]) -> Classification:
        import numpy as np
        start = time.perf_counter()
        query = self._embed(text)
        scores = np.array([float(query @ self._embed(label.replace("_", " "))) for label in labels])
        exps = np.exp((scores - scores.max()) * 10)
        probs = dict(zip(labels, (exps / exps.sum()).tolist()))
        return Classification(max(probs, key=probs.get), probs, (time.perf_counter()-start)*1000)


def create_backend(name: str = "sklearn", model: str | None = None, endpoint: str | None = None,
                   api_key: str | None = None) -> BaseClassifierBackend:
    name = name.lower().strip()
    if name == "sklearn":
        return SklearnBackend()
    if name == "local-slm" and endpoint and not endpoint.rstrip("/").endswith("api/generate"):
        local_endpoint = endpoint.rstrip("/")
        if not local_endpoint.endswith("chat/completions"):
            if not local_endpoint.endswith("/v1"):
                local_endpoint += "/v1"
            local_endpoint += "/chat/completions"
        return LocalHTTPBackend(model or "local-model", local_endpoint)
    if name in ("ollama", "local-slm"):
        return OllamaBackend(model or "qwen2.5:1.5b", endpoint or "http://localhost:11434/api/generate")
    if name == "openrouter":
        return OpenRouterBackend(model or "google/gemini-2.5-flash-lite", api_key=api_key,
                                 endpoint=endpoint or "https://openrouter.ai/api/v1/chat/completions")
    if name == "onnx":
        if not model:
            raise ValueError("onnx backend requires a local model directory")
        return OnnxBackend(model)
    raise ValueError(f"Unknown classifier backend: {name}")
