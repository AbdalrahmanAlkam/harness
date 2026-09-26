"""Provider settings for OpenAI-compatible chat and tool-call endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    env_key: str | None
    default_model: str
    context_window: int


PROVIDERS = {
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                           "stealth/space-bunny-alpha", 1_000_000),
    "anthropic": Provider("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
                          "claude-sonnet-4-6", 200_000),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
                       "gpt-4o", 128_000),
    "deepseek": Provider("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                         "deepseek-chat", 128_000),
    "google": Provider("google", "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "GEMINI_API_KEY", "gemini-2.5-flash", 1_000_000),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                     "llama-3.3-70b-versatile", 128_000),
    "local": Provider("local", "http://localhost:11434/v1", None, "qwen3.5:4b", 8_000),
}

PROVIDER_TIERS = {
    "anthropic": {"fast": "claude-haiku-4-5", "standard": "claude-sonnet-4-6",
                  "reasoning": "claude-sonnet-4-6"},
    "openai": {"fast": "gpt-4o-mini", "standard": "gpt-4o", "reasoning": "o3-mini"},
    "deepseek": {"fast": "deepseek-chat", "standard": "deepseek-chat", "reasoning": "deepseek-reasoner"},
    "google": {"fast": "gemini-2.5-flash-lite", "standard": "gemini-2.5-flash",
               "reasoning": "gemini-2.5-pro"},
    "groq": {"fast": "llama-3.1-8b-instant", "standard": "llama-3.3-70b-versatile",
             "reasoning": "llama-3.3-70b-versatile"},
    "local": {"fast": "qwen3.5:4b", "standard": "qwen3.5:4b", "reasoning": "qwen3.5:4b"},
}


def provider_for_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    for name, provider in PROVIDERS.items():
        if urlparse(provider.base_url).hostname == host:
            return name
    return "custom"


def context_window(model: str, provider: str = "openrouter", override: int | None = None) -> int:
    """Conservative fallback; precise OpenRouter catalog lengths can override it."""
    if override and override > 0:
        return override
    lowered = model.lower()
    if lowered == "stealth/space-bunny-alpha":
        return 1_000_000
    if "claude" in lowered:
        return 200_000
    if "gemini" in lowered:
        return 1_000_000
    if "gpt-4o" in lowered or "o3-mini" in lowered:
        return 128_000
    return PROVIDERS.get(provider, PROVIDERS["openrouter"]).context_window
