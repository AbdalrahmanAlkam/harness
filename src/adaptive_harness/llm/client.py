"""LLM client supporting OpenRouter and OpenAI-compatible endpoints with model tiering."""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from adaptive_harness.llm.mock_client import LLMResponse, MockLLMClient, ToolCall


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_TIERS = {
    "fast": "google/gemini-2.5-flash-lite",
    "standard": "z-ai/glm-5.3-flash",
    "reasoning": "anthropic/claude-sonnet-4",
}


class LLMClient:
    """Multi-tier LLM Client with OpenRouter support and seamless offline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        force_mock: bool = False,
    ):
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
        endpoint_host = urlparse(self.base_url).hostname
        local_endpoint = endpoint_host in {"localhost", "127.0.0.1", "::1"}
        self.api_key = api_key or ("local" if local_endpoint else
                                   os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        self.default_model = default_model or MODEL_TIERS["standard"]
        self.force_mock = force_mock
        self.mock_client = MockLLMClient(default_model=self.default_model)

        self._openai_client: Optional[OpenAI] = None
        if self.api_key and not self.force_mock:
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/adaptive-agent-harness",
                    "X-Title": "Adaptive Agent Harness",
                },
            )

    @property
    def is_mock(self) -> bool:
        return self._openai_client is None

    def get_model_for_tier(self, tier: str) -> str:
        """Maps an abstract tier (fast, standard, reasoning) to a concrete model ID."""
        return MODEL_TIERS.get(tier.lower(), self.default_model)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Executes a chat completion call with automatic model selection and tool handling."""
        selected_model = model or (self.get_model_for_tier(tier) if tier else self.default_model)

        # Use mock client if offline or no key configured
        if self.is_mock:
            return self.mock_client.complete(messages=messages, tools=tools, model=selected_model)

        kwargs: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        model_name = selected_model.lower()
        is_openrouter = self.base_url.rstrip("/").startswith("https://openrouter.ai")
        is_claude_reasoning = (model_name.startswith("anthropic/claude") and any(
            marker in model_name for marker in ("claude-3.7", "claude-3-7", "claude-sonnet-4",
                                                 "claude-opus-4", "claude-haiku-4", "claude-4", "claude-5")))
        supports_reasoning = (is_claude_reasoning or
                              any(marker in model_name for marker in
                                  ("gemini-2.5", "gemini-3", "deepseek-r1", "o3-mini", "reasoning",
                                   "z-ai/glm-5.3")))
        if is_openrouter and supports_reasoning:
            if reasoning_budget_tokens == 0:
                kwargs["extra_body"] = {"reasoning": {"enabled": False} if is_claude_reasoning
                                        else {"effort": "none"}}
                kwargs.pop("temperature", None)
            elif reasoning_effort:
                if reasoning_budget_tokens and is_claude_reasoning:
                    budget = max(1024, reasoning_budget_tokens)
                    kwargs["extra_body"] = {"reasoning": {"max_tokens": budget}}
                    kwargs["max_completion_tokens"] = budget + max(1024, budget // 4)
                else:
                    kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
                kwargs.pop("temperature", None)

        try:
            assert self._openai_client is not None
            response = self._openai_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls: List[ToolCall] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        if not isinstance(args, dict):
                            args = {"raw": args}
                    except Exception:
                        args = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        )
                    )

            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }

            return LLMResponse(
                content=msg.content,
                tool_calls=tool_calls,
                model=response.model or selected_model,
                finish_reason=choice.finish_reason or "stop",
                usage=usage_dict,
            )

        except Exception as e:
            return LLMResponse(
                content=f"API call failed: {type(e).__name__}: {str(e)}",
                tool_calls=[],
                model=selected_model,
                finish_reason="error",
            )
