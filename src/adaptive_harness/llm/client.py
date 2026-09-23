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
                timeout=45.0,
                max_retries=1,
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
        legacy_thinking_history = bool(is_claude_reasoning and reasoning_effort and reasoning_budget_tokens != 0 and any(
            message.get("role") == "assistant" and message.get("tool_calls") and
            not message.get("reasoning_details") for message in messages))
        if is_openrouter and supports_reasoning:
            if legacy_thinking_history:
                # Older saved turns contain tool calls without the opaque
                # thinking signature OpenRouter needs to replay with Claude.
                # Use provider defaults for this conversation until it resets.
                pass
            elif reasoning_budget_tokens == 0:
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

        # OpenRouter advances Anthropic's cache breakpoint as the conversation grows.
        # Other providers handle compatible prompt prefixes implicitly.
        if is_openrouter and model_name.startswith("anthropic/claude"):
            kwargs.setdefault("extra_body", {})["cache_control"] = {"type": "ephemeral"}

        try:
            assert self._openai_client is not None
            thinking_fallback = legacy_thinking_history
            try:
                response = self._openai_client.chat.completions.create(**kwargs)
            except Exception as exc:
                detail = str(exc).lower()
                status = getattr(exc, "status_code", None)
                history_rejected = status in {400, 422} and (
                    "thinking block" in detail or "thinking is enabled" in detail)
                reasoning_rejected = history_rejected or (status in {400, 422} and any(
                    token in detail for token in ("reasoning", "max_tokens", "effort", "unsupported parameter")))
                if not reasoning_rejected or "reasoning" not in kwargs.get("extra_body", {}):
                    raise
                # Keep the chosen model and request intact; retry once with the
                # provider's default thinking behavior after a parameter rejection.
                fallback_kwargs = dict(kwargs)
                extra_body = dict(fallback_kwargs.get("extra_body") or {})
                extra_body.pop("reasoning", None)
                if extra_body:
                    fallback_kwargs["extra_body"] = extra_body
                else:
                    fallback_kwargs.pop("extra_body", None)
                fallback_kwargs.pop("max_completion_tokens", None)
                fallback_kwargs["temperature"] = temperature
                response = self._openai_client.chat.completions.create(**fallback_kwargs)
                thinking_fallback = True
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
            if response.usage:
                details = getattr(response.usage, "prompt_tokens_details", None)
                usage_dict["cached_tokens"] = int(
                    getattr(details, "cached_tokens", None) or
                    getattr(response.usage, "cache_read_input_tokens", None) or 0)

            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details is None:
                extra = getattr(msg, "model_extra", None)
                if isinstance(extra, dict):
                    reasoning_details = extra.get("reasoning_details")

            return LLMResponse(
                content=msg.content,
                tool_calls=tool_calls,
                model=response.model or selected_model,
                finish_reason=choice.finish_reason or "stop",
                usage=usage_dict,
                metadata={"thinking_fallback": thinking_fallback,
                          "thinking_fallback_reason": "legacy_tool_history" if legacy_thinking_history else
                              "provider_rejected_thinking" if thinking_fallback else "",
                          "reasoning_details": self._serialize_reasoning_details(reasoning_details)},
            )

        except Exception as e:
            # Never fabricate successful work after a failed live request. The
            # current task stops; subsequent requests can use the offline engine.
            detail = str(e).replace(self.api_key, "[redacted]") if self.api_key else str(e)
            self._openai_client = None
            return LLMResponse(
                content=f"API call failed: {type(e).__name__}: {detail}. Switched to Offline Mock Engine; reconnect with /key or retry from the CLI.",
                tool_calls=[],
                model=selected_model,
                finish_reason="error",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

    @staticmethod
    def _serialize_reasoning_details(details):
        """Keep OpenRouter's opaque reasoning blocks intact for tool-call replay."""
        if details is None:
            return None
        serialized = []
        for item in details:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = item.dict(exclude_none=True)
            serialized.append(item)
        return serialized
