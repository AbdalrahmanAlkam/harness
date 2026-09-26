"""LLM client supporting OpenRouter and OpenAI-compatible endpoints with model tiering."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from adaptive_harness.llm.mock_client import LLMResponse, MockLLMClient, ToolCall
from adaptive_harness.llm.providers import PROVIDERS, PROVIDER_TIERS, provider_for_url
from adaptive_harness.llm.effort import EFFORTS, supported_efforts


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_TIERS = {
    "fast": "google/gemini-2.5-flash-lite",
    "standard": "stealth/space-bunny-alpha",
    "reasoning": "anthropic/claude-sonnet-4",
}


def _fenced_tool_calls(content: str | None, tools: List[Dict[str, Any]] | None) -> List[ToolCall]:
    """Recover explicit JSON tool requests from providers without native calls.

    Plain JSON examples are deliberately ignored. Only a ``tool_call`` fence
    or a JSON object with a ``tool_call`` wrapper may initiate a tool.
    """
    if not content or not tools:
        return []
    available = {item.get("function", {}).get("name") for item in tools}
    calls = []
    for index, match in enumerate(re.finditer(r"```(tool_call|tool|json)\s*\n(.*?)\n```",
                                           content, flags=re.I | re.S), start=1):
        try:
            payload = json.loads(match.group(2))
        except (ValueError, TypeError):
            continue
        if match.group(1).lower() == "json":
            payload = payload.get("tool_call") if isinstance(payload, dict) else None
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool")
        args = payload.get("arguments", payload.get("args"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        if name in available and isinstance(args, dict):
            calls.append(ToolCall(id=f"fenced_tool_{index}", name=name, arguments=args))
    return calls


class LLMClient:
    """Multi-tier LLM Client with OpenRouter support and seamless offline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        force_mock: bool = False,
        provider: Optional[str] = None,
        provider_keys: Optional[Dict[str, str]] = None,
        backup_providers: tuple[str, ...] = (),
    ):
        if provider and provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider or provider_for_url(base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL)
        self.provider_keys = dict(provider_keys or {})
        self.backup_providers = tuple(name for name in backup_providers if name in PROVIDERS)
        self.base_url = base_url or (PROVIDERS[self.provider].base_url if provider else
                                     os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL)
        endpoint_host = urlparse(self.base_url).hostname
        local_endpoint = endpoint_host in {"localhost", "127.0.0.1", "::1"}
        env_name = PROVIDERS[self.provider].env_key if self.provider in PROVIDERS else "OPENROUTER_API_KEY"
        self.api_key = api_key or self.provider_keys.get(self.provider) or ("local" if local_endpoint else
                                   os.environ.get(env_name or "") or
                                   (os.environ.get("OPENAI_API_KEY") if self.provider == "openrouter" else None))
        self.default_model = default_model or (PROVIDERS[self.provider].default_model if provider else MODEL_TIERS["standard"])
        self.force_mock = force_mock
        self._temporary_offline = False
        self.mock_client = MockLLMClient(default_model=self.default_model)

        self._openai_client: Optional[OpenAI] = None
        self._make_client()

    def _make_client(self) -> None:
        self._openai_client = None
        if self.api_key and not self.force_mock:
            headers = ({"HTTP-Referer": "https://github.com/adaptive-agent-harness",
                        "X-Title": "Adaptive Agent Harness"} if self.provider == "openrouter" else
                       {"x-goog-api-client": "adaptive-harness/0.1"} if self.provider == "google" else {})
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=45.0,
                max_retries=1,
                default_headers=headers,
            )

    def switch_provider(self, provider: str, *, model: str | None = None,
                        base_url: str | None = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider
        self.base_url = base_url or PROVIDERS[provider].base_url
        self.api_key = (self.provider_keys.get(provider) or
                        ("local" if provider == "local" else os.environ.get(PROVIDERS[provider].env_key or "")))
        self.default_model = model or PROVIDERS[provider].default_model
        self._temporary_offline = False
        self._make_client()

    def set_provider_key(self, provider: str, key: str | None) -> None:
        if provider not in PROVIDERS or provider == "local":
            raise ValueError(f"Unknown key provider: {provider}")
        if key:
            self.provider_keys[provider] = key
        else:
            self.provider_keys.pop(provider, None)
        if self.provider == provider:
            self.switch_provider(provider, model=self.default_model)

    @property
    def is_mock(self) -> bool:
        return self.force_mock or self._openai_client is None or self._temporary_offline

    def get_model_for_tier(self, tier: str) -> str:
        """Maps an abstract tier (fast, standard, reasoning) to a concrete model ID."""
        return PROVIDER_TIERS.get(self.provider, MODEL_TIERS).get(tier.lower(), self.default_model)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
        _allow_failover: bool = True,
    ) -> LLMResponse:
        """Executes a chat completion call with automatic model selection and tool handling."""
        selected_model = model or (self.get_model_for_tier(tier) if tier else self.default_model)

        # Use mock client if offline or no key configured
        if self.force_mock or self._openai_client is None:
            return self.mock_client.complete(messages=messages, tools=tools, model=selected_model)
        # A prior failure only changed the visible status to offline. Keep the
        # credential/client and retry it once on the next task automatically.
        self._temporary_offline = False

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
        available_efforts = supported_efforts(selected_model, self.provider)
        if reasoning_effort in EFFORTS and available_efforts and reasoning_effort not in available_efforts:
            # Automatic task classification uses common effort names. Some
            # models expose only a subset; choose the nearest supported one.
            target = EFFORTS.index(reasoning_effort)
            reasoning_effort = min(available_efforts,
                key=lambda effort: (abs(EFFORTS.index(effort) - target), -EFFORTS.index(effort)))
        is_claude_reasoning = ((model_name.startswith("anthropic/claude") or
                               (self.provider == "anthropic" and model_name.startswith("claude"))) and any(
            marker in model_name for marker in ("claude-3.7", "claude-3-7", "claude-sonnet-4",
                                                 "claude-opus-4", "claude-haiku-4", "claude-4", "claude-5")))
        supports_reasoning = (bool(available_efforts) or is_claude_reasoning or
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
            elif reasoning_budget_tokens == 0 and model_name != "stealth/space-bunny-alpha":
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

        if self.provider == "openai" and model_name.startswith(("o1", "o3", "o4")):
            kwargs.pop("temperature", None)
            if reasoning_effort in {"low", "medium", "high"}:
                kwargs["reasoning_effort"] = reasoning_effort
        elif self.provider == "anthropic" and reasoning_budget_tokens and reasoning_effort and not legacy_thinking_history:
            kwargs.pop("temperature", None)
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": "enabled", "budget_tokens": max(1024, reasoning_budget_tokens)}
            kwargs["max_completion_tokens"] = max(1024, reasoning_budget_tokens) + 2048

        # OpenRouter advances Anthropic's cache breakpoint as the conversation grows.
        # Other providers handle compatible prompt prefixes implicitly.
        if is_openrouter and model_name.startswith("anthropic/claude"):
            kwargs.setdefault("extra_body", {})["cache_control"] = {"type": "ephemeral"}
        if is_openrouter:
            # The OpenAI SDK has no `usage` keyword. `extra_body` sends this
            # OpenRouter extension as a top-level JSON field instead.
            kwargs.setdefault("extra_body", {})["usage"] = {"include": True}

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
            else:
                tool_calls = _fenced_tool_calls(msg.content, tools)

            usage_dict = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
            if response.usage:
                prompt_details = getattr(response.usage, "prompt_tokens_details", None)
                completion_details = getattr(response.usage, "completion_tokens_details", None)
                def field(obj, name):
                    direct = getattr(obj, name, None) if obj is not None else None
                    if direct is not None:
                        return direct
                    extras = getattr(obj, "model_extra", None) if obj is not None else None
                    if not isinstance(extras, dict) and obj is not None:
                        extras = getattr(obj, "__dict__", {})
                    return extras.get(name) if isinstance(extras, dict) else None
                usage_dict["cached_tokens"] = int(field(prompt_details, "cached_tokens") or
                    field(response.usage, "cache_read_input_tokens") or 0)
                usage_dict["cache_write_tokens"] = int(field(prompt_details, "cache_write_tokens") or 0)
                usage_dict["reasoning_tokens"] = int(field(completion_details, "reasoning_tokens") or 0)
                if not usage_dict["total_tokens"]:
                    usage_dict["total_tokens"] = usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]
                reported_cost = field(response.usage, "cost")
                cost_details = field(response.usage, "cost_details")
                if reported_cost is None:
                    reported_cost = field(cost_details, "upstream_inference_cost")
                    if reported_cost is None:
                        reported_cost = field(cost_details, "total_cost")
                if reported_cost is not None:
                    usage_dict["cost_usd"] = float(reported_cost)

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
                metadata={"provider": self.provider, "thinking_fallback": thinking_fallback,
                          "thinking_fallback_reason": "legacy_tool_history" if legacy_thinking_history else
                              "provider_rejected_thinking" if thinking_fallback else "",
                          "reasoning_details": self._serialize_reasoning_details(reasoning_details)},
            )

        except Exception as e:
            # Never fabricate successful work after a failed live request. The
            # current task stops; subsequent requests can use the offline engine.
            detail = str(e).replace(self.api_key, "[redacted]") if self.api_key else str(e)
            self._temporary_offline = True
            status = getattr(e, "status_code", None)
            if status in {401, 403}:
                hint = "The saved API key was rejected; it remains saved. Check its permissions/account, and replace it only if it was revoked."
            elif status == 429:
                hint = "The provider rate-limited this request or has no available quota; your saved key is unchanged."
            elif status in {400, 422}:
                hint = "The provider rejected this request; your saved key is unchanged."
            else:
                hint = "The request failed; your saved key is unchanged and will be retried on the next task."
            if _allow_failover and status in {429, 500, 502, 503, 504}:
                primary_provider = self.provider
                primary_model = self.default_model
                primary_url = self.base_url
                primary_key = self.api_key
                for backup in self.backup_providers:
                    if backup == self.provider or (backup != "local" and not
                        (self.provider_keys.get(backup) or os.environ.get(PROVIDERS[backup].env_key or ""))):
                        continue
                    self.switch_provider(backup)
                    fallback = self.complete(messages, tools, model=self.default_model,
                        temperature=temperature, reasoning_effort=reasoning_effort,
                        reasoning_budget_tokens=reasoning_budget_tokens, _allow_failover=False)
                    if fallback.finish_reason != "error":
                        fallback.metadata = {**(fallback.metadata or {}), "failed_over_from": primary_provider}
                        return fallback
                self.switch_provider(primary_provider, model=primary_model, base_url=primary_url)
                self.api_key = primary_key
                self._make_client()
                self._temporary_offline = True
            return LLMResponse(
                content=f"API call failed: {type(e).__name__}: {detail}. {hint} This task stopped without claiming success.",
                tool_calls=[],
                model=selected_model,
                finish_reason="error",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                metadata={"provider": self.provider, "error_status": status},
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
