"""Reasoning effort choices exposed by known task models."""

from __future__ import annotations


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def supported_efforts(model: str, provider: str = "openrouter") -> tuple[str, ...]:
    """Return selectable efforts; unknown models keep provider defaults.

    Provider catalogs advertise whether reasoning exists but generally do not
    expose its exact effort enum, so keep this list conservative.
    """
    name = model.lower().strip()
    if provider == "openrouter" and name == "stealth/space-bunny-alpha":
        return EFFORTS
    if name in {"z-ai/glm-5.3", "z-ai/glm-5.3-prime"}:
        return ("low", "high", "max")
    if name.startswith(("z-ai/glm-5.3-flash", "openai/o3-mini", "o3-mini")):
        return ("low", "medium", "high")
    if "claude" in name or "gemini-2.5" in name or "gemini-3" in name:
        return ("low", "medium", "high")
    if "deepseek-r1" in name or name.startswith(("o3", "o4", "openai/o3", "openai/o4")):
        return ("low", "medium", "high")
    return ()
