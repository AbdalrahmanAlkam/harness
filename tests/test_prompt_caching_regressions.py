"""Prompt caching must mark a real prefix boundary.

Anthropic prompt caching is opt-in per content block: the provider hashes
everything up to and including the last block carrying ``cache_control``. The
harness's system prompt and tool definitions are byte-identical on every step of
a run, so they are the ideal cached prefix.

``cache_control`` used to be sent as a top-level request field, which marks
nothing. Every request then paid full price for the system prompt and the tool
schemas -- in a 20-step tool loop, twenty times over. It is the single largest
recurring cost in the product.
"""

from __future__ import annotations

import copy

from adaptive_harness.llm.client import _with_cache_breakpoint


def _system_prompt_size() -> int:
    """The system prompt is the bulk of what this is saving."""
    from adaptive_harness.prompts import get_default_registry

    return len(get_default_registry().get("system.default"))


def test_the_breakpoint_lands_on_the_system_message():
    messages = [
        {"role": "system", "content": "You are a careful agent."},
        {"role": "user", "content": "do the thing"},
    ]
    marked = _with_cache_breakpoint(messages)

    system = marked[0]
    assert system["role"] == "system"
    assert isinstance(system["content"], list), "a cache breakpoint needs a content block"
    block = system["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "You are a careful agent."
    assert block["cache_control"] == {"type": "ephemeral"}


def test_only_one_breakpoint_is_placed():
    """A second marker on a later block would split the cacheable prefix."""
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
    ]
    marked = _with_cache_breakpoint(messages)
    assert isinstance(marked[0]["content"], list)
    assert marked[2]["content"] == "c", "a later system message must stay a plain string"


def test_the_caller_message_list_is_not_mutated():
    """self.messages is reused for the whole session; rewriting it in place
    would corrupt every later request."""
    messages = [
        {"role": "system", "content": "stable prefix"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    before = copy.deepcopy(messages)
    marked = _with_cache_breakpoint(messages)

    assert messages == before, "the input list was modified in place"
    assert marked[1:] == messages[1:], "everything after the system prompt must be untouched"


def test_a_conversation_with_no_system_message_is_returned_unchanged():
    messages = [{"role": "user", "content": "x"}]
    assert _with_cache_breakpoint(messages) == messages


def test_a_system_message_that_is_already_blocks_is_flattened_safely():
    messages = [{"role": "system", "content": [{"type": "text", "text": "already blocks"}]}]
    marked = _with_cache_breakpoint(messages)
    text = "".join(block.get("text", "") for block in marked[0]["content"])
    assert text == "already blocks"
    assert marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_the_cached_prefix_is_the_part_that_never_changes():
    """Quantifies what the breakpoint is protecting: the system prompt is the
    dominant per-request cost and is identical on every step."""
    assert _system_prompt_size() > 500, (
        "the system prompt is large enough for caching to matter; if this fails "
        "the prompt was shortened and the saving shrank with it")
