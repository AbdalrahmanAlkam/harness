"""Classifier-gated filtering of verbose tool output.

The main model's context is the scarcest resource in the agent loop: every tool
result is replayed on *every* subsequent request, so one verbose ``run_pytest``
or ``search_files`` is paid for again and again until compaction rescues it.

This module asks the cheap local classifier whether a result is worth that cost.
Only when it is judged noise is a **secondary** model asked to compress it, which
keeps the expensive model out of the loop for the boring cases. The raw text is
never destroyed: it is archived behind a short token that the model can spend to
read the whole thing when the summary is not enough.

Three properties matter more than the saving:

* **Errors are never filtered.** A failure is the evidence the agent needs; only
  successful output is a candidate.
* **The gate fails open.** If the classifier errors, the backend cannot judge
  these labels, or the summariser returns nothing useful, the original output is
  passed through untouched. Losing context is worse than wasting it.
* **Bounded memory.** The archive is a fixed-size LRU, so a long run cannot grow
  without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from adaptive_harness.classifiers.engine import OUTPUT_VALUE_LABELS

#: Prefix for the tokens the model spends to retrieve a filtered result. Chosen
#: to be unmistakable in a transcript and impossible to confuse with a filename.
TOKEN_PREFIX = "OUTPUT"

#: The notice is deliberately terse. It competes for the same context the
#: filtering was meant to save, so it states only the fact, the size, and the
#: token -- no explanation and no restatement of the summary.
NOTICE = ("[{chars:,} chars of {tool} output were compressed to {kept:,} chars; "
          "they were judged low-value. Call read_full_output with token {token} "
          "to see everything.]")


class ToolOutputArchive:
    """A bounded, insertion-ordered store of raw tool output.

    Least-recently-used entries are evicted once ``limit`` is reached. A token
    that has been evicted is reported as unknown rather than returning a stale or
    wrong result: the model is told to re-run the tool instead.
    """

    def __init__(self, limit: int = 12):
        if limit < 1:
            raise ValueError("Archive limit must be at least 1")
        self.limit = limit
        self._entries: "OrderedDict[str, str]" = OrderedDict()
        self._counter = 0

    def store(self, tool_name: str, output: str) -> str:
        """Archive ``output`` and return the token that retrieves it."""
        self._counter += 1
        token = f"{TOKEN_PREFIX}-{self._counter}"
        self._entries[token] = f"tool={tool_name}\n{output}"
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)
        return token

    def fetch(self, token: str) -> str | None:
        """Return the archived text, or ``None`` if the token is unknown or evicted."""
        if token not in self._entries:
            return None
        self._entries.move_to_end(token)
        return self._entries[token]

    def tokens(self) -> list[str]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._counter = 0


@dataclass(frozen=True)
class FilterOutcome:
    """What the agent should put in the transcript for one tool result."""

    content: str
    filtered: bool
    token: str | None = None
    raw_chars: int = 0
    kept_chars: int = 0
    reason: str = ""

    @property
    def saved_chars(self) -> int:
        return max(0, self.raw_chars - self.kept_chars)


@dataclass
class ToolOutputFilter:
    """Decide whether a tool result is noise and, if so, compress it.

    Args:
        classifier: Anything with ``classify(text, labels)`` returning an object
            with ``label`` and ``probabilities``. A backend that cannot judge
            these labels raises, which is handled as "not noise".
        summarize: ``(tool_name, raw_output) -> str``. Injected so this module
            carries no model dependency and the caller decides which model pays.
        min_chars: Results shorter than this are never filtered; the saving would
            not pay for the notice that has to replace them.
        noise_threshold: Confidence in ``noise`` required before filtering.
        max_summary_chars: Hard cap on what the secondary model may return.
    """

    classifier: object
    summarize: Callable[[str, str], str]
    min_chars: int = 1200
    noise_threshold: float = 0.5
    max_summary_chars: int = 1500
    archive: ToolOutputArchive = field(default_factory=ToolOutputArchive)

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        if not 0.0 < self.noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be in (0, 1]")

    def judge(self, output: str) -> tuple[bool, str]:
        """Return ``(is_noise, reason)``; never raises."""
        try:
            prediction = self.classifier.classify(output[:20_000], OUTPUT_VALUE_LABELS)
        except Exception as exc:  # noqa: BLE001 - any failure means "keep it"
            return False, f"classifier unavailable ({type(exc).__name__})"
        probabilities = getattr(prediction, "probabilities", None) or {}
        confidence = float(probabilities.get("noise", 0.0))
        is_noise = str(getattr(prediction, "label", "")) == "noise" and confidence >= self.noise_threshold
        return is_noise, f"noise p={confidence:.2f}"

    def process(self, tool_name: str, output: str, *, success: bool = True) -> FilterOutcome:
        """Return the transcript content for one tool result."""
        raw_chars = len(output)
        keep = FilterOutcome(content=output, filtered=False, raw_chars=raw_chars,
                             kept_chars=raw_chars)
        # A failure is the evidence the agent needs; never compress it.
        if not success or raw_chars < self.min_chars:
            return keep
        is_noise, reason = self.judge(output)
        if not is_noise:
            return FilterOutcome(content=output, filtered=False, raw_chars=raw_chars,
                                 kept_chars=raw_chars, reason=reason)
        try:
            summary = (self.summarize(tool_name, output) or "").strip()
        except Exception as exc:  # noqa: BLE001 - fall back to the raw output
            return FilterOutcome(content=output, filtered=False, raw_chars=raw_chars,
                                 kept_chars=raw_chars,
                                 reason=f"{reason}; summariser failed ({type(exc).__name__})")
        # Judge the summariser's real output *before* applying the cap: a model
        # that merely echoes its input would otherwise be truncated into looking
        # like a genuine summary.
        if len(summary) >= raw_chars:
            return FilterOutcome(content=output, filtered=False, raw_chars=raw_chars,
                                 kept_chars=raw_chars, reason=f"{reason}; summary not shorter")
        summary = summary[:self.max_summary_chars]
        token = self.archive.store(tool_name, output)
        content = (summary + "\n\n" + NOTICE.format(chars=raw_chars, tool=tool_name,
                                                     kept=len(summary), token=token))
        # The notice is paid for out of the same budget, so this only counts as
        # a win if the transcript actually got smaller.
        if len(content) >= raw_chars:
            return FilterOutcome(content=output, filtered=False, raw_chars=raw_chars,
                                 kept_chars=raw_chars, reason=f"{reason}; notice ate the saving")
        return FilterOutcome(content=content, filtered=True, token=token,
                             raw_chars=raw_chars, kept_chars=len(content), reason=reason)
