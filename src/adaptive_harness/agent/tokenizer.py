"""Token estimation with pluggable backends.

`len(json)//4` is a decent rough guess for English prose and a bad one for
source code and CJK, where it under-counts by 10–15%. That matters because the
compaction threshold is expressed as a fraction of the context window: an
estimator that under-counts by 15% lets a request run 15% over the limit before
anything compacts, and the provider rejects it.

The default backend is still the zero-dependency character estimate — a product
that cannot run without `tiktoken` installed is not a product — but a better
estimator is one `pip install` and a setting away.

Selection is a plugin setting, so the core only knows the interface.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional

#: Backends, in the order they are tried. Each is a callable taking the
#: serialized request and returning a token count. `None` means unavailable.
_BACKENDS: Dict[str, Callable[[str], int]] = {}

#: The estimator in use. Swapped by the plugin setting; kept module-level so
#: every call site observes the change without being passed a new object.
_ACTIVE: str = "chars4"


def _register(name: str) -> Callable[[Callable[[str], int]], Callable[[str], int]]:
    def decorate(function: Callable[[str], int]) -> Callable[[str], int]:
        _BACKENDS[name] = function
        return function
    return decorate


# --- backends --------------------------------------------------------------

@_register("chars4")
def _chars4(serialized: str) -> int:
    """The zero-dependency default: four characters per token.

    Deliberately biased slightly high. Under-counting means a request exceeds
    the window before the threshold notices; over-counting merely compacts a
    little early, which costs a little context and nothing else.
    """
    if not serialized:
        return 0
    # CJK and other wide scripts average closer to one token per character, so
    # a flat /4 under-counts them badly. Count them separately rather than
    # making the default estimator depend on a tokenizer.
    cjk = sum(1 for character in serialized if _is_wide(character))
    remainder = len(serialized) - cjk
    return max(1, math.ceil(remainder / 4) + cjk)


@_register("tiktoken")
def _tiktoken(serialized: str) -> Optional[int]:  # type: ignore[override]
    """A real BPE count, when tiktoken happens to be installed."""
    encoding = _tiktoken_encoding()
    if encoding is None:
        return None
    return len(encoding.encode(serialized, disallowed_special=()))


@_register("transformers")
def _transformers(serialized: str) -> Optional[int]:  # type: ignore[override]
    """A real tokenizer, when transformers and a local model happen to exist."""
    tokenizer = _transformers_tokenizer()
    if tokenizer is None:
        return None
    return len(tokenizer.encode(serialized))


_TIKTOKEN_ENCODING: Any = None
_TIKTOKEN_TRIED = False


def _tiktoken_encoding() -> Any:
    global _TIKTOKEN_ENCODING, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken

            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - an optional backend that is absent
            _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


_TRANSFORMERS_TOKENIZER: Any = None
_TRANSFORMERS_TRIED = False


def _transformers_tokenizer() -> Any:
    global _TRANSFORMERS_TOKENIZER, _TRANSFORMERS_TRIED
    if not _TRANSFORMERS_TRIED:
        _TRANSFORMERS_TRIED = True
        try:
            from transformers import AutoTokenizer

            _TRANSFORMERS_TOKENIZER = AutoTokenizer.from_pretrained(
                "gpt2", local_files_only=True)
        except Exception:  # noqa: BLE001 - an optional backend that is absent
            _TRANSFORMERS_TOKENIZER = None
    return _TRANSFORMERS_TOKENIZER


# --- CJK detection ---------------------------------------------------------

#: Ranges whose characters cost roughly one token each rather than four
#: characters per token.
_WIDE_RANGES = (
    (0x1100, 0x115F),    # Hangul Jamo
    (0x2E80, 0x303E),    # CJK radicals, Kangxi, punctuation
    (0x3041, 0x33FF),    # Hiragana through CJK compatibility
    (0x3400, 0x4DBF),    # CJK extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xA000, 0xA4CF),    # Yi
    (0xAC00, 0xD7A3),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0x20000, 0x2A6DF),  # CJK extension B
)


def _is_wide(character: str) -> bool:
    point = ord(character)
    return any(low <= point <= high for low, high in _WIDE_RANGES)


# --- selection -------------------------------------------------------------

def available_backends() -> List[str]:
    """Backend names that would actually work right now."""
    usable = ["chars4"]
    if _tiktoken_encoding() is not None:
        usable.append("tiktoken")
    if _transformers_tokenizer() is not None:
        usable.append("transformers")
    return usable


def active_backend() -> str:
    return _ACTIVE


def set_backend(name: str) -> str:
    """Select an estimator. Falls back to the default when unavailable.

    Falling back rather than raising is deliberate: a user who selected
    `tiktoken` and then uninstalled it should get a slightly less accurate
    estimate, not a context window that stopped working.
    """
    global _ACTIVE
    if name == _ACTIVE:
        return _ACTIVE
    if name not in _BACKENDS:
        return _ACTIVE
    # Probe it once; a backend that raises on a probe is treated as absent.
    try:
        if _BACKENDS[name]("hello") is None:
            return _ACTIVE
    except Exception:  # noqa: BLE001 - a broken optional backend is absent
        return _ACTIVE
    _ACTIVE = name
    return _ACTIVE


def reset_backend() -> None:
    """Restore the default. Used by tests and by `harness doctor`."""
    global _ACTIVE
    _ACTIVE = "chars4"


# --- the call signature every consumer uses ---------------------------------

def count(messages: list[dict[str, Any]],
          tools: list[dict[str, Any]] | None = None) -> int:
    """Estimate the tokens a request would use.

    The signature is unchanged from the original `estimate_tokens`, so no call
    site had to move when the backends were introduced.
    """
    serialized = json.dumps({"messages": messages, "tools": tools or []},
                            ensure_ascii=False, default=str)
    backend = _BACKENDS.get(_ACTIVE, _chars4)
    try:
        count_value = backend(serialized)
    except Exception:  # noqa: BLE001 - an estimator must never break a request
        count_value = _chars4(serialized)
    if count_value is None:
        count_value = _chars4(serialized)
    return max(1, int(count_value))


def relative_to_chars4(text: str) -> float:
    """How much the active backend disagrees with the default, for `doctor`."""
    exact = count([{"role": "user", "content": text}])
    return exact / max(1, _chars4(text))
