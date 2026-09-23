"""Deterministic, loss-aware compaction of tool evidence before model reuse."""

from __future__ import annotations

import re
import unicodedata


ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def clean_output(value: str) -> str:
    value = ANSI.sub("", value)
    return "".join(character for character in value
                   if character in "\n\t" or unicodedata.category(character) != "Cc")


def _collapse_repeated_lines(value: str) -> str:
    lines = value.splitlines()
    result: list[str] = []
    previous = None
    repeats = 0
    for line in lines:
        if line == previous:
            repeats += 1
            continue
        if repeats:
            result.append(f"… previous line repeated {repeats} more times")
            repeats = 0
        result.append(line)
        previous = line
    if repeats:
        result.append(f"… previous line repeated {repeats} more times")
    return "\n".join(result)


def compact_tool_output(tool_name: str, output: str, *, limit: int = 1500) -> str:
    """Preserve failure evidence and bounds while dropping repetitive chatter."""
    if limit < 200:
        raise ValueError("Output limit must be at least 200 characters")
    cleaned = clean_output(output)
    if tool_name in {"run_bash", "search_files"}:
        cleaned = _collapse_repeated_lines(cleaned)
    if tool_name == "run_pytest":
        lines = cleaned.splitlines()
        summary = next((line.strip() for line in reversed(lines)
                        if re.search(r"\b(?:\d+ failed|\d+ passed|no tests ran)\b", line)), "")
        failure_start = next((i for i, line in enumerate(lines) if " FAILURES " in line), None)
        if failure_start is not None:
            failure_end = next((i for i in range(failure_start + 1, len(lines))
                                if " short test summary info " in lines[i]), len(lines))
            failures = [line for line in lines[failure_start:failure_end]
                        if not re.search(r"\sPASSED\s|\sSKIPPED\s", line)]
            cleaned = "\n".join(failures + ([summary] if summary else []))
        elif summary and len(cleaned) > limit:
            cleaned = summary
    if len(cleaned) <= limit:
        return cleaned
    # Keep both the beginning and end: shell/test failures frequently end with the decisive error.
    head = int(limit * 0.55)
    tail = limit - head - 70
    result = cleaned[:head].rstrip() + f"\n… {len(cleaned)-head-tail} characters omitted …\n" + cleaned[-tail:].lstrip()
    return result[:limit]


def rank_search_results(output: str, task: str, semif_engine=None, *, max_files: int = 3) -> str:
    """Reduce a many-file grep result to representative matches from relevant files."""
    by_path: dict[str, list[str]] = {}
    for line in output.splitlines():
        path, separator, rest = line.partition(":")
        if separator and rest.partition(":")[0].isdigit():
            by_path.setdefault(path, []).append(line)
    if len(by_path) <= max_files:
        return output
    terms = set(re.findall(r"[a-z0-9_]+", task.lower()))
    lexical = {path: len(terms.intersection(set(re.findall(r"[a-z0-9_]+", path.lower()))))
               + 0.1 * len(terms.intersection(set(re.findall(r"[a-z0-9_]+", " ".join(by_path[path][:2]).lower()))))
               for path in by_path}
    candidates = sorted(by_path, key=lambda path: (-lexical[path], path))[:26]
    scored: dict[str, float] = {}
    if semif_engine is not None:
        try:
            options = {f"F{i}": f"{path}: {' '.join(by_path[path][:2])[:240]}"
                       for i, path in enumerate(candidates)}
            decision = semif_engine.decide(f"Which file best answers this task? {task[:500]}", options)
            scored = {path: decision.probabilities[f"F{i}"] for i, path in enumerate(candidates)}
        except Exception:
            scored = {}
    if not scored:
        scored = {path: lexical[path] for path in candidates}
    chosen = sorted(candidates, key=lambda path: (-scored[path], path))[:max_files]
    lines = [line for path in chosen for line in by_path[path][:3]]
    lines.append(f"… showing {len(chosen)} of {len(by_path)} matching files; narrow the search to inspect others.")
    return "\n".join(lines)
