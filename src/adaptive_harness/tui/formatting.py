"""Small, dependency-free cleanup for model Markdown and displayable formulas."""
from __future__ import annotations

import re


_LATEX = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\approx": "≈", r"\infty": "∞", r"\rightarrow": "→",
    r"\to": "→", r"\Rightarrow": "⇒", r"\pi": "π", r"\theta": "θ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\omega": "ω",
    r"\Delta": "Δ", r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\sqrt": "√", r"\partial": "∂", r"\nabla": "∇", r"\forall": "∀",
    r"\exists": "∃", r"\in": "∈", r"\notin": "∉",
}
_SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPT = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


def _replace_tex_command(source: str, command: str, arity: int, render) -> str:
    """Replace a TeX command with balanced arguments, including nested braces."""
    output = []
    cursor = 0
    while True:
        start = source.find(command, cursor)
        if start < 0:
            output.append(source[cursor:])
            return "".join(output)
        output.append(source[cursor:start])
        position = start + len(command)
        if position < len(source) and source[position].isalpha():
            output.append(command)
            cursor = position
            continue
        args = []
        for _ in range(arity):
            while position < len(source) and source[position].isspace():
                position += 1
            if position >= len(source) or source[position] != "{":
                break
            depth = 1
            arg_start = position + 1
            position += 1
            while position < len(source) and depth:
                if source[position] == "{":
                    depth += 1
                elif source[position] == "}":
                    depth -= 1
                position += 1
            if depth:
                break
            args.append(source[arg_start:position - 1])
        if len(args) != arity:
            output.append(source[start:start + len(command)])
            cursor = start + len(command)
            continue
        output.append(render(args))
        cursor = position


def _format_formula(formula: str) -> str:
    value = formula.strip()
    for command in (r"\text", r"\mathrm", r"\operatorname", r"\boxed", r"\mathbf", r"\mathit"):
        value = _replace_tex_command(value, command, 1, lambda args: args[0])
    value = _replace_tex_command(value, r"\frac", 2,
        lambda args: "(" + _format_formula(args[0]) + ")/(" + _format_formula(args[1]) + ")")
    value = _replace_tex_command(value, r"\sqrt", 1, lambda args: "√(" + _format_formula(args[0]) + ")")
    value = re.sub(r"\^\{([^{}]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPT), value)
    value = re.sub(r"_\{([^{}]+)\}", lambda m: m.group(1).translate(_SUBSCRIPT), value)
    value = re.sub(r"\^([0-9+-])", lambda m: m.group(1).translate(_SUPERSCRIPT), value)
    value = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUBSCRIPT), value)
    value = value.replace(r"\{", "{").replace(r"\}", "}")
    value = re.sub(r"\\begin\{(?:aligned|align|cases|equation\*?)\}|\\end\{(?:aligned|align|cases|equation\*?)\}", "", value)
    value = value.replace(r"\\", "\n").replace("&", " ")
    value = re.sub(r"\\(?:left|right|displaystyle|textstyle|quad|qquad|,|;|:|!)", " ", value)
    value = re.sub(r"\\mathbb\{([RNCQ])\}", lambda m: {"R": "ℝ", "N": "ℕ", "C": "ℂ", "Q": "ℚ"}[m.group(1)], value)
    for command, symbol in _LATEX.items():
        value = value.replace(command, symbol)
    value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r" *\n *", "\n", re.sub(r"[ \t]+", " ", value)).strip()


def format_model_markdown(source: str) -> str:
    """Remove leaked channel markers and render common LaTeX in readable Unicode."""
    text = source or ""
    # Keep code blocks byte-for-byte while transforming prose and math outside them.
    chunks = re.split(r"(```.*?```|~~~.*?~~~)", text, flags=re.S)
    for index in range(0, len(chunks), 2):
        part = chunks[index]
        part = re.sub(r"<\|[^|]+\|>", "", part, flags=re.I)
        part = re.sub(r"\[(?:analysis|assistant/analysis|final|assistant/final|thought)\]", "", part, flags=re.I)
        part = re.sub(r"<think>.*?</think>", "", part, flags=re.I | re.S)
        part = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n" + _format_formula(m.group(1)) + "\n", part, flags=re.S)
        part = re.sub(r"\\\((.*?)\\\)", lambda m: _format_formula(m.group(1)), part, flags=re.S)
        part = re.sub(r"\$\$(.*?)\$\$", lambda m: "\n" + _format_formula(m.group(1)) + "\n", part, flags=re.S)
        part = re.sub(r"\$(?!\$)([^$\n]+)\$", lambda m: _format_formula(m.group(1)), part)
        chunks[index] = part
    return "".join(chunks)
