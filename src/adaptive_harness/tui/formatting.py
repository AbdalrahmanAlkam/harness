"""Small, dependency-free cleanup for model Markdown and displayable formulas."""
from __future__ import annotations

import re
from dataclasses import dataclass


_LATEX = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\approx": "≈", r"\infty": "∞", r"\rightarrow": "→",
    r"\to": "→", r"\Rightarrow": "⇒", r"\pi": "π", r"\theta": "θ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\omega": "ω",
    r"\Delta": "Δ", r"\Omega": "Ω", r"\Sigma": "Σ", r"\Gamma": "Γ",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\oint": "∮",
    r"\sqrt": "√", r"\partial": "∂", r"\nabla": "∇", r"\forall": "∀",
    r"\exists": "∃", r"\in": "∈", r"\notin": "∉",
}
_SUPERSCRIPT = str.maketrans(dict(zip(
    "0123456789+-=()abdefghijklmnoprstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")))
_SUBSCRIPT = str.maketrans(dict(zip("0123456789+-=()aehi jklmnoprstuvx".replace(" ", ""),
                                    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")))


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
    value = re.sub(r"\^([0-9+-])", lambda m: m.group(1).translate(_SUPERSCRIPT), value)
    value = re.sub(r"\^\{([^{}]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPT), value)
    value = re.sub(r"_\{([^{}]+)\}", lambda m: m.group(1).translate(_SUBSCRIPT), value)
    value = re.sub(r"_([0-9aehijklmnoprstuvx])", lambda m: m.group(1).translate(_SUBSCRIPT), value)
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


@dataclass(frozen=True)
class _MathBox:
    lines: tuple[str, ...]
    baseline: int = 0

    @property
    def width(self) -> int:
        return max((len(line) for line in self.lines), default=0)


def _read_braced(source: str, position: int) -> tuple[str, int] | None:
    while position < len(source) and source[position].isspace():
        position += 1
    if position >= len(source) or source[position] != "{":
        return None
    depth, start = 1, position + 1
    position += 1
    while position < len(source) and depth:
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
        position += 1
    return (source[start:position - 1], position) if depth == 0 else None


def _join_boxes(boxes: list[_MathBox]) -> _MathBox:
    if not boxes:
        return _MathBox(("",))
    above = max(box.baseline for box in boxes)
    below = max(len(box.lines) - box.baseline - 1 for box in boxes)
    lines = []
    for row in range(above + below + 1):
        segments = []
        for box in boxes:
            source_row = row - above + box.baseline
            segment = box.lines[source_row] if 0 <= source_row < len(box.lines) else ""
            segments.append(segment.ljust(box.width))
        lines.append("".join(segments).rstrip())
    return _MathBox(tuple(lines), above)


def _display_box(formula: str) -> _MathBox:
    """Build a small Unicode fraction layout without evaluating model text."""
    boxes: list[_MathBox] = []
    cursor = 0
    for match in re.finditer(r"\\(?:dfrac|tfrac|frac)(?![A-Za-z])", formula):
        if match.start() < cursor:
            continue
        numerator = _read_braced(formula, match.end())
        denominator = _read_braced(formula, numerator[1]) if numerator else None
        if not numerator or not denominator:
            continue
        raw_prefix = formula[cursor:match.start()]
        prefix = _format_formula(raw_prefix)
        if prefix:
            boxes.append(_MathBox((prefix + (" " if raw_prefix[-1:].isspace() else ""),)))
        top = _display_box(numerator[0])
        bottom = _display_box(denominator[0])
        width = max(top.width, bottom.width, 1)
        top_lines = tuple(line.center(width) for line in top.lines)
        bottom_lines = tuple(line.center(width) for line in bottom.lines)
        boxes.append(_MathBox((*top_lines, "─" * width, *bottom_lines), len(top_lines)))
        cursor = denominator[1]
    suffix = _format_formula(formula[cursor:])
    if suffix:
        boxes.append(_MathBox(tuple(suffix.splitlines())))
    return _join_boxes(boxes)


def _display_math(formula: str) -> str:
    value = re.sub(r"\\(?:begin|end)\{(?:aligned|align|equation\*?)\}", "", formula)
    rows = re.split(r"\\\\|\n", value)
    return "\n".join("\n".join(_display_box(row.replace("&", " ")).lines).strip("\n")
                     for row in rows if row.strip())


def format_model_markdown(source: str, *, plain: bool = False) -> str:
    """Remove leaked channel markers and render common LaTeX in readable Unicode."""
    text = source or ""
    # Keep code blocks byte-for-byte while transforming prose and math outside them.
    chunks = re.split(r"(```.*?```|~~~.*?~~~)", text, flags=re.S)
    for index in range(0, len(chunks), 2):
        part = chunks[index]
        part = re.sub(r"<\|[^|]+\|>", "", part, flags=re.I)
        part = re.sub(r"\[(?:analysis|assistant/analysis|final|assistant/final|thought)\]", "", part, flags=re.I)
        part = re.sub(r"<think>.*?</think>", "", part, flags=re.I | re.S)
        def block(match: re.Match[str]) -> str:
            rendered = _display_math(match.group(1))
            return "\n" + rendered + "\n" if plain else "\n\n```text\n" + rendered + "\n```\n\n"
        part = re.sub(r"\\\[(.*?)\\\]", block, part, flags=re.S)
        part = re.sub(r"\\\((.*?)\\\)", lambda m: _format_formula(m.group(1)), part, flags=re.S)
        part = re.sub(r"\$\$(.*?)\$\$", block, part, flags=re.S)
        part = re.sub(r"\$(?!\$)([^$\n]+)\$", lambda m: _format_formula(m.group(1)), part)
        chunks[index] = part
    return "".join(chunks)
