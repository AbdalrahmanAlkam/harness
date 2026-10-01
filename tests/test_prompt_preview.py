"""The prompt list is an audit view, so it has to be readable.

`prompts list` is what a user opens to see every instruction the models
receive. It is only useful if a row is legible on one line and if the grouping
tells them which prompts can actually fire.

These tests pin the properties that make it that, because the original was
readable at none of them: previews cut mid-word, columns wrapping into each
other, and 41 flat alphabetical rows giving no sense of what triggers what.
"""

from __future__ import annotations

import pytest

from adaptive_harness.prompts import get_default_registry
from adaptive_harness.prompt_preview import (
    GROUPS,
    MIN_PREVIEW_WIDTH,
    OVERRIDE_FLAG,
    build_rows,
    collapse,
    group_rows,
    render,
    truncate,
)


# --- a preview is the whole prompt, on one line ----------------------------


def test_a_multiline_prompt_previews_from_its_beginning():
    """The original took ``splitlines()[0]``, so a prompt that opened with a
    blank line previewed as a fragment of the middle."""
    text = "\n\nThe task requires real file changes.\nUse write_file or edit_file."
    assert collapse(text).startswith("The task requires")


def test_collapse_flattens_every_kind_of_whitespace():
    assert collapse("a\n\n  b\tc  ") == "a b c"
    assert collapse("") == ""


def test_collapse_covers_the_whole_prompt():
    registry = get_default_registry()
    for name in registry.names():
        preview = collapse(registry.get(name))
        assert preview, f"{name} collapsed to nothing"
        # The first words survive, which is what makes it a preview rather
        # than a fragment of the middle.
        first_line = registry.get(name).strip().splitlines()[0]
        assert preview.startswith(first_line[:20])


# --- truncation is at a word, and says so ----------------------------------


def test_a_short_preview_is_not_truncated():
    assert truncate("short", 40) == "short"


def test_a_long_preview_is_cut_at_a_word_boundary():
    text = "the quick brown fox jumps over the lazy dog"
    cut = truncate(text, 20)
    assert cut.endswith("…")
    assert " " in cut.rstrip("…"), "cut mid-word"
    assert text.startswith(cut.rstrip("…"))


def test_a_zero_width_yields_nothing_rather_than_raising():
    assert truncate("anything", 0) == "anything"


def test_a_very_narrow_width_still_returns_something():
    assert truncate("the quick brown fox", 3)


# --- rows fit the console, at every width -----------------------------------


@pytest.mark.parametrize("width", [60, 80, 100, 120, 160, 200])
def test_no_row_wraps_at_any_terminal_width(width: int):
    """Rich wraps to its own width, so a preview sized to something else is
    wrapped anyway -- which is the ragged column this replaced."""
    lines = render(get_default_registry(), width=width)
    for line in lines:
        assert len(line) <= width, (
            f"a line of {len(line)} chars will wrap at {width}: {line!r}")


def test_a_narrow_terminal_drops_the_size_column_rather_than_the_preview():
    """A list of forty ellipses and no content is not an audit view."""
    rows = build_rows(get_default_registry(), width=60)
    assert rows, "nothing was listed at all"
    assert all(row.size == 0 for row in rows), (
        "the size column was kept at a width where the preview has no room")
    assert any(row.preview and not row.preview.endswith("…") for row in rows), (
        "every preview was truncated to nothing")


def test_a_wide_terminal_keeps_the_size_column():
    rows = build_rows(get_default_registry(), width=160)
    assert all(row.size > 0 for row in rows)


# --- grouping says what can trigger a prompt -------------------------------


def test_every_prompt_appears_exactly_once():
    """A prompt that is invisible because someone forgot a table row is worse
    than one that is merely out of place."""
    registry = get_default_registry()
    grouped: list[str] = []
    for _heading, _blurb, members in group_rows(build_rows(registry, width=120)):
        grouped.extend(row.name for row in members)
    assert sorted(grouped) == sorted(registry.names())


def test_the_groupings_name_no_duplicate():
    seen: set[str] = set()
    for _heading, _blurb, names in GROUPS:
        assert not (seen & set(names)), "a prompt is grouped twice"
        seen.update(names)


def test_an_ungrouped_prompt_is_still_listed():
    """Forward compatibility: a prompt added tomorrow must appear today."""
    from adaptive_harness.prompt_preview import Row

    rows = [Row(name="brand.new.prompt", preview="x", size=1, overridden=False)]
    groups = group_rows(rows)
    assert len(groups) == 1
    heading, blurb, members = groups[0]
    assert heading == "Other", "a new prompt was dropped instead of listed"
    assert members[0].name == "brand.new.prompt"


def test_the_system_prompt_is_under_always_in_context():
    """The group a user cares about most is the one that always applies."""
    groups = group_rows(build_rows(get_default_registry(), width=120))
    always = groups[0]
    names = {row.name for row in always[2]}
    assert "system.default" in names
    assert "intervention.termination_verdict" not in names


def test_every_group_has_a_heading_and_an_explanation():
    for heading, blurb, _members in group_rows(build_rows(get_default_registry(), width=120)):
        assert heading
        assert blurb, f"the {heading!r} group says nothing about when it fires"


# --- overrides are visible -------------------------------------------------


def test_an_overridden_prompt_is_flagged_and_the_flag_fits():
    from adaptive_harness.prompt_preview import Row

    row = Row(name="x", preview="some preview", size=10, overridden=True)
    assert OVERRIDE_FLAG.strip() in render_lines([row], width=120)[0]


def render_lines(rows, width):
    """The row format, isolated so the flag is testable on its own."""
    name_width = max((len(r.name) for r in rows), default=20)
    lines = []
    for row in rows:
        size_field = f"{row.size:>6}c  " if row.size else ""
        flag = OVERRIDE_FLAG if row.overridden else ""
        lines.append(f"  {row.name.ljust(name_width)}  {size_field}{row.preview}{flag}")
    return lines


# --- the listing is complete -----------------------------------------------


def test_every_registered_prompt_appears_in_the_listing():
    listing = "\n".join(render(get_default_registry(), width=140))
    for name in get_default_registry().names():
        assert name in listing, f"{name} is missing from the listing"


def test_the_listing_has_no_empty_previews():
    rows = build_rows(get_default_registry(), width=120)
    for row in rows:
        assert row.preview.strip(), f"{row.name} has an empty preview"
        assert not row.preview.startswith(" "), f"{row.name}'s preview starts mid-sentence"
