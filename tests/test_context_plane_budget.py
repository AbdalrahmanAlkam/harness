"""Phase 1 — the Context Plane budget.

The plane is the only thing standing between a plugin and the model's context
window, so these tests are mostly about what it *refuses*. A budget that leaks
is not a budget.

The properties held here:
- no fragment escapes the available budget;
- a pinned fragment is never evicted, whatever it costs;
- every admission and every rejection carries a reason;
- admission is two-stage and the first stage is free;
- a plugin cannot understate its own size to buy its way in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_harness.agent.context_plane import (
    ContextPlane,
    build_plane,
    project_instruction_files,
)
from adaptive_harness.plugins.types import (
    ContextFragment,
    PRIORITY_OPPORTUNISTIC,
    PRIORITY_PINNED,
    PRIORITY_PROJECT,
    PRIORITY_REFERENCE,
    PRIORITY_REQUIREMENT,
    Trigger,
)


def _fragment(source: str, tokens: int, **kwargs) -> ContextFragment:
    kwargs.setdefault("priority", PRIORITY_REFERENCE)
    kwargs.setdefault("trigger", "always")
    return ContextFragment(source=source, content="x" * max(1, tokens),
                           tokens=tokens, **kwargs)


# --- the budget is a budget -------------------------------------------------


def test_nothing_is_admitted_beyond_the_budget():
    plane = build_plane(1000, [_fragment(f"f{i}", 200) for i in range(20)])
    report = plane.allocate()
    assert report.used <= report.available, (
        f"admitted {report.used} tokens into {report.available} available")
    assert len(report.admitted) + len(report.rejected) == 20, (
        "a fragment was neither admitted nor explained")


def test_every_rejection_says_why():
    plane = build_plane(1000, [_fragment("big", 9000)])
    report = plane.allocate()
    assert report.rejected
    source, reason = report.rejected[0]
    assert source == "big" and reason, "a silent rejection is indistinguishable from a bug"


def test_a_turn_with_no_fragments_is_fine():
    report = ContextPlane(1000).allocate()
    assert report.admitted == [] and report.rejected == []


def test_the_budget_reserves_room_for_the_final_summary():
    """A request that fills the window completely has nowhere to put an answer."""
    from adaptive_harness.agent.context_plane import RESERVED_FOR_SUMMARY_FRACTION

    plane = ContextPlane(1000)
    assert plane.reserved == int(1000 * RESERVED_FOR_SUMMARY_FRACTION)
    assert plane.available < plane.capacity, "the whole window was handed out"


# --- pinned means pinned ----------------------------------------------------


def test_a_pinned_fragment_survives_a_crowded_budget():
    plane = build_plane(1000, [
        _fragment("pinned", 500, priority=PRIORITY_PINNED, pinned=True),
        _fragment("big-1", 400), _fragment("big-2", 400),
    ])
    report = plane.allocate()
    assert "pinned" in report.admitted, "a pinned fragment was evicted"
    assert report.rejected, "and the space went to something else"


def test_a_pinned_fragment_survives_even_when_it_exceeds_the_window():
    """It is pinned. The alternative is a contract that silently does not hold."""
    plane = build_plane(1000, [
        _fragment("project:AGENTS.md", 5000, priority=PRIORITY_PINNED, pinned=True),
        _fragment("other", 10),
    ])
    report = plane.allocate()
    assert "project:AGENTS.md" in report.admitted
    assert [source for source, _ in report.rejected] == ["other"], (
        "a pinned fragment must still cost the others their space")


def test_pinned_fragments_are_ordered_first():
    plane = build_plane(1000, [
        _fragment("low-priority", 100),
        _fragment("pinned", 100, priority=PRIORITY_PINNED, pinned=True),
    ])
    assert plane.allocate().admitted[0] == "pinned"


# --- priority order ---------------------------------------------------------


def test_guaranteed_fragments_beat_opportunistic_ones():
    plane = build_plane(1000, [
        _fragment("opportunistic", 100, priority=PRIORITY_OPPORTUNISTIC, guaranteed=False),
        _fragment("requirement", 100, priority=PRIORITY_REQUIREMENT),
    ])
    admitted = plane.allocate().admitted
    assert admitted.index("requirement") < admitted.index("opportunistic")


def test_a_project_fragment_outranks_a_reference_one():
    plane = build_plane(1000, [
        _fragment("file:x.py", 100, priority=PRIORITY_REFERENCE),
        _fragment("project:conventions", 100, priority=PRIORITY_PROJECT),
    ])
    admitted = plane.allocate().admitted
    assert admitted.index("project:conventions") < admitted.index("file:x.py")


# --- a plugin cannot understate its own size --------------------------------


def test_the_measured_cost_beats_the_declared_cost():
    """A fragment claiming to be tiny but carrying a lot of text is measured,
    not believed."""
    liar = ContextFragment(source="liar", content="x" * 20000, tokens=1,
                           priority=PRIORITY_REFERENCE, trigger=Trigger.parse("always"))
    plane = build_plane(1000, [liar])
    report = plane.allocate()
    assert "liar" not in report.admitted, (
        "a plugin bought its way into the context by understating its size")


def test_a_measured_cost_is_never_zero():
    plane = ContextPlane(1000)
    plane.contribute(_fragment("f", 0))
    assert plane._measure(plane.fragments["f"]) >= 1


# --- two-stage admission ----------------------------------------------------


def test_the_trigger_stage_is_free_and_local():
    """Stage one must not consult a model -- sacred invariant 1. A relevance
    callback wired in must not be called during selection."""
    calls = []
    plane = ContextPlane(1000, relevance=lambda fragment, text: calls.append(1) or 1.0)
    plane.contribute(_fragment("f", 10, trigger="keyword:refactor"))
    plane.select(text="please refactor this")
    assert calls == [], "the trigger stage called the classifier"


def test_a_non_matching_fragment_is_never_scored():
    plane = ContextPlane(1000)
    plane.contribute(_fragment("only-on-deploy", 10, trigger="keyword:deploy"))
    assert plane.select(text="an ordinary request") == []
    assert plane.select(text="time to deploy") != []


def test_a_pinned_fragment_is_selected_regardless_of_its_trigger():
    plane = ContextPlane(1000)
    plane.contribute(_fragment("always-in", 10, priority=PRIORITY_PINNED, pinned=True,
                               trigger="keyword:never-typed"))
    assert len(plane.select(text="anything")) == 1


def test_with_no_classifier_the_plane_defers_opportunistic_rather_than_inventing():
    """The honest default: admit what a plugin marked durable, defer what it
    marked optional, rather than pretending to know which matters."""
    plane = ContextPlane(1000)
    plane.contribute(_fragment("guaranteed", 10, guaranteed=True))
    plane.contribute(_fragment("optional", 10, guaranteed=False))
    scores = plane.score(plane.select(), "text")
    assert scores["guaranteed"] > scores["optional"]


def test_a_relevance_classifier_is_used_when_supplied():
    # Score on the fragment's own content, not the turn text, so the two
    # fragments are genuinely distinguishable.
    plane = ContextPlane(1000, relevance=lambda fragment, text:
                         0.9 if "git" in fragment.content else 0.1)
    plane.contribute(ContextFragment(
        source="about-git", content="git workflow", tokens=10,
        priority=PRIORITY_REFERENCE, trigger=Trigger.parse("always")))
    plane.contribute(ContextFragment(
        source="about-other", content="something else", tokens=10,
        priority=PRIORITY_REFERENCE, trigger=Trigger.parse("always")))
    scores = plane.score(plane.select(), "an ordinary turn")
    assert scores["about-git"] == pytest.approx(0.9)
    assert scores["about-other"] == pytest.approx(0.1)


def test_a_classifier_that_raises_does_not_break_the_turn():
    """A classifier is a local model that can fail to load, or a plugin backend
    that can crash. Neither may take down a request."""
    def exploding(fragment, text):
        raise RuntimeError("classifier is down")

    plane = ContextPlane(1000, relevance=exploding)
    plane.contribute(_fragment("f", 10))
    report = plane.allocate(text="hello")
    assert report.admitted == ["f"], (
        "a broken relevance classifier must fall back, not lose the fragment")


# --- attribution ------------------------------------------------------------


def test_every_rendered_fragment_names_its_provenance():
    """Anything the model reads can be traced back to whoever put it there."""
    fragment = ContextFragment(source="skill:refactor", content="do it well",
                               tokens=5, priority=PRIORITY_REFERENCE,
                               trigger=Trigger.parse("always"),
                               provenance="plugin:helper")
    rendered = ContextPlane(1000).render([fragment])
    assert "skill:refactor" in rendered and "plugin:helper" in rendered


def test_rendering_nothing_produces_nothing():
    plane = ContextPlane(1000)
    assert plane.render([]) == ""
    assert plane.as_messages([]) == []


# --- re-contribution replaces rather than duplicates ------------------------


def test_recontributing_a_source_replaces_it():
    plane = ContextPlane(1000)
    plane.contribute(_fragment("dup", 100))
    plane.contribute(_fragment("dup", 200))
    assert len(plane.fragments) == 1
    assert plane.allocate().admitted == ["dup"]


def test_a_fragment_can_be_forgotten():
    plane = build_plane(1000, [_fragment("temp", 10)])
    assert plane.forget("temp")
    assert not plane.forget("temp")
    assert plane.allocate().admitted == []


def test_contribute_rejects_a_non_fragment():
    """A raw string is a bug, and must be caught where it is made."""
    with pytest.raises(TypeError):
        ContextPlane(1000).contribute("just some text")


# --- project instructions ---------------------------------------------------


def test_project_instructions_are_loaded_and_pinned(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("Use tabs. Run the tests.", encoding="utf-8")
    instructions = tmp_path / ".harness" / "instructions"
    instructions.mkdir(parents=True)
    (instructions / "style.md").write_text("Two spaces for Python.", encoding="utf-8")

    fragments = project_instruction_files(tmp_path)
    sources = {fragment.source for fragment in fragments}
    assert sources == {"project:AGENTS.md", "project:style.md"}
    assert all(fragment.pinned for fragment in fragments), (
        "a project instruction the harness could drop is worse than none")
    assert all(fragment.trigger.matches(text="anything") for fragment in fragments)


def test_a_workspace_with_no_instructions_loads_nothing(tmp_path: Path):
    assert project_instruction_files(tmp_path) == []


# --- the describe() surface, which /context renders --------------------------


def test_describe_reports_what_is_held_and_why():
    plane = build_plane(1000, [_fragment("held", 40, priority=PRIORITY_PROJECT,
                                          trigger="keyword:go")])
    plane.allocate(text="go")
    described = plane.describe()
    assert described["capacity"] == 1000
    assert "held" in described["held"]
    assert described["held"]["held"]["provenance"] == ""
    assert "go" in described["held"]["held"]["trigger"]
    assert described["last"]["admitted"] == ["held"]


# --- 1.1 the one rule: nothing writes into the message list but the plane --


def test_only_the_agent_loop_and_the_interface_write_conversation_messages():
    """The rule this module exists to enforce, checked against the source.

    If content can enter the transcript by the back door, "how much am I paying
    for context" has no answer -- and that is the only question an operator has
    about context.

    Two writers are legitimate and are named here deliberately:

    - `agent/agent.py`, which appends the conversation itself: the user's turn,
      the model's replies, tool results, nudges. That is the transcript, not
      context.
    - `tui/app.py`, which *renders* that transcript and owns the saved session.

    Anything else that appends to a message list is a way for content to reach
    the model without passing the plane, and is the thing this test exists to
    catch when someone adds a new one.
    """
    import re

    package = Path(__file__).resolve().parent.parent / "src" / "adaptive_harness"
    allowed = {"agent/agent.py", "tui/app.py"}
    offenders = []
    for path in package.rglob("*.py"):
        relative = str(path.relative_to(package))
        if relative in allowed or path.name == "context_plane.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        code = re.sub(r'"""(?:.|\n)*?"""', "", source)
        code = re.sub(r"#.*", "", code)
        # `self.messages` is the agent transcript; a swarm's `messages` is an
        # inter-agent ledger and is not context at all.
        for match in re.finditer(r"self\.messages\.(append|insert|extend)\(", code):
            offenders.append(f"{relative}: self.messages.{match.group(1)}")
    assert not offenders, (
        f"these modules write conversation messages outside the agent loop: {offenders}")


def test_the_agent_loop_admits_context_through_the_plane():
    """The complement of the rule above: the agent's own context injection goes
    through the plane, not by appending a string it built itself."""
    source = (Path(__file__).resolve().parent.parent / "src" / "adaptive_harness"
              / "agent" / "agent.py").read_text(encoding="utf-8")
    assert "as_messages(admitted_fragments)" in source, (
        "admitted context must be rendered by the plane, not by the loop")
    assert "context_budget" in source, "the budget report must reach the interface"
