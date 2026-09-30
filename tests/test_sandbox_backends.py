"""The sandbox plugin's backends.

Everything here is tested **without root, without a real sandbox, and without
attempting to escape one**. What is testable is the part that decides whether
the rest can be trusted: detection tells the truth, the honest backend says so
in plain words, a backend name the plugin does not implement is refused, an
argv list is never handed to a shell, and the timeout kills the whole process
group. The one test that needs a real backend is marked skipped when there is
not one, and even when it runs it only executes ``echo`` -- a sandbox suite
that tries to break out of a sandbox is a different, and much more dangerous,
piece of software.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

from adaptive_harness.plugins.host import PluginHost

# The core ships with no plugins, so the path is resolved rather than assumed.
from plugin_paths import plugin_path  # noqa: E402

PLUGIN_DIRECTORY = plugin_path("sandbox")


def _plugin_module():
    """The plugin's module, imported the way the host imports it.

    The host registers the module in ``sys.modules`` before executing it, and
    that registration is not optional: ``@dataclass`` resolves annotations
    through the module's namespace, so a module loaded without it fails on the
    first decorator.
    """
    path = PLUGIN_DIRECTORY / "plugin.py"
    spec = importlib.util.spec_from_file_location("_adaptive_plugin_sandbox", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sandbox = _plugin_module()


# --- the plugin itself ------------------------------------------------------


def test_the_plugin_loads_and_declares_what_it_actually_uses(tmp_path: Path):
    # The core ships with no plugins, so discovery is pointed at the directory
    # this plugin actually lives in.
    host = PluginHost(project_root=tmp_path)
    host.discovery_roots = lambda: [PLUGIN_DIRECTORY.parent]
    host.discover()
    plugin = next(p for p in host.plugins if p.name == "sandbox")
    assert plugin.ok, plugin.error

    assert {"tools", "subprocess"} <= set(plugin.permissions), (
        "it registers a tool and it runs external programs, so it says so")
    assert [tool.name for tool in plugin.tools] == ["sandbox_run"]
    assert plugin.tools[0].risk == "exec", (
        "running a command must be reviewed by the strict profile")
    assert host.is_read_only("sandbox_run") is False, (
        "a tool that executes cannot opt out of the gate by declaring itself read")


# --- detection tells the truth ---------------------------------------------


def test_every_backend_reports_an_availability_with_a_reason():
    statuses = sandbox.detect()
    assert [status.name for status in statuses] == [b.name for b in sandbox.BACKENDS]
    for status in statuses:
        assert isinstance(status.available, bool)
        assert status.reason.strip(), f"{status.name} reported nothing about itself"
        assert status.summary.strip()
        assert isinstance(status.auto_selectable, bool)


def test_detection_agrees_with_the_thing_it_claims_to_probe():
    """Landlock's availability is a fact about the kernel, so the reported
    answer has to be the one the syscall actually gave."""
    backend = sandbox.LandlockBackend()
    available, reason = backend.available()
    abi, probe_reason = sandbox._landlock_probe()
    assert available is (abi is not None)
    if not available:
        assert reason == probe_reason
    assert reason.strip()


def test_a_backend_off_this_platform_says_so_rather_than_pretending(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    for name in ("bubblewrap", "container"):
        available, reason = sandbox._BY_NAME[name]().available()
        assert not available, f"{name} claimed to work on {sys.platform}"
        assert "linux" in reason.lower()

    monkeypatch.setattr(sys, "platform", "linux")
    available, reason = sandbox.SeatbeltBackend().available()
    assert not available
    assert "macos" in reason.lower(), "the reason says why, not just 'unavailable'"


def test_a_probe_that_raises_becomes_an_unavailability_not_a_crash(monkeypatch):
    """Detection runs inside a tool call. A backend whose detection blows up
    must report itself unavailable, because the alternative is a traceback
    where a sentence belongs."""
    class Exploding(sandbox.Backend):
        name = "exploding"
        summary = "A backend whose probe raises."

        def available(self, *, deep: bool = False):
            raise RuntimeError("probe exploded")

        def wrap(self, argv, workspace):  # pragma: no cover - never reached
            raise AssertionError("an unavailable backend must not be wrapped")

        def confinement(self, workspace):  # pragma: no cover - never reached
            return "nothing"

    status = Exploding().status()
    assert status.available is False
    assert "probe exploded" in status.reason
    assert "RuntimeError" in status.reason


def test_an_explicitly_requested_unusable_backend_is_refused_not_downgraded(monkeypatch):
    """Asking for a backend and quietly getting another one is the failure this
    whole plugin is arranged to prevent, so it raises instead."""
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError) as refusal:
        sandbox.select("seatbelt")
    assert "seatbelt" in str(refusal.value)
    assert "nothing was run" in str(refusal.value).lower()


# --- the honest default -----------------------------------------------------


def test_none_backend_is_available_and_admits_it_confines_nothing():
    backend = sandbox.NoneBackend()
    available, reason = backend.available()
    assert available is True
    assert "confines nothing" in reason
    assert "NO CONFINEMENT" in backend.confinement(Path.cwd())


def test_the_none_backend_result_says_in_plain_words_that_nothing_was_confined(tmp_path: Path):
    result = sandbox.sandbox_run(["echo", "hello"], backend="none",
                                 timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert "hello" in result["output"]
    assert "NO CONFINEMENT" in result["output"]
    assert "Nothing was restricted" in result["output"]
    assert result["metadata"]["backend"] == "none"
    assert result["metadata"]["confinement_applied"] is False


def test_auto_reports_no_confinement_rather_than_picking_a_winner(monkeypatch, tmp_path: Path):
    """With every real backend off, ``auto`` still runs the command -- refusing
    outright would push people to switch the sandbox off silently -- but the
    result leads with the fact that nothing confined it."""
    for backend in sandbox.BACKENDS:
        monkeypatch.setattr(backend, "available",
                            lambda **_: (False, "forced off for this test"), raising=False)
    monkeypatch.setattr(sandbox.NoneBackend, "available",
                        lambda **_: (True, "the last resort"), raising=False)

    result = sandbox.sandbox_run(["echo", "unconfined"], timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert result["metadata"]["backend"] == "none"
    assert result["metadata"]["confinement_applied"] is False
    assert "NO CONFINEMENT" in result["output"]
    assert "unconfined" in result["output"], "the command still ran, unconfined"


def test_a_confinement_claim_always_says_what_was_left_open(tmp_path: Path):
    """Every backend's own description names what it does not restrict, so a
    reader is never left to assume the whole machine was sealed."""
    for backend in sandbox.BACKENDS:
        text = backend().confinement(tmp_path)
        assert len(text) > 40, f"{backend.name} does not describe itself"
    assert "NOT restricted" in sandbox.LandlockBackend().confinement(tmp_path)
    assert "no network" in sandbox.BubblewrapBackend().confinement(tmp_path)
    assert "network" in sandbox.SeatbeltBackend().confinement(tmp_path)
    assert "no network" in sandbox.ContainerBackend().confinement(tmp_path)


# --- a backend the plugin does not implement is refused ---------------------


def test_an_unknown_backend_is_refused_and_nothing_runs(tmp_path: Path):
    marker = tmp_path / "should-not-exist"
    result = sandbox.sandbox_run(["touch", str(marker)], backend="chroot",
                                 timeout=30, workspace=str(tmp_path))
    assert result["success"] is False
    assert "chroot" in result["error"]
    assert "not a backend this plugin implements" in result["error"]
    assert not marker.exists(), "a refused backend must not run the command"


def test_every_backend_name_the_tool_advertises_is_one_it_implements():
    manifest = (PLUGIN_DIRECTORY / "plugin.plugin.json").read_text(encoding="utf-8")
    import json
    schema = json.loads(manifest)["tools"][0]["parameters"]
    advertised = set(schema["properties"]["backend"]["enum"])
    assert advertised == set(sandbox._BY_NAME) | {"auto"}


# --- the command is never handed to a shell ---------------------------------


def test_argv_is_not_shell_interpolated(tmp_path: Path):
    """A list is execed directly, so the metacharacters are characters. If this
    ever went through a shell, the file would exist."""
    marker = tmp_path / "pwned"
    payload = f"; touch {marker}"
    result = sandbox.sandbox_run(["echo", payload, "$(whoami)", "*"],
                                 backend="none", timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert payload in result["output"], "the argument was echoed literally"
    assert "$(whoami)" in result["output"], "no expansion happened"
    assert not marker.exists()
    assert result["metadata"]["argv"] == ["echo", payload, "$(whoami)", "*"]


def test_a_string_carrying_shell_syntax_is_refused_not_guessed_at(tmp_path: Path):
    marker = tmp_path / "pwned"
    result = sandbox.sandbox_run(f"echo hi && touch {marker}",
                                 timeout=30, workspace=str(tmp_path))
    assert result["success"] is False
    assert "shell" in result["error"]
    assert not marker.exists()


def test_a_plain_string_is_split_without_a_shell(tmp_path: Path):
    result = sandbox.sandbox_run("echo one two", backend="none",
                                 timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert "one two" in result["output"]
    assert result["metadata"]["argv"] == ["echo", "one", "two"]


def test_an_empty_command_is_refused(tmp_path: Path):
    for command in ([], "   "):
        result = sandbox.sandbox_run(command, timeout=30, workspace=str(tmp_path))
        assert result["success"] is False
        assert "empty" in result["error"]


# --- the timeout bounds the run and the whole process group -----------------

#: Spawns a grandchild that inherits the process group, records its pid, then
#: waits. If only the direct child is killed on timeout, the grandchild outlives
#: the timeout -- which is the whole failure this tests.
_SPAWNER = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))
time.sleep(30)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not our process
        return True
    return True


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="no process groups on this platform")
def test_the_timeout_kills_the_whole_process_group(tmp_path: Path):
    pidfile = tmp_path / "grandchild.pid"
    started = time.monotonic()
    result = sandbox.sandbox_run([sys.executable, "-c", _SPAWNER, str(pidfile)],
                                 backend="none", timeout=2, workspace=str(tmp_path))
    elapsed = time.monotonic() - started

    assert result["metadata"]["timed_out"] is True
    assert result["success"] is False
    assert "timeout" in result["error"]
    assert elapsed < 20, "the run was bounded by the timeout, not by the sleep"
    assert pidfile.exists(), "the child never got as far as forking a grandchild"

    grandchild = int(pidfile.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild), (
        "a descendant survived the timeout; only the process that was started was killed")


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="no process groups on this platform")
def test_a_command_that_finishes_in_time_is_not_killed(tmp_path: Path):
    result = sandbox.sandbox_run([sys.executable, "-c", "print('done')"],
                                 backend="none", timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert result["metadata"]["timed_out"] is False
    assert "done" in result["output"]


# --- arguments that would make the result a lie ------------------------------


def test_the_filesystem_root_is_refused_as_a_workspace(tmp_path: Path):
    """Confining writes to ``/`` restricts nothing, so it is refused rather
    than reported as a sandbox."""
    result = sandbox.sandbox_run(["echo", "hi"], backend="none",
                                 timeout=30, workspace=tmp_path.anchor)
    assert result["success"] is False
    assert "refused" in result["error"]


def test_a_workspace_that_is_not_a_directory_is_refused(tmp_path: Path):
    missing = tmp_path / "nope"
    result = sandbox.sandbox_run(["echo", "hi"], backend="none",
                                 timeout=30, workspace=str(missing))
    assert result["success"] is False
    assert "not a directory" in result["error"]


@pytest.mark.parametrize("timeout", [0, -1, 500, "soon", None])
def test_an_unusable_timeout_is_refused(timeout, tmp_path: Path):
    result = sandbox.sandbox_run(["echo", "hi"], timeout=timeout, workspace=str(tmp_path))
    assert result["success"] is False
    assert "timeout" in result["error"]


def test_a_failing_command_reports_its_status_without_pretending_it_worked(tmp_path: Path):
    result = sandbox.sandbox_run([sys.executable, "-c", "raise SystemExit(3)"],
                                 backend="none", timeout=30, workspace=str(tmp_path))
    assert result["success"] is False
    assert "status 3" in result["error"]
    assert result["metadata"]["returncode"] == 3


def test_a_command_that_cannot_start_is_reported_as_not_started(tmp_path: Path):
    result = sandbox.sandbox_run(["definitely-not-a-real-binary-42"],
                                 backend="none", timeout=30, workspace=str(tmp_path))
    assert result["success"] is False
    assert "could not be started" in result["error"]
    assert "NO CONFINEMENT" in result["output"], (
        "even a failure reports what confined it, which was nothing")


# --- the argv each backend builds, checked as data --------------------------


def test_the_bubblewrap_argv_is_the_documented_one(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/bwrap")
    argv = sandbox.BubblewrapBackend().wrap(["echo", "hi"], tmp_path).argv
    assert argv[:2] == ["/usr/bin/bwrap", "--unshare-net"]
    assert "--ro-bind" in argv and argv[argv.index("--ro-bind") + 1:argv.index("--ro-bind") + 3] == ["/", "/"]
    assert argv[argv.index("--bind") + 1:argv.index("--bind") + 3] == [str(tmp_path)] * 2
    assert argv[argv.index("--") + 1:] == ["echo", "hi"], (
        "the command is the last thing in the argv, after --, and is never a shell string")


def test_the_bubblewrap_argv_ends_the_wrapper_with_the_parent(monkeypatch, tmp_path: Path):
    """Without --die-with-parent a killed wrapper leaves the confined process
    running with nobody left to reap it."""
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/bwrap")
    argv = sandbox.BubblewrapBackend().wrap(["echo"], tmp_path).argv
    assert "--die-with-parent" in argv


def test_the_container_argv_has_no_network_and_a_read_write_workspace(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sandbox.shutil, "which",
                        lambda name: "/usr/bin/docker" if name == "docker" else None)
    backend = sandbox.ContainerBackend()
    argv = backend.wrap(["echo", "hi"], tmp_path).argv
    assert argv[0] == "/usr/bin/docker"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--volume") + 1] == f"{tmp_path}:{tmp_path}"
    assert argv[argv.index("--workdir") + 1] == str(tmp_path)
    assert argv[-2:] == ["echo", "hi"]


def test_the_container_backend_is_never_selected_automatically():
    """A container is a different trust domain with a different failure mode.
    Choosing one is the operator's call, made by naming it."""
    assert sandbox.ContainerBackend.auto_selectable is False
    assert "container" not in sandbox._AUTO_ORDER
    for status in sandbox.detect():
        if status.name == "container":
            assert "opt-in" in status.reason
    monkeypatch_needed = {status.name for status in sandbox.detect() if status.auto_selectable}
    assert "container" not in monkeypatch_needed


def test_auto_prefers_landlock_then_bubblewrap_then_seatbelt(monkeypatch, tmp_path: Path):
    """The order is a preference, and it is a stated one."""
    calls: list[str] = []

    class Recorder(sandbox.Backend):
        def __init__(self, name, available):
            self.name, self._available = name, available

        def available(self, *, deep=False):
            calls.append(self.name)
            return (self._available, "recording")

        def wrap(self, argv, workspace):  # pragma: no cover - not reached
            raise AssertionError("no command is run by this test")

        def confinement(self, workspace):  # pragma: no cover - not reached
            return "nothing"

    for name, is_available in (("landlock", False), ("bubblewrap", True),
                               ("seatbelt", True)):
        monkeypatch.setitem(sandbox._BY_NAME, name, lambda n=name, a=is_available: Recorder(n, a))
    chosen, _ = sandbox.select("auto")
    assert chosen.name == "bubblewrap", calls
    assert calls == ["landlock", "bubblewrap"], "seatbelt was never consulted"


def test_the_seatbelt_profile_denies_the_network_and_writes_outside(tmp_path: Path):
    """The profile is generated data, so it is checked as data. Nothing here
    needs macOS, and nothing here runs anything."""
    profile = sandbox.SeatbeltBackend().profile(tmp_path)
    lines = [line.strip() for line in profile.splitlines() if line.strip()]
    assert lines[0] == "(version 1)"
    assert lines.index("(allow default)") < lines.index("(deny network*)"), (
        "the denials come after the default allow, because the last match wins")
    assert "(deny network*)" in lines
    assert "(deny file-write*)" in lines
    assert f'(allow file-write* (subpath "{tmp_path}"))' in lines
    assert lines.index("(deny file-write*)") < lines.index(
        f'(allow file-write* (subpath "{tmp_path}"))'), (
        "the workspace is re-allowed after the denial, or the denial wins and the "
        "command cannot write anything at all")


def test_the_seatbelt_profile_escapes_a_quote_in_the_path():
    escaped = sandbox.SeatbeltBackend._escape('/tmp/a "quoted" dir')
    assert escaped == '/tmp/a \\"quoted\\" dir', (
        "an unescaped quote would end the SBPL string and the rest of the path "
        "would be read as profile source")
    assert sandbox.SeatbeltBackend._escape("back\\slash") == "back\\\\slash"


def test_the_landlock_backend_applies_its_rules_in_the_child_not_the_parent(tmp_path: Path):
    """Landlock confines the calling process, so applying it in the parent
    would confine the harness. It must be a preexec hook."""
    invocation = sandbox.LandlockBackend().wrap(["echo", "hi"], tmp_path)
    assert invocation.argv == ["echo", "hi"], "the argv is untouched"
    assert callable(invocation.preexec), "the ruleset is applied between fork and exec"


def test_landlock_only_asks_the_kernel_for_bits_the_kernel_has():
    """A bit the running kernel does not know about is EINVAL, not a sandbox.
    The rights are therefore gated on the ABI version that was actually
    reported."""
    abi1 = sandbox._landlock_bits(1)
    abi2 = sandbox._landlock_bits(2)
    abi3 = sandbox._landlock_bits(3)
    abi5 = sandbox._landlock_bits(5)
    assert not abi1 & sandbox._LL_REFER, "REFER arrived in ABI 2"
    assert abi2 & sandbox._LL_REFER
    assert not abi2 & sandbox._LL_TRUNCATE, "TRUNCATE arrived in ABI 3"
    assert abi3 & sandbox._LL_TRUNCATE
    assert abi5 & sandbox._LL_IOCTL_DEV, "IOCTL_DEV arrived in ABI 5"
    for abi in (1, 2, 3, 4, 5, 9):
        bits = sandbox._landlock_bits(abi)
        assert bits & sandbox._RO_ACCESS == sandbox._RO_ACCESS, (
            "read and execute everywhere is the read-only bind of the whole filesystem")


# --- the one test that needs a real backend ---------------------------------


@pytest.mark.skipif(
    all(status.name != "landlock" or not status.available
        for status in sandbox.detect()),
    reason="no real backend on this machine: this test runs a command through a "
           "genuine sandbox, and this kernel offers none, so there is nothing "
           "to run it under")
def test_a_real_backend_runs_a_harmless_command(tmp_path: Path):
    """Deliberately the least interesting command there is.

    This verifies that a real backend's argv is accepted and its ruleset is
    actually applied -- not that the sandbox holds. A test suite that tries to
    escape a sandbox is a different program with a different risk profile, and
    a failure there tells you about the kernel, not about this plugin.
    """
    result = sandbox.sandbox_run(["echo", "harmless"], timeout=30, workspace=str(tmp_path))
    assert result["success"] is True
    assert result["metadata"]["backend"] != "none", (
        "a real backend was available, so none was not the answer")
    assert result["metadata"]["confinement_applied"] is True
    assert "harmless" in result["output"]
