"""The core ships with no plugins, and must stay that way.

An install of the harness is a working agent and nothing else. Every extra
capability is something the user chooses, because a toolbelt nobody chose and
cannot find is the most common way an agent product feels cluttered.

These tests pin the guarantee rather than the intention. They fail if a plugin
is added back into the package, if a build starts shipping plugin data, or if
the harness stops working with an empty plugin directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_harness.plugins.host import PluginHost
from adaptive_harness.plugins.registry import OFFICIAL, Registry, InstallError

REPO = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "adaptive_harness.cli", *args],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "TERM": "dumb"})


# --- nothing is bundled -----------------------------------------------------


def test_the_package_ships_no_plugin_directory():
    """A plugin inside the package is a plugin every user has and did not ask
    for. The official ones live beside the harness, not inside it."""
    packaged = REPO / "src" / "adaptive_harness" / "plugins" / "bundled"
    assert not packaged.exists(), (
        f"plugins were moved back into the package: {sorted(p.name for p in packaged.iterdir())}")


def test_the_package_ships_no_plugin_data():
    """Even as data rather than modules, shipping them would install them."""
    content = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "bundled" not in content, (
        "pyproject still packages the bundled plugin directory")


def test_a_fresh_install_has_no_plugins(tmp_path: Path):
    """The guarantee as a user meets it."""
    result = _run("plugin", "list")
    assert result.returncode == 0
    assert "No plugins found" in result.stdout or "0 found" in result.stdout


# --- the core works with none ----------------------------------------------


def test_the_host_is_happy_with_nothing_installed(tmp_path: Path):
    host = PluginHost(project_root=tmp_path)
    assert host.discover() == []
    assert host.build_tools(tmp_path) == []
    assert host.skills() == {}
    assert host.settings() == []
    assert host.context_fragments() == []
    assert host.load_errors == []


def test_a_run_works_with_no_plugins(tmp_path: Path):
    result = _run("dev", "say hi", "--offline", "--workspace", str(tmp_path),
                  "--db", str(tmp_path / "e.db"))
    assert result.returncode == 0, f"a bare install could not run a task: {result.stderr[-400:]}"


# --- installing is how capability arrives ---------------------------------


def test_every_official_plugin_validates(tmp_path: Path):
    """Nothing is offered that would not install."""
    from plugin_paths import plugin_path

    from adaptive_harness.plugins.lifecycle import validate

    for item in OFFICIAL:
        directory = plugin_path(item["name"])
        report = validate(directory)
        assert report.ok, f"{item['name']} would not install:\n{report.render()}"


def test_an_official_plugin_declares_what_it_does(tmp_path: Path):
    for item in OFFICIAL:
        assert item["summary"].endswith("."), f"{item['name']}: summary is not a sentence"
        assert item["why"], f"{item['name']} has no reason for existing"
        assert item["min_harness"], f"{item['name']} declares no minimum harness version"


def test_install_then_uninstall_is_a_clean_round_trip(tmp_path: Path):
    from plugin_paths import plugin_path

    registry = Registry(directory=tmp_path / "plugins")
    assert registry.list() == []

    installed = registry.install(plugin_path("todo-scan"))
    assert installed.name == "todo-scan"
    assert registry.is_installed("todo-scan")

    loaded = [item for item in registry.list() if item.name == "todo-scan"]
    assert loaded and loaded[0].valid, loaded[0].error if loaded else "not listed"

    assert registry.uninstall("todo-scan")
    assert not registry.is_installed("todo-scan")
    assert registry.list() == []


def test_installing_twice_is_refused_rather_than_silently_replacing(tmp_path: Path):
    from plugin_paths import plugin_path

    registry = Registry(directory=tmp_path / "plugins")
    registry.install(plugin_path("todo-scan"))
    with pytest.raises(InstallError):
        registry.install(plugin_path("todo-scan"))
    # And explicitly allowed, so updating is one flag rather than a deletion.
    registry.install(plugin_path("todo-scan"), overwrite=True)


def test_an_invalid_plugin_is_not_installed(tmp_path: Path):
    """A plugin that would not load must not reach the plugin directory at all."""
    broken = tmp_path / "broken"
    broken.mkdir(parents=True)
    (broken / "broken.plugin.json").write_text(json.dumps({
        "name": "broken", "version": "1.0", "permissions": ["tools"],
        "tools": [{"name": "t", "description": "x", "handler": "not_defined",
                   "risk": "read", "parameters": {}}],
    }), encoding="utf-8")
    (broken / "plugin.py").write_text("def other():\n    return {}\n", encoding="utf-8")

    registry = Registry(directory=tmp_path / "plugins")
    with pytest.raises(InstallError):
        registry.install(broken)
    assert not registry.is_installed("broken")


def test_an_uninstallable_name_is_reported_rather_than_silently_succeeding(tmp_path: Path):
    assert Registry(directory=tmp_path / "plugins").uninstall("never-installed") is False


# --- no privileged tier ----------------------------------------------------


def test_an_official_plugin_is_not_preferred_over_a_community_one(tmp_path: Path):
    """Trust is earned by what a plugin does, not by who wrote it.

    A plugin that deserves more trust belongs in the core. One that does not is
    whatever it says on its tin, whoever wrote it.
    """
    from adaptive_harness.plugins.host import discovery_roots_for

    roots = discovery_roots_for(tmp_path, allow_project_plugins=False)
    assert all("official" not in str(root).lower() for root in roots), (
        "official plugins are getting a privileged directory community ones do not")
