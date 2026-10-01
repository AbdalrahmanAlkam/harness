# Plugins

A plugin adds a capability to the harness without editing the harness. It can
contribute:

- a **tool** the model can call,
- a **skill** with its own instructions and verification checks,
- a **setting** that persists and appears on the settings screen,
- a **prompt override**,
- a **slash command**,
- a **hook** that observes the agent's event stream.

Anything most users will not use belongs here rather than in
`src/adaptive_harness`, which is what keeps the core small and lets the tool
surface stay tight.

## The trust model

**A plugin is third-party code that runs with your privileges.** Read what it
asks for before you install it:

```bash
adaptive-harness plugins
```

```
Plugins (1 found)
  ● todo-scan 1.0.0 — Find TODO/FIXME markers in the workspace.
    permissions: tools
    provides: 1 tool(s), scan_markers
```

A manifest declares the permissions it wants. The rule is that **anything not
declared is refused, not merely unused**:

| Permission | Grants |
| :--- | :--- |
| `tools` | register a callable tool |
| `skills` | register a skill |
| `settings` | contribute a persistent setting |
| `prompts` | override a prompt |
| `commands` | add a slash command |
| `hooks` | observe the agent event stream |
| `net` | make network requests |
| `subprocess` | execute external programs |
| `fs_home` | read or write outside the workspace |
| `env` | read the process environment |

Beyond the declared grant, three things are enforced regardless:

- **Path arguments are contained.** A `path`, `file_path`, `directory`, `target`
  or `filename` argument is resolved against the workspace and refused if it
  escapes, by symlink or by `..`.
- **A declared risk drives the safety gate.** A tool marked `read` is treated
  like `read_file` by the strict profile; anything else is assumed to mutate.
  A plugin cannot reach execution by calling itself something other than
  `run_bash`.
- **A broken plugin cannot stop the harness.** A manifest that will not parse, a
  module that will not import, a tool with no handler — each is reported by name
  and skipped, and the rest still load.

`subprocess`, `fs_home` and `env` are declared but not yet enforced by a
sandbox. A plugin holding `subprocess` can do what any Python program can. That
is the honest limit, which is why the permission is listed rather than assumed.

## The core ships with none

An install of the harness is a working agent and nothing else. Every extra
capability is something you choose, because a toolbelt nobody chose and cannot
find is the most common way an agent product feels cluttered.

```bash
adaptive-harness plugin available   # what there is, and why each exists
adaptive-harness plugin install web  # add one
adaptive-harness plugin list        # what you have
adaptive-harness plugin uninstall web
```

`install` takes an official name or any path, validates before copying, and
never overwrites without `--overwrite`. It copies rather than references, so a
plugin installed from a git checkout keeps working when the checkout moves.

**There is no privileged tier.** An official plugin installs into the same
directory as a community one and is trusted exactly as much, because a plugin
that deserves more trust belongs in the core and one that does not is whatever
it says on its tin.

## Layout

A plugin is a directory containing a manifest and, if it needs any code, a
Python file. The manifest may be named `plugin.plugin.json` or after the plugin
itself (`web.plugin.json`); any `*.plugin.json` is read.

```
~/.config/adaptive-harness/plugins/
└── my-plugin/
    ├── plugin.plugin.json
    └── plugin.py
```

Two roots are searched, in increasing precedence:

1. `~/.config/adaptive-harness/plugins/` — installed plugins, ours and the
   community's alike.
2. `<project>/.harness/plugins/` — a repository's own, opt-in, because a
   repository you merely cloned must not be able to run code.

**Root 3 is off unless you turn it on.** This is a coding agent that people
point at repositories they did not write, so a repository you have merely
cloned must not be able to execute anything by being the working directory. Opt
in per run:

```bash
adaptive-harness plugins --with-project-plugins
```

or set `allow_project_plugins` in the config. When project plugins are found
but not loaded, the count is reported rather than staying silent — a user who
installed one should not be left wondering why it is not running. Roots 1 and
2 need no gate: the first ships inside the package you installed, the second is
code you put there yourself.

## A complete example

`plugin.plugin.json`:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "What it does, in one line.",
  "author": "you@example.com",
  "permissions": ["tools"],
  "tools": [
    {
      "name": "do_a_thing",
      "description": "The model reads this to decide when to call it.",
      "handler": "do_a_thing",
      "risk": "read",
      "domains": ["coding"],
      "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  ]
}
```

`plugin.py`:

```python
def do_a_thing(path):
    from pathlib import Path
    target = Path(path)
    if not target.is_file():
        return {"success": False, "error": f"No such file: {path}"}
    return {"success": True, "output": target.read_text()[:2000]}
```

`name` must be a valid Python identifier and must not shadow a built-in tool.
`risk` is required and must be one of `read`, `write`, `exec`, `net` — it has no
default because it decides whether the strict safety profile asks first.

A handler returns a `ToolResult`, a dict with `success`/`output`/`error`, or a
plain string. Raising is fine: the error is caught, reported with the plugin's
name, and returned to the model as a failed tool call rather than crashing the
run.

## Adding more than a tool

A **skill** is a factory returning a `BaseSkill`:

```json
"permissions": ["skills"],
"skills": [{"name": "data_auditor", "factory": "build_skill"}]
```

```python
from adaptive_harness.skills.registry import BaseSkill

def build_skill(source="plugin"):
    return BaseSkill(
        name="data_auditor", title="Data Auditor", category="Data",
        trigger="profile a dataset, check a data file",
        instructions="Profile the data before analysing it.",
        tools=("do_a_thing",), invariants=("evidence_checked",),
        source=source)
```

`invariants` may only name checks that already exist in `SkillVerifier`, so a
skill cannot invent a gate that is never evaluated.

A **setting** becomes a row on the settings screen:

```json
"permissions": ["settings"],
"settings": [{
  "key": "depth", "label": "Depth", "kind": "cycle",
  "choices": ["fast", "thorough"], "default": "fast"
}]
```

A **prompt override** and a **command** are declarative:

```json
"permissions": ["prompts", "commands"],
"prompts": {"system.default": "Extra instruction."},
"commands": {"/my-command": "Do the thing"}
```

A **hook** is a module-level function:

```python
def on_agent_event(event):
    if event.event_type == "tool_call":
        print(event.payload["name"])
```

Hooks run on every step, so they must be fast. One that raises is removed and
reported; it cannot break the run.

## Subagents

A subagent is a markdown file with a frontmatter block, under
`.harness/agents/` in a project or `~/.config/adaptive-harness/agents/` for your
own. The frontmatter keys match the ones Claude Code documents, so a definition
written for either works in both.

```markdown
---
name: db-reviewer
description: Reviews a migration for safety and reversibility.
tools: read_file, search_files, run_bash
permissionMode: plan
maxTurns: 12
isolation: worktree
background: false
---

You review database migrations. ...
```

Acted on here: `tools`, `disallowedTools`, `model`, `permissionMode`, `maxTurns`,
`effort`, `isolation`, `background`. Everything else is carried, and anything
carried is listed by `harness agent list` so a key this build ignores is
visible rather than silently dropped.

The definition is honoured, not merely read: its tool list becomes the agent's
surface, its system prompt is what the subagent is given, and
`permissionMode: plan` means the agent cannot write at all.

```bash
harness agent init db-reviewer --description "Reviews a migration."
harness agent list
harness agent show db-reviewer
```

`/tasks` shows both what is running and what you can spawn. The four built-in
roles are definitions too, and a project file may shadow one deliberately.

A tool this build does not provide is dropped and reported — in the result, in
the tool metadata, and on screen. A definition where *no* requested tool exists
is refused outright, because an agent that cannot do anything is not worth
starting.

### Background agents

`background: true`, or `wait: false` on `delegate_subagent`, returns an agent id
immediately and runs the subagent without holding your turn. Collect it later:

```python
tool.background_result(agent_id)
```

A background agent still registers normally, so `/tasks` sees it start *and*
finish. Asking for an id that is not running says so rather than hanging.

### Subagent hooks

`subagent_start` and `subagent_stop` are named after Claude Code's documented
`SubagentStart` / `SubagentStop`. A stop hook may return a string, which is fed
back to the subagent — that is what makes it a control rather than an observer:

```python
def check_it_finished_the_work(record):
    if record.status != "completed":
        return f"{record.id} ended as {record.status}. Say what went wrong."
    if not record.tools:
        return f"{record.id} did nothing. Do the task, or say why you cannot."
    return ""
```

A hook that raises is dropped and reported; it never fails the run it observes.

## Safety and limits

The plugin surface is a convenience, not a security boundary. It exists so that
capabilities which most users do not need do not have to live in the core. The
workspace containment and the safety gate apply to plugin tools exactly as they
apply to built-in ones, but a plugin with `subprocess` can do anything a Python
program can, and no manifest makes that safe.
