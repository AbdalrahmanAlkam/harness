# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's "Report a vulnerability" button on the Security tab of this
repository, which opens a private advisory to the maintainers.

Please include: what you found, the version or commit, the steps to reproduce,
and the impact you believe it has. We aim to acknowledge a report within three
business days.

## What this tool does, and what that means for your machine

Adaptive Agent Harness runs a language model that can execute shell commands,
read and write files, and run code on your machine. **That is the product.** A
user who grants it access to a repository is granting an LLM the ability to
run commands with their own privileges.

Before you run it, understand these properties:

- **`run_bash` is not sandboxed.** It executes shell commands with your full
  privileges, full filesystem access, and full network access. There is no
  container, no seccomp, no user namespace. Treat every command it runs as a
  command you ran yourself.
- **The model can read your environment.** Child processes inherit your
  environment, which may contain API keys. Do not point the harness at a
  directory containing credentials you would not paste into a chat window.
- **It runs on repositories you do not fully control.** Content in a file, a
  README, a web page, or a search result can contain instructions addressed to
  the model. The harness has prompt-injection defences (a safe expression
  evaluator, workspace containment on file tools, destructive-action gating,
  a risk classifier), but no harness can fully eliminate this class of attack.
  Review what the agent is about to do, and use the `cautious` or `strict`
  safety profile when you want confirmation before anything irreversible.
- **Verified mathematics means verified by this harness, not by a human.** The
  research swarm checks its own results with executed derivations, SymPy, and
  Lean 4. A `PROVEN` verdict is a machine-checked result against the claims
  the run actually made. It is not a peer review.

## What is enforced

- All file tools resolve paths and refuse anything outside the workspace root,
  including via symlinks.
- The Lean prover is contained to the workspace, and will not certify a file
  that declares no theorem.
- The science sandbox (`run_python_repl`) runs under Bubblewrap with networking
  unshared and no workspace mount, and fails closed when `bwrap` is absent.
- Swarm workers are confined to the artifact paths they have leased; a worker
  cannot write to a sibling's file through either the file tools or the shell.
- Credentials and config are written atomically at mode `0600`, and symlinked
  config files are refused.
- API keys are redacted from error messages before they are shown or logged.
- Typst is never installed without an explicit `--install-typst`, and is
  version-pinned when it is.

## What is not enforced

- The `run_bash` blocklist is a guard against accidents, not a sandbox. It is
  not a security boundary and should not be relied on as one.
- The research swarm writes to `research/<topic>/` and `output/`. It does not
  reach outside the workspace, but it does create files.
- There is no telemetry. Nothing is sent anywhere except the model provider you
  configure, and optionally a search API if you supply its key.

## Supported versions

The `main` branch and the most recent release are supported. Please upgrade
before reporting an issue against an older commit.
