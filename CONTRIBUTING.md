# Contributing

Thanks for helping. This project is a working agent harness, so the bar for a
change is that it makes the harness more trustworthy or more useful without
making it harder to reason about.

## Getting set up

```bash
git clone https://github.com/AbdalrahmanAlkam/harness.git
cd harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.12 or newer. Linux, macOS, or WSL.

## Running the tests

```bash
pytest -m "not slow"   # the development loop, about 90 seconds
pytest                 # the full gate, including real Lean 4 compilation
```

The `slow` marker covers tests that compile real Lean proofs or drive a full
research swarm. They are excluded from the quick run because they dominate the
wall clock, not because they are unreliable — please run the full suite before
opening a pull request.

Some tests need external tools and skip cleanly without them: Lean 4, Typst,
Node, and Bubblewrap. If one of those is installed, more of the suite runs.

## What we care about

**Never assert something that did not happen.** This is the project's central
invariant and the thing most worth protecting. A receipt must not say Lean
checked a file it did not check. A paper must not print a finding the run did
not decide. A verifier must not return success for a value it did not verify.
A verdict must not be `PROVEN` when an invariant failed. When you find a path
where the harness can lie, fix it and add a test that reproduces the lie — the
regression test is the point, not an afterthought.

**Fail closed, and say why.** An error the user can act on beats a silent
fallback. If a tool cannot do the right thing, it should say so rather than
substitute something plausible.

**Prefer the smallest correct change.** Match the surrounding style. The
codebase has a deliberately plain idiom; a clever abstraction that nobody can
follow is a regression even when it is shorter.

## Before you open a pull request

- The full test suite passes.
- New behaviour has a test that fails without your change.
- A bug fix includes a regression test reproducing the original failure.
- If you changed a user-visible string, command, or config key, the README and
  `CHANGELOG.md` are updated in the same commit.
- If you touched anything that could affect the research swarm's honesty
  (proofs, gates, the ledger, paper rendering), say so explicitly in the PR
  description, because those are the paths we review most carefully.

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, and the
harness version (`adaptive-harness --version`). If the problem is that the
harness reported something untrue, that is a security issue — see
[SECURITY.md](SECURITY.md) and report it privately.

## License

MIT. See [LICENSE](LICENSE).
