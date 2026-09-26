# Running the Seymour counterexample search

```bash
./research/snc.sh
```

The topic is in `scripts/snc_topic.txt`, not on the command line. That is not
stylistic: a double-quoted topic has every `$D`, `$v` and `$A` expanded away by
the shell before the harness ever sees it, and `Seymour's` makes single-quoting
impossible. `"$(cat scripts/snc_topic.txt)"` survives both.

## The claim ladder

Attacking an open problem is the point, so the topic is written as a **ladder**
rather than as one claim. A claim with `kind: "conjecture"` states the problem
itself: it may have no exact expression, it is carried as the target, and it is
**never adjudicated or marked proven**. Beneath it sit the decidable rungs the
run can actually settle — exhaustive small cases, an exact verifier for a
proposed counterexample, boundary cases, a Lean formalisation.

So the run reports *progress on the sub-claims* and states the gap explicitly.
The progress report has a "What remains open" section naming the conjecture and
any unsettled rung. A run that proves a small-case lemma while saying the
conjecture is unsolved is the correct outcome, not a disappointing one.

## What counts as a finding

A candidate counterexample counts only if `|N+(v)|` and `|N++(v)|` are
**recomputed from the adjacency matrix by exact code**, not asserted. An
unverifiable candidate is recorded as a failed candidate, not a partial result.

Expect `UNSOLVED` with a progress report. That is the honest outcome, and the
report names which invariant blocked it and what remains open.

## Reading the output

- `research/snc-counterexample-search-for-seymours-second-neighborhood-conjecture/`
- `proof_receipts.json` — the exact bytes executed, with SHA-256
- `task_board.json` — every task, its owner, and what became of it
- `comm_ledger.jsonl` — hash-chained; verifies its own integrity

## Cost

24 cycles x 8 workers x 4 concurrent is substantial. Start lower to see whether
it is making progress before spending the rest: `--max-cycles 4 --parallel-workers 2`.
