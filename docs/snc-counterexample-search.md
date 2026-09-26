# Running the Seymour counterexample search

```bash
./research/snc.sh
```

The topic is in `scripts/snc_topic.txt`, not on the command line. That is not
stylistic: a double-quoted topic has every `$D`, `$v` and `$A` expanded away by
the shell before the harness ever sees it, and `Seymour's` makes single-quoting
impossible. `"$(cat scripts/snc_topic.txt)"` survives both.

## What this can and cannot do

Seymour's Second Neighborhood Conjecture is **open**. This system cannot prove
or disprove it, and a run that reports success has not settled the conjecture.

What it can do is real: generate candidate 2-cycle-free digraphs and
**machine-verify** each one by recomputing `|N+(v)|` and `|N++(v)|` for every
vertex from the adjacency matrix. A reported counterexample is a finding only if
that recomputation passes. An unverifiable candidate is recorded as a failed
candidate, not as a partial result.

Expect `UNSOLVED` with a progress report. That is the honest outcome, and the
report names which invariant blocked it.

## Reading the output

- `research/snc-counterexample-search-for-seymours-second-neighborhood-conjecture/`
- `proof_receipts.json` — the exact bytes executed, with SHA-256
- `task_board.json` — every task, its owner, and what became of it
- `comm_ledger.jsonl` — hash-chained; verifies its own integrity

## Cost

24 cycles x 8 workers x 4 concurrent is substantial. Start lower to see whether
it is making progress before spending the rest: `--max-cycles 4 --parallel-workers 2`.
