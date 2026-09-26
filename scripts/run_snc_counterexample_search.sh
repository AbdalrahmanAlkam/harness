#!/usr/bin/env bash
# Counterexample search for Seymour's Second Neighborhood Conjecture.
#
# The topic lives in a file, not in the command line. Two reasons, both learned
# the hard way: a double-quoted topic has every $D, $v and $A eaten by the shell,
# and "Seymour's" makes single-quoting impossible. "$(cat ...)" survives both.
#
# HONEST EXPECTATION: this conjecture is open. Nothing here can prove or disprove
# it. What the system can do is search candidate digraphs and MACHINE-VERIFY each
# one by recomputing every neighbourhood count from the adjacency matrix. A
# reported counterexample is a real finding only if that verification passes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC="$(cat "$HERE/snc_topic.txt")"
exec adaptive-harness research "$TOPIC" \
  --root research \
  --max-cycles 24 \
  --absolute-ceiling 24 \
  --patience 4 \
  --worker-steps 40 \
  --max-workers 8 \
  --parallel-workers 4 \
  --worker-timeout 3600 \
  --task-lease 5400 \
  --task-attempts 6 \
  --overseer-stop 5
