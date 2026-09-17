# Exploratory experiment history and cost record

This file records work that preceded the clean formal protocol in
`EXPERIMENT_PLAN.md`. It is retained to account for exploratory cost and to make
clear that discarded runs are not part of the submitted formal evidence.

## Recorded exploratory cost

The detailed rows are in `code/EXPLORATORY_RUNS.csv`.

- 14 exploratory training runs have surviving `metrics.json` evidence.
- Their summed training time is 38,326.481 seconds (10.64625 job-hours).
- Their summed whole-process time is 39,908.333 seconds (11.08565 job-hours).
- They processed 157,532,160 next-token training targets in total.
- 10 standalone evaluations have surviving JSON timing evidence and sum to
  157.053 seconds (0.04363 job-hours).
- Therefore the evidenced exploratory total is at least 40,065.386 job-seconds
  (11.12927 job-hours).  This is a sum across jobs, not elapsed wall-clock time;
  some jobs may have overlapped.

Six cache searches also survive as result grids: 42 + 168 + 56 + 49 + 30 +
28 = 373 tried cache configurations.  Their elapsed time was not recorded and
is therefore reported as unknown rather than zero.  Failed or interrupted jobs
without artifacts likewise cannot be reconstructed completely.

## Relationship to the formal evidence

All submitted training and model selection used the supplied train and
validation splits only. The formal-development tree excluded the test file, and
the complete method and checkpoint were frozen and hash-recorded before one
final full-test evaluation; no choice changed afterward.

An earlier discarded exploratory workspace had accessed the public test split,
but those runs are not used as formal evidence. The raw provenance rows remain
in `code/EXPLORATORY_RUNS.csv`; they are retained for cost accounting rather
than for method selection, comparison or any reported formal conclusion.

## Formal costs

Formal job costs are recorded separately in `code/FORMAL_RUNS.csv` and the
immutable JSON artifacts. The one post-freeze formal test obtained
1.6160026140038766 BPB in 41.520049 scorer seconds and 50.96 external elapsed
seconds. No method choice was changed afterward. The final report combines the
formal and exploratory ledgers while keeping unlike timing scopes explicit.
