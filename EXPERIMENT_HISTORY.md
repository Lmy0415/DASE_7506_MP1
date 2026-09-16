# Experiment-history and cost disclosure

This file records work that preceded the clean formal protocol in
`EXPERIMENT_PLAN.md`.  It is retained so that unsuccessful trials and historical
test access are not hidden by the final rerun.

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

## Historical test access

Before the formal protocol was preregistered, five standalone test evaluations
were run on discarded exploratory predictors:

| Predictor | Test BPB | Scoring seconds |
|---|---:|---:|
| continuous-cache 128x4 | 2.010090 | 11.674 |
| gate-off 128x4 | 2.101265 | 8.580 |
| gate-on 128x4 | 2.091555 | 8.921 |
| 320x8 plus cache, seed 17 | 1.622126 | 37.244 |
| 192x6 plus cache, seed 17 | 1.655538 | 17.582 |

This exposure cannot be undone by extracting another copy of the starter.  The
formal mitigation is narrower: the formal development directory physically
omits the test text; B0, B1, P0, seed 7506, the 3,000-step stopping point, the
4,800-step cosine horizon, and cache settings 255/13/0.065 were fixed before
formal validation; and no formal choice is changed after the frozen predictor's
single final-test run.  The report must not claim that the public test set was
historically unseen.

## Formal costs

Formal job costs are recorded separately in `code/FORMAL_RUNS.csv` and the
immutable JSON artifacts. The one post-freeze formal test obtained
1.6160026140038766 BPB in 41.520049 scorer seconds and 50.96 external elapsed
seconds. No method choice was changed afterward. The final report combines the
formal and exploratory ledgers while keeping unlike timing scopes explicit.
