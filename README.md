# DASE7506 MP1: Small Language Model Challenge

**LI Maoyuan · UID 3035947946**

This repository contains my individual MP1 submission.  It trains a causal
language model from random initialization on the supplied WikiText-2 training
text and reports bits per UTF-8 byte (BPB; lower is better) under the fixed
BPE-2048, 256-token-window protocol.

The submitted predictor combines an eight-block, width-320 GPT with a strictly
causal continuous cache confined to the current evaluation window.  The cache
does not change the trained weights and is reset for every independent window.

The public repository does not redistribute the course benchmark text or
tokenizer. Copy the unchanged `tokenizer.json` and `wikitext_*.txt` files from
the supplied MP1 starter into `code/data/` before training or evaluation; their
required SHA-256 values remain recorded in `code/data/manifest.json`.

## Headline results

| Run | Predictor | Training targets | Validation BPB | Official test BPB | Parameters |
|---|---|---:|---:|---:|---:|
| B0 | 128x4 baseline, default LR horizon | 9,830,400 | 2.075891 | not run | 1,088,256 |
| B1 | 128x4 baseline, matched LR horizon | 9,830,400 | 1.997825 | not run | 1,088,256 |
| P0 @ 1,200 | 320x8, cache off | 9,830,400 | 1.780348 | not run | 10,601,600 |
| P0 @ 3,000 | 320x8, cache off | 24,576,000 | 1.634624 | not run | 10,601,600 |
| **Submitted P0** | **320x8, cache on** | **24,576,000** | **1.598754** | **1.616003** | **10,601,600** |

Final checkpoint SHA-256:
`25e2c6adcdc312726317e6abe06df6f7035bde5675ae70d5eb7f51978db71152`.
The immutable release commit is linked from the submission Issue; the
pre-test method boundary is recorded in `FINAL_FREEZE.json`.

No external text or pretrained weights are used.  The supplied data, tokenizer,
causal window construction and official evaluator are unchanged.

## What each comparison establishes

- **B0** is the initial classroom baseline: 1,200 updates along a 1,200-step
  cosine learning-rate horizon.
- **B0 versus B1** isolates the schedule-horizon effect at the same architecture
  and 9,830,400 processed targets.
- **B1 versus P0 @ 1,200** is the required same-target architecture comparison:
  it matches seed, sampled batches, optimizer, schedule horizon and targets.
- **P0 @ 1,200 versus P0 @ 3,000** describes the contribution of additional
  training along the predeclared trajectory.
- **P0 @ 3,000 cache off versus cache on** is a paired mechanism ablation with
  identical neural weights and training cost.

All formal runs use seed `7506`, batch size `32`, context length `256`, FP32 and
four CPU threads.  This is a controlled single-seed study; it does not estimate
variance across random initializations.

## Repository map

- [`REPORT.pdf`](REPORT.pdf) and [`REPORT.tex`](REPORT.tex): ten-page final
  report and its standard LaTeX source.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): predeclared formal protocol.
- [`EXPERIMENT_HISTORY.md`](EXPERIMENT_HISTORY.md): exploratory cost and
  provenance record, including how discarded runs relate to formal evidence.
- [`SOURCE_INTEGRITY.md`](SOURCE_INTEGRITY.md): clean-copy provenance.
- [`code/README.md`](code/README.md): exact install, training, validation,
  freezing, resource-measurement and final-test commands.
- [`code/FORMAL_RUNS.csv`](code/FORMAL_RUNS.csv) and
  [`code/EXPLORATORY_RUNS.csv`](code/EXPLORATORY_RUNS.csv): cost ledgers.
- [`checkpoint/`](checkpoint/): frozen checkpoint bundle and hashes.

## Resource compliance

Paired measurements use the same machine, validation corpus, FP32 precision and
four threads.  The `5x` check uses the evaluator's shared score-loop timing. As
a secondary end-to-end observation, the corresponding external elapsed times
were 42.88 s for P0 and 16.55 s for B0, a ratio of 2.590937x. The official
full-test result is 1.6160026140038766 BPB (41.520049 scorer seconds; 50.96 s
external elapsed; 1.550232 GiB peak RSS).

| Quantity | Recorded result | Limit |
|---|---:|---:|
| Proposed validation score-loop time | 34.338785 s | -- |
| Paired B0 validation score-loop time | 8.209438 s | -- |
| Proposed / B0 score-loop ratio | 4.182842x | at most `5x` |
| Proposed peak resident memory | 1.517380 GiB | at most `4 GiB` |
| Uncompressed inference assets | 40.478294 MiB | at most `64 MiB` |

Aggregate JSON and resource logs needed to audit the reported score are
retained under `code/artifacts/`.  Per-window arrays remain in the private
experiment archive and are not required to run the supplied evaluator.

## Student-led work with AI assistance

This project was directed and evaluated by LI Maoyuan. I defined the research
question, made the final decisions on architecture, training protocol and
experimental controls, ran and interpreted the experiments, verified the
implementation against the assignment requirements, and reviewed every claim
in the final report.

OpenAI Codex/ChatGPT was used as an auxiliary coding and discussion tool. It
helped explain the baseline and BPB metric, survey relevant published methods,
suggest candidate experiments, review code and tests, diagnose implementation
issues, organize experimental records, and edit documentation. AI suggestions
were not treated as experimental evidence and did not replace my judgement or
verification. Every reported result is supported by commands and artifacts
that I checked, and I remain responsible for the submitted work.

Tool and period: OpenAI Codex/ChatGPT, September 2026.

## Formal data separation and holdout protocol

All submitted training runs use only the supplied training split, while
architecture, training schedule, stopping point, cache configuration and
checkpoint selection are based on validation results. The formal-development
tree physically excludes the test file, and its training and validation
programs provide no test-split option.

Before final evaluation, the complete method, checkpoint, configuration and
implementation were frozen and hash-recorded. The frozen predictor was then
evaluated once on the full test split solely to obtain the reported final BPB,
and no model or configuration choice was changed in response to that result.

An earlier discarded exploratory workspace had accessed the public test split.
Those runs are not used as formal experimental evidence. The defensible holdout
claim is therefore that the complete formal rerun and all submitted development
comparisons were validation-only, and that the submitted predictor was frozen
before its final test evaluation—not that the test split was historically
unseen throughout the entire project.
