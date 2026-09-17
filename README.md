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
  historical test-access disclosure.
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

## AI-use disclosure

I used OpenAI Codex/ChatGPT as an AI coding and discussion assistant.  It helped
explain the baseline architecture and BPB, search and summarize published
methods, scaffold and review the resumable trainer and tests, orchestrate
commands, diagnose implementation issues, and draft documentation.  I chose
the research question and formal protocol, reviewed the generated code and
explanations, interpreted the experiments, and remain responsible for the
submitted implementation, citations, claims and results.  AI suggestions were
not treated as experimental evidence; every reported number is tied to a
recorded command and artifact in this repository.

Tool and period: OpenAI Codex/ChatGPT, September 2026.

## Historical test-exposure disclosure

Before the clean formal protocol was established, five test evaluations were
run during exploratory AI-assisted work on discarded gate, cache and scaled-GPT
predictors.  Their BPB values were 2.010090, 2.101265, 2.091555, 1.622126 and
1.655538.  This was a protocol mistake, so I do not claim that the public test
text was historically unseen to the project.

Extracting a new starter copy cannot erase information already observed.  The
mitigation was therefore prospective: I created a development copy with the
test text physically absent, predeclared B0/B1/P0 and all cache constants, used
only validation-capable development code, and made no formal choice in response
to the frozen predictor's single final-test result.  The limitation remains
explicit in the report and experiment history.
