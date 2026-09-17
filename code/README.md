# Reproducing the MP1 submission

Run every command in this `code/` directory. The protocol has two phases:
development may read only train and validation data; the official test program
is run once after the submitted predictor is frozen.

This public submission intentionally omits the course-distributed raw text and
tokenizer. Before running the commands, copy `tokenizer.json`,
`wikitext_train.txt`, `wikitext_validation.txt`, and `wikitext_test.txt` from
the unchanged MP1 starter into `data/`. `data/manifest.json` records the hashes
that the loaders require.

Do **not** use `evaluate.py` during development. Its fixed loader opens every
benchmark split even when validation is requested. Use `train.py`'s validation
records or `score_validation.py`, whose loader has no test-split API.

## 1. Install and test

Python 3.12 and the pinned packages were used. On macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux, install the CPU PyTorch wheel first if desired:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

Run the contract and protocol tests:

```bash
python -m unittest discover -s tests -v
```

Expected result: `Ran 24 tests` followed by `OK`. These tests use synthetic or
development data; they do not run official test evaluation.

## 2. Formal training

Every command requires a new, empty `--run-dir`.

### B0: initial baseline

```bash
python train.py \
  --implementation model \
  --config configs/baseline.json \
  --seed 7506 \
  --steps 1200 --schedule-steps 1200 \
  --batch-size 32 \
  --save-steps 1200 --eval-steps 1200 \
  --device cpu --precision fp32 --threads 4 \
  --run-dir runs/formal-baseline-default-s7506
```

### B1: baseline with the proposed schedule horizon

```bash
python train.py \
  --implementation model \
  --config configs/baseline.json \
  --seed 7506 \
  --steps 1200 --schedule-steps 4800 \
  --batch-size 32 \
  --save-steps 1200 --eval-steps 1200 \
  --device cpu --precision fp32 --threads 4 \
  --run-dir runs/formal-baseline-matched-s7506
```

### P0: proposed 320x8 neural model

```bash
python train.py \
  --implementation student \
  --config configs/final_320x8_train.json \
  --seed 7506 \
  --steps 3000 --schedule-steps 4800 \
  --batch-size 32 \
  --save-steps 1200,3000 --eval-steps 1200,3000 \
  --device cpu --precision fp32 --threads 4 \
  --run-dir runs/formal-final-320x8-s7506
```

B0 and B1 each process 9,830,400 training targets. P0 has a model-only
snapshot at that same target count (step 1,200) and finishes with 24,576,000
targets (step 3,000). `--schedule-steps` fixes the learning-rate trajectory
independently of the actual stop.

At every requested save step the trainer writes both `checkpoint-step-N.pt`
(model-only inference state) and `checkpoint-step-N.resume.pt` (model,
optimizer, global step, target count, batch generator, Torch RNG and history).
To resume after a saved step, repeat every original training argument, select a
new empty output directory, and add:

```bash
--resume-from runs/<original-run>/checkpoint-step-N.resume.pt
```

The trainer rejects an incompatible recipe, and the resume path is tested for
bit-exact agreement on a stochastic synthetic run.

## 3. Validation-only scoring

The following scorer imports the numerical `score()` loop unchanged from
`evaluate.py`, but obtains data only from `dev_data.py`. It refuses to
overwrite evidence and emits both aggregate JSON and FP64 per-window NLL.

```bash
mkdir -p validation frozen artifacts

python score_validation.py \
  --checkpoint runs/formal-baseline-default-s7506/checkpoint-step-1200.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/B0-step1200.json

python score_validation.py \
  --checkpoint runs/formal-baseline-matched-s7506/checkpoint-step-1200.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/B1-step1200.json

python score_validation.py \
  --checkpoint runs/formal-final-320x8-s7506/checkpoint-step-1200.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/P0-step1200-cache-off.json

python score_validation.py \
  --checkpoint runs/formal-final-320x8-s7506/checkpoint-step-3000.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/P0-step3000-neural.json
```

Each JSON records checkpoint, canonical config, data, implementation, official
evaluator and validation-driver hashes. Training already produces the same
neural validation values; these commands provide uniform auditable artifacts.

## 4. Freeze the paired cache ablation

Freezing changes metadata only. Both artifacts below contain byte-identical
neural state dictionaries from the same step-3,000 checkpoint.

```bash
python freeze_cache_checkpoint.py \
  --source runs/formal-final-320x8-s7506/checkpoint-step-3000.pt \
  --output frozen/P0-step3000-cache-off.pt --mode off

python freeze_cache_checkpoint.py \
  --source runs/formal-final-320x8-s7506/checkpoint-step-3000.pt \
  --output frozen/P0-step3000-cache-on.pt --mode on

python score_validation.py \
  --checkpoint frozen/P0-step3000-cache-off.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/P0-step3000-cache-off.json

python score_validation.py \
  --checkpoint frozen/P0-step3000-cache-on.pt \
  --device cpu --precision fp32 --threads 4 \
  --output validation/P0-step3000-cache-on.json
```

The frozen cache is `window=255`, `theta=13`, `lambda=0.065`. It may use only
an earlier hidden state and the successor token already observed by the current
query position. It has no persistent state and resets on every scorer call.

## 5. Paired resource measurement on validation

The resource comparison uses the same validation workload, machine, four
threads and FP32 for B0 and the frozen final predictor. On macOS:

```bash
/usr/bin/time -l python score_validation.py \
  --checkpoint runs/formal-baseline-default-s7506/checkpoint-step-1200.pt \
  --device cpu --precision fp32 --threads 4 \
  --output artifacts/B0-validation-resource.json \
  2> artifacts/B0-validation-resource.txt

/usr/bin/time -l python score_validation.py \
  --checkpoint frozen/P0-step3000-cache-on.pt \
  --device cpu --precision fp32 --threads 4 \
  --output artifacts/P0-validation-resource.json \
  2> artifacts/P0-validation-resource.txt
```

On Linux substitute `/usr/bin/time -v`. The JSON `seconds` field times the
shared scoring loop; the external time log provides peak resident memory.

Measure the uncompressed inference assets actually required by the checkpoint:

```bash
python - <<'PY'
from pathlib import Path
files = [Path('frozen/P0-step3000-cache-on.pt'), Path('cache_model.py'), Path('model.py')]
size = sum(path.stat().st_size for path in files)
print({'files': [str(path) for path in files], 'bytes': size, 'MiB': size / 2**20})
PY
```

For GitHub web delivery, split the approximately 40.5 MiB checkpoint into
independently hashed parts smaller than GitHub's browser-upload limit:

```bash
python checkpoint_bundle.py create \
  --checkpoint frozen/P0-step3000-cache-on.pt \
  --output-dir ../checkpoint \
  --asset cache_model.py --asset model.py
```

This writes a manifest with part hashes, the complete checkpoint SHA-256 and
the exact uncompressed inference-asset total. A reviewer reconstructs it with:

```bash
python checkpoint_bundle.py restore \
  --bundle-dir ../checkpoint \
  --output ../checkpoint/P0-step3000-cache-on.pt
```

Restoration verifies every part and the full file before atomically exposing
the checkpoint. It does not retrain or alter model contents.

Recorded final measurements:

| Quantity | Result | Requirement |
|---|---:|---:|
| P0 validation score-loop time | 34.338784708001185 s | -- |
| B0 validation score-loop time | 8.209438499994576 s | -- |
| P0/B0 score-loop ratio | 4.182842067459775x | at most `5x` |
| P0 peak RSS | 1.5173797607421875 GiB | at most `4 GiB` |
| Inference assets | 40.478294372558594 MiB | at most `64 MiB` |

The `5x` requirement is evaluated using the common scorer's score-loop
seconds. For completeness, the paired external elapsed times were 42.88 s for
P0 and 16.55 s for B0, giving an end-to-end ratio of
2.5909365558912385x.

## 6. Freeze declaration and single official test

Before the command below, the repository records:

```text
FINAL_CHECKPOINT=frozen/P0-step3000-cache-on.pt
FINAL_CHECKPOINT_SHA256=25e2c6adcdc312726317e6abe06df6f7035bde5675ae70d5eb7f51978db71152
FROZEN_AT=2026-09-16T18:06:24Z
FORMAL_SELECTION_CHANGED_AFTER_THIS_POINT=false
```

Only after that declaration, run the fixed official full-test scorer once:

```bash
/usr/bin/time -l python evaluate.py \
  --checkpoint frozen/P0-step3000-cache-on.pt \
  --device cpu --precision fp32 --threads 4 \
  --split test \
  --output artifacts/P0-final-test.json \
  2> artifacts/P0-final-test-resource.txt
```

On Linux use `/usr/bin/time -v`. Do not test B0, B1, the step-1,200 P0 model,
or cache-off: their roles are established on validation. The submission BPB is
`1.6160026140038766` from the complete final-test JSON. That single frozen-
predictor run used 41.520049 score-loop seconds, 50.96 external elapsed seconds
and 1.550232 GiB peak RSS.

## 7. Results and provenance

| Predictor | Updates / LR horizon | Targets | Cache | Validation BPB | Test BPB |
|---|---:|---:|---|---:|---:|
| B0 | 1,200 / 1,200 | 9,830,400 | off | 2.075891 | not run |
| B1 | 1,200 / 4,800 | 9,830,400 | off | 1.997825 | not run |
| P0 same-target | 1,200 / 4,800 | 9,830,400 | off | 1.780348 | not run |
| P0 longer | 3,000 / 4,800 | 24,576,000 | off | 1.634623765423197 | not run |
| **P0 submitted** | **3,000 / 4,800** | **24,576,000** | **255/13/0.065** | **1.598754227700291** | **1.6160026140038766** |

The definitive formal ledger is `FORMAL_RUNS.csv`. Discarded exploratory runs
are retained for provenance and cost accounting in `../EXPERIMENT_HISTORY.md`
and `EXPLORATORY_RUNS.csv`, but are excluded from formal evidence. Preserve
every JSON, `.window-nll.npy`, resource log, checkpoint hash, exact source
revision and package version with the submission.

## 8. Student-led work, AI assistance and data attribution

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

All submitted training and model selection used the supplied train and
validation splits only. The formal-development tree excluded the test file, and
the complete method and checkpoint were frozen and hash-recorded before one
final full-test evaluation; no choice changed afterward. An earlier discarded
exploratory workspace had accessed the public test split, but those runs are
not used as formal evidence.

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher in *Pointer Sentinel Mixture Models* (ICLR 2017). The text is by
Wikipedia contributors. Hashes and upstream licensing notices are retained in
`data/manifest.json` and the supplied guide. No external training text or
pretrained weights are used.
