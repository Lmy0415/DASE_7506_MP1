# MP1 Formal Experiment Plan

**Status:** preregistered before formal training and final evaluation  
**Primary seed:** `7506`  
**Development rule:** all model assessment and comparison uses the validation split only  
**Final-test rule:** after the predictor is frozen, run exactly one full-test evaluation

## 1. Purpose and scope

This plan defines the formal, reduced-cost experiment used to support the MP1
submission. It replaces exploratory search with three prespecified training
runs and a paired cache ablation. The formal experiment uses one seed; it does
not make a multi-seed stability claim.

The final candidate is a 320-wide, eight-layer causal GPT with ten attention
heads. It uses the supplied WikiText-2 training text, fixed BPE-2048 tokenizer,
fixed evaluator, context length 256, and a causal continuous cache confined to
the current independent evaluation window.

No external training text, pretrained weights, validation/test answer cache,
future-token access, cross-window state, or evaluation-time network access is
permitted.

## 2. Frozen common settings

Unless a row below explicitly says otherwise, formal runs use:

- seed: `7506`
- batch size: `32`
- context length: `256`
- targets per update: `32 * 256 = 8,192`
- optimizer: supplied AdamW recipe
- peak learning rate: `0.001`
- learning-rate schedule: cosine
- minimum learning-rate ratio: `0.1`
- gradient clipping: supplied maximum norm `1.0`
- training data and batch sampling: supplied training split and sampler
- development scoring: FP32 validation scoring with the fixed evaluator
- ranked scoring: FP32 CPU scoring with the fixed evaluator

`Schedule horizon` means the total step count used to compute the learning-rate
trajectory. `Actual updates` means the number of optimizer updates performed.
Thus a run may stop before its schedule horizon while retaining the learning
rates defined by that longer horizon.

## 3. Prespecified training matrix

| Run ID | Model | Actual updates | Schedule horizon | Saved checkpoints | Processed targets used by checkpoint |
|---|---|---:|---:|---|---:|
| `B0-initial` | baseline 128x4, 4 heads | 1,200 | 1,200 | step 1,200 | 9,830,400 |
| `B1-matched` | baseline 128x4, 4 heads | 1,200 | 4,800 | step 1,200 | 9,830,400 |
| `P0-320x8` | proposed 320x8, 10 heads | 3,000 | 4,800 | steps 1,200 and 3,000 | 9,830,400 and 24,576,000 |

All three runs use seed `7506`. `B1-matched` and `P0-320x8` must use the same
batch-start sequence and the same learning-rate trajectory through step 1,200.
The only intended difference in their step-1,200 comparison is architecture.

The proposed run stops at step 3,000. It must not be extended or have its
checkpoint selected after inspecting formal validation results. Step 3,000 is
fixed in advance.

## 4. Required comparisons

### 4.1 Initial baseline

`B0-initial` establishes the supplied 1,200-step baseline using its normal
1,200-step horizon. Report its validation BPB, parameter count, processed
targets, training time, and validation time.

This run is the initial baseline result; it is not the matched architecture
control because its learning-rate horizon differs from the proposed run.

### 4.2 Same-target matched control

Compare:

- `B1-matched`, step 1,200; and
- `P0-320x8`, step 1,200.

Both checkpoints have processed exactly `9,830,400` training targets and share
seed, batches, optimizer settings, and schedule horizon. Report the absolute
validation BPB values and the signed difference

```text
delta_architecture = BPB(P0 step 1200) - BPB(B1 step 1200)
```

Lower values are better. Because this protocol has one seed, do not report a
standard deviation, confidence interval, or claim statistical stability.

### 4.3 Prespecified longer-training result

Evaluate `P0-320x8` at step 3,000 on validation. Compare it descriptively with
its own step-1,200 checkpoint:

```text
delta_training = BPB(P0 step 3000) - BPB(P0 step 1200)
```

This comparison measures the combined effect of additional processed targets
along the preregistered training trajectory. It is not an architecture
ablation.

## 5. Frozen cache ablation

The cache settings are fixed before formal validation:

- cache window: `255`
- cosine-similarity temperature: `13`
- interpolation weight: `0.065`

Do not run a new cache grid or adjust these values using formal validation.

Using the exact same `P0-320x8` step-3,000 neural checkpoint, evaluate on
validation:

1. **Cache off:** interpolation weight `0`.
2. **Cache on:** window `255`, temperature `13`, interpolation weight `0.065`.

Report:

```text
delta_cache = BPB(cache on) - BPB(cache off)
```

This is the formal mechanism ablation. It changes no trained neural weights and
adds no training targets. The cache may use only earlier token/value pairs in
the current 256-token evaluator window and must reset between examples,
windows, and scoring calls.

## 6. Validation-only development and freeze procedure

Before the final test:

1. Complete the three formal training runs.
2. Run the specified validation evaluations only.
3. Run contract tests for causality, normalized probabilities, example
   independence, gradient flow, and state reset.
4. Measure CPU scoring time, peak evaluation RAM, and uncompressed inference
   asset size for the final cache-on predictor.
5. Confirm the final predictor is within 5x paired local baseline CPU scoring
   time, 4 GiB peak evaluation RAM, and 64 MiB uncompressed inference assets.
6. Record the final code, implementation, evaluator, tokenizer, data, config,
   and checkpoint hashes.
7. Freeze the final predictor as `P0-320x8` step 3,000 with cache
   `255 / 13 / 0.065`.

Formal validation results are used to describe performance, not to change the
prespecified architecture, step, schedule, seed, or cache settings. If the
prespecified method fails, report the failure; do not silently replace it.

## 7. Single final-test rule

After completing the freeze checklist, run exactly one new full-test scoring
job for:

```text
P0-320x8 step 3000
seed 7506
cache window 255
theta 13
lambda 0.065
FP32 CPU
```

Do not run formal test evaluations for:

- either baseline checkpoint;
- the proposed step-1,200 checkpoint;
- the cache-off ablation;
- another seed, architecture, schedule, checkpoint, or cache setting.

The single final-test BPB is reported as one value. No test mean or test
standard deviation is reported.

## 8. Historical test-exposure disclosure

Before this formal protocol was created, exploratory work in other working
directories had already evaluated earlier gate and cache-based predictors on
the public test split. That exposure cannot be undone and must be disclosed in
the README and report.

The formal protocol does not claim that the public test split was historically
unseen. It claims only that, after this preregistration, no formal architecture,
training, checkpoint, or cache decision is changed using test results, and that
the single frozen formal predictor is newly scored on the full test once.

Suggested disclosure:

> Earlier exploratory work accessed the public test split. The formal protocol
> was subsequently preregistered and fixed all architecture, schedule,
> checkpoint, seed, and cache choices before formal validation and the single
> final-test evaluation. No formal choice was changed in response to a formal
> test result.

## 9. Formal cost and provenance record

Create one log row for every formal training, validation, resource-measurement,
and final-test job. Record at least:

- run ID and purpose (`baseline`, `matched control`, `proposed`, `ablation`,
  `resource check`, or `final test`)
- command line
- start and end timestamps with timezone
- exit status and whether the run completed
- seed
- model configuration and parameter count
- actual updates and schedule horizon
- batch size and processed training targets
- optimizer, schedule, peak LR, and minimum LR ratio
- device, device name, precision, and thread count
- preparation seconds
- training seconds
- validation/evaluation seconds
- total process seconds
- peak evaluation RSS
- uncompressed inference-asset bytes
- validation or test BPB, split name, target count, and UTF-8 byte count
- checkpoint ancestry and checkpoint step
- checkpoint SHA-256
- implementation SHA-256
- evaluator SHA-256
- tokenizer SHA-256
- data-manifest SHA-256
- configuration-file SHA-256
- notes for warnings, failures, reruns, or deviations

Report costs separately as:

1. formal training cost;
2. formal validation and resource-measurement cost;
3. cache-ablation cost;
4. single final-test cost;
5. prior exploratory/search cost, including unsuccessful work;
6. total project cost.

A rerun does not erase the cost of the earlier attempt. Any deviation from this
plan must be timestamped, justified without reference to a test result, and
retained in the repository.

## 10. Formal result table template

| Result | Seed | Actual updates / horizon | Targets | Cache | Validation BPB | Test BPB |
|---|---:|---:|---:|---|---:|---:|
| Initial baseline `B0` | 7506 | 1200 / 1200 | 9,830,400 | off | pending | not run |
| Matched baseline `B1` | 7506 | 1200 / 4800 | 9,830,400 | off | pending | not run |
| Proposed same-target | 7506 | 1200 / 4800 | 9,830,400 | off | pending | not run |
| Proposed longer, ablation | 7506 | 3000 / 4800 | 24,576,000 | off | pending | not run |
| Frozen final predictor | 7506 | 3000 / 4800 | 24,576,000 | 255/13/0.065 | pending | one final run only |

