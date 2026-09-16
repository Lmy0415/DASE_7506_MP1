"""Development trainer with reproducible interruption and resume support."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, make_model, setup, sha
from dev_data import load_development_data
from evaluate import score


CHECKPOINT_FORMAT = 1
PEAK_LR = .001
MIN_LR_RATIO = .1
WARMUP_STEPS = 100
WEIGHT_DECAY = .1
GRAD_CLIP = 1.


def parse_step_set(value):
    """Parse a comma-separated set of positive global step numbers."""
    if not value:
        return set()
    try:
        steps = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Step lists must contain integers.") from exc
    if any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError("Saved/evaluated steps must be positive.")
    return steps


def learning_rate_at(step, schedule_steps, peak_lr=PEAK_LR,
                     min_lr_ratio=MIN_LR_RATIO, warmup_steps=WARMUP_STEPS):
    """Cosine LR at a zero-based global update, independent of process stop."""
    if not 0 <= step < schedule_steps:
        raise ValueError("Global step must lie inside the LR schedule horizon.")
    warmup = min(1., (step + 1) / warmup_steps)
    decay = min_lr_ratio + (1 - min_lr_ratio) * .5 * (
        1 + math.cos(math.pi * step / schedule_steps)
    )
    return peak_lr * warmup * decay


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def atomic_torch_save(payload, path):
    """Write a torch artifact atomically in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_checkpoint(model, implementation, config, seed, train_targets):
    """Small inference artifact; intentionally excludes optimizer/RNG state."""
    return {
        "protocol": PROTOCOL,
        "implementation": implementation,
        "config": config,
        "model": _cpu_copy(model.state_dict()),
        "seed": seed,
        "train_tokens": train_targets,
    }


def resume_checkpoint(model, optimizer, completed_steps, train_targets,
                      batch_rng, signature, history, validation_history,
                      train_seconds, validation_seconds):
    """Complete state needed for an exact continuation of training."""
    return {
        "protocol": PROTOCOL,
        "format_version": CHECKPOINT_FORMAT,
        "signature": signature,
        "model": _cpu_copy(model.state_dict()),
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "completed_steps": completed_steps,
        "train_targets": train_targets,
        "batch_rng_state": batch_rng.get_state().cpu(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state_all": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
        "history": history,
        "validation_history": validation_history,
        "train_seconds": train_seconds,
        "intermediate_validation_seconds": validation_seconds,
    }


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(state, model, optimizer, batch_rng, signature, device):
    """Validate and restore a complete resume checkpoint."""
    if state.get("protocol") != PROTOCOL or state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("Resume checkpoint belongs to a different protocol/format.")
    if state.get("signature") != signature:
        raise ValueError("Resume checkpoint is incompatible with this training recipe.")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    _optimizer_to(optimizer, device)
    batch_rng.set_state(state["batch_rng_state"])
    # Restore global RNG last so reconstruction cannot perturb stochastic layers.
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda":
        saved = state.get("cuda_rng_state_all", [])
        if len(saved) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state does not match the available devices.")
        torch.cuda.set_rng_state_all(saved)
    return {
        "completed_steps": int(state["completed_steps"]),
        "train_targets": int(state["train_targets"]),
        "history": list(state.get("history", [])),
        "validation_history": list(state.get("validation_history", [])),
        "train_seconds": float(state.get("train_seconds", 0.)),
        "intermediate_validation_seconds": float(
            state.get("intermediate_validation_seconds", 0.)
        ),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="student")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/baseline-s7506")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--steps", type=int, default=1200,
                        help="Actual global optimizer step at which this process stops.")
    parser.add_argument("--schedule-steps", type=int,
                        help="Cosine LR horizon; defaults to --steps.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-steps", type=parse_step_set, default=set(),
                        help="Comma-separated global steps for model and resume snapshots.")
    parser.add_argument("--eval-steps", type=parse_step_set, default=set(),
                        help="Comma-separated global validation steps.")
    parser.add_argument("--resume-from", type=Path,
                        help="Complete .resume.pt checkpoint to continue in a new run directory.")
    return parser.parse_args(), parser


def main():
    total_started = time.perf_counter()
    args, parser = _arguments()
    schedule_steps = args.schedule_steps or args.steps
    if args.steps < 1 or args.batch_size < 1 or schedule_steps < 1:
        parser.error("Batch size and step counts must be positive.")
    if args.steps > schedule_steps:
        parser.error("--steps cannot exceed --schedule-steps.")
    if any(step > args.steps for step in args.save_steps | args.eval_steps):
        parser.error("Saved/evaluated steps cannot exceed --steps.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Run directory already contains results. Use a new --run-dir.")

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data, data_fingerprints = load_development_data(ROOT)
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY
    )
    tokens = data["train"][0].to(device)
    batch_rng = torch.Generator().manual_seed(args.seed)
    signature = {
        "protocol": PROTOCOL,
        "implementation": args.implementation,
        "implementation_sha256": implementation_sha,
        "config": config,
        "data": data_fingerprints,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "context": 256,
        "precision": precision,
        "device_type": device.type,
        "threads": args.threads,
        "schedule": "cosine",
        "schedule_steps": schedule_steps,
        "peak_lr": PEAK_LR,
        "min_lr_ratio": MIN_LR_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
    }
    state = {
        "completed_steps": 0,
        "train_targets": 0,
        "history": [],
        "validation_history": [],
        "train_seconds": 0.,
        "intermediate_validation_seconds": 0.,
    }
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        state = restore_training_state(
            saved, model, optimizer, batch_rng, signature, device
        )
    if state["completed_steps"] > args.steps:
        parser.error("Resume checkpoint is already beyond --steps.")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    session_validation_seconds = 0.

    def cumulative_train_seconds():
        return (
            state["train_seconds"] + time.perf_counter() - started
            - session_validation_seconds
        )

    for step in range(state["completed_steps"], args.steps):
        starts = torch.randint(
            len(tokens) - 257, (args.batch_size,), generator=batch_rng
        ).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        learning_rate = learning_rate_at(step, schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(
                model(batch[:, :-1]).flatten(0, 1).float(),
                batch[:, 1:].flatten(),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        completed = step + 1
        state["train_targets"] += batch[:, 1:].numel()

        if completed % 100 == 0 or completed == args.steps:
            row = {
                "step": completed,
                "loss": loss.item(),
                "learning_rate": learning_rate,
                "seconds": cumulative_train_seconds(),
            }
            state["history"].append(row)
            print(json.dumps(row), flush=True)

        if completed in args.eval_steps:
            intermediate = score(model, *data["validation"], device, "fp32")
            intermediate.pop("window_nll_nats")
            session_validation_seconds += intermediate["seconds"]
            state["validation_history"].append({"step": completed, **intermediate})
            print(json.dumps({"validation": state["validation_history"][-1]}), flush=True)

        if completed in args.save_steps:
            train_seconds_now = cumulative_train_seconds()
            atomic_torch_save(
                model_checkpoint(
                    model, args.implementation, config, args.seed,
                    state["train_targets"],
                ),
                args.run_dir / f"checkpoint-step-{completed}.pt",
            )
            atomic_torch_save(
                resume_checkpoint(
                    model, optimizer, completed, state["train_targets"],
                    batch_rng, signature, state["history"],
                    state["validation_history"], train_seconds_now,
                    state["intermediate_validation_seconds"]
                    + session_validation_seconds,
                ),
                args.run_dir / f"checkpoint-step-{completed}.resume.pt",
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = cumulative_train_seconds()
    state["completed_steps"] = args.steps
    state["train_seconds"] = train_seconds
    state["intermediate_validation_seconds"] += session_validation_seconds

    if state["validation_history"] and state["validation_history"][-1]["step"] == args.steps:
        validation = {
            key: value for key, value in state["validation_history"][-1].items()
            if key != "step"
        }
    else:
        validation = score(model, *data["validation"], device, "fp32")
        validation.pop("window_nll_nats")

    checkpoint = args.run_dir / "checkpoint.pt"
    atomic_torch_save(
        model_checkpoint(
            model, args.implementation, config, args.seed, state["train_targets"]
        ),
        checkpoint,
    )
    result = {
        "protocol": PROTOCOL,
        "implementation": args.implementation,
        "config": config,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": precision,
        "completed_steps": args.steps,
        "schedule_steps": schedule_steps,
        "train_tokens": state["train_targets"],
        "preparation_seconds": preparation_seconds,
        "train_seconds": train_seconds,
        "validation": validation,
        "history": state["history"],
        "validation_history": state["validation_history"],
        "intermediate_validation_seconds": state["intermediate_validation_seconds"],
        "process_seconds": time.perf_counter() - total_started,
        "torch_version": str(torch.__version__),
        "threads": args.threads,
        "checkpoint_sha256": sha(checkpoint),
        "implementation_sha256": implementation_sha,
        **device_metrics(device),
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result | {"history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
