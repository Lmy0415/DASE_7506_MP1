"""Score one checkpoint on validation without opening the test split.

The numerical scoring loop is imported unchanged from ``evaluate.py``.  This
development-only entry point differs only in data access: ``dev_data`` verifies
and loads the supplied tokenizer, training text, and validation text, and has no
API for the test split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL, ROOT, device_metrics, make_model, setup, sha
from dev_data import load_development_data
from evaluate import score


def _canonical_config_sha256(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def evaluate_validation(checkpoint_path, device_name="cpu", precision="fp32", threads=4):
    """Return validation metrics and per-window losses for one checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    device, precision = setup(device_name, precision, threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint belongs to a different course protocol.")

    required = ("implementation", "config", "model")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    model, implementation_sha = make_model(
        checkpoint["implementation"], checkpoint["config"], device
    )
    model.load_state_dict(checkpoint["model"])
    data, fingerprints = load_development_data(ROOT)
    if fingerprints["protocol"] != PROTOCOL:
        raise ValueError("Development data belongs to a different course protocol.")

    result = score(model, *data["validation"], device, precision)
    losses = result.pop("window_nll_nats")
    result.update(
        protocol=PROTOCOL,
        split="validation",
        precision=precision,
        checkpoint_sha256=sha(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        source_checkpoint_sha256=checkpoint.get("source_checkpoint_sha256"),
        implementation=checkpoint["implementation"],
        implementation_sha256=implementation_sha,
        config_sha256=_canonical_config_sha256(checkpoint["config"]),
        validation_scorer_sha256=sha(Path(__file__)),
        official_evaluator_sha256=sha(ROOT / "evaluate.py"),
        data_sha256=fingerprints["sha256"],
        seed=checkpoint.get("seed"),
        train_targets=checkpoint.get("train_tokens"),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        **device_metrics(device),
    )
    return result, losses


def write_result_artifacts(output, result, losses):
    """Write immutable JSON and float64 per-window NLL artifacts."""
    output = Path(output)
    window_output = output.with_suffix(".window-nll.npy")
    existing = [path for path in (output, window_output) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite validation artifact: {existing[0]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(losses, dtype=np.float64)
    np.save(window_output, values)
    if not np.isclose(values.sum(), result["nll_nats"], rtol=1e-12, atol=1e-9):
        window_output.unlink()
        raise ValueError("Per-window losses do not sum to aggregate NLL.")
    result = dict(result)
    result.update(
        window_nll_path=window_output.name,
        window_nll_sha256=sha(window_output),
        window_count=int(values.size),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="fp32",
        help="Use FP32 for the formal CPU comparison.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result, losses = evaluate_validation(
        args.checkpoint, args.device, args.precision, args.threads
    )
    result = write_result_artifacts(args.output, result, losses)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
