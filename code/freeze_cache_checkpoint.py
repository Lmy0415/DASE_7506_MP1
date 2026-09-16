"""Freeze a trained GPT checkpoint into a cache-on or cache-off predictor.

This utility transforms checkpoint metadata only. It does not load any dataset,
run inference, or alter the trained model state.
"""
import argparse
from pathlib import Path

import torch

from common import sha


CACHE_WINDOW = 255
CACHE_THETA = 13.0
CACHE_LAMBDA = 0.065
REQUIRED_KEYS = ("protocol", "config", "model", "seed", "train_tokens")


def freeze_checkpoint(source: Path, output: Path, mode: str) -> dict:
    """Create a model-only inference checkpoint with frozen cache settings."""
    source = Path(source)
    output = Path(output)
    if mode not in {"on", "off"}:
        raise ValueError("mode must be 'on' or 'off'.")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output checkpoints must be different files.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"Source checkpoint is missing required keys: {missing}")
    if not isinstance(checkpoint["config"], dict):
        raise ValueError("Source checkpoint config must be a dictionary.")
    if not isinstance(checkpoint["model"], dict):
        raise ValueError("Source checkpoint model must be a state dictionary.")

    config = dict(checkpoint["config"])
    config.update(
        cache_enabled=mode == "on",
        cache_window=CACHE_WINDOW,
        cache_theta=CACHE_THETA,
        cache_lambda=CACHE_LAMBDA if mode == "on" else 0.0,
    )
    frozen = {
        "protocol": checkpoint["protocol"],
        "implementation": "cache_model",
        "config": config,
        "model": checkpoint["model"],
        "seed": checkpoint["seed"],
        "train_tokens": checkpoint["train_tokens"],
        "source_checkpoint_sha256": sha(source),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frozen, output)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Model-only checkpoint produced by training.")
    parser.add_argument("--output", required=True, type=Path,
                        help="New cache inference checkpoint; must not exist.")
    parser.add_argument("--mode", required=True, choices=("on", "off"))
    args = parser.parse_args()
    frozen = freeze_checkpoint(args.source, args.output, args.mode)
    print({
        "output": str(args.output),
        "mode": args.mode,
        "source_checkpoint_sha256": frozen["source_checkpoint_sha256"],
        "train_tokens": frozen["train_tokens"],
    })


if __name__ == "__main__":
    main()
