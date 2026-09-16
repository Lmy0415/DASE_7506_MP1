"""Development-only data loader.

This module deliberately has no test-split option. Training and model selection
must only open the train and validation text files.
"""
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer


DEVELOPMENT_SPLITS = ("train", "validation")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_development_data(root):
    """Load and verify only train and validation data."""
    root = Path(root)
    data_dir = root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    filenames = ["tokenizer.json", *[f"wikitext_{s}.txt" for s in DEVELOPMENT_SPLITS]]
    fingerprints = {}
    for filename in filenames:
        actual = _sha256(data_dir / filename)
        expected = manifest["sha256"][filename]
        if actual != expected:
            raise ValueError(f"Changed benchmark file: {filename}")
        fingerprints[filename] = actual

    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    data = {}
    for split in DEVELOPMENT_SPLITS:
        raw = (data_dir / f"wikitext_{split}.txt").read_bytes()
        ids = tokenizer.encode(raw.decode("utf-8")).ids
        data[split] = (torch.tensor(ids, dtype=torch.long), len(raw))
    return data, {"protocol": manifest["protocol"], "sha256": fingerprints}
