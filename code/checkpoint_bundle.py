"""Create or restore a hashed, GitHub-web-sized checkpoint bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path


FORMAT_VERSION = 1
DEFAULT_CHUNK_BYTES = 20 * 1024 * 1024


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_bundle(checkpoint, output_dir, assets=(), chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Split a checkpoint, hash every part, and write a reconstruction manifest."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    assets = [Path(asset) for asset in assets]
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    with checkpoint.open("rb") as source:
        for index in range(10000):
            content = source.read(chunk_bytes)
            if not content:
                break
            name = f"{checkpoint.name}.part-{index:03d}"
            path = output_dir / name
            path.write_bytes(content)
            parts.append({"name": name, "bytes": len(content), "sha256": _sha256(path)})
    if not parts:
        raise ValueError("Checkpoint is empty.")

    manifest = {
        "format_version": FORMAT_VERSION,
        "checkpoint_name": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint),
        "chunk_bytes": chunk_bytes,
        "parts": parts,
        "required_assets": [
            {"path": str(asset), "bytes": asset.stat().st_size, "sha256": _sha256(asset)}
            for asset in assets
        ],
        "uncompressed_inference_bytes": checkpoint.stat().st_size
        + sum(asset.stat().st_size for asset in assets),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore_bundle(bundle_dir, output):
    """Verify parts, reconstruct the checkpoint atomically, and verify its SHA."""
    bundle_dir = Path(bundle_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint-bundle format.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            for part in manifest["parts"]:
                path = bundle_dir / part["name"]
                if path.stat().st_size != part["bytes"] or _sha256(path) != part["sha256"]:
                    raise ValueError(f"Checkpoint part failed verification: {part['name']}")
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
        if temporary.stat().st_size != manifest["checkpoint_bytes"]:
            raise ValueError("Reconstructed checkpoint has the wrong size.")
        if _sha256(temporary) != manifest["checkpoint_sha256"]:
            raise ValueError("Reconstructed checkpoint failed SHA-256 verification.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--checkpoint", required=True, type=Path)
    create.add_argument("--output-dir", required=True, type=Path)
    create.add_argument("--asset", action="append", default=[], type=Path)
    create.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)

    restore = commands.add_parser("restore")
    restore.add_argument("--bundle-dir", required=True, type=Path)
    restore.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "create":
        result = create_bundle(args.checkpoint, args.output_dir, args.asset, args.chunk_bytes)
    else:
        result = restore_bundle(args.bundle_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
