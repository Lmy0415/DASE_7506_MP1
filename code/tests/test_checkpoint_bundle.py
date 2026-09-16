"""Tests for deterministic checkpoint bundle creation and restoration."""
from pathlib import Path
import tempfile
import unittest

from checkpoint_bundle import create_bundle, restore_bundle


class CheckpointBundleTests(unittest.TestCase):
    def test_round_trip_and_asset_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(bytes(range(251)) * 17)
            asset = root / "model.py"
            asset.write_text("MODEL = True\n")
            bundle = root / "bundle"
            manifest = create_bundle(checkpoint, bundle, [asset], chunk_bytes=997)
            restored = root / "restored.pt"
            restored_manifest = restore_bundle(bundle, restored)

            self.assertEqual(restored.read_bytes(), checkpoint.read_bytes())
            self.assertEqual(restored_manifest, manifest)
            self.assertEqual(len(manifest["parts"]), 5)
            self.assertEqual(
                manifest["uncompressed_inference_bytes"],
                checkpoint.stat().st_size + asset.stat().st_size,
            )

    def test_corruption_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"a" * 100)
            bundle = root / "bundle"
            manifest = create_bundle(checkpoint, bundle, chunk_bytes=30)
            (bundle / manifest["parts"][1]["name"]).write_bytes(b"corrupt")
            output = root / "restored.pt"
            with self.assertRaisesRegex(ValueError, "failed verification"):
                restore_bundle(bundle, output)
            self.assertFalse(output.exists())

    def test_refuses_nonempty_bundle_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "keep").write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "non-empty bundle"):
                create_bundle(checkpoint, bundle)

            clean_bundle = root / "clean-bundle"
            create_bundle(checkpoint, clean_bundle)
            output = root / "output.pt"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "overwrite checkpoint"):
                restore_bundle(clean_bundle, output)


if __name__ == "__main__":
    unittest.main()
