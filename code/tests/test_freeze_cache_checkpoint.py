import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from freeze_cache_checkpoint import freeze_checkpoint


class FreezeCacheCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.pt"
        self.state = {
            "token.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "norm.weight": torch.tensor([1.0, 2.0, 3.0]),
        }
        torch.save({
            "protocol": "7506-mp1-wt2-v2",
            "implementation": "student",
            "config": {
                "vocab": 2048,
                "width": 320,
                "heads": 10,
                "depth": 8,
                "context": 256,
            },
            "model": self.state,
            "seed": 7506,
            "train_tokens": 24_576_000,
            "optimizer": {"must_not_be_copied": True},
            "history": ["must_not_be_copied"],
        }, self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def source_sha(self):
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def assert_common_frozen_fields(self, checkpoint):
        self.assertEqual(checkpoint["protocol"], "7506-mp1-wt2-v2")
        self.assertEqual(checkpoint["implementation"], "cache_model")
        self.assertEqual(checkpoint["seed"], 7506)
        self.assertEqual(checkpoint["train_tokens"], 24_576_000)
        self.assertEqual(checkpoint["source_checkpoint_sha256"], self.source_sha())
        self.assertNotIn("optimizer", checkpoint)
        self.assertNotIn("history", checkpoint)
        self.assertEqual(set(checkpoint["model"]), set(self.state))
        for name, tensor in self.state.items():
            torch.testing.assert_close(checkpoint["model"][name], tensor,
                                       rtol=0, atol=0)

    def test_freezes_cache_on_without_changing_model_weights(self):
        output = self.root / "cache-on.pt"
        freeze_checkpoint(self.source, output, "on")
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        self.assert_common_frozen_fields(checkpoint)
        self.assertTrue(checkpoint["config"]["cache_enabled"])
        self.assertEqual(checkpoint["config"]["cache_window"], 255)
        self.assertEqual(checkpoint["config"]["cache_theta"], 13.0)
        self.assertEqual(checkpoint["config"]["cache_lambda"], 0.065)

    def test_freezes_cache_off_without_changing_model_weights(self):
        output = self.root / "cache-off.pt"
        freeze_checkpoint(self.source, output, "off")
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        self.assert_common_frozen_fields(checkpoint)
        self.assertFalse(checkpoint["config"]["cache_enabled"])
        self.assertEqual(checkpoint["config"]["cache_window"], 255)
        self.assertEqual(checkpoint["config"]["cache_theta"], 13.0)
        self.assertEqual(checkpoint["config"]["cache_lambda"], 0.0)

    def test_refuses_to_overwrite_output(self):
        output = self.root / "existing.pt"
        output.write_bytes(b"do not replace")
        with self.assertRaises(FileExistsError):
            freeze_checkpoint(self.source, output, "on")
        self.assertEqual(output.read_bytes(), b"do not replace")


if __name__ == "__main__":
    unittest.main()
