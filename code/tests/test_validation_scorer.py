"""Synthetic tests for the validation-only scoring driver."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import score_validation


class DummyModel:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state

    def parameters(self):
        return iter((torch.zeros(3), torch.zeros(5)))


class ValidationScorerTests(unittest.TestCase):
    def make_checkpoint(self, directory, protocol=None):
        path = Path(directory) / "checkpoint.pt"
        torch.save(
            {
                "protocol": protocol or score_validation.PROTOCOL,
                "implementation": "cache_model",
                "config": {"b": 2, "a": 1},
                "model": {"weight": torch.tensor([1.0])},
                "seed": 7506,
                "train_tokens": 1234,
            },
            path,
        )
        return path

    def test_scores_only_validation_with_official_score_function(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            model = DummyModel()
            validation = (torch.tensor([4, 5]), 17)
            fingerprints = {
                "protocol": score_validation.PROTOCOL,
                "sha256": {
                    "tokenizer.json": "tok",
                    "wikitext_train.txt": "train",
                    "wikitext_validation.txt": "validation",
                },
            }
            scored = {
                "bpb": 1.5,
                "token_ppl": 4.0,
                "nll_nats": 3.0,
                "targets": 1,
                "utf8_bytes": 17,
                "seconds": 0.1,
                "window_nll_nats": [1.0, 2.0],
            }

            with (
                mock.patch.object(
                    score_validation, "setup", return_value=(torch.device("cpu"), "fp32")
                ),
                mock.patch.object(
                    score_validation, "make_model", return_value=(model, "implementation-sha")
                ),
                mock.patch.object(
                    score_validation,
                    "load_development_data",
                    return_value=({"validation": validation}, fingerprints),
                ) as load_development_data,
                mock.patch.object(score_validation, "score", return_value=scored) as score,
                mock.patch.object(score_validation, "sha", side_effect=lambda path: f"sha-{Path(path).name}"),
                mock.patch.object(
                    score_validation,
                    "device_metrics",
                    return_value={"device": "cpu", "device_name": "CPU"},
                ),
            ):
                result, losses = score_validation.evaluate_validation(checkpoint)

            load_development_data.assert_called_once_with(score_validation.ROOT)
            score.assert_called_once_with(model, *validation, torch.device("cpu"), "fp32")
            self.assertEqual(losses, [1.0, 2.0])
            self.assertEqual(result["split"], "validation")
            self.assertEqual(result["parameters"], 8)
            self.assertNotIn("wikitext_test.txt", result["data_sha256"])
            canonical = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
            self.assertEqual(result["config_sha256"], canonical)
            self.assertEqual(model.loaded, {"weight": torch.tensor([1.0])})

    def test_rejects_wrong_protocol_before_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory, protocol="wrong")
            with (
                mock.patch.object(
                    score_validation, "setup", return_value=(torch.device("cpu"), "fp32")
                ),
                mock.patch.object(score_validation, "load_development_data") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "different course protocol"):
                    score_validation.evaluate_validation(checkpoint)
            loader.assert_not_called()

    def test_writes_float64_losses_hash_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "validation.json"
            result = {"nll_nats": 3.0, "bpb": 1.0}
            written = score_validation.write_result_artifacts(output, result, [1.0, 2.0])
            window_output = output.with_suffix(".window-nll.npy")
            values = np.load(window_output)
            self.assertEqual(values.dtype, np.float64)
            self.assertAlmostEqual(values.sum(), 3.0)
            self.assertEqual(written["window_count"], 2)
            self.assertEqual(written["window_nll_sha256"], score_validation.sha(window_output))
            self.assertEqual(json.loads(output.read_text()), written)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                score_validation.write_result_artifacts(output, result, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
