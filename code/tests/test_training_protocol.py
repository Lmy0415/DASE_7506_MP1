"""Synthetic/development-only tests for the formal training protocol."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

import dev_data
from common import ROOT
from train import (
    atomic_torch_save,
    learning_rate_at,
    model_checkpoint,
    restore_training_state,
    resume_checkpoint,
)


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_actual_stop_does_not_change_schedule_prefix(self):
        short = [learning_rate_at(step, 4800) for step in range(3000)]
        prefix_of_full = [learning_rate_at(step, 4800) for step in range(3000)]
        self.assertEqual(short, prefix_of_full)
        self.assertNotEqual(learning_rate_at(2999, 4800), learning_rate_at(2999, 3000))

    def test_formal_target_counts(self):
        targets_per_step = 32 * 256
        self.assertEqual(1200 * targets_per_step, 9_830_400)
        self.assertEqual(3000 * targets_per_step, 24_576_000)

    def test_development_loader_never_opens_test_text(self):
        opened = []
        original = Path.read_bytes

        def guarded(path):
            opened.append(Path(path).name)
            if Path(path).name == "wikitext_test.txt":
                raise AssertionError("Development loader touched the test split.")
            return original(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            data, fingerprints = dev_data.load_development_data(ROOT)
        self.assertEqual(set(data), {"train", "validation"})
        self.assertNotIn("wikitext_test.txt", opened)
        self.assertNotIn("wikitext_test.txt", fingerprints["sha256"])

    def test_resume_is_bit_exact_on_synthetic_training(self):
        signature = {
            "protocol": "synthetic",
            "seed": 7506,
            "schedule_steps": 8,
            "batch_size": 4,
        }

        def construct():
            torch.manual_seed(7506)
            model = nn.Sequential(
                nn.Linear(5, 7), nn.GELU(), nn.Dropout(.2), nn.Linear(7, 3)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=.1
            )
            batch_rng = torch.Generator().manual_seed(7506)
            return model, optimizer, batch_rng

        def update(model, optimizer, batch_rng, step):
            x = torch.randn(4, 5, generator=batch_rng)
            y = torch.randint(0, 3, (4,), generator=batch_rng)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, 8)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            return y.numel()

        full_model, full_optimizer, full_rng = construct()
        full_targets = sum(
            update(full_model, full_optimizer, full_rng, step) for step in range(8)
        )

        partial_model, partial_optimizer, partial_rng = construct()
        partial_targets = sum(
            update(partial_model, partial_optimizer, partial_rng, step)
            for step in range(3)
        )
        payload = resume_checkpoint(
            partial_model, partial_optimizer, 3, partial_targets, partial_rng,
            signature, [], [], 0., 0.,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.resume.pt"
            atomic_torch_save(payload, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)

        resumed_model, resumed_optimizer, resumed_rng = construct()
        restored = restore_training_state(
            saved, resumed_model, resumed_optimizer, resumed_rng, signature,
            torch.device("cpu"),
        )
        resumed_targets = restored["train_targets"]
        for step in range(restored["completed_steps"], 8):
            resumed_targets += update(
                resumed_model, resumed_optimizer, resumed_rng, step
            )

        self.assertEqual(full_targets, resumed_targets)
        for expected, actual in zip(full_model.parameters(), resumed_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        expected_state = full_optimizer.state_dict()["state"]
        actual_state = resumed_optimizer.state_dict()["state"]
        self.assertEqual(expected_state.keys(), actual_state.keys())
        for key in expected_state:
            for name, expected in expected_state[key].items():
                actual = actual_state[key][name]
                if torch.is_tensor(expected):
                    self.assertTrue(torch.equal(expected, actual))
                else:
                    self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(full_rng.get_state(), resumed_rng.get_state()))

    def test_resume_rejects_changed_recipe_and_model_checkpoint_is_small_state(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        rng = torch.Generator().manual_seed(1)
        signature = {"seed": 7506, "schedule_steps": 4800}
        state = resume_checkpoint(
            model, optimizer, 0, 0, rng, signature, [], [], 0., 0.
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_training_state(
                state, model, optimizer, rng,
                {"seed": 17, "schedule_steps": 4800}, torch.device("cpu"),
            )
        inference = model_checkpoint(model, "student", {"width": 3}, 7506, 0)
        self.assertNotIn("optimizer", inference)
        self.assertNotIn("torch_rng_state", inference)
        self.assertIn("optimizer", state)
        self.assertIn("torch_rng_state", state)


if __name__ == "__main__":
    unittest.main()
