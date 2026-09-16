import unittest

import torch
from torch.nn import functional as F

from cache_model import build_model


class CacheContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7506)
        self.model = build_model(
            dict(
                vocab=2048,
                width=32,
                heads=4,
                depth=2,
                context=256,
                cache_enabled=True,
                cache_lambda=0.065,
                cache_theta=13.0,
                cache_window=255,
            )
        ).eval()

    def test_defaults_match_frozen_method(self):
        torch.manual_seed(7506)
        model = build_model(
            dict(vocab=2048, width=32, heads=4, depth=1, context=256)
        )
        self.assertTrue(model.cache_enabled)
        self.assertEqual(model.cache_window, 255)
        self.assertEqual(model.cache_theta, 13.0)
        self.assertEqual(model.cache_lambda, 0.065)

    def test_cache_is_causal_and_normalized(self):
        x = torch.randint(0, 2048, (2, 12))
        changed = x.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            original = self.model.predict_log_probs(x)
            future_changed = self.model.predict_log_probs(changed)
        torch.testing.assert_close(
            original[:, :7], future_changed[:, :7], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            original.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6
        )

    def test_examples_are_independent_and_state_resets(self):
        x = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            together = self.model.predict_log_probs(x)
            alone = self.model.predict_log_probs(x[:1])
            self.model.predict_log_probs((x + 31) % 2048)
            again = self.model.predict_log_probs(x)
        torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(together, again, atol=1e-6, rtol=1e-6)

    def test_cache_off_is_plain_neural_distribution(self):
        x = torch.randint(0, 2048, (2, 12))
        self.model.cache_enabled = False
        with torch.no_grad():
            actual = self.model.predict_log_probs(x)
            expected = F.log_softmax(self.model(x).float(), dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_training_forward_does_not_depend_on_cache_switch(self):
        x = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            cache_on_logits = self.model(x)
            self.model.cache_enabled = False
            cache_off_logits = self.model(x)
        torch.testing.assert_close(
            cache_on_logits, cache_off_logits, atol=0.0, rtol=0.0
        )


if __name__ == "__main__":
    unittest.main()
