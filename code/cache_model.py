"""GPT with an optional causal, within-window continuous neural cache."""
import torch
from torch.nn import functional as F

from model import GPT


class ContinuousCacheGPT(GPT):
    """Use hidden-state similarity to recall next tokens seen earlier in a window.

    ``forward`` is inherited unchanged from ``GPT`` so training remains ordinary
    next-token maximum likelihood.  The cache exists only inside one call to
    ``predict_log_probs`` and is discarded when that call returns.
    """

    def __init__(self, config):
        super().__init__(config)
        self.cache_enabled = bool(config.get("cache_enabled", True))
        self.cache_lambda = float(config.get("cache_lambda", 0.065))
        self.cache_theta = float(config.get("cache_theta", 13.0))
        self.cache_window = int(config.get("cache_window", self.context - 1))
        if not 0.0 <= self.cache_lambda < 1.0:
            raise ValueError("cache_lambda must be in [0, 1).")
        if self.cache_theta <= 0.0:
            raise ValueError("cache_theta must be positive.")
        if not 1 <= self.cache_window <= self.context - 1:
            raise ValueError("cache_window must be in [1, context - 1].")

    def cache_probabilities(self, hidden, ids, theta=None):
        """Return cache distributions using only memory positions before queries."""
        batch, length, _ = hidden.shape
        theta = self.cache_theta if theta is None else float(theta)
        if theta <= 0.0:
            raise ValueError("theta must be positive.")

        normalized = F.normalize(hidden.float(), dim=-1)
        scores = theta * (normalized @ normalized.transpose(-1, -2))

        query = torch.arange(length, device=ids.device)[:, None]
        memory = torch.arange(length, device=ids.device)[None, :]
        allowed = (memory < query) & (memory >= query - self.cache_window)

        weights = torch.zeros_like(scores)
        if length > 1:
            weights[:, 1:] = F.softmax(
                scores[:, 1:].masked_fill(~allowed[1:], -torch.inf), dim=-1
            )

        # Memory position m stores the observed successor ids[m + 1].  Since
        # m < query, that successor is never later than the current input token.
        labels = torch.cat([ids[:, 1:], ids.new_zeros(batch, 1)], dim=1)
        cache = hidden.new_zeros(
            batch, length, self.config["vocab"], dtype=torch.float32
        )
        cache.scatter_add_(
            -1, labels[:, None, :].expand(batch, length, length), weights
        )
        return cache

    def predict_components(self, ids, theta=None):
        hidden = self.features(ids)
        neural_logp = F.log_softmax(self.head(hidden).float(), dim=-1)
        cache_probs = self.cache_probabilities(hidden, ids, theta)
        return neural_logp, cache_probs

    def predict_log_probs(self, ids):
        hidden = self.features(ids)
        neural_logp = F.log_softmax(self.head(hidden).float(), dim=-1)
        if not self.cache_enabled or self.cache_lambda == 0.0:
            return neural_logp

        cache_probs = self.cache_probabilities(hidden, ids)
        neural_probs = neural_logp.exp()
        mixed = (
            (1.0 - self.cache_lambda) * neural_probs
            + self.cache_lambda * cache_probs
        )
        # No earlier memory exists at the first position.
        mixed[:, 0] = neural_probs[:, 0]
        return mixed.clamp_min(torch.finfo(torch.float32).tiny).log()


def build_model(config):
    return ContinuousCacheGPT(config)
