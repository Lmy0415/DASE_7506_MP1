"""Final MP1 model factory.

Training uses the ordinary GPT forward pass.  During evaluation, the model can
mix its neural distribution with a strictly causal continuous cache constructed
only from earlier positions in the current independent evaluation window.
"""
from cache_model import ContinuousCacheGPT


def build_model(config):
    return ContinuousCacheGPT(config)
