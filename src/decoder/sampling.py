"""Turning logits into text: greedy decoding, temperature, top-k, top-p, and the KV cache.

Sampling controls
-----------------
The model outputs logits z; probabilities are softmax(z / τ).

* **Temperature τ.** τ -> 0 is greedy (always the most likely token); τ = 1 samples the
  model's own distribution; τ > 1 flattens it. Physically: a Boltzmann distribution with
  energies -z at temperature τ.
* **Top-k** keeps only the k most likely tokens, cutting the long tail of individually
  unlikely tokens that together carry real probability mass (and produce nonsense).
* **Top-p (nucleus)** keeps the smallest set of tokens whose probabilities sum to >= p.
  Unlike top-k it adapts: few candidates when the model is confident, many when not.

Filters are applied in that order: temperature, top-k, top-p, then renormalise and draw.
"""

import torch
from torch import Tensor

from decoder.model import Decoder, KVCache


def filter_logits(logits: Tensor, top_k: int | None = None, top_p: float | None = None) -> Tensor:
    """Set excluded tokens' logits to -inf. logits: (B, V)."""
    if top_k is not None:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Drop a token if the tokens *before* it already reach p; the most likely token
        # is therefore always kept.
        mass_before = probs.cumsum(dim=-1) - probs
        sorted_logits = sorted_logits.masked_fill(mass_before >= top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """One token per row of logits (B, V) -> (B,)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(filter_logits(logits.float() / temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: Decoder,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Extend prompt (B, T0) by max_new_tokens tokens.

    With the cache: one forward pass over the prompt ("prefill"), then one pass per new
    token over that token alone ("decode"). Without: a full pass over the whole sequence
    for every token. Both give the same logits; ``tests/test_sampling.py`` checks it.
    """
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt + new tokens exceed max_seq_len")
    model.eval()
    tokens = prompt
    if use_cache:
        cache = KVCache(model.config, prompt.shape[0], prompt.device, model.embed.weight.dtype)
        logits = model(prompt, start=0, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
            nxt = sample_next(logits, temperature, top_k, top_p, generator)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            logits = model(nxt[:, None], start=tokens.shape[1] - 1, cache=cache)[:, -1]
    else:
        for _ in range(max_new_tokens):
            logits = model(tokens)[:, -1]
            nxt = sample_next(logits, temperature, top_k, top_p, generator)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
    return tokens
