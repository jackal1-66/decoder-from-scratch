"""Greedy, temperature, top-k and top-p sampling and generation, as in ``decoder.sampling``.

Randomness comes only from explicit PRNG keys (``mx.random.key(seed)``), never from MLX's
global generator: the same key gives the same tokens. ``generate`` splits its key into
one subkey per new token.
"""

import mlx.core as mx

from decoder_mlx.model import Decoder, KVCache


def _check_filters(top_k: int | None, top_p: float | None, vocab_size: int) -> None:
    if top_k is not None and not 1 <= top_k <= vocab_size:
        raise ValueError(f"top_k must be in [1, {vocab_size}], got {top_k}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")


def _check_sampling(
    temperature: float, top_k: int | None, top_p: float | None, key: mx.array | None, vocab: int
) -> None:
    if not temperature >= 0.0:  # also rejects NaN
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    _check_filters(top_k, top_p, vocab)
    if temperature > 0.0 and key is None:
        raise ValueError("sampling with temperature > 0 needs a PRNG key: mx.random.key(seed)")


def filter_logits(
    logits: mx.array, top_k: int | None = None, top_p: float | None = None
) -> mx.array:
    """Set excluded tokens' logits to -inf. logits: (B, V)."""
    _check_filters(top_k, top_p, logits.shape[-1])
    if top_k is not None:
        # mx.topk returns the k largest values unordered, so the k-th is their minimum.
        kth = mx.min(mx.topk(logits, top_k, axis=-1), axis=-1, keepdims=True)
        logits = mx.where(logits < kth, -mx.inf, logits)
    if top_p is not None:
        order = mx.argsort(-logits, axis=-1)  # descending
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        # Drop a token if the tokens before it already reach p; the most likely token is
        # therefore always kept.
        mass_before = mx.cumsum(probs, axis=-1) - probs
        sorted_logits = mx.where(mass_before >= top_p, -mx.inf, sorted_logits)
        logits = mx.put_along_axis(logits, order, sorted_logits, axis=-1)
    return logits


def sample_next(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    key: mx.array | None = None,
) -> mx.array:
    """One token per row of logits (B, V) -> (B,) uint32. ``key`` is needed if temperature > 0."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (B, V), got {logits.shape}")
    _check_sampling(temperature, top_k, top_p, key, logits.shape[-1])
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    filtered = filter_logits(logits.astype(mx.float32) / temperature, top_k, top_p)
    return mx.random.categorical(filtered, axis=-1, key=key)


def generate(
    model: Decoder,
    prompt: mx.array,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    key: mx.array | None = None,
) -> mx.array:
    """Extend prompt (B, T0) of token ids by max_new_tokens tokens; returns (B, T0 + new).

    Puts the model in eval mode. With the cache: one forward pass over the prompt, then
    one pass per new token over that token alone; without: a full pass over the whole
    sequence for every token. Both give the same tokens (``tests/test_mlx_sampling.py``).
    """
    config = model.config
    if prompt.ndim != 2 or prompt.shape[1] == 0 or not mx.issubdtype(prompt.dtype, mx.integer):
        raise ValueError(
            f"prompt must be a non-empty integer array (B, T0), got {prompt.dtype} {prompt.shape}"
        )
    if mx.any((prompt < 0) | (prompt >= config.vocab_size)).item():
        raise ValueError(f"prompt has token ids outside [0, {config.vocab_size})")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if prompt.shape[1] + max_new_tokens > config.max_seq_len:
        raise ValueError(
            f"prompt ({prompt.shape[1]}) + new tokens ({max_new_tokens}) exceed "
            f"max_seq_len ({config.max_seq_len})"
        )
    _check_sampling(temperature, top_k, top_p, key, config.vocab_size)

    model.eval()
    keys = mx.random.split(key, max_new_tokens) if key is not None else None

    def next_token(logits: mx.array, i: int) -> mx.array:
        step_key = keys[i] if keys is not None else None
        nxt = sample_next(logits, temperature, top_k, top_p, step_key)
        return nxt.astype(prompt.dtype)[:, None]

    # mx.eval every step keeps the lazy graph, and with it memory, from growing.
    tokens = prompt
    if use_cache:
        cache = KVCache(config, prompt.shape[0], model.embed.weight.dtype)
        logits = model(prompt, start=0, cache=cache)[:, -1]
        for i in range(max_new_tokens):
            nxt = next_token(logits, i)
            tokens = mx.concatenate([tokens, nxt], axis=1)
            logits = model(nxt, start=tokens.shape[1] - 1, cache=cache)[:, -1]
            mx.eval(tokens, logits)
    else:
        for i in range(max_new_tokens):
            tokens = mx.concatenate([tokens, next_token(model(tokens)[:, -1], i)], axis=1)
            mx.eval(tokens)
    return tokens
