"""The properties checked in test_model.py, for the MLX model, plus fast-kernel equivalence."""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from decoder.config import Config
from decoder_mlx.model import (
    Decoder,
    KVCache,
    RMSNorm,
    apply_rope,
    flat_parameters,
    rope_tables,
)


def random_tokens(config: Config, shape: tuple[int, int], seed: int = 0) -> mx.array:
    return mx.random.randint(0, config.vocab_size, shape, key=mx.random.key(seed))


def assert_close(actual: mx.array, expected: mx.array, atol: float = 1e-5) -> None:
    # rtol as torch.testing.assert_close uses for float32.
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1.3e-6, atol=atol)


def twin_models(config: Config) -> tuple[Decoder, Decoder]:
    """The same weights with fast and with explicit RoPE/attention."""
    mx.random.seed(0)
    fast = Decoder(config, use_fast_kernels=True)
    explicit = Decoder(config, use_fast_kernels=False)
    explicit.update(fast.parameters())
    fast.eval()
    explicit.eval()
    return fast, explicit


def test_output_shape(mlx_model: Decoder, config: Config) -> None:
    assert mlx_model(random_tokens(config, (3, 10))).shape == (3, 10, config.vocab_size)


def test_causality_future_tokens_do_not_change_past_logits(
    mlx_model: Decoder, config: Config
) -> None:
    a = random_tokens(config, (1, 12))
    b = mx.concatenate([a[:, :7], (a[:, 7:] + 1) % config.vocab_size], axis=1)
    la, lb = mlx_model(a), mlx_model(b)
    assert_close(la[:, :7], lb[:, :7])
    assert not mx.allclose(la[:, 7:], lb[:, 7:]).item()


@pytest.mark.parametrize("use_fast_kernels", [True, False])
def test_incremental_decoding_with_cache_matches_full_forward(
    config: Config, use_fast_kernels: bool
) -> None:
    mx.random.seed(0)
    model = Decoder(config, use_fast_kernels)
    model.eval()
    tokens = random_tokens(config, (2, 16))
    full = model(tokens)
    cache = KVCache(config, batch=2)
    prefill = model(tokens[:, :5], start=0, cache=cache)
    steps = [model(tokens[:, t : t + 1], start=t, cache=cache) for t in range(5, 16)]
    assert_close(mx.concatenate([prefill, *steps], axis=1), full)


def test_fast_kernels_match_explicit_rope_and_attention(config: Config) -> None:
    fast, explicit = twin_models(config)
    tokens = random_tokens(config, (2, 16))
    assert_close(fast(tokens), explicit(tokens))
    # With a cache, queries start at an offset: checks RoPE's offset and the mask alignment.
    caches = KVCache(config, batch=2), KVCache(config, batch=2)
    for model, cache in zip((fast, explicit), caches, strict=True):
        model(tokens[:, :9], start=0, cache=cache)
    assert_close(
        fast(tokens[:, 9:12], start=9, cache=caches[0]),
        explicit(tokens[:, 9:12], start=9, cache=caches[1]),
    )


@pytest.mark.parametrize("offset", [0, 1, 17])
def test_fast_rope_is_the_half_split_rotation(offset: int) -> None:
    D, T = 16, 8
    x = mx.random.normal((2, 3, T, D), key=mx.random.key(0))
    cos, sin = rope_tables(D, 64, 10_000.0)
    expected = apply_rope(x, cos[offset : offset + T], sin[offset : offset + T])
    fast = mx.fast.rope(x, D, traditional=False, base=10_000.0, scale=1.0, offset=offset)
    assert_close(fast, expected)
    interleaved = mx.fast.rope(x, D, traditional=True, base=10_000.0, scale=1.0, offset=offset)
    assert offset == 0 or not mx.allclose(interleaved, expected, atol=1e-3).item()


def test_attention_dropout_falls_back_to_the_explicit_path() -> None:
    config = Config(vocab_size=50, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64,
                    max_seq_len=32, dropout=0.5)  # fmt: skip
    fast, explicit = twin_models(config)
    fast.train()
    explicit.train()
    tokens = random_tokens(config, (2, 16))
    # Same seed, same sequence of random draws: identical dropout masks on both paths.
    mx.random.seed(1)
    out_fast = fast(tokens)
    mx.random.seed(1)
    assert_close(out_fast, explicit(tokens))


def test_rope_scores_depend_only_on_relative_position() -> None:
    D = 16
    cos, sin = rope_tables(D, 64, 10_000.0)
    q, k = (
        mx.random.normal((D,), key=mx.random.key(1)),
        mx.random.normal((D,), key=mx.random.key(2)),
    )

    def score(m: int, n: int) -> float:
        qm = apply_rope(q.reshape(1, 1, 1, D), cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k.reshape(1, 1, 1, D), cos[n : n + 1], sin[n : n + 1])
        return float(np.array((qm * kn).sum()))

    assert score(10, 7) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(10, 7) == pytest.approx(score(3, 0), abs=1e-4)
    assert score(10, 7) != pytest.approx(score(10, 2), abs=1e-4)


def test_rope_is_a_rotation_and_preserves_norms() -> None:
    cos, sin = rope_tables(8, 16, 10_000.0)
    x = mx.random.normal((1, 1, 16, 8), key=mx.random.key(0))
    norm = mx.linalg.norm
    assert_close(norm(apply_rope(x, cos, sin), axis=-1), norm(x, axis=-1))


def test_rmsnorm_gives_unit_rms() -> None:
    x = 5 * mx.random.normal((4, 7, 32), key=mx.random.key(0))
    y = RMSNorm(32, eps=1e-6)(x)
    assert_close(mx.sqrt(mx.mean(y * y, axis=-1)), mx.ones((4, 7)), atol=1e-4)


def test_initial_loss_is_close_to_uniform(mlx_model: Decoder, config: Config) -> None:
    tokens = random_tokens(config, (8, 32))
    logits = mlx_model(tokens[:, :-1])
    loss = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="mean")
    assert loss.item() == pytest.approx(math.log(config.vocab_size), rel=0.05)


def test_can_overfit_a_single_batch(config: Config) -> None:
    mx.random.seed(0)
    model = Decoder(config)
    tokens = random_tokens(config, (4, 32))
    optimizer = optim.AdamW(learning_rate=3e-3)

    def loss_fn(model: Decoder, x: mx.array, y: mx.array) -> mx.array:
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(150):
        loss, grads = loss_and_grad(model, tokens[:, :-1], tokens[:, 1:])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    assert loss.item() < 0.1


def test_parameters_are_named_as_in_the_pytorch_state_dict(mlx_model: Decoder) -> None:
    names = set(flat_parameters(mlx_model.parameters()))
    assert {
        "embed.weight",
        "norm.weight",
        "blocks.1.attn.wk.weight",
        "blocks.0.ff.w3.weight",
    } <= names
    assert not any(name.startswith(("head", "_")) for name in names)  # tied head, RoPE tables


def test_inputs_are_validated(mlx_model: Decoder, config: Config) -> None:
    with pytest.raises(ValueError, match="integer"):
        mlx_model(mx.zeros((1, 4)))
    with pytest.raises(ValueError, match=r"shape \(B, T\)"):
        mlx_model(mx.zeros((4,), dtype=mx.int32))
    with pytest.raises(ValueError, match="max_seq_len"):
        mlx_model(mx.zeros((1, 4), dtype=mx.int32), start=config.max_seq_len - 2)
    with pytest.raises(ValueError, match="cache batch"):
        mlx_model(mx.zeros((1, 4), dtype=mx.int32), cache=KVCache(config, batch=2))
