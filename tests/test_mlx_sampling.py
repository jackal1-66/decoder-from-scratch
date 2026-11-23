"""The properties checked in test_sampling.py, for MLX, plus PRNG keys and argument checks."""

import mlx.core as mx
import numpy as np
import pytest

from decoder.config import Config
from decoder_mlx.model import Decoder
from decoder_mlx.sampling import filter_logits, generate, sample_next


def kept(
    logits: mx.array, top_k: int | None = None, top_p: float | None = None
) -> list[list[bool]]:
    return np.isfinite(np.array(filter_logits(logits, top_k, top_p))).tolist()


def test_greedy_is_argmax() -> None:
    logits = mx.array([[0.1, 2.0, -1.0], [3.0, 0.0, 2.9]])
    assert sample_next(logits, temperature=0.0).tolist() == [1, 0]
    assert sample_next(logits, top_k=1, key=mx.random.key(0)).tolist() == [1, 0]


def test_top_k_keeps_k_largest() -> None:
    assert kept(mx.array([[1.0, 4.0, 3.0, 2.0]]), top_k=2) == [[False, True, True, False]]


def test_top_p_keeps_smallest_nucleus() -> None:
    logits = mx.log(mx.array([[0.5, 0.3, 0.15, 0.05]]))
    assert kept(logits, top_p=0.5) == [[True, False, False, False]]
    assert kept(logits, top_p=0.7) == [[True, True, False, False]]
    assert kept(logits, top_p=0.9) == [[True, True, True, False]]
    assert kept(logits, top_p=1e-9) == [[True, False, False, False]]  # top token survives


def test_top_p_restores_the_original_token_order() -> None:
    logits = mx.log(mx.array([[0.05, 0.3, 0.5, 0.15]]))
    assert kept(logits, top_p=0.7) == [[False, True, True, False]]


def test_sampling_follows_the_distribution() -> None:
    probs = mx.array([0.6, 0.3, 0.1])
    logits = mx.broadcast_to(mx.log(probs), (20_000, 3))
    draws = np.array(sample_next(logits, key=mx.random.key(0)))
    freq = np.bincount(draws, minlength=3) / 20_000
    np.testing.assert_allclose(freq, np.array(probs), atol=0.015, rtol=0)


def test_low_temperature_concentrates_on_the_mode() -> None:
    logits = mx.broadcast_to(mx.array([1.0, 0.5, 0.0]), (5_000, 3))
    cold_key, hot_key = mx.random.split(mx.random.key(0))
    cold = sample_next(logits, temperature=0.1, key=cold_key)
    hot = sample_next(logits, temperature=10.0, key=hot_key)
    assert float(np.array(mx.mean(cold == 0))) > 0.99
    assert float(np.array(mx.mean(hot == 0))) < 0.4


def test_the_same_key_gives_the_same_tokens(mlx_model: Decoder, config: Config) -> None:
    prompt = mx.random.randint(0, config.vocab_size, (2, 5), key=mx.random.key(1))

    def sample(seed: int) -> list[list[int]]:
        out = generate(mlx_model, prompt, 20, temperature=1.0, top_k=20, key=mx.random.key(seed))
        return np.array(out).tolist()

    assert sample(0) == sample(0)
    assert sample(0) != sample(1)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_cached_and_uncached_generation_agree(
    mlx_model: Decoder, config: Config, temperature: float
) -> None:
    prompt = mx.random.randint(0, config.vocab_size, (2, 5), key=mx.random.key(1))
    key = mx.random.key(0)
    with_cache = generate(mlx_model, prompt, 20, temperature, key=key)
    without = generate(mlx_model, prompt, 20, temperature, use_cache=False, key=key)
    assert mx.array_equal(with_cache, without).item()
    assert mx.array_equal(with_cache[:, :5], prompt).item()
    assert with_cache.dtype == prompt.dtype


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "max_new_tokens", "message"),
    [
        (1.0, None, None, 4, "needs a PRNG key"),
        (-1.0, None, None, 4, "temperature"),
        (float("nan"), None, None, 4, "temperature"),
        (0.0, 0, None, 4, "top_k"),
        (0.0, 51, None, 4, "top_k"),
        (0.0, None, 0.0, 4, "top_p"),
        (0.0, None, 1.5, 4, "top_p"),
        (0.0, None, None, -1, "max_new_tokens"),
        (0.0, None, None, 28, "max_seq_len"),
    ],
)
def test_generate_rejects_bad_arguments(
    mlx_model: Decoder,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    max_new_tokens: int,
    message: str,
) -> None:
    prompt = mx.zeros((1, 5), dtype=mx.int32)
    with pytest.raises(ValueError, match=message):
        generate(mlx_model, prompt, max_new_tokens, temperature, top_k, top_p)


@pytest.mark.parametrize(
    "prompt",
    [
        mx.zeros((5,), dtype=mx.int32),
        mx.zeros((1, 0), dtype=mx.int32),
        mx.zeros((1, 5)),
        mx.array([[0, 1, 50]]),
        mx.array([[0, -1, 2]]),
    ],
)
def test_generate_rejects_bad_prompts(mlx_model: Decoder, prompt: mx.array) -> None:
    with pytest.raises(ValueError, match="prompt"):
        generate(mlx_model, prompt, 4, temperature=0.0)
